import copy
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import List, Mapping, Optional

import numpy as np

from tensorrt_llm.scaffolding import (Controller, NativeGenerationController,
                                      ParallelProcess, StreamGenerationTask,
                                      Task, extract_answer_from_boxed)


@dataclass
class ConfidenceInfo:
    conf_grouped: float = 0.0
    conf_group_size: int = 128
    conf_threshold: float = 0.0
    conf_list: list[float] = field(default_factory=list)
    conf_group_list: deque[float] = field(default_factory=deque)
    avg_conf_group_list: list[float] = field(default_factory=list)

    def get_min_conf_grouped(self):
        if len(self.avg_conf_group_list) == 0:
            print(
                f"Warning: no valid conf group yet, maybe you should decrease conf_group_size"
            )
            return self.conf_threshold
        return min(self.avg_conf_group_list)

    def should_stop(self):
        return self.avg_conf_group_list[-1] < self.conf_threshold if len(
            self.conf_list) >= self.conf_group_size else False

    def get_statistics(self,
                       tail_tokens: int = 2048,
                       bottom_percent: float = 0.1):
        # global mean confidence
        self.mean_conf = np.mean(self.conf_list)

        # tail mean confidence
        tail_conf_list = self.conf_list[-tail_tokens:] if len(
            self.conf_list) > tail_tokens else self.conf_list
        self.tail_mean_conf = np.mean(tail_conf_list)

        # bottom window mean confidence and min window confidence
        if len(self.avg_conf_group_list) == 0:
            self.bottom_window_mean_conf = np.mean(self.conf_list)
            self.min_window_conf = self.bottom_window_mean_conf
        else:
            num_bottom = max(
                1, int(len(self.avg_conf_group_list) * bottom_percent))
            if num_bottom == 1:
                self.bottom_window_mean_conf = np.min(self.avg_conf_group_list)
            else:
                self.bottom_window_mean_conf = np.mean(
                    np.partition(self.avg_conf_group_list,
                                 num_bottom - 1)[:num_bottom])
            self.min_window_conf = np.min(self.avg_conf_group_list)

    def update_confidence_info(self, token_dict: Mapping[int, 'Logprob'],
                               token_id: int):
        mean_logprob = np.mean(
            [logprob_obj.logprob for logprob_obj in token_dict.values()])
        new_conf = round(-mean_logprob, 3)
        self.conf_list.append(new_conf)
        self.conf_grouped += new_conf
        self.conf_group_list.append(new_conf)
        if len(self.conf_group_list) > self.conf_group_size:
            self.conf_grouped -= self.conf_group_list.popleft()
        if len(self.conf_group_list) == self.conf_group_size:
            self.avg_conf_group_list.append(self.conf_grouped /
                                            self.conf_group_size)


def basic_majority_vote(tasks: List[Task], **kwargs) -> Task:
    answers = kwargs['answers']
    majority_answer = Counter(answers).most_common(1)[0][0]
    return tasks[answers.index(majority_answer)]


def weighted_majority_vote(type: str, filter_top_percent: float = 1.0):

    def impl(tasks: List[Task], **kwargs) -> Task:
        answers = kwargs['answers']
        if type == 'mean_confidence_weighted':
            confidences = kwargs['mean_confidences']
        elif type == 'tail_confidence_weighted':
            confidences = kwargs['tail_confidences']
        elif type == 'bottom_window_weighted':
            confidences = kwargs['bottom_window_confidences']
        elif type == 'min_window_weighted':
            confidences = kwargs['min_window_confidences']
        else:
            raise ValueError(f"Invalid type: {type}")

        if filter_top_percent < 1:
            num_keep = max(1, int(len(confidences) * filter_top_percent))
            sorted_indices = np.argsort(confidences)[::-1]
            save_indices = sorted_indices[:num_keep].tolist()

            tasks = [tasks[i] for i in save_indices]
            answers = [answers[i] for i in save_indices]
            confidences = [confidences[i] for i in save_indices]

        print(f"{type=} {filter_top_percent=} has {len(tasks)} valid tasks",
              flush=True)

        answer_to_weights = {}
        for answer, confidence in zip(answers, confidences):
            answer_to_weights[answer] = answer_to_weights.get(answer,
                                                              0.0) + confidence
        majority_answer = max(answer_to_weights, key=answer_to_weights.get)
        return tasks[answers.index(majority_answer)]

    return impl


VOTE_POLICY_IMPL = {
    'majority':
    basic_majority_vote,
    "mean_confidence_weighted":
    weighted_majority_vote('mean_confidence_weighted'),
    "tail_confidence_weighted":
    weighted_majority_vote('tail_confidence_weighted'),
    "bottom_window_weighted":
    weighted_majority_vote('bottom_window_weighted'),
    "min_window_weighted":
    weighted_majority_vote('min_window_weighted'),
    "top10_tail_filtered":
    weighted_majority_vote('tail_confidence_weighted', filter_top_percent=0.1),
    "top10_bottom_window_filtered":
    weighted_majority_vote('bottom_window_weighted', filter_top_percent=0.1)
}


def vote(tasks: List[Task]):
    for task in tasks:
        task.costimized_result_fields[
            'extracted_answer'] = extract_answer_from_boxed(task.output_str)
        task.costimized_result_fields['confidence_info'].get_statistics()
    valid_tasks = [
        task for task in tasks
        if task.costimized_result_fields['extracted_answer']
    ]
    if len(valid_tasks) == 0:
        print(
            "Warning: No valid tasks, maybe you should increase max_output_len, a random task will be returned"
        )
        return {
            policy_name: random.choice(tasks)
            for policy_name, policy_impl in VOTE_POLICY_IMPL.items()
        }

    answers = [
        task.costimized_result_fields['extracted_answer']
        for task in valid_tasks
    ]
    confidences = [
        task.costimized_result_fields['confidence_info'] for task in valid_tasks
    ]
    mean_confidences = [conf.mean_conf for conf in confidences]
    tail_confidences = [conf.tail_mean_conf for conf in confidences]
    bottom_window_confidences = [
        conf.bottom_window_mean_conf for conf in confidences
    ]
    min_window_confidences = [conf.min_window_conf for conf in confidences]

    return {
        policy_name:
        policy_impl(valid_tasks,
                    answers=answers,
                    mean_confidences=mean_confidences,
                    tail_confidences=tail_confidences,
                    bottom_window_confidences=bottom_window_confidences,
                    min_window_confidences=min_window_confidences)
        for policy_name, policy_impl in VOTE_POLICY_IMPL.items()
    }


class DeepConfOfflineController(NativeGenerationController):

    def __init__(self,
                 conf_group_size: int,
                 conf_threshold: float,
                 logprobs_topk: int = 20,
                 sampling_params: dict = None,
                 streaming: bool = False,
                 **kwargs):
        super().__init__(sampling_params, streaming)
        self.logprobs_topk = logprobs_topk
        self.confidence_info = ConfidenceInfo(conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepConfOfflineController only supports one task"
        yield from super().process(tasks, **kwargs)
        for logprobs_dict, token_id in zip(tasks[0].logprobs,
                                           tasks[0].output_tokens):
            self.confidence_info.update_confidence_info(logprobs_dict, token_id)
        tasks[0].costimized_result_fields[
            'confidence_info'] = self.confidence_info
        print(f"end process, generated {len(tasks[0].output_tokens)} tokens",
              flush=True)


class DeepConfOfflineMajorityVoteController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self,
                 sample_num: int,
                 conf_group_size: int,
                 conf_threshold: float,
                 vote_policy: str = 'majority',
                 logprobs_topk: int = 20,
                 sampling_params: dict = None,
                 **kwargs):
        super().__init__()
        self.sample_num = sample_num
        self.conf_group_size = conf_group_size
        self.conf_threshold = conf_threshold
        self.vote_policy = vote_policy
        self.logprobs_topk = logprobs_topk
        self.sampling_params = sampling_params

        self.generation_controller = DeepConfOfflineController(
            conf_group_size=conf_group_size,
            conf_threshold=conf_threshold,
            logprobs_topk=logprobs_topk,
            sampling_params=sampling_params)

    def clone(self):
        return DeepConfOfflineMajorityVoteController(
            self.sample_num, self.conf_group_size, self.conf_threshold,
            self.vote_policy, self.logprobs_topk, self.sampling_params)

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepConfMajorityVoteController only supports one task"

        generation_controllers = [
            self.generation_controller.clone() for _ in range(self.sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(self.sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(kwargs) for _ in range(self.sample_num)
        ]
        yield ParallelProcess(generation_controllers, tasks_list,
                              generation_kwargs_list)

        task_list = [ttasks[0] for ttasks in tasks_list]
        tasks[0].result = vote(task_list)[self.vote_policy].result


class DeepConfOnlineController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self,
                 conf_group_size: int,
                 conf_threshold: float,
                 logprobs_topk: int = 20,
                 sampling_params: dict = None):
        super().__init__()
        self.sampling_params = sampling_params
        self.logprobs_topk = logprobs_topk
        self.confidence_info = ConfidenceInfo(conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
        print(f"begin online controller", flush=True)
        assert len(
            tasks) == 1, "DeepThinkOnlineController only supports one task"
        online_task = StreamGenerationTask.create_from_generation_task(tasks[0])
        online_task.streaming_step = self.confidence_info.conf_group_size
        online_task.worker_tag = self.WorkerTag.GENERATION

        for key, value in self.sampling_params.items():
            if getattr(online_task, key) is None:
                setattr(online_task, key, value)

        last_step_index = 0
        while online_task.end_flag == False:
            yield [online_task]
            for i in range(last_step_index, len(online_task.output_tokens)):
                logprobs_dict, token_id = online_task.logprobs[
                    i], online_task.output_tokens[i]
                self.confidence_info.update_confidence_info(
                    logprobs_dict, token_id)
                if self.confidence_info.should_stop():
                    print(
                        f"early stop, {self.confidence_info.avg_conf_group_list[-1]} < {self.confidence_info.conf_threshold}, generated {len(self.confidence_info.conf_list)} tokens",
                        flush=True)
                    online_task.cancel_flag = True
                    yield [online_task]
                    break
            print(
                f"continue to generate, generated {len(online_task.output_tokens)} tokens",
                flush=True)
            last_step_index = len(online_task.output_tokens)
        tasks[0].result = online_task.result
        tasks[0].costimized_result_fields[
            'confidence_info'] = self.confidence_info
        print(
            f"end online controller, generated {len(online_task.output_tokens)} tokens",
            flush=True)


class DeepConfOnlineMajorityVoteController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self,
                 sample_num: int,
                 conf_group_size: int,
                 conf_threshold: float,
                 vote_policy: str = 'majority',
                 warmup_sample_num: int = 5,
                 confidence_percentile: int = 90,
                 logprobs_topk: int = 20,
                 sampling_params: dict = None,
                 **kwargs):
        super().__init__()
        self.sample_num = sample_num
        self.warmup_sample_num = warmup_sample_num
        assert sample_num >= warmup_sample_num, f"{sample_num=} must be greater than {warmup_sample_num=}"
        self.final_sample_num = sample_num - warmup_sample_num
        self.conf_group_size = conf_group_size
        self.conf_threshold = conf_threshold
        self.confidence_percentile = confidence_percentile
        self.vote_policy = vote_policy
        self.logprobs_topk = logprobs_topk
        self.sampling_params = sampling_params

        self.warmup_generation_controller = DeepConfOfflineController(
            conf_group_size=self.conf_group_size,
            conf_threshold=self.conf_threshold,
            logprobs_topk=self.logprobs_topk,
            sampling_params=self.sampling_params)
        self.final_generation_controller = DeepConfOnlineController(
            conf_group_size=self.conf_group_size,
            conf_threshold=self.conf_threshold,
            logprobs_topk=self.logprobs_topk,
            sampling_params=self.sampling_params)

    def clone(self):
        return DeepConfOnlineMajorityVoteController(
            sample_num=self.sample_num,
            conf_group_size=self.conf_group_size,
            conf_threshold=self.conf_threshold,
            vote_policy=self.vote_policy,
            warmup_sample_num=self.warmup_sample_num,
            confidence_percentile=self.confidence_percentile,
            logprobs_topk=self.logprobs_topk,
            sampling_params=self.sampling_params)

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks
        ) == 1, "DeepConfOnlineMajorityVoteController only supports one task"
        # warm up to get conf_threshold
        if self.warmup_sample_num > 0:
            print(f"begin warmup", flush=True)
            warmup_tasks_list = yield from self.parallel_generate(tasks,
                                                                  warmup=True,
                                                                  **kwargs)
            print(f"end warmup", flush=True)

            min_confs = [
                tasks[0].costimized_result_fields['confidence_info'].
                get_min_conf_grouped() for tasks in warmup_tasks_list
            ]
            conf_bar = float(
                np.percentile(min_confs, 100 - self.confidence_percentile))
        else:
            warmup_tasks_list = []
            conf_bar = self.conf_threshold
        print(f"conf_bar={conf_bar}", flush=True)

        if self.final_sample_num > 0:
            print(f"begin final", flush=True)
            final_tasks_list = yield from self.parallel_generate(
                tasks, warmup=False, conf_bar=conf_bar, **kwargs)
            print(f"end final", flush=True)
        else:
            final_tasks_list = []

        finished_task_list = [
            finished_tasks[0]
            for finished_tasks in chain(warmup_tasks_list, final_tasks_list)
        ]
        print(f"len(finished_task_list)={len(finished_task_list)}", flush=True)
        tasks[0].result = vote(finished_task_list)[self.vote_policy].result
        print(f"end vote", flush=True)

    def parallel_generate(self,
                          tasks: List[Task],
                          warmup: bool,
                          conf_bar: Optional[float] = None,
                          **kwargs):
        # warmup will not stop early
        generation_controller = self.warmup_generation_controller if warmup else self.final_generation_controller
        sample_num = self.warmup_sample_num if warmup else self.final_sample_num

        generation_controllers = [
            generation_controller.clone() for _ in range(sample_num)
        ]
        if conf_bar is not None:
            for generation_controller in generation_controllers:
                generation_controller.confidence_info.conf_threshold = conf_bar
        tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(kwargs) for _ in range(sample_num)
        ]
        yield ParallelProcess(generation_controllers, tasks_list,
                              generation_kwargs_list)

        return tasks_list
