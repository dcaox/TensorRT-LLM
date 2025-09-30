import copy
import random
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Tuple

from tensorrt_llm.scaffolding import (Controller, NativeGenerationController,
                                      ParallelProcess, StreamGenerationTask,
                                      Task, extract_answer_from_boxed)


@dataclass
class ConfidenceInfo:
    conf_grouped: float
    conf_list: list[float]
    conf_group_list: deque[float]
    conf_group_size: int
    conf_threshold: float


def update_confidence_info(confidence_info: ConfidenceInfo,
                           token_dict: Mapping[int, 'Logprob'], token_id: int):
    new_conf = -sum(logprob_obj.logprob
                    for logprob_obj in token_dict.values()) / len(token_dict)
    confidence_info.conf_list.append(new_conf)
    confidence_info.conf_group_list.append(new_conf)
    confidence_info.conf_grouped += new_conf
    if len(confidence_info.conf_group_list) > confidence_info.conf_group_size:
        confidence_info.conf_grouped -= confidence_info.conf_group_list.popleft(
        )


def basic_majority_vote(tasks: List[Task]) -> Task:
    answers = [
        task.costimized_result_fields['extracted_answer'] for task in tasks
    ]
    majority_answer = Counter(answers).most_common(1)[0][0]
    return tasks[answers.index(majority_answer)]


VOTE_POLICY_IMPL = {
    'majority': basic_majority_vote,
}


def vote(tasks: List[Task]):
    for task in tasks:
        task.costimized_result_fields[
            'extracted_answer'] = extract_answer_from_boxed(task.output_str)
    valid_tasks = [
        task for task in tasks
        if task.costimized_result_fields['extracted_answer']
    ]
    random_return = False
    if len(valid_tasks) == 0:
        print(
            "No valid tasks, maybe you should increase max_output_len, a random task will be returned"
        )
        random_return = True
    return {
        policy_name:
        policy_impl(valid_tasks) if not random_return else random.choice(tasks)
        for policy_name, policy_impl in VOTE_POLICY_IMPL.items()
    }


class DeepConfOfflineController(NativeGenerationController):

    def __init__(self,
                 conf_group_size: int,
                 conf_threshold: float,
                 logprobs_topk: int = 20,
                 sampling_params: dict = None,
                 streaming: bool = False):
        super().__init__(sampling_params, streaming)
        self.logprobs_topk = logprobs_topk
        self.confidence_info = ConfidenceInfo(conf_grouped=0.0,
                                              conf_list=[],
                                              conf_group_list=deque([]),
                                              conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepConfOfflineController only supports one task"
        yield from super().process(tasks, **kwargs)
        for logprobs_dict, token_id in zip(tasks[0].logprobs,
                                           tasks[0].output_tokens):
            update_confidence_info(self.confidence_info, logprobs_dict,
                                   token_id)
        tasks[0].costimized_result_fields[
            'confidence_info'] = self.confidence_info


class DeepConfOfflineMajorityVoteController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self,
                 generation_controller: Controller,
                 sample_num: int,
                 conf_group_size: int,
                 conf_threshold: float,
                 vote_policy: str = 'majority'):
        super().__init__()
        self.generation_controller = generation_controller
        self.sample_num = sample_num
        self.conf_group_size = conf_group_size
        self.conf_threshold = conf_threshold
        self.confidence_info = [
            ConfidenceInfo(conf_grouped=0.0,
                           conf_list=[],
                           conf_group_list=deque([]),
                           conf_group_size=conf_group_size,
                           conf_threshold=conf_threshold)
            for _ in range(self.sample_num)
        ]
        self.vote_policy = vote_policy

    def clone(self):
        return DeepConfOfflineMajorityVoteController(
            self.generation_controller.clone(), self.sample_num,
            self.conf_group_size, self.conf_threshold, self.vote_policy)

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

        task_list = [tasks[0] for tasks in tasks_list]
        for i, task in enumerate(task_list):
            for logprobs_dict, token_id in zip(task.logprobs,
                                               task.output_tokens):
                update_confidence_info(self.confidence_info[i], logprobs_dict,
                                       token_id)
            task.costimized_result_fields[
                'confidence_info'] = self.confidence_info[i]
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
        self.confidence_info = ConfidenceInfo(conf_grouped=0.0,
                                              conf_list=[],
                                              conf_group_list=deque([]),
                                              conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
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
                update_confidence_info(self.confidence_info, logprobs_dict,
                                       token_id)
            if self.should_stop(self.confidence_info):
                online_task.cancel_flag = True
                yield [online_task]
                break
            last_step_index = len(online_task.output_tokens)
        tasks[0].result = online_task.result
        tasks[0].costimized_result_fields[
            'confidence_info'] = self.confidence_info

    # TODO: implement should_stop
    def should_stop(self, confidence_info: ConfidenceInfo) -> bool:
        if len(confidence_info.conf_group_list
               ) >= confidence_info.conf_group_size:
            avg_conf = confidence_info.conf_grouped / len(
                confidence_info.conf_group_list)
            return avg_conf < confidence_info.conf_threshold
        return False


class DeepConfOnlineMajorityVoteController(Controller):

    def __init__(self, generation_controller: Controller,
                 sample_num_per_round: int, max_sample_num: float):
        super().__init__()
        self.generation_controller = generation_controller
        self.sample_num_per_round = sample_num_per_round
        self.max_sample_num = max_sample_num

    def clone(self):
        return DeepConfOnlineMajorityVoteController(
            self.generation_controller.clone(), self.sample_num,
            self.max_sample_num)

    def process(self, tasks: List[Task], **kwargs):
        candidates = []
        should_continue = True
        sample_next_round = self.sample_num_per_round
        while len(candidates) < self.max_sample_num and should_continue:
            sample_num = sample_next_round
            generation_controllers = [
                self.generation_controller.clone() for _ in range()
            ]
            tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
            generation_kwargs_list = [
                copy.deepcopy(kwargs) for _ in range(sample_num)
            ]
            yield ParallelProcess(generation_controllers, tasks_list,
                                  generation_kwargs_list)

            should_continue, sample_next_round = self.next_round_judge(
                candidates, tasks_list)
            candidates.extend(tasks_list)

        return self.majority_vote(candidates, **kwargs)

    def next_round_judge(self, candidates: List[Task],
                         tasks_list: List[Task]) -> Tuple[bool, int]:
        pass

    def majority_vote(self, candidates: List[Task],
                      **kwargs) -> Tuple[int, str]:
        pass
