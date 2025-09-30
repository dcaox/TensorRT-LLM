import argparse

from tensorrt_llm.scaffolding import (NativeGenerationController,
                                      ScaffoldingLlm, TRTLLMWorker)
from tensorrt_llm.scaffolding.contrib.DeepConf import (
    DeepConfOfflineController, DeepConfOfflineMajorityVoteController,
    DeepConfOnlineController)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument('--run_type',
                        type=str,
                        required=True,
                        help="Type of the run")
    args = parser.parse_args()
    return args


def run_scaffolding_llm(prompts, proposer_worker, controller):
    llm = ScaffoldingLlm(
        controller,
        {
            controller.WorkerTag.GENERATION: proposer_worker,
            NativeGenerationController.WorkerTag.GENERATION: proposer_worker,
        },
    )
    results = llm.generate(prompts)
    for i, result in enumerate(results):
        print(f"result {i}:\n{result.outputs[0].text}")
    llm.shutdown(shutdown_workers=False)


def test_offline_controller(prompts, proposer_worker, logprobs_topk=20):
    prototype_controller = DeepConfOfflineController(
        conf_group_size=10,
        conf_threshold=0.5,
        logprobs_topk=logprobs_topk,
        sampling_params={
            "temperature": 0.9,
            "max_tokens": 1024,
            "num_logprobs": logprobs_topk,
        })

    run_scaffolding_llm(prompts, proposer_worker, prototype_controller)


def test_online_controller(prompts, proposer_worker, logprobs_topk=20):
    prototype_controller = DeepConfOnlineController(conf_group_size=10,
                                                    conf_threshold=0.5,
                                                    logprobs_topk=logprobs_topk,
                                                    sampling_params={
                                                        "temperature":
                                                        0.9,
                                                        "max_tokens":
                                                        1024,
                                                        "num_logprobs":
                                                        logprobs_topk,
                                                    })

    run_scaffolding_llm(prompts, proposer_worker, prototype_controller)


def test_offline_majority_vote_controller(prompts, proposer_worker, **kwargs):

    prototype_generation_controller = NativeGenerationController(
        sampling_params={
            "max_tokens": 8192,
            "temperature": 0.9,
            "num_logprobs": kwargs.get("logprobs_topk", 20),
        })
    majority_vote_controller = DeepConfOfflineMajorityVoteController(
        generation_controller=prototype_generation_controller,
        sample_num=kwargs.get("sample_num", 10),
        conf_group_size=kwargs.get("conf_group_size", 10),
        conf_threshold=kwargs.get("conf_threshold", 0.5),
        vote_policy=kwargs.get("vote_policy", "majority"),
    )

    run_scaffolding_llm(prompts, proposer_worker, majority_vote_controller)


def main():
    args = parse_arguments()
    logprobs_topk = 20

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=32,
        max_num_tokens=8192,
    )
    if args.run_type == "offline":
        test_offline_controller(prompts,
                                llm_worker,
                                logprobs_topk=logprobs_topk)
    elif args.run_type == "online":
        test_online_controller(prompts, llm_worker, logprobs_topk=logprobs_topk)
    elif args.run_type == "offline_majority_vote":
        test_offline_majority_vote_controller(prompts, llm_worker)
    else:
        raise ValueError(f"Invalid run type: {args.run_type}")

    llm_worker.shutdown()
    print('llm worker shutdown done')


if __name__ == "__main__":
    main()
