import os
import logging
import datasets


def is_main_process() -> bool:
    # accelerate/torchrun set LOCAL_RANK; when not distributed it's -1/absent
    return int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0)


def quiet_progress_bars_on_non_main() -> None:
    """Silence tqdm progress bars on every rank except the main one.

    Under `accelerate launch --multi_gpu` all N ranks run this code, so bars like
    "Resolving data files" (datasets) and "Loading checkpoint shards" (transformers)
    are drawn N times and interleave into unreadable noise. The `[Dataset] ...` text
    logs are already gated on is_main_process(); this gives the bars the same
    treatment. Each disable is independent so one missing/renamed API can't stop
    the others from taking effect.
    """
    if is_main_process():
        return
    for disable in (
        lambda: __import__("huggingface_hub.utils", fromlist=["x"]).disable_progress_bars(),
        lambda: datasets.disable_progress_bars(),
        lambda: __import__("transformers.utils.logging", fromlist=["x"]).disable_progress_bar(),
    ):
        try:
            disable()
        except Exception:
            pass


def silent():
    for name in [
        "transformers",
        "datasets",
        "accelerate",
        "vllm",
        "uvicorn",
        "httpx",
        "urllib3",
        "vllm.logger",
        "vllm.worker",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.sampling",
        "vllm.sampling.topk_topp_sampler",
        "transformers.trainer",
        "transformers.training_args",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False
    datasets.logging.set_verbosity(datasets.logging.ERROR)