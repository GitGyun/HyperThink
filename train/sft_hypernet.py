from __future__ import annotations
import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# Non-main ranks: kill the tqdm bars ("Resolving data files", "Loading checkpoint
# shards") so a multi-GPU launch draws them once instead of once per rank. Must run
# before the first datasets/transformers/hub call below.
from utils.train_utils import quiet_progress_bars_on_non_main
quiet_progress_bars_on_non_main()

# On a cold cache the parallel accelerate processes hit the Hub at once (HTTP 429).
# Make every hub call retry with backoff so the run survives the rate limit.
try:
    from utils.hf_http import enable_hf_retries
    enable_hf_retries()
except Exception:
    pass
import gc
import torch
import wandb
from transformers import set_seed

# PyTorch 2.6+ changed torch.load default to weights_only=True,
# which breaks HF Trainer's resume (rng_state contains numpy objects).
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': kw.get('weights_only', False)})

from utils.train_utils import is_main_process
from models.wrappers import load_model
from data.sft_data import setup_data_for_training
from train.args import parse_args, setup_training_args
from train.sft_trainer import setup_trainer


# Main
def main():
    # parse args
    args = parse_args()

    # setup wandb
    if args.use_wandb and is_main_process():
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            entity=args.entity_name,
            config=args,
        )

    # setup training args
    training_args = setup_training_args(args)
    set_seed(training_args.seed)

    # setup data
    train_dataset, train_collator = setup_data_for_training(args)

    # load model
    model = load_model(args)
    model = model.to(getattr(args, 'torch_dtype', torch.bfloat16))

    # Trainer setup
    trainer = setup_trainer(
        model,
        args,
        training_args,
        train_dataset,
        train_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    torch.cuda.empty_cache()
    gc.collect()

    if "RANK" in os.environ:
        trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
