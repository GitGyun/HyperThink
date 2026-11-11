from utils.dataset_registry import TRAIN_TASKS
import argparse
from utils.train_utils import is_main_process, silent
from transformers import TrainingArguments
import os
import torch


def str2bool(v):
    """Parse a CLI boolean. `type=bool` is broken (bool('False') is True), so use
    this for value-style bool flags like --use_text_embedding False."""
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{v}'")


from utils.model_registry import num_layers, output_root, model_tag


def set_run_name(args):
    args.run_name = f"HyperNet{args.name_postfix}"


def set_bias_layers(args):
    N_LAYERS = num_layers(args.base_model)

    if args.bias_layers_mode == "all":
        args.bias_layers_to_transform = list(range(N_LAYERS))  # for Qwen 0.6B
    elif args.bias_layers_mode == "first-half":
        args.bias_layers_to_transform = list(range(N_LAYERS // 2))
    elif args.bias_layers_mode == "last-half":
        args.bias_layers_to_transform = list(range(N_LAYERS // 2, N_LAYERS))
    elif args.bias_layers_mode == "first8":
        args.bias_layers_to_transform = list(range(8))
    elif args.bias_layers_mode == "last8":
        args.bias_layers_to_transform = list(range(N_LAYERS - 8, N_LAYERS))
    elif args.bias_layers_mode == "first4":
        args.bias_layers_to_transform = list(range(4))
    elif args.bias_layers_mode == "last4":
        args.bias_layers_to_transform = list(range(N_LAYERS - 4, N_LAYERS))
    elif args.bias_layers_mode == "last_third":
        args.bias_layers_to_transform = list(range(N_LAYERS * 2 // 3, N_LAYERS))
    else:
        pass  # use provided list


def set_debug_mode(args):
    args.use_wandb = False
    args.max_steps = 10
    args.save_every = 5
    args.run_name = "debugging" + args.name_postfix


def parse_args(is_shell=True):
    ap = argparse.ArgumentParser(description="Train the HyperThink hypernetwork without validation.")

    # core IO / naming
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--name_postfix", "-ptf", type=str, default="")
    ap.add_argument("--project_name", type=str, default="hyperthink", help="Weights & Biases project.")
    ap.add_argument("--entity_name", type=str, default=None,
                    help="Weights & Biases entity (user or team); default = your W&B default entity.")
    ap.add_argument("--use_wandb", action="store_true", default=False,
                    help="Enable Weights & Biases logging (disabled by default).")
    ap.add_argument("--resume", action="store_true", default=False)

    # dataset
    ap.add_argument("--train_datasets", choices=TRAIN_TASKS, type=str, default=["gsm8k"], nargs="+")
    ap.add_argument("--max_train_length", type=int, default=8192)


    # hypernet config
    ap.add_argument("--text_encoder_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--text_embed_dim", '-ted', type=int, default=128)
    ap.add_argument("--param_hidden", '-ph', type=int, default=128)
    ap.add_argument("--dit_layers", type=int, default=3)
    ap.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "sdpa"])
    ap.add_argument("--num_codes", '-nc', type=int, default=256)
    ap.add_argument("--code_dim", '-cd', type=int, default=512)
    ap.add_argument("--use_text_embedding", type=str2bool, default=None)

    # which bias layers/modules to control
    ap.add_argument("--bias_layers_to_transform", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25, 26, 27])
    ap.add_argument("--bias_target_modules", type=str, nargs="+", default=["q_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"])
    ap.add_argument("--bias_layers_mode", type=str, default="selected", choices=["selected", "all", "first-half", "last-half", "first8", "last8", "first4", "last4", "last_third"])

    # LoRA injection (alternative to bias injection)
    ap.add_argument("--use_lora_injection", action="store_true", default=False, help="Use LoRA injection instead of bias injection")
    ap.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA matrices when using LoRA injection")

    # debug mode
    ap.add_argument("--debug_mode", "-debug", action="store_true", default=False)

    # dataset presets
    ap.add_argument("--extended_datasets", action="store_true", default=False)
    ap.add_argument("--math_datasets", action="store_true", default=False)
    ap.add_argument("--sm3b", action="store_true", default=False)
    ap.add_argument("--om7b", action="store_true", default=False)

    # checkpoint saving
    ap.add_argument("--save_every", type=int, default=5000)

    # training hyperparams
    ap.add_argument("--batch_size", "-bs", type=int, default=8, help="per-device train batch size")
    ap.add_argument("--grad_accum", "-ga", type=int, default=1, help="gradient_accumulation_steps")
    ap.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", "-ms", type=int, default=100000)
    ap.add_argument("--warmup_steps", "-ws", type=int, default=100)
    ap.add_argument("--lr_scheduler_type", "-lrs", type=str, default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce_weight", "-cew", type=float, default=1.0, help="Weight for the cross-entropy loss term.")
    ap.add_argument("--vq_weight", "-vqw", type=float, default=100, help="Weight for the VQ loss term.")
    ap.add_argument("--vq_beta", "-vqb", type=float, default=0.25, help="Beta commit loss weight in VQ-VAE.")
    ap.add_argument("--vq_gumbel", type=str2bool, default=True,
                    help="Use stochastic Gumbel assignments at train and eval time; False selects nearest codes.")
    ap.add_argument("--vq_soft", type=str2bool, default=True,
                    help="Use soft Gumbel assignments or, without Gumbel, soft straight-through gradients. False uses hard assignments.")
    ap.add_argument("--vq_usage_weight", "-vquw", type=float, default=0.1, help="Weight for VQ code usage loss (perplexity regularizer).")

    # precision / distrib knobs
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--ddp_backend", type=str, default=None, choices=["nccl", "gloo", "mpi"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true",
                    help="Keep library (transformers/datasets/accelerate) log output; silenced by default.")

    if is_shell:
        args = ap.parse_args()
    else:
        args = ap.parse_args([])

    if args.sm3b:
        args.base_model = 'HuggingFaceTB/SmolLM3-3B'
        args.text_encoder_name = 'HuggingFaceTB/SmolLM3-3B'
        if args.bias_layers_mode == 'selected':
            args.bias_layers_mode = 'last-half'

    if args.om7b:
        args.base_model = 'allenai/Olmo-3-7B-Think'
        args.text_encoder_name = 'allenai/Olmo-3-7B-Think'
        if args.bias_layers_mode == 'selected':
            args.bias_layers_mode = 'last-half'


    if args.use_text_embedding is None:
        if 'Embedding' in args.text_encoder_name:
            args.use_text_embedding = True
        else:
            args.use_text_embedding = False


    # dataset presets
    if args.extended_datasets:
        args.train_datasets = ["gsm8k", "deepscaler", "codeforces-cots", "logiqa", "openbookqa", "qasc"]
    if args.math_datasets:
        args.train_datasets = ["gsm8k", "deepscaler"]

    # Fixed public distillation layout.
    cot_data_root = os.path.join("outputs/distill_data", model_tag(args.base_model))
    args.cot_solution_multi_paths = {
        name: os.path.join(cot_data_root, name, "outputs_n*_seed*.jsonl") for name in sorted(set(args.train_datasets))
    }
    # run naming
    set_run_name(args)

    # bias layers to transform
    set_bias_layers(args)

    # debug mode settings
    if args.debug_mode:
        set_debug_mode(args)

    return args


def setup_training_args(args) -> TrainingArguments:
    output_dir = os.path.join(output_root(args.base_model), "hypernet_outputs", args.run_name)
    report_to_value = "wandb" if args.use_wandb else "none"
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.project_name

    use_bf16 = use_fp16 = False
    args.torch_dtype = torch.float32
    if args.bf16 and torch.cuda.is_available():
        use_bf16 = True
        args.torch_dtype = torch.bfloat16
    elif args.fp16 and torch.cuda.is_available():
        use_fp16 = True
        args.torch_dtype = torch.float16

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="no",
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=1,
        # safetensors disallows two tensors sharing memory; Qwen3-0.6B ties
        # embed_tokens/lm_head. Fall back to torch.save (pickle) to bypass the check.
        save_safetensors=False,
        report_to=report_to_value,
        remove_unused_columns=False,
        disable_tqdm=False,
        ddp_find_unused_parameters=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_drop_last=True,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=args.tf32,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        label_names=["labels"],
    )

    if not args.verbose:
        silent()
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    training_args.run_name = args.run_name
    os.makedirs(output_dir, exist_ok=True)
    if is_main_process():
        print(f"[Log] saving outputs to {output_dir}")

    return training_args
