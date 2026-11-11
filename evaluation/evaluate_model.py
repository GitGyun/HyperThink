from utils.dataset_registry import TEST_TASKS
import argparse

import torch
from transformers import set_seed

from train.args import set_bias_layers, str2bool
from utils.train_utils import silent
from utils.eval_utils import eval_loop
from data.sft_data import setup_data_for_evaluation
from models.wrappers import load_model

# Retry Hub calls on 429/transient errors (parallel eval shards on a cold cache).
try:
    from utils.hf_http import enable_hf_retries
    enable_hf_retries()
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate hypernet with chunked sampling.")

    # base model
    ap.add_argument("--base_model", type=str, default='Qwen/Qwen3-0.6B')
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--bias_layers_to_transform", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25, 26, 27])
    ap.add_argument("--bias_target_modules", type=str, nargs="+", default=["q_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"])
    ap.add_argument("--bias_layers_mode", type=str, default="selected",
                    choices=["selected", "all", "first-half", "last-half", "first8", "last8", "first4", "last4"])
    ap.add_argument("--sm3b", action="store_true", default=False)
    ap.add_argument("--om7b", action="store_true", default=False)

    # LoRA injection (alternative to bias injection)
    ap.add_argument("--use_lora_injection", action="store_true", default=False, help="Use LoRA injection instead of bias injection")
    ap.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA matrices when using LoRA injection")

    # hypernet
    ap.add_argument("--param_hidden", '-ph', type=int, default=128)
    ap.add_argument("--dit_layers", type=int, default=3)
    ap.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "sdpa"])
    ap.add_argument("--text_encoder_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--text_embed_dim", '-ted', type=int, default=128)
    ap.add_argument("--use_text_embedding", type=str2bool, default=None)
    ap.add_argument("--num_codes", '-nc', type=int, default=256)
    ap.add_argument('--code_dim', '-cd', type=int, default=512)
    ap.add_argument("--vq_gumbel", type=str2bool, default=True,
                    help="Use stochastic Gumbel assignments. Set False for nearest-code checkpoints.")
    ap.add_argument("--vq_soft", type=str2bool, default=True,
                    help="Use soft Gumbel assignments. Match the training setting; without Gumbel, both values select nearest codes.")
    # dataset
    # Any task in utils.dataset_registry.DATASET_REGISTRY is accepted.
    ap.add_argument('--task', choices=TEST_TASKS, type=str, default='gsm8k-test')
    ap.add_argument("--ckpt", type=str, default=None, help="Path to trained hypernet checkpoint. If None or 'None', use randomly initialized hypernet.")
    ap.add_argument("--n_tasks", type=int, default=None, help="Number of tasks to evaluate. Default None means all.")
    ap.add_argument('--task_offset', type=int, default=0, help="Start evaluating from this example index.")
    ap.add_argument('--end_offset', type=int, default=None)
    ap.add_argument('--reset', action='store_true', default=False)

    # decoding
    ap.add_argument("--samples_per_adapter", type=int, default=5, help="How many stochastic samples to draw per adapter/step")
    ap.add_argument("--eval_chunk_n", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument('--nothink_mode', action='store_true', default=False,
                    help="Disable thinking in the chat template. Thinking budgets are ignored in this mode.")
    ap.add_argument("--thinking_budgets", type=int, default=None,
                    help="Think for up to N tokens, then force an answer for sequences that hit the cap, are empty, or do not close </think>.")
    ap.add_argument("--thinking_budgets_force_tokens", type=int, default=100,
                    help="Maximum additional tokens for the forced-answer phase (default: 100).")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    # other
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Default: outputs_<model_tag>/hypernet_evaluation (see utils/model_registry.py).")
    ap.add_argument("--output_postfix", '-ptf', type=str, default='')
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards to split adapter examples into.")
    ap.add_argument("--shard_id", type=int, default=0, help="This process shard id in [0, num_shards-1].")

    args = ap.parse_args()

    if args.thinking_budgets is not None and args.thinking_budgets <= 0:
        ap.error("--thinking_budgets must be positive")
    if args.thinking_budgets_force_tokens <= 0:
        ap.error("--thinking_budgets_force_tokens must be positive")

    # if 'Extended' in args.ckpt:
    #     args.vq_gumbel = False
    #     args.vq_soft = False

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

    if args.output_dir is None:
        args.output_dir = "outputs/results"

    if args.ckpt is not None and args.ckpt.lower() == 'none':
        args.ckpt = None

    if args.use_text_embedding is None:
        if 'Embedding' in args.text_encoder_name:
            args.use_text_embedding = True
        else:
            args.use_text_embedding = False

    # Test queries are loaded from Hugging Face by data.sft_data.setup_data_for_evaluation.
    print(f"[eval] task={args.task} split=test")

    # bias layers to transform
    set_bias_layers(args)

    if args.dtype == "float16":
        args.torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        args.torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    silent()

    # Load model
    print("Initializing base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        model = load_model(args, hypernet_state=state['state_dict']).to(device).eval()
        print(f"Loaded Hypernet from {args.ckpt}")
    else:
        model = load_model(args, hypernet_state=None).to(device).eval()
        print("Using randomly initialized Hypernet (zero bias)")

    # Load dataset
    ds = setup_data_for_evaluation(args)

    # run evaluation loop
    eval_loop(ds, model, args, device)


if __name__ == "__main__":
    main()
