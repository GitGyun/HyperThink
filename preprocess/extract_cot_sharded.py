import argparse
import os
# Quiet noisy startup logs (set before torch/transformers/vLLM import):
#  - c10d "socket.cpp ... Address family not supported" IPv6 warning from vLLM's
#    distributed init  -> TORCH_CPP_LOG_LEVEL (libtorch C++ log level)
#  - transformers "`torch_dtype` is deprecated! Use `dtype` instead!" warning_once
#    -> TRANSFORMERS_VERBOSITY
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# Disable vLLM's FP8 DeepGEMM kernels and flashinfer's JIT sampler. BF16 inference
# uses neither, and both need extra system libraries (libnvrtc, ninja).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import sys
import json
import random
from typing import Optional, List
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.dataset_registry import to_chat_prompt, META_INFO_DIR  # to_chat_prompt shared with utils/eval_utils.py so train == eval prompts

# Retry Hub calls on 429/transient errors (parallel shards on a cold cache).
try:
    from utils.hf_http import enable_hf_retries
    enable_hf_retries()
except Exception:
    pass

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def _hf_retry(fn, *a, _retries=6, _base=2.0, **kw):
    """Call an HF hub function, retrying transient rate-limit errors (HTTP 429)
    with exponential backoff + jitter. Aggregate loaders (bbh / combined AIME)
    fan out many API calls; running 8 shards in parallel easily trips the
    anonymous rate limit, so retry instead of crashing the whole shard."""
    import random, time
    for i in range(_retries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e)
            transient = '429' in msg or 'too many requests' in msg.lower() or 'rate limit' in msg.lower()
            if not transient or i == _retries - 1:
                raise
            delay = _base * (2 ** i) + random.uniform(0, _base)
            print(f"[HF retry] {type(e).__name__}: rate-limited (attempt {i+1}/{_retries}); "
                  f"sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)


# Substrings indicating a port-bind or distributed-init collision, matched
# case-insensitively against the exception message and its __cause__ chain.
# Kept tight to avoid swallowing unrelated errors (no bare "port" token).
#
# vLLM V1 caveat: when the EngineCore subprocess fails (most often EADDRINUSE
# during ProcessGroup init), the real `DistNetworkError` text only reaches the
# subprocess's stderr. The parent raises a generic
# `RuntimeError("Engine core initialization failed. See root cause above.")`
# with no chained cause, so vLLM's parent-side phrases are listed too. Other
# subprocess crashes (OOM during weight load, a bad config) can produce them as
# well, but a bounded number of retries (`max_retries`) is better than failing
# on every port collision.
_PORT_ERR_TOKENS = (
    "address already in use",
    "eaddrinuse",
    "errno 98",
    "code: -98",                              # torch.distributed DistNetworkError format
    "bind:",
    "bind error",
    "tcpstore",
    "distnetworkerror",
    "the server socket has failed",
    "failed to bind",
    "failed to listen",
    # vLLM V1 parent-side messages: subprocess actual cause is hidden, so any of
    # these signals a likely (but not guaranteed) port issue worth retrying.
    "engine core initialization failed",
    "engine core failed to start",
    "engine process failed to start",
    "executor failed",
)


def _is_port_error(exc: BaseException) -> bool:
    """True when `exc` (or any cause/context in its chain) looks like a port-bind
    / distributed-rendezvous collision. Checks (in order, on every node of the
    exception chain):
      1. `OSError` with `errno == 98` (Linux EADDRINUSE).
      2. vLLM `EngineDeadError`: the subprocess died for any reason
         (commonly a port collision between parallel shards).
      3. Substring match against `_PORT_ERR_TOKENS` on `str(exc).lower()`."""
    # Lazy-import the vLLM exception so this helper still works in unit tests
    # that stub out vllm (or in environments where it isn't installed).
    EngineDeadError = None
    try:
        from vllm.v1.engine.exceptions import EngineDeadError as _EDE
        EngineDeadError = _EDE
    except Exception:
        pass

    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError) and getattr(cur, "errno", None) == 98:
            return True
        if EngineDeadError is not None and isinstance(cur, EngineDeadError):
            return True
        msg = str(cur).lower()
        if any(t in msg for t in _PORT_ERR_TOKENS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _llm_init_with_port_retry(LLM_cls, *, max_retries: int = 8,
                              base_delay: float = 1.5, **llm_kwargs):
    """Construct vLLM's LLM, retrying on port-bind collisions with a fresh port
    each time. Race-prone because the port chosen by `pick_free_port` can be
    grabbed by another shard between check and bind.

    Strategy:
      * attempt 0: use whatever VLLM_PORT is already in os.environ (set by the
        caller via pick_free_port).
      * attempt N>0: rebind to a random port inside the kernel reserved range
        (ip_local_reserved_ports) so vLLM's secondary `bind(("", 0))` calls,
        which the kernel routes outside the reserved range, cannot collide.
        Falls back to socket(0) OS-assignment if reserved range is unavailable.

    Non-port errors are re-raised immediately. Raises the last port error
    after `max_retries` exhausted.
    """
    import random as _rand
    import socket as _sock
    import time as _time
    from utils.dist_port import _parse_reserved_ranges

    # Read kernel reserved range once: pick rebind ports from inside it so they
    # don't race with vLLM's OS-random IPC ports (which land just above the
    # reserved range).
    _reserved_ranges: List[tuple[int, int]] = []
    try:
        with open("/proc/sys/net/ipv4/ip_local_reserved_ports") as _f:
            _reserved_ranges = _parse_reserved_ranges(_f.read())
    except OSError:
        pass

    def _fresh_free_port() -> int:
        # Try a random port inside the reserved range first.
        for lo, hi in _reserved_ranges:
            if hi - lo < 32:
                continue
            for _ in range(20):
                cand = _rand.randint(lo + 1, hi - 1)
                s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                try:
                    s.bind(("", cand))
                    return s.getsockname()[1]
                except OSError:
                    continue
                finally:
                    s.close()
        # Last-resort fallback: OS-assigned (will land outside reserved range).
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        try:
            s.bind(("", 0))
            return s.getsockname()[1]
        finally:
            s.close()

    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        if attempt > 0:
            new_port = _fresh_free_port()
            os.environ["VLLM_PORT"] = str(new_port)
            delay = base_delay * (2 ** (attempt - 1)) + _rand.uniform(0, base_delay)
            print(f"[port retry] LLM init attempt {attempt + 1}/{max_retries} "
                  f"after port-collision error: {type(last_err).__name__}: "
                  f"{str(last_err)[:160]} | rebound VLLM_PORT={new_port}, "
                  f"sleeping {delay:.1f}s", flush=True)
            _time.sleep(delay)
        try:
            return LLM_cls(**llm_kwargs)
        except Exception as e:
            if not _is_port_error(e):
                raise
            last_err = e
    raise RuntimeError(
        f"vLLM init failed after {max_retries} port-retry attempts. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    ) from last_err


from utils.dataset_registry import TRAIN_TASKS, TEST_TASKS, DATASET_REGISTRY as TASKS, MCQ_TASKS, normalize_mcq_row, DATASET_REVISION
from utils.eval_utils import split_think, is_generation_correct

# Code tasks have no scalar answer: a training row is kept when its post-thinking
# answer is non-empty (unit tests are not executed at extraction time).
CODE_TASKS = ("codeforces-cots",)

# Appended to a generation that exhausted --thinking_budgets without closing its
# thinking block, to force the final answer in a short second pass.
STOP_PROMPT = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\nFinal Answer:"

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    # vllm
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)  # higher values can run out of memory

    # dataset and subsampling
    p.add_argument("--task", type=str, default=None, choices=list(TASKS.keys()))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)

    # sampling params
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_samples", type=int, default=1,
                   help="CoTs sampled per problem in this run (vLLM n). With K>1 the record's "
                        "full_generated_text/degenerate become K-lists and the file names carry _n<K>. "
                        "Seeds (--seed) are the other way to collect several CoTs per problem.")

    # output
    p.add_argument("--distill", action="store_true", help="Write training samples instead of evaluation records.")
    p.add_argument("--save_incorrect", action="store_true")
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Base-generation records for scoring. Default: outputs/results "
                        "(see utils/model_registry.py).")
    p.add_argument("--train_output_dir", type=str, default=None,
                   help="Where the training rows {index, solution, thinking, answer} go, split into "
                        "<stem>.jsonl (correct) and <stem>_incorrect.jsonl. Default: outputs/distill_data, "
                        "Training reads outputs/distill_data by default. Enabled by --distill.")

    # sharding: each process handles one shard of [start, end)
    p.add_argument("--shard_id", type=int, default=0, help="this shard index (0-based)")
    p.add_argument("--num_shards", type=int, default=1, help="total number of shards")

    # other
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable_thinking", default=False, action="store_true",
                   help="Native no-think mode (chat template enable_thinking=False).")
    p.add_argument("--thinking_budgets", type=int, default=None,
                   help="Optional thinking budget (tokens). Generations that exhaust it without "
                        "closing </think> get STOP_PROMPT appended and a short forced-answer pass.")
    p.add_argument("--thinking_budgets_force_tokens", type=int, default=100,
                    help="Max tokens for the forced-answer phase under --thinking_budgets "
                         "(after STOP_PROMPT is appended to budget-exhausted generations). "
                         "Default 100 works for Qwen3 (commits to \\boxed{} quickly) but is "
                         "too short for OLMo3-7B-Think (drifts back into 'Wait, ...' thinking "
                         "patterns); raise to 512+ for those.")
    p.add_argument("--reset", action='store_true', default=False, help="whether to overwrite existing output files")

    args = p.parse_args()
    if args.task is None:
        args.task = "gsm8k" if args.distill else "gsm8k-test"
    allowed = TRAIN_TASKS if args.distill else TEST_TASKS
    if args.task not in allowed:
        p.error(f"--task {args.task!r} is not supported in {'distill' if args.distill else 'evaluation'} mode; choose from {allowed}")
    return args


# Main
def main():
    """Generate test scoring records, or correct training rows with --distill.

    Distillation drops degenerate generations and automatically merges the
    correct shard files. Incorrect/raw diagnostics are opt-in.
    """
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = "outputs/results"
    if args.train_output_dir is None:
        args.train_output_dir = "outputs/distill_data"
    if args.num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {args.num_samples}")
    random.seed(args.seed)

    # Per-shard distributed port so parallel vLLM engines don't collide (EADDRINUSE).
    from utils.dist_port import pick_free_port
    os.environ["VLLM_PORT"] = str(pick_free_port(args.shard_id))

    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}")

    # load and subset dataset with idxs
    repo, subset, split, prompt_fn = TASKS[args.task]
    if subset == "_AGGREGATE_BBH_":
        from datasets import concatenate_datasets
        from utils.dataset_registry import BBH_PER_SUBTASK_CAP, get_bbh_configs
        cfgs = get_bbh_configs(repo)   # local meta_info cache; no per-run tree-listing API call
        parts = []
        for cfg in cfgs:
            d = _hf_retry(load_dataset, repo, cfg, split=split)
            if BBH_PER_SUBTASK_CAP and len(d) > BBH_PER_SUBTASK_CAP:
                d = d.select(range(BBH_PER_SUBTASK_CAP))
            d = d.add_column("subtask", [cfg] * len(d))
            parts.append(d)
        ds = concatenate_datasets(parts)
    elif subset == "_COMBINED_AIME_24_25_":
        from datasets import concatenate_datasets
        d1 = _hf_retry(load_dataset, "Maxwell-Jia/AIME_2024", split="train")
        d1 = d1.rename_column("Problem", "problem").rename_column("Answer", "answer")
        d1 = d1.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True)
        d1 = d1.select_columns(["problem", "answer"])
        d2 = _hf_retry(load_dataset, "MathArena/aime_2025", split="train")
        d2 = d2.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True)
        d2 = d2.select_columns(["problem", "answer"])
        ds = concatenate_datasets([d1, d2])
    else:
        ds = load_dataset(repo, subset, split=split,
                          revision=DATASET_REVISION.get(args.task))
    if args.task == 'deepscaler':
        ds = ds.add_column("index", list(range(len(ds))))
        with open(os.path.join(META_INFO_DIR, 'deepscaler_train_ids.json'), 'r') as f:
            train_ids = set(json.load(f))
        ds = ds.filter(lambda x: x['index'] in train_ids)

    N = len(ds)
    global_start = max(0, args.start)
    global_end = N if args.end is None else min(args.end, N)
    if global_start >= global_end:
        raise ValueError(f"invalid range: start={global_start}, end={global_end}, N={N}")

    idxs = list(range(global_start, global_end))
    total = len(idxs)

    # shard [global_start, global_end) into num_shards
    shard_size = (total + args.num_shards - 1) // args.num_shards  # ceil
    shard_start = args.shard_id * shard_size
    shard_end = min(total, (args.shard_id + 1) * shard_size)

    if shard_start >= shard_end:
        print(f"[INFO] Shard {args.shard_id}/{args.num_shards}: empty slice, nothing to do.")
        return

    local_idxs = idxs[shard_start:shard_end]
    print(
        f"[INFO] Task={args.task}, global range=[{global_start},{global_end}) "
        f"(total={total}) | shard {args.shard_id}/{args.num_shards}: "
        f"local indices [{shard_start},{shard_end}) (count={len(local_idxs)})"
    )

    # task-specific preprocessing
    if args.task in ['gsm8k', 'gsm8k-test']:
        ds = ds.rename_column("question", "problem")
        ds = ds.rename_column("answer", "solution")

        def _map(batch):
            cots = batch["solution"]
            answer = [
                re.match(r"^\s*####\s*(.+?)\s*$", cot.strip().splitlines()[-1]).group(1).strip()
                for cot in cots
            ]
            return {"answer": answer}

        ds = ds.map(_map, batched=True, num_proc=1)

    elif args.task == 'bbh':
        # lighteval/bbh rows -> shared MCQ schema (n_choices varies per row).
        def _map_bbh(batch):
            problems, choices_list, labels_list, answers = [], [], [], []
            n = len(batch["input"])
            for i in range(n):
                problems.append(batch["input"][i])
                ch = list(batch["choices"][i] or [])
                n_ch = len(ch)
                lbls = [chr(ord('A') + j) for j in range(n_ch)]
                choices_list.append(ch)
                labels_list.append(lbls)
                tgt = int(batch["target_idx"][i])
                answers.append(lbls[tgt] if 0 <= tgt < n_ch else "")
            return {"problem": problems, "choices": choices_list,
                    "choice_labels": labels_list, "answer": answers}
        ds = ds.map(_map_bbh, batched=True, num_proc=1,
                    remove_columns=[c for c in ds.column_names if c != "subtask"])
        ds = ds.add_column("index", list(range(len(ds))))

    elif args.task == 'aime_24_25':
        # Already normalized to {problem, answer} (combined tasks did it during concat).
        ds = ds.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True, num_proc=1)
        ds = ds.add_column("index", list(range(len(ds))))

    elif args.task in MCQ_TASKS:
        # Normalize OpenBookQA / QASC / LogiQA / CommonsenseQA into a shared schema.
        ds = normalize_mcq_row(args.task, ds)

    elif args.task == 'livecodebench':
        # livecodebench/code_generation: 400 competitive-programming problems with
        # stdin/stdout test cases. Build the problem from question_content + starter
        # code if present; persist test cases so downstream eval can run them.
        def _map_lcb(batch):
            problems, solutions, answers = [], [], []
            n = len(batch["question_content"])
            for i in range(n):
                title = (batch.get("question_title") or [""] * n)[i] or ""
                qc = batch["question_content"][i] or ""
                starter = (batch.get("starter_code") or [""] * n)[i] or ""
                parts = [f"# {title}", "", qc]
                if starter.strip():
                    parts += ["", "Use this starter code:", "```python", starter, "```"]
                problems.append("\n".join(parts).strip())
                solutions.append("")
                answers.append("")
            return {"problem": problems, "solution": solutions, "answer": answers}
        keep_cols = {"question_id", "platform", "difficulty",
                     "public_test_cases", "private_test_cases", "contest_date"}
        ds = ds.map(_map_lcb, batched=True, num_proc=1,
                    remove_columns=[c for c in ds.column_names if c not in keep_cols])
        ds = ds.add_column("index", list(range(len(ds))))

    elif args.task == 'codeforces-cots':
        # open-r1/codeforces-cots (solutions_decontaminated): Codeforces problems + DeepSeek-R1
        # Python CoT solutions. Build `problem` from description + IO format + examples;
        # store R1's `generation` as the reference `solution` (gold for distillation).
        def _map_cf(batch):
            problems, solutions, answers = [], [], []
            for i in range(len(batch["description"])):
                title = batch.get("title", [""] * len(batch["description"]))[i] or ""
                desc = batch["description"][i] or ""
                inp = batch["input_format"][i] or ""
                outp = batch["output_format"][i] or ""
                note = batch.get("note", [None] * len(batch["description"]))[i] or ""
                examples = batch.get("examples", [None] * len(batch["description"]))[i] or []
                ex_text = ""
                for j, ex in enumerate(examples):
                    ex_text += f"\nExample {j + 1}:\nInput:\n{ex.get('input', '')}\nOutput:\n{ex.get('output', '')}\n"
                parts = [f"# {title}", "", desc, "", f"Input format: {inp}", f"Output format: {outp}"]
                if note:
                    parts.append(f"\nNote: {note}")
                if ex_text:
                    parts.append(ex_text)
                problems.append("\n".join(parts).strip())
                solutions.append(batch["generation"][i] or "")
                answers.append("")
            return {"problem": problems, "solution": solutions, "answer": answers}
        ds = ds.map(_map_cf, batched=True, num_proc=1,
                    remove_columns=[c for c in ds.column_names if c not in ("id",)])
        ds = ds.add_column("index", list(range(len(ds))))

    # One directory per global slice; one file per shard. File stem:
    #   <task>[_n<K>]_shard<I>_of_<N>[_seed<S>].jsonl   (no _seed suffix for the default seed 42)
    # The same stem is used for the training rows under --train_output_dir.
    from utils.model_registry import model_tag
    from contextlib import nullcontext
    args.output_dir = os.path.join(args.output_dir, model_tag(args.model), args.task)
    mode = "nothink" if args.disable_thinking else "think"
    out_name = f"{mode}_shard_{args.shard_id}_of_{args.num_shards}"
    out_path = os.path.join(args.output_dir, out_name + ".jsonl")
    write_train = args.distill
    train_root = os.path.join(args.train_output_dir, model_tag(args.model), args.task)
    train_dir = os.path.join(train_root, ".shards", str(args.num_shards), str(args.shard_id))
    train_name = f"outputs_n{args.num_samples}_seed{args.seed}.jsonl"
    train_ok_path = os.path.join(train_dir, train_name)
    train_bad_path = train_ok_path.replace(".jsonl", "_incorrect.jsonl")
    target = train_ok_path if write_train else out_path
    if os.path.exists(target) and not args.reset:
        raise FileExistsError(f"{target} exists; use --reset to overwrite")
    if write_train and os.path.exists(train_ok_path + ".done"):
        os.remove(train_ok_path + ".done")
    if not write_train or args.save_raw:
        os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    llm = _llm_init_with_port_retry(
        LLM,
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=True,  # skip CUDA graphs and torch.compile, so no C compiler is needed
    )

    if args.thinking_budgets is not None:
        max_tokens = args.thinking_budgets
    else:
        max_tokens = args.max_new_tokens
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        n=args.num_samples,
    )

    # build prompts for this shard only
    prompts: List[str] = []
    base_queries: List[str] = []
    # All supported backbones (Qwen3 / SmolLM3 / OLMo3) have chat templates that accept the
    # `enable_thinking` kwarg; --disable_thinking flips it cleanly.
    enable_thinking = not args.disable_thinking
    print(f"[INFO] enable_thinking: {enable_thinking}")

    for idx in local_idxs:
        row = ds[idx]
        base_queries.append(prompt_fn(row))

    for query in base_queries:
        chat_text = to_chat_prompt(tok, query, enable_thinking=enable_thinking)
        if (enable_thinking is False) and chat_text.endswith('<think>'):
            # native enable_thinking=False but the template still appended bare <think>
            # (some chat template revisions), so close it manually.
            chat_text += '\n\n</think>\n\n'
        prompts.append(chat_text)

    if prompts:
        print('\nSample Prompt (this shard):')
        print(prompts[0])
    
    # inference for this shard
    K = args.num_samples
    print(f"[INFO] Running vLLM generate on {len(prompts)} prompts x {K} sample(s) for shard {args.shard_id}/{args.num_shards}...")
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(local_idxs)
    # texts[i][j] / finished[i][j]: sample j of prompt i and its vLLM finish reason
    texts = [[c.text for c in out.outputs] for out in outputs]
    finished = [[c.finish_reason for c in out.outputs] for out in outputs]
    assert all(len(t) == K for t in texts), "vLLM returned a different number of samples than requested"

    def _is_degenerate(text, finish_reason):
        # 1. hit the token limit  2. empty output  3. (thinking mode) never closed </think>
        return (finish_reason == "length"
                or text.strip() == ""
                or ((not args.disable_thinking) and text.find('</think>') <= 0))

    if args.thinking_budgets is not None:
        # Budget-exhausted generations get STOP_PROMPT appended and a short forced-answer pass.
        post_sampling_params = SamplingParams(
            max_tokens=args.thinking_budgets_force_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        pending = [(i, j) for i in range(len(texts)) for j in range(K)
                   if _is_degenerate(texts[i][j], finished[i][j])]
        if pending:
            post_prompts = [prompts[i] + texts[i][j] + STOP_PROMPT for i, j in pending]
            print(f"[INFO] Running post-thinking vLLM generate on {len(post_prompts)} budget-exhausted generations...")
            post_outputs = llm.generate(post_prompts, post_sampling_params)
            print('\nSample Degenerate Prompt (this shard):')
            print(post_prompts[0], '\n')
            for (i, j), post_out in zip(pending, post_outputs):
                texts[i][j] += STOP_PROMPT + post_out.outputs[0].text
                finished[i][j] = post_out.outputs[0].finish_reason

    # save outputs for this shard: the scoring records (+ the training rows)
    num_degenerate = 0
    n_ok = n_bad = n_dropped = 0
    f_ok = f_bad = None
    if write_train:
        os.makedirs(train_dir, exist_ok=True)
        f_ok = open(train_ok_path, "w", encoding="utf-8")
        f_bad = open(train_bad_path, "w", encoding="utf-8") if args.save_incorrect else None
    with (open(out_path, "w", encoding="utf-8") if not write_train or args.save_raw else nullcontext(None)) as fout:
        for i in range(len(outputs)):
            ds_idx = local_idxs[i]
            row = ds[ds_idx]
            index = row["index"] if "index" in row else ds_idx
            degenerate = [_is_degenerate(t, f) for t, f in zip(texts[i], finished[i])]
            num_degenerate += sum(degenerate)

            record = {
                "index": index,
                "problem": row["problem"],
                "solution": row.get("solution", None),
                "answer": row.get("answer", None),
                "full_generated_text": texts[i][0] if K == 1 else texts[i],
                "degenerate": bool(degenerate[0]) if K == 1 else [bool(d) for d in degenerate],
            }
            if args.task in MCQ_TASKS:
                record["choices"] = row["choices"]
                record["choice_labels"] = row["choice_labels"]
                if args.task == 'bbh' and 'subtask' in row:
                    record["subtask"] = row["subtask"]
            elif args.task == 'livecodebench':
                # Do not store private_test_cases here: it is ~98% of each record
                # (multi-MB) and identical across every seed/model run. Test cases
                # are benchmark data, so scoring loads them once by index from a
                # compact cache (evaluation.code_exec.get_livecodebench_tests).
                # Keep the tiny public_test_cases for convenience.
                record["public_test_cases"] = row.get("public_test_cases")
                record["difficulty"] = row.get("difficulty")
                record["platform"] = row.get("platform")
                record["question_id"] = row.get("question_id")
                # contest_date is a datetime; convert it to an ISO string for JSON
                cd = row.get("contest_date")
                record["contest_date"] = cd.isoformat() if hasattr(cd, "isoformat") else cd

            if fout is not None:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not write_train:
                continue
            # Training rows: drop degenerate generations, split the rest into
            # (thinking, answer) and file them by correctness. The correctness rule is
            # the one print_base_accuracy.py applies to the post-</think> text.
            n_choices = len(row["choice_labels"]) if args.task in MCQ_TASKS else 0
            for text, is_degen in zip(texts[i], degenerate):
                thinking, solution = split_think(text)
                if is_degen or thinking is None:
                    n_dropped += 1
                    continue
                train_rec = {"index": index, "solution": solution, "thinking": thinking,
                             "answer": row.get("answer", None)}
                if args.task in CODE_TASKS:
                    ok = bool(solution.strip())
                else:
                    ok = is_generation_correct(solution, row.get("answer"), n_choices)
                target_file = f_ok if ok else f_bad
                if target_file is not None:
                    target_file.write(json.dumps(train_rec, ensure_ascii=False) + "\n")
                n_ok += int(ok)
                n_bad += int(not ok)
    if write_train:
        f_ok.close()
        if f_bad is not None:
            f_bad.close()
        from utils.artifact_paths import merge_distill_shards
        merge_distill_shards(train_root, train_name, args.num_shards, args.shard_id)

    n_gen = len(outputs) * K
    if num_degenerate > 0:
        print(
            f"[WARNING][Shard {args.shard_id}/{args.num_shards}] "
            f"{num_degenerate}/{n_gen} generations are degenerate."
        )
    if not write_train or args.save_raw:
        print(
            f"[DONE][Shard {args.shard_id}/{args.num_shards}] "
            f"saved {len(outputs)} examples x {K} sample(s) → {out_path}"
        )
    if write_train:
        print(
            f"[DONE][Shard {args.shard_id}/{args.num_shards}] training rows: "
            f"{n_ok} correct → {train_root}/{train_name}; {n_bad} incorrect (save={args.save_incorrect}), "
            f"{n_dropped} dropped (degenerate / unclosed <think>)"
        )

if __name__ == "__main__":
    from time import time

    start_t = time()
    main()
    end_t = time()
    print(f"Total time: {(end_t - start_t) / 60:.2f} minutes")
