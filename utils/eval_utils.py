import numpy as np
import torch
import os
import json
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer

import re
from utils.dataset_registry import to_chat_prompt
from evaluation.evaluator import extract_final_answer, is_correct


def _build_task_query(example) -> str:
    """Return the repository-standard task prompt before chat templating."""
    from utils.dataset_registry import DATASET_REGISTRY

    ds_name = example.get('dataset_name') if isinstance(example, dict) else None
    if ds_name and ds_name in DATASET_REGISTRY:
        prompt_fn = DATASET_REGISTRY[ds_name][3]
        return prompt_fn(example)

    query = example.get('query_text') if isinstance(example, dict) else None
    if query is None:
        query = example.get('problem', '') if isinstance(example, dict) else str(example)
    return query


def build_lm_prompt(tokenizer, example, enable_thinking: bool = True) -> str:
    """Build the chat-formatted LM prompt for one example.

    Mirrors extract_cot_sharded.py exactly: dispatch to the task's
    prompt_fn via DATASET_REGISTRY, then apply the chat template via
    to_chat_prompt. `example` is a row dict; it must carry `dataset_name` so we
    can look up the prompt_fn, and falls back to `query_text` / `problem`.
    """
    query = _build_task_query(example)
    return to_chat_prompt(tokenizer, query, enable_thinking=enable_thinking)


def resolve_eos_ids(model, tokenizer):
    """Use model.generation_config.eos_token_id (multi-stop list) with fallback to tokenizer."""
    module = getattr(model, 'module', model)
    eos = getattr(getattr(module, 'generation_config', None), 'eos_token_id', None)
    if eos is None:
        base = getattr(getattr(module, 'base_model', module), 'base_model', module)
        eos = getattr(getattr(base, 'generation_config', None), 'eos_token_id', None)
    return eos or tokenizer.eos_token_id


def gen_sampling_kwargs(temperature: float = 0.6, top_p: float = 0.95) -> dict:
    """Sampling kwargs for HF generate, matching the extraction defaults."""
    if temperature > 0:
        return dict(do_sample=True, temperature=temperature, top_p=top_p)
    return dict(do_sample=False)


def extract_letter_answer(text: str, n_choices: int = 4) -> str:
    """Extract MCQ letter answer from model output, generalized to arbitrary number of choices (A..H supported).

    Works for OpenBookQA/LogiQA (A-D), CommonsenseQA (A-E), BBH (up to A-G), QASC (A-H).
    """
    raw = extract_final_answer(text)
    if raw is None:
        return ""
    s = raw.strip().strip('*').strip()
    upper = chr(ord('A') + n_choices - 1)
    cls = f"A-{upper}a-{upper.lower()}"

    # Single letter already
    if re.fullmatch(f'[{cls}]', s):
        return s.upper()

    # "B. 4", "\\text{C}", "C. \\ i" etc.
    head_re = re.compile(rf'^[\\]*(?:text\{{)?([{cls}])[.,)}}\s\\]', re.IGNORECASE)
    m = head_re.match(s)
    if m:
        return m.group(1).upper()

    # "(B)", "( B )"
    m = re.fullmatch(rf'\(\s*([{cls}])\s*\)', s)
    if m:
        return m.group(1).upper()

    # Last resort: first standalone valid letter in the string
    m = re.search(rf'\b([{cls}])\b', s)
    if m:
        return m.group(1).upper()

    return s.strip().upper()


def split_think(text: str):
    """Split a thinking-mode generation into (thinking, answer).

    Returns (None, None) when the <think> block was never closed (the generation
    was truncated or degenerate), so callers can drop it."""
    if not text or "</think>" not in text:
        return None, None
    head, tail = text.split("</think>", 1)
    return head.split("<think>", 1)[-1].strip(), tail.strip()


def is_generation_correct(text: str, answer, n_choices: int = 0) -> bool:
    """The single correctness rule shared by base scoring (print_base_accuracy.py),
    training-row filtering at extraction (extract_cot_sharded.py) and the hypernet
    evaluation loop: letter match for MCQ tasks (n_choices > 0), otherwise the
    final boxed answer checked with the sympy-backed matcher."""
    if n_choices > 0:
        return extract_letter_answer(text, n_choices=n_choices) == str(answer).strip().upper()
    extracted = extract_final_answer(text)
    if not extracted:
        return False
    return bool(is_correct(extracted, answer))


def print_stats(stats, stat_name):
    average_stats_per_question = [np.mean(c) for c in stats]
    std_stats_per_question = [np.std(c) for c in stats]
    average_stats = np.mean(average_stats_per_question)
    std_stats = np.mean(std_stats_per_question)
    print(f"\nAverage {stat_name}: {average_stats:.4f} ± {std_stats:.4f}")


@torch.no_grad()
def _hf_generate_texts(
    model,
    tokenizer,
    prompt: str,
    total_n: int,
    base_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    **model_kwargs,
):
    """Sequential (one example at a time) generation; used when --eval_chunk_n is 1."""
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    gen_out = model.generate(
        **model_kwargs,
        **inputs,
        **gen_sampling_kwargs(temperature, top_p),
        max_new_tokens=max_new_tokens,
        num_return_sequences=total_n,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=resolve_eos_ids(model, tokenizer),
        use_cache=True,
    )
    start = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(gen_out[:, start:], skip_special_tokens=True)


def setup_output_path(args):
    dtag = args.task
    from utils.model_registry import model_tag
    out_path = os.path.join(args.output_dir, model_tag(args.base_model), dtag,
        f"hyperthink{args.output_postfix}_shard_{args.shard_id}_of_{args.num_shards}.jsonl")
    if (not args.reset) and os.path.exists(out_path):
        print(f"Output path {out_path} exists. Exiting to avoid overwriting. Use --reset to overwrite.")
        exit(1)

    return out_path


def _evaluate_texts(texts_k, ex, args):
    """Per-sample correctness and degeneracy for one example's generations.

    Code tasks are scored here as math (no exec); their real scoring is the
    exec-based pass in print_accuracy.py."""
    n_choices = (ex.get('n_choices') or 0) if isinstance(ex, dict) else 0   # >0 for MCQ tasks
    corrects_k = [is_generation_correct(t, ex['answer'], n_choices) for t in texts_k]
    degens_k = [t.strip() == "" for t in texts_k]
    return corrects_k, degens_k


def _prepare_example(ex, args, tokenizer, device):
    """Build the LM prompt and the hypernet inputs for a single example.

    Prompt construction is delegated to build_lm_prompt (the extraction prompt): the
    task's registry prompt_fn is applied and nothing is appended afterwards.
    """
    # Ensure dataset_name is set so build_lm_prompt can dispatch (args.task is the dataset name)
    if isinstance(ex, dict) and 'dataset_name' not in ex:
        ex = dict(ex); ex['dataset_name'] = getattr(args, 'task', None)
    prompt = build_lm_prompt(
        tokenizer, ex, enable_thinking=not getattr(args, 'nothink_mode', False))

    # HFRolloutDataset supplies the legacy per-task hypernet query, including
    # choices/instructions where the original evaluation used them.
    # Fall back to the raw problem only for callers without query_text.
    hypernet_text = ex.get('query_text') or ex.get('problem', '')

    model_kwargs = {}
    if not hasattr(args, '_query_tokenizer'):
        args._query_tokenizer = AutoTokenizer.from_pretrained(
            getattr(args, 'text_encoder_name', args.base_model), use_fast=True, trust_remote_code=True)
    tok = args._query_tokenizer(
        hypernet_text, padding=True, truncation=True,
        max_length=args.max_new_tokens, return_tensors="pt",
    )
    model_kwargs['query_input_ids'] = tok['input_ids'].to(device)
    model_kwargs['query_attention_mask'] = tok['attention_mask'].to(device)
    model_kwargs['query_task_ids'] = torch.tensor([int(ex['index'])], device=device)
    model_kwargs['query_text'] = [hypernet_text]

    return prompt, model_kwargs


def _left_pad(tokenizer, prompts, device):
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tokenizer.padding_side = orig_padding_side
    return inputs


def _seed_for(args, batch):
    base_seed = args.seed + batch[0][0]['index']
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)


# Preserve the continuation prompt used by the original evaluation implementation.
THINKING_BUDGET_STOP_PROMPT = (
    "\nConsidering the limited time by the user, I have to give the solution based on "
    "the thinking directly now.\n</think>.\n\nFinal Answer:"
)


def _active_thinking_budget(args):
    """Legacy no-think mode bypasses both phases of budgeted reasoning."""
    if getattr(args, 'nothink_mode', False):
        return None
    return getattr(args, 'thinking_budgets', None)


def _hit_length_flags(gen_tokens, eos):
    """A generated row without any EOS token exhausted its token budget."""
    if isinstance(eos, (list, tuple, set)):
        eos_ids = {int(e) for e in eos if e is not None}
    else:
        eos_ids = {int(eos)} if eos is not None else set()
    return [not any(int(tok) in eos_ids for tok in row) for row in gen_tokens.tolist()]


def _thinking_budget_finalize(prompts, think_texts, K, continue_generate, force_n, hit_length=None):
    """Continue capped, empty, or unclosed traces; keep completed answers unchanged."""
    finals = list(think_texts)
    pending, cont_prompts = [], []
    for s, text in enumerate(think_texts):
        if ((hit_length is not None and hit_length[s])
                or not text.strip() or text.find('</think>') <= 0):
            pending.append(s)
            cont_prompts.append(prompts[s // K] + text + THINKING_BUDGET_STOP_PROMPT)
    if pending:
        continuations = continue_generate(cont_prompts, pending, force_n)
        for s, continuation in zip(pending, continuations):
            finals[s] = think_texts[s] + THINKING_BUDGET_STOP_PROMPT + continuation
    return finals


def _finish_thinking_budget(model, tokenizer, batch, args, device, texts, gen_tokens, eos,
                            bias_vec=None):
    """Generate answers only for pending samples, reusing their original deltas."""
    def continue_generate(prompts, pending, max_tokens):
        inputs = _left_pad(tokenizer, prompts, device)
        model_kwargs = {'bias_vec': bias_vec[pending]} if bias_vec is not None else {}
        outputs = model.generate(
            **inputs, **model_kwargs,
            **gen_sampling_kwargs(args.temperature, args.top_p),
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos,
            use_cache=True,
        )
        return tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)

    return _thinking_budget_finalize(
        [item[1] for item in batch], texts, args.samples_per_adapter,
        continue_generate, getattr(args, 'thinking_budgets_force_tokens', 100),
        _hit_length_flags(gen_tokens, eos),
    )


@torch.no_grad()
def _generate_batch_hypernet(model, tokenizer, batch, args, device):
    """Generate texts for a batch of examples using batched bias injection.

    Args:
        model: SFTWrapper (has .hypernet and .base_model)
        batch: list of (example, prompt, model_kwargs) tuples
    Returns:
        list of list[str], one per example in the batch
    """
    B = len(batch)
    K = args.samples_per_adapter

    # 1. Run the hypernet on the batched queries (text-encoder tokenizer, as in training)
    query_texts_list = [mk.get('query_text', [''])[0] for _, _, mk in batch]
    if not hasattr(args, '_query_tokenizer'):
        args._query_tokenizer = AutoTokenizer.from_pretrained(
            getattr(args, 'text_encoder_name', args.base_model), use_fast=True, trust_remote_code=True)
    hp_tok = args._query_tokenizer(
        query_texts_list, padding=True, truncation=True,
        max_length=args.max_new_tokens, return_tensors="pt",
    )
    hp_out = model.hypernet(
        input_ids=hp_tok['input_ids'].to(device),
        attention_mask=hp_tok['attention_mask'].to(device),
        task_ids=torch.tensor([int(ex['index']) for ex, _, _ in batch], device=device),
        query_text=query_texts_list,
    )
    bias_vec = hp_out['pred_delta']                              # [B, D]
    bias_vec_expanded = bias_vec.repeat_interleave(K, dim=0)     # [B*K, D]

    # 2. Generate with the per-example deltas injected
    inputs = _left_pad(tokenizer, [item[1] for item in batch], device)
    _seed_for(args, batch)
    prompt_len = inputs['input_ids'].shape[1]

    gen_out = model.base_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        bias_vec=bias_vec_expanded,
        **gen_sampling_kwargs(args.temperature, args.top_p),
        max_new_tokens=_active_thinking_budget(args) or args.max_new_tokens,
        num_return_sequences=K,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=resolve_eos_ids(model, tokenizer),
        use_cache=True,
    )
    texts = tokenizer.batch_decode(gen_out[:, prompt_len:], skip_special_tokens=True)
    if _active_thinking_budget(args) is not None:
        texts = _finish_thinking_budget(
            model.base_model, tokenizer, batch, args, device, texts,
            gen_out[:, prompt_len:], resolve_eos_ids(model, tokenizer),
            bias_vec=bias_vec_expanded,
        )
    return [texts[i * K:(i + 1) * K] for i in range(B)]


def eval_loop(ds, model, args, device):
    out_path = setup_output_path(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_generated_texts: List[List[List[str]]] = []
    all_correctness: List[List[List[float]]] = []
    all_degeneracy: List[List[List[float]]] = []
    chunk_size = getattr(args, 'eval_chunk_n', None) or 1
    # The batch path also handles one example and preserves its bias across phases.
    use_batched = (chunk_size > 1) or _active_thinking_budget(args) is not None
    if use_batched:
        print(f"[eval] Batched generation enabled: chunk_size={chunk_size}")

    chunk_buf = []  # list of (example, prompt, model_kwargs)
    pbar = tqdm(ds, total=len(ds), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar:40}")

    def _flush_chunk():
        """Process accumulated chunk and append results."""
        if not chunk_buf:
            return
        batch_results = _generate_batch_hypernet(model, tokenizer, chunk_buf, args, device)
        for (ex_i, _, _), texts_k in zip(chunk_buf, batch_results):
            corrects_k, degens_k = _evaluate_texts(texts_k, ex_i, args)
            all_generated_texts.append(texts_k)
            all_correctness.append(corrects_k)
            all_degeneracy.append(degens_k)

    for ex in pbar:
        prompt, model_kwargs = _prepare_example(ex, args, tokenizer, device)

        if use_batched:
            chunk_buf.append((ex, prompt, model_kwargs))
            if len(chunk_buf) >= chunk_size:
                _flush_chunk()
                chunk_buf = []
        else:
            texts_k = _hf_generate_texts(
                model, tokenizer, prompt,
                total_n=args.samples_per_adapter,
                base_seed=args.seed + int(ex['index']),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                **model_kwargs,
            )
            corrects_k, degens_k = _evaluate_texts(texts_k, ex, args)
            all_generated_texts.append(texts_k)
            all_correctness.append(corrects_k)
            all_degeneracy.append(degens_k)

    if use_batched and chunk_buf:
        _flush_chunk()
        chunk_buf = []

    # Save results + generic columns as JSONL (the format print_accuracy.py reads back).
    assert len(ds) == len(all_generated_texts) == len(all_correctness) == len(all_degeneracy), \
        f"result/row length mismatch: {len(ds)} rows vs {len(all_generated_texts)} results"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            rec = dict(ds[i])
            rec[f"hypernet_eval_generated_texts"] = all_generated_texts[i]
            rec[f"hypernet_eval_correctness"] = all_correctness[i]
            rec[f"hypernet_eval_degeneracy"] = all_degeneracy[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved final dataset as JSONL to: {out_path}")

    print_stats(all_correctness, "correctness")
    print_stats(all_degeneracy, "degeneracy")
    all_lengths = [[len(text) for text in ex_texts] for ex_texts in all_generated_texts]
    print_stats(all_lengths, "length")
