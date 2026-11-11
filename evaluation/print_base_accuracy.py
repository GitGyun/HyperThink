from utils.dataset_registry import TEST_TASKS
import os
import argparse
import json
import hashlib
import multiprocessing as mp
import numpy as np
from glob import glob
from tqdm import tqdm
from evaluation.code_exec import CODE_TASKS, TEST_FIELDS, score_code_task
from utils.eval_utils import is_generation_correct


# Tasks that use letter MCQ scoring + (n_choices, default split size for output paths)
MCQ_TASK_INFO = {'commonsenseqa': (5, 1221), 'bbh': (7, 420)}
MATH_TASK_INFO = {'gsm8k-test': 1319, 'math500': 500, 'aime_24_25': 60}

# Combined tasks → [(subsplit_name, index_lo, index_hi)] for a per-component breakdown.
# index is the combined-dataset position (2024 = [0,30), 2025 = [30,60)).
SUBSPLITS = {
    'aime_24_25': [('AIME 2024', 0, 30), ('AIME 2025', 30, 60)],
}
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Silence the per-call "sequence length longer than max" tokenizer warning we
# trigger on purpose when batch-tokenizing full generations for length counting.
hf_logging.set_verbosity_error()


def _load_slim_jsonl(load_arg):
    """Stream a JSONL keeping only the keys scoring needs. A file that holds several
    seeds is subsampled back to one row per problem, then --n_tasks is applied.

    Reading the file directly skips large unused columns such as livecodebench's
    private_test_cases. Returns (records, original_row_count).
    """
    path, needed_keys, default_size, n_tasks = load_arg
    recs = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            recs.append({k: obj.get(k) for k in needed_keys})
    orig_len = len(recs)
    if orig_len > default_size and orig_len % default_size == 0:
        step = orig_len // default_size
        recs = recs[step - 1::step]
    if n_tasks > 0:
        recs = recs[:n_tasks]
    return recs, orig_len


def _expand_samples(recs, key):
    """A record's generation field is one string (one sample per run) or a list of K
    strings (`--num_samples K` at extraction). Expand a file into K per-sample record
    lists so every sample is scored like a separate seed (pass@k counts them all)."""
    if not recs or not isinstance(recs[0].get(key), list):
        return [recs]
    K = len(recs[0][key])
    return [[{**r, key: (r[key][j] if isinstance(r[key], list) and j < len(r[key]) else '')}
             for r in recs] for j in range(K)]


def _cache_key(args, eval_outputs):
    """Fingerprint every input that affects the printed metrics: the requested
    model/task/outputs_tag, the other result-affecting flags, and a (name, size,
    mtime) signature of the matched files (so regenerated/added shards miss)."""
    sig = {
        'model': args.model, 'task': args.task, 'outputs_tag': args.outputs_tag,
        'key': args.key, 'n_tasks': args.n_tasks,
        'eval_samples': args.eval_samples, 'major_vote': args.major_vote,
        'max_private_tests': args.max_private_tests,
        'exec_timeout': args.exec_timeout,
        'files': sorted((os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f)))
                        for f in eval_outputs),
    }
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def _print_metrics(m):
    """Print the metrics dict. Fresh runs and cache hits both use it, so the output
    is identical either way."""
    dset, n_total, n_correct = m['dset_tag'], m['n_total'], m['n_correct']
    if m['is_code']:
        print(f"All-tests-pass accuracy in {dset}: {n_correct:.2f}/{n_total} = {n_correct/n_total*100:.2f}%"
              f"  (private cap={m['max_private_tests']})")
    else:
        print(f"Accuracy in {dset}: {n_correct:.2f}/{n_total} = {n_correct/n_total*100:.2f}%")
    print(f"Pass@{m['pass_k']} Accuracy in {dset}: {m['n_pass_correct']:.2f}/{n_total} = {m['n_pass_correct']/n_total*100:.2f}%")
    print(f"Average tokens per example: {m['avg_tokens']:.2f}")
    print(f"Degeneracy rate: {m['degen_rate']:.2f}%")
    if m['is_code']:
        print(f"Has-code rate: {m['has_code_rate']:.2f}%")
    for name, ss in (m.get('subsplits') or {}).items():
        print(f"  [{name}] Accuracy: {ss['acc']:.2f}%  Pass@{m['pass_k']}: {ss['pass']:.2f}%  "
              f"avg tokens: {ss['avg_tokens']:.2f}  (n={ss['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=TEST_TASKS, type=str, default='gsm8k-test')
    ap.add_argument('--key', type=str, default='full_generated_text')
    ap.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B')
    ap.add_argument('--n_tasks', type=int, default=-1)
    ap.add_argument('--eval_samples', type=int, default=-1)
    ap.add_argument('--results_root', default='outputs/results')
    ap.add_argument('--outputs_tag', '-ot', type=str, default='')
    ap.add_argument('--major_vote', action='store_true', default=False)
    # Code-task (livecodebench) exec scoring.
    ap.add_argument('--max_private_tests', type=int, default=50,
                    help='Cap on private (LCB) test cases per example. -1 = no cap.')
    ap.add_argument('--exec_timeout', type=float, default=6.0,
                    help='Per-test wall-clock timeout (seconds) for code execution.')
    ap.add_argument('--exec_workers', type=int, default=None,
                    help='Thread workers for code execution (default min(32, cpu*4)).')
    ap.add_argument('--refresh', action='store_true', default=False,
                    help='Ignore any cached metrics and recompute (then refresh the cache).')
    args = ap.parse_args()

    dset_tag = args.task
    default_size = {**MATH_TASK_INFO, 'livecodebench': 400,
                    'commonsenseqa': 1221, 'bbh': 420}[args.task]

    from utils.model_registry import model_tag
    outputs_root = os.path.join(args.results_root, model_tag(args.model))
    mode = "nothink" if "nothink" in args.outputs_tag else "think"
    args.outputs = f"{outputs_root}/{args.task}/{mode}_shard_*_of_*.jsonl"
    print(f"Loading outputs from {args.outputs}...")
    eval_outputs = sorted(glob(args.outputs), key=lambda p: int(p.rsplit("_shard_", 1)[1].split("_")[0]))
    if len(eval_outputs) == 0:
        raise ValueError(f'No outputs found for pattern: {args.outputs}')

    # Skip empty files, such as a shard that was created but never written.
    empty = [f for f in eval_outputs if os.path.getsize(f) == 0]
    if empty:
        print(f'Skipping {len(empty)} empty file(s): ' +
              ', '.join(os.path.basename(f) for f in empty))
    eval_outputs = [f for f in eval_outputs if os.path.getsize(f) > 0]
    if not eval_outputs:
        raise ValueError(f'All files matching {args.outputs} are empty.')

    # Metrics cache: skip loading and scoring when this exact
    # (model, task, outputs_tag, flags, input files) was already scored.
    cache_dir = os.path.join(outputs_root, '.metrics_cache')
    cache_file = os.path.join(cache_dir, f'{_cache_key(args, eval_outputs)}.json')
    if not args.refresh and os.path.exists(cache_file):
        with open(cache_file) as fh:
            cached = json.load(fh)
        print(f"[cache] hit ({os.path.relpath(cache_file)}); pass --refresh to recompute")
        _print_metrics(cached)
        return

    # Only needed on a cache miss (loaded after the cache check so warm hits skip it).
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    # Only these fields are read below; everything else (incl. multi-MB code-test
    # columns) is skipped during load. See _load_slim_jsonl.
    needed_keys = [args.key, 'answer']
    if args.task in SUBSPLITS:
        needed_keys.append('index')   # to split combined tasks (e.g. aime_24_25) by component

    # Each shard contains different problems, so retain its test-case fields.
    test_fields = TEST_FIELDS.get(args.task, []) if args.task in CODE_TASKS else []
    load_args = [(f, needed_keys + test_fields, default_size, -1)
                 for j, f in enumerate(eval_outputs)]
    n_workers = min(len(load_args), os.cpu_count() or 1, 8)
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            loaded = pool.map(_load_slim_jsonl, load_args)
    else:
        loaded = [_load_slim_jsonl(a) for a in load_args]

    rows = [row for recs, _ in loaded for row in recs]
    if args.n_tasks > 0:
        rows = rows[:args.n_tasks]
    ds_seeds = _expand_samples(rows, args.key)
    if args.eval_samples > 0:
        ds_seeds = ds_seeds[:args.eval_samples]
    
    print(f'Loaded {len(ds_seeds)} seeds for evaluation ({len(eval_outputs)} file(s)).')

    n_correct = 0
    n_pass_correct = 0
    n_tokens = 0
    n_degen = 0
    n_has_code = 0
    # Batch-tokenize the scored field once per seed instead of calling
    # tokenizer.tokenize() per example. Token counts always use the full raw
    # generation. Scoring text:
    #   thinking mode, </think> missing -> '' (fails; the generation never finished thinking)
    #   thinking mode, </think> present -> the text after </think>
    #   no-think mode                   -> the full raw text
    think_filter = 'nothink' not in args.outputs_tag

    def _score_text(s):
        s = s or ''
        if not think_filter:
            return s                                # nothink: full raw
        if '</think>' not in s:
            return ''                               # incomplete thinking -> fail
        return s.rsplit('</think>', 1)[-1]          # thinking: post-</think>


    tok_len_seeds = []
    for ds in ds_seeds:
        texts = [(r.get(args.key) or '') for r in ds]
        tok_len_seeds.append(list(
            tokenizer(texts, add_special_tokens=False, return_length=True)['length']
        ))

    # Code tasks (livecodebench): execute the extracted code against the (capped)
    # test cases.
    code_pass = code_hascode = None
    if args.task in CODE_TASKS:
        # Scoring text follows _score_text; token counts above use the raw text.
        code_texts = [[_score_text(ds[i][args.key]) for i in range(len(ds))]
                      for ds in ds_seeds]
        max_tests = None if (args.max_private_tests is not None and args.max_private_tests < 0) \
            else args.max_private_tests
        # The per-seed JSONLs only carry `public_test_cases` (private was dropped at
        # extraction time to keep file sizes manageable), so pull pub+priv from the
        # shared cache instead, indexed by record idx.
        from evaluation.code_exec import get_livecodebench_tests
        tests_by_idx = get_livecodebench_tests(
            max_private=(max_tests if max_tests is not None else -1))
        payloads = [tests_by_idx[int(ds_seeds[0][i].get('index', i))]
                    for i in range(len(ds_seeds[0]))]
        print(f'[code-exec] livecodebench: cap={max_tests} timeout={args.exec_timeout}s '
              f'workers={args.exec_workers or "auto"} (pub+priv from cache) ...')
        code_pass, code_hascode = score_code_task(
            args.task, code_texts, payloads=payloads,
            max_tests=max_tests, timeout=args.exec_timeout, workers=args.exec_workers,
        )

    sub = {name: {'c': 0.0, 'p': 0, 't': 0.0, 'n': 0} for name, _, _ in SUBSPLITS.get(args.task, [])}
    for index in tqdm(range(len(ds_seeds[0])), desc='Evaluating outputs... ',
                      bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar:40}"):
        raw_preds = [(ds[index][args.key] or '') for ds in ds_seeds]
        # Scoring text follows _score_text; token counts above use the raw text.
        preds = [_score_text(p) for p in raw_preds]

        if args.task in CODE_TASKS:
            correct = [code_pass[s][index] for s in range(len(ds_seeds))]
            n_has_code += np.mean([code_hascode[s][index] for s in range(len(ds_seeds))])
        else:
            ans = ds_seeds[0][index]['answer']
            n_choices = MCQ_TASK_INFO[args.task][0] if args.task in MCQ_TASK_INFO else 0
            correct = [is_generation_correct(pred, ans, n_choices) for pred in preds]

        ex_pass = int(sum(correct) > 0)
        n_pass_correct += ex_pass

        if args.major_vote:
            correct = int(np.mean(correct) > 0.5)
        else:
            correct = np.mean(correct)

        # Token counts use the raw length of every generation, filtered or not.
        tok_counts = [tok_len_seeds[s][index] for s in range(len(ds_seeds))]
        ex_tok = float(np.mean(tok_counts))
        n_tokens += ex_tok
        n_correct += correct

        if sub:   # per-component accumulation for combined tasks (e.g. aime_24_25)
            ridx = int(ds_seeds[0][index].get('index', index))
            for sname, lo, hi in SUBSPLITS[args.task]:
                if lo <= ridx < hi:
                    s = sub[sname]
                    s['c'] += correct; s['p'] += ex_pass; s['t'] += ex_tok; s['n'] += 1
                    break

        if 'nothink' not in args.outputs_tag:
            n_degen += np.mean([('</think>' not in r) or (not r.strip()) for r in raw_preds])
        else:
            n_degen += np.mean([not r.strip() for r in raw_preds])


    n_total = len(ds_seeds[0])
    is_code = args.task in CODE_TASKS
    metrics = {
        'task': args.task, 'dset_tag': dset_tag, 'is_code': is_code,
        'n_correct': float(n_correct), 'n_total': int(n_total),
        'n_pass_correct': float(n_pass_correct), 'pass_k': len(ds_seeds),
        'avg_tokens': float(n_tokens / n_total), 'degen_rate': float(n_degen / n_total * 100),
        'has_code_rate': float(n_has_code / n_total * 100) if is_code else 0.0,
        'max_private_tests': args.max_private_tests,
        'subsplits': ({name: {'acc': s['c'] / s['n'] * 100, 'pass': s['p'] / s['n'] * 100,
                              'avg_tokens': s['t'] / s['n'], 'n': s['n']}
                       for name, s in sub.items() if s['n'] > 0} or None),
    }
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as fh:
        json.dump(metrics, fh)
    _print_metrics(metrics)

if __name__ == "__main__":
    main()
