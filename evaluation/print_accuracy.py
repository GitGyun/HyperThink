from utils.dataset_registry import TEST_TASKS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from evaluation.evaluator import extract_final_answer, is_correct
from evaluation.code_exec import CODE_TASKS, score_code_task, load_code_tests_by_index, get_livecodebench_tests
from utils.eval_utils import extract_letter_answer
from transformers import AutoTokenizer
from itertools import combinations
import signal
from contextlib import contextmanager
import sys
import os
import json

# Number of options for each letter-MCQ task, used when a record has no n_choices field.
MCQ_NCHOICES = {'commonsenseqa': 5, 'bbh': 7}


class _ScoreTimeout(BaseException):
    """Raised by the SIGALRM handler. Subclasses BaseException so is_correct's
    `except (ValueError, ...)` / `except Exception` clauses never swallow it."""


@contextmanager
def _time_limit(seconds: float):
    """Per-call wall-clock timeout via SIGALRM, without spawning a process.
    Works on the main thread on Unix, which is how this script runs."""
    def _handler(signum, frame):
        raise _ScoreTimeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _score_code_outputs(ds, args, dset_tag, tokenizer):
    """Exec-based scoring for code tasks (livecodebench), reporting the same metrics
    as evaluation/print_base_accuracy.py: all-tests-pass accuracy, Pass@K, average
    tokens, degeneracy rate, and has-code rate. Test cases are loaded by index from
    --lcb_tests, the livecodebench cache, or the base-model output, so they match the
    ones print_base_accuracy.py scores against."""
    key = args.key
    examples = list(ds)
    n = len(examples)
    if args.lcb_tests:
        src = args.lcb_tests
        tests_by_idx = load_code_tests_by_index(args.task, src, args.max_private_tests)
    else:
        src = 'livecodebench tests cache'
        tests_by_idx = get_livecodebench_tests(args.max_private_tests)
    if not tests_by_idx:
        print(f"[error] no test cases found ({src}). Pass --lcb_tests <glob>.")
        return

    # Hypernetwork and distillation outputs carry no thinking trace, so scoring and
    # token counts use the full generation. A sample is degenerate when its scored
    # text is empty.
    samp_cap = args.eval_samples if args.eval_samples > 0 else None

    K = max((len(ex[key]) if isinstance(ex[key], list) else 1) for ex in examples)
    if samp_cap:
        K = min(K, samp_cap)
    payloads, missing = [], 0
    raw_seeds = [[""] * n for _ in range(K)]     # unfiltered, for tokens
    texts_seeds = [[""] * n for _ in range(K)]   # full generation, for scoring
    for i, ex in enumerate(examples):
        pl = tests_by_idx.get(int(ex['index']))
        if pl is None:
            missing += 1
            pl = {'kind': 'stdin', 'cases': [], 'fn': None}   # no tests → fails
        payloads.append(pl)
        preds = ex[key] if isinstance(ex[key], list) else [ex[key]]
        if samp_cap:
            preds = preds[:samp_cap]
        for k in range(K):
            raw = (preds[k] if k < len(preds) else "") or ""
            raw_seeds[k][i] = raw
            texts_seeds[k][i] = raw                 # full raw for scoring

    correct, has_code = score_code_task(args.task, texts_seeds, payloads=payloads,
                                        timeout=args.exec_timeout, workers=args.exec_workers)

    # Token lengths of the raw generations, batch-tokenized once per seed.
    tok_len_seeds = [list(tokenizer([raw_seeds[k][i] for i in range(n)],
                                    add_special_tokens=False, return_length=True)['length'])
                     for k in range(K)]

    n_correct = n_pass = n_has = n_tokens = n_degen = 0.0
    for i in range(n):
        cor = [correct[k][i] for k in range(K)]
        n_pass += int(sum(cor) > 0)
        n_correct += int(np.mean(cor) > 0.5) if args.major_vote else float(np.mean(cor))
        n_has += float(np.mean([has_code[k][i] for k in range(K)]))
        n_tokens += np.mean([tok_len_seeds[k][i] for k in range(K)])
        n_degen += np.mean([not texts_seeds[k][i].strip() for k in range(K)])

    print(f"All-tests-pass accuracy in {dset_tag}: {n_correct:.2f}/{n} = {n_correct/n*100:.2f}%"
          f"  (private cap={args.max_private_tests})")
    print(f"Pass@{K} Accuracy in {dset_tag}: {n_pass:.2f}/{n} = {n_pass/n*100:.2f}%")
    print(f"Average tokens per example: {n_tokens/n:.2f}")
    print(f"Degeneracy rate: {n_degen/n*100:.2f}%")
    print(f"Has-code rate: {n_has/n*100:.2f}%")
    if missing:
        print(f"[warn] {missing}/{n} examples had no matching tests (scored as fail)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', type=str, default=None,
                    help='Eval run directory written by evaluation/evaluate_model.py '
                         '(<output_dir>/<run_tag>), or a single JSONL file.')
    ap.add_argument('--task', choices=TEST_TASKS, type=str, default='gsm8k-test')
    ap.add_argument('--key', type=str, default='hypernet_eval_generated_texts')
    ap.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B')
    ap.add_argument("--output_postfix", '-ptf', type=str, default='')
    ap.add_argument('--n_tasks', type=int, default=-1)
    ap.add_argument('--eval_samples', type=int, default=-1)
    ap.add_argument('--major_vote', action='store_true', default=False)
    # Code-task (livecodebench) exec scoring.
    ap.add_argument('--max_private_tests', type=int, default=50,
                    help='Cap on private (LCB) test cases per example. -1 = no cap.')
    ap.add_argument('--exec_timeout', type=float, default=6.0,
                    help='Per-test wall-clock timeout (seconds) for code execution.')
    ap.add_argument('--exec_workers', type=int, default=None,
                    help='Thread workers for code execution (default min(32, cpu*4)).')
    ap.add_argument('--lcb_tests', type=str, default='',
                    help='Glob of JSONL(s) carrying code test cases keyed by index '
                         '(default: the base-model output for --model/--task).')
    args = ap.parse_args()
    from utils.model_registry import model_tag
    if args.outputs is None:
        args.outputs = f"outputs/results/{model_tag(args.model)}/{args.task}"


    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    dset_tag = args.task   # default; overridden below for the special-cased tasks
    if not args.outputs.endswith('.jsonl'):
        patterns = [
            f'{args.outputs}/{dset_tag}*/dataset{args.output_postfix}.jsonl',
            f'{args.outputs}/{dset_tag}*/dataset{args.output_postfix}_shard*.jsonl',
            f'{args.outputs}/*{args.output_postfix}.jsonl',
        ]
        public_pattern = f"{args.outputs}/hyperthink{args.output_postfix}_shard_*_of_*.jsonl"
        if glob(public_pattern):
            patterns = [public_pattern]
        outputs = sorted(set(sum([glob(pattern) for pattern in patterns], [])))
        if len(outputs) == 0:
            print(f'No match in {patterns}')
            sys.exit()
    else:
        outputs = args.outputs
        if not os.path.exists(outputs):
            print(f'No paths found in {args.outputs}')
            sys.exit()

    # Load outputs as plain dicts (json.loads per line) rather than
    # datasets.load_dataset, whose pyarrow schema inference fails when a column
    # like 'answer' mixes int-like and string values (e.g. AIME '336^\circ').
    _paths = outputs if isinstance(outputs, list) else [outputs]
    ds = []
    for _p in _paths:
        with open(_p) as _fh:
            ds += [json.loads(_l) for _l in _fh if _l.strip()]
    # Keep the first record of each index.
    seen = set()
    unique_ids = []
    for i, r in enumerate(ds):
        v = r['index']
        if v not in seen:
            seen.add(v)
            unique_ids.append(i)
    ds = [ds[i] for i in unique_ids]

    if args.n_tasks > 0:
        ds = ds[:args.n_tasks]

    # Code tasks (livecodebench) are scored by executing the generated programs.
    if args.task in CODE_TASKS:
        _score_code_outputs(ds, args, dset_tag, tokenizer)
        return

    n_correct = 0
    n_pass_correct = 0
    n_tokens = 0
    max_tokens = 0
    n_degen = 0
    eval_pass_k = False
    n_eval_samples = 1   # set to len(pred) for list preds below; 1 keeps scalar-pred runs safe
    task_n_choices = MCQ_NCHOICES.get(args.task, 0)
    def _is_correct_with_timeout(pred_text, answer, timeout=5):
        """Run is_correct with a cheap SIGALRM timeout (no per-call process spawn)."""
        extracted = extract_final_answer(pred_text)
        if not extracted:
            return False
        try:
            with _time_limit(timeout):
                return bool(is_correct(extracted, answer))
        except _ScoreTimeout:
            return False
        except Exception:
            return False

    def _score_one(pred_text, ex):
        """Per-generation correctness: letter-MCQ (n_choices) / math."""
        nc = ex.get('n_choices') or task_n_choices
        if nc and nc > 0:
            return extract_letter_answer(pred_text, n_choices=nc) == str(ex['answer']).strip().upper()
        return _is_correct_with_timeout(pred_text, ex['answer'])

    for index in tqdm(range(len(ds)), desc='Evaluating outputs... ',
                      bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar:40}"):
        ex = ds[index]
        try:
            pred = ex[args.key]
        except:
            print(f'Key {args.key} not found in the dataset.')
            print(f'Available keys: {list(ex.keys())}')
            sys.exit()

        if isinstance(pred, list):
            raw_pred = list(pred)         # full generations, for degeneracy
            n_degen += np.mean([int(('<think>' in p or '</think>' in p) and not '<think>\n\n</think>' in p) for p in raw_pred])
            score_pred = list(raw_pred)                # full raw for scoring

            if args.major_vote:
                eval_samples = len(score_pred) if args.eval_samples < 0 else args.eval_samples
                vote_accuracy = 0
                for pred_comb in combinations(score_pred, eval_samples):
                    votes = [_score_one(p, ex) for p in pred_comb]
                    vote_accuracy += np.mean(votes) > 0.5
                correct = vote_accuracy / len(list(combinations(score_pred, eval_samples)))
                n_eval_samples = 1
            else:
                if args.eval_samples > 0:
                    score_pred = score_pred[:args.eval_samples]
                    raw_pred = raw_pred[:args.eval_samples]
                correct = [_score_one(p, ex) for p in score_pred]

                if not eval_pass_k:
                    eval_pass_k = True
                    n_eval_samples = len(score_pred)
                n_pass_correct += int(sum(correct) > 0)
                correct = np.mean(correct)

            # Token counts use the full generation.
            lengths = [len(tokenizer.tokenize(p)) for p in score_pred]
            n_tokens += np.mean(lengths)
            max_tokens = max(max_tokens, max(lengths, default=0))

        else:
            raw_pred = pred
            n_degen += int(('<think>' in pred or '</think>' in pred) and not '<think>\n\n</think>' in pred)
            score_pred = pred                          # full raw for scoring
            correct = _score_one(score_pred, ex)
            length = len(tokenizer.tokenize(score_pred))
            n_tokens += length
            max_tokens = max(max_tokens, length)

        n_correct += correct

    n_total = len(ds)

    print(f"Accuracy in test split: {n_correct:.2f}/{n_total} = {n_correct/n_total*100:.2f}%")
    if n_eval_samples > 1:
        print(f"Pass@{n_eval_samples} Accuracy in test split: {n_pass_correct/n_total*100:.2f}%")
    print(f"Average tokens per example: {n_tokens/len(ds):.2f}")
    print(f"Max tokens over example: {max_tokens}")
    print(f"Degenerate answers: {n_degen:.2f}")

if __name__ == "__main__":
    main()
