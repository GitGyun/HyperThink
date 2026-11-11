"""Exec-based scoring for code-generation tasks (livecodebench).

A generation is correct when the Python code extracted from it passes all of the
(capped) test cases for its problem, the standard code-benchmark metric.

Two execution modes:
  * stdin:       run the program, feed test input on stdin, compare stdout.
                 (livecodebench testtype=="stdin")
  * functional:  import the solution as a module, call the entry point
                 (a `Solution` method, an explicit fn_name, or the last
                 top-level function) with JSON-decoded args, compare the
                 JSON-encoded return value. (livecodebench testtype=="functional")

Generated programs run in subprocesses, dispatched from a thread pool, with a
per-test timeout and a short-circuit on the first failing test.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

CODE_TASKS = {"livecodebench"}

# Which extra (test-case) columns each task needs loaded from its output JSONL.
TEST_FIELDS = {
    "livecodebench": ["public_test_cases", "private_test_cases"],
}


# code extraction
def extract_python_code(text: str) -> str:
    """Pull the *last* python code block out of a generation (the final block is
    almost always the submitted solution); fall back to any fenced block, then
    to a def-starting span."""
    if not text:
        return ""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"(?:^|\n)(def\s+\w+\([^)]*\)[\s\S]+)", text)
    return m.group(1).strip() if m else ""


def _loads(x):
    """Tolerant JSON-string -> python; passes through already-parsed values."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None


def _norm_out(s: str) -> str:
    """Normalize program stdout / expected output for comparison (strip trailing
    whitespace per line, drop leading/trailing blank lines)."""
    return "\n".join(line.rstrip() for line in (s or "").strip().split("\n"))


# runners (each executes the program in a subprocess)
def _run_stdin(code: str, cases: list, timeout: float) -> bool:
    """Run program once per case, feed `input` on stdin, compare stdout. All
    cases must pass; short-circuits on the first failure."""
    if not cases:
        return False
    for c in cases:
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                input=str(c.get("input", "")),
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, Exception):
            return False
        if r.returncode != 0:
            return False
        if _norm_out(r.stdout) != _norm_out(str(c.get("output", ""))):
            return False
    return True


# Harness run in a fresh interpreter: import the solution as a module (so the
# model's own `if __name__ == "__main__"` block does not execute), locate the
# entry point, then run every (capped) test, short-circuiting on first mismatch.
_FUNC_RUNNER = r"""
import sys, json, os, importlib.util
d = sys.argv[1]
fn = sys.argv[2] if len(sys.argv) > 2 else ""
spec = importlib.util.spec_from_file_location("sol", os.path.join(d, "sol.py"))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print("PASS 0 0"); sys.exit(0)
g = vars(mod)
call = None
if fn and fn in g and callable(g[fn]):
    call = g[fn]
elif "Solution" in g:
    try:
        inst = g["Solution"]()
        ms = [m for m in dir(inst) if not m.startswith("_") and callable(getattr(inst, m))]
        call = getattr(inst, ms[0]) if ms else None
    except Exception:
        call = None
if call is None:
    fns = [v for k, v in g.items()
           if callable(v) and getattr(v, "__module__", "") == "sol" and not k.startswith("_")]
    call = fns[-1] if fns else None
try:
    tests = json.load(open(os.path.join(d, "tests.json")))
except Exception:
    tests = []
if call is None:
    print("PASS 0 %d" % len(tests)); sys.exit(0)
passed = 0
for t in tests:
    try:
        inp = str(t.get("input", ""))
        args = [json.loads(x) for x in inp.split("\n") if x.strip() != ""]
        res = call(*args)
        exp = json.loads(t.get("output", ""))
        if json.loads(json.dumps(res)) == exp:
            passed += 1
        else:
            break
    except Exception:
        break
print("PASS %d %d" % (passed, len(tests)))
"""


def _run_functional(code: str, cases: list, fn_name: str, timeout: float) -> bool:
    """All cases run inside one subprocess (one timeout for the whole example)."""
    if not cases:
        return False
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "sol.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "tests.json"), "w") as f:
            json.dump(cases, f)
        try:
            r = subprocess.run(
                [sys.executable, "-c", _FUNC_RUNNER, d, fn_name or ""],
                capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, Exception):
            return False
    for line in reversed((r.stdout or "").strip().splitlines()):
        if line.startswith("PASS "):
            try:
                _, p, n = line.split()
                return int(p) == int(n) and int(n) > 0
            except Exception:
                return False
    return False


# payload building + per-example judge
def build_payload(task: str, rec: dict, max_tests: int) -> dict:
    """Parse a record's test cases into a small, picklable payload (already
    capped, so the multi-MB private_test_cases never reaches a worker)."""
    if task == "livecodebench":
        pub = _loads(rec.get("public_test_cases")) or []
        priv = _loads(rec.get("private_test_cases")) or []
        if max_tests is not None and max_tests >= 0:
            priv = priv[:max_tests]
        cases = list(pub) + list(priv)
        kind = "functional" if (cases and cases[0].get("testtype") == "functional") else "stdin"
        return {"kind": kind, "cases": cases, "fn": None}

    return {"kind": "stdin", "cases": [], "fn": None}


def _judge(job) -> bool:
    """job = (code, payload, timeout). Returns all-tests-pass."""
    code, payload, timeout = job
    if not code:
        return False
    kind = payload["kind"]
    cases = payload.get("cases") or []
    if kind == "functional":
        # one process for the whole example → allow a larger wall budget
        return _run_functional(code, cases, payload.get("fn"),
                               min(60.0, timeout + 0.3 * len(cases)))
    return _run_stdin(code, cases, timeout)


# orchestrator
def load_code_tests_by_index(task, tests_glob, max_tests=50):
    """Build {index: payload} for a code task by reading test cases from JSONL(s)
    that carry the test-case columns (e.g. the base-model output). Reused by the
    training rollout and print_accuracy.py so neither re-downloads the dataset.
    """
    import glob as _glob
    paths = sorted(_glob.glob(tests_glob)) if any(c in tests_glob for c in "*?[") else [tests_glob]
    fields = TEST_FIELDS.get(task, [])
    by_idx = {}
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = obj.get("index")
                if idx is None or int(idx) in by_idx:
                    continue
                rec = {k: obj.get(k) for k in fields}
                by_idx[int(idx)] = build_payload(task, rec, max_tests)
    return by_idx


# The single tests cache stores up to this many private tests/problem; scoring
# caps further at read time (accuracy plateaus well below this).
LCB_CACHE_CAP = 50


def get_livecodebench_tests(max_private=50, cache_dir=".cache/code_tests", rebuild=False):
    """Return {index: payload} for livecodebench from a compact local cache.

    Test cases are benchmark data. Instead of storing the multi-MB
    private_test_cases in every (per-seed, per-model) result file, we keep one
    compact cache (public + first LCB_CACHE_CAP private per problem) built once
    from the locally-cached HF dataset; the requested `max_private` is applied at
    read time. `index` is the position in the registry's 'test' split (matches
    extraction / HFRolloutDataset). Subsequent calls just read the cache (no ~75s
    HF arrow rebuild).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "livecodebench_tests.jsonl")
    read_cap = (LCB_CACHE_CAP if (max_private is None or max_private < 0
                                  or max_private > LCB_CACHE_CAP) else max_private)

    by_idx = {}
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0 and not rebuild:
        with open(cache_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                by_idx[int(o["index"])] = build_payload("livecodebench", o, read_cap)
        return by_idx

    # Build the cache (store up to LCB_CACHE_CAP private) from the cached HF dataset.
    from datasets import load_dataset
    from utils.dataset_registry import DATASET_REGISTRY
    repo, subset, split, _ = DATASET_REGISTRY["livecodebench"]
    ds = load_dataset(repo, subset, split=split)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        for i, r in enumerate(ds):
            priv = (_loads(r.get("private_test_cases")) or [])[:LCB_CACHE_CAP]
            rec = {"index": i,
                   "public_test_cases": r.get("public_test_cases") or "[]",
                   "private_test_cases": json.dumps(priv)}
            fh.write(json.dumps(rec) + "\n")
            by_idx[i] = build_payload("livecodebench", rec, read_cap)
    os.replace(tmp, cache_path)
    return by_idx


def score_code_task(task, texts_seeds, records=None, payloads=None, max_tests=50,
                    timeout=6.0, workers=None):
    """Score every (seed, example) by executing its extracted code.

    Args:
        task:         one of CODE_TASKS.
        texts_seeds:  list (per seed) of generation strings (already think-filtered).
        records:      list of record dicts (one per example) carrying the test
                      cases, taken from a single seed (tests are the same for every seed).
        payloads:     pre-built per-example payloads (alternative to `records`).
        max_tests:    cap on private (LCB) test cases per example.
        timeout:      per-test (stdin) wall-clock timeout in seconds.
        workers:      thread count (default min(32, cpu_count*4)).

    Returns:
        (correct, has_code): two [n_seeds][n_examples] bool matrices.
    """
    n_seeds = len(texts_seeds)
    n = len(texts_seeds[0])
    if workers is None:
        workers = min(32, (os.cpu_count() or 8) * 4)

    # Parse + cap each example's tests once (tests are identical across seeds).
    if payloads is None:
        payloads = [build_payload(task, records[i], max_tests) for i in range(n)]

    jobs, meta = [], []
    for s in range(n_seeds):
        for i in range(n):
            code = extract_python_code(texts_seeds[s][i])
            jobs.append((code, payloads[i], timeout))
            meta.append((s, i, bool(code)))

    # Each job is dominated by a blocking subprocess.run, which releases the GIL,
    # so threads give real parallelism without pickling or fork-after-tokenizer
    # hazards. The subprocess provides the isolation.
    if workers > 1 and jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_judge, jobs))
    else:
        results = [_judge(j) for j in jobs]

    correct = [[False] * n for _ in range(n_seeds)]
    has_code = [[False] * n for _ in range(n_seeds)]
    for (s, i, hc), r in zip(meta, results):
        correct[s][i] = bool(r)
        has_code[s][i] = hc
    return correct, has_code
