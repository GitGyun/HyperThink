# Dataset registry: dataset metadata and prompt templates shared by extraction,
# training, and evaluation.
import os
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_INFO_DIR = os.path.join(REPO_ROOT, "meta_info")   # small cached index files that ship with the repo


# Prompt templates

# copied from deepseekr1 model card
MATH_QUERY_TEMPLATE = """
{Question}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".strip()


def to_chat_prompt(tokenizer, query: str, enable_thinking: Optional[bool] = True) -> str:
    """Wrap a user query in the tokenizer's chat template (with the generation prompt).
    Shared by extraction, training, and evaluation so prompts match exactly.
    `enable_thinking=None` omits the kwarg entirely."""
    messages = [{"role": "user", "content": query}]

    kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if enable_thinking is not None:
        kwargs['enable_thinking'] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


BBH_PER_SUBTASK_CAP = 20  # per-subtask sample cap → total 420 across 21 subtasks
                          # (raw lighteval/bbh is ~78k, dominated by hyperbaton/formal_fallacies).
                          # Take the first N rows of each subtask for reproducibility.


def get_bbh_configs(repo: str = "lighteval/bbh"):
    """Return bbh's subtask config names from a local meta_info cache, hitting the
    Hub at most once.

    bbh has no central config metadata, so `get_dataset_config_names` enumerates
    configs by recursively listing the repo tree per subtask. That burst of Hub API
    calls trips the rate limit (HTTP 429) when several shards run in parallel,
    and `datasets` does not resolve the configs offline (it returns ['default']).
    We cache the list once under meta_info/hf_meta/bbh_configs.json and read it
    locally thereafter, so subsequent runs make zero metadata requests for it.
    """
    import json
    cache = os.path.join(META_INFO_DIR, "hf_meta", "bbh_configs.json")
    if os.path.exists(cache):
        with open(cache) as f:
            cfgs = json.load(f)
        if cfgs:
            return cfgs
    from datasets import get_dataset_config_names
    cfgs = sorted(get_dataset_config_names(repo))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(cfgs, f, indent=2)
    return cfgs

def _format_letter_list(labels):
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + ", or " + labels[-1]


CODE_QUERY_TEMPLATE = """{Question}

Please reason about the problem, then provide a complete Python solution that reads from stdin and writes to stdout. Put your final solution inside a ```python ... ``` code block.""".strip()


CODE_QUERY_TEMPLATE_WITH_STARTER = """{Question}

Use the following starter code:
```python
{StarterCode}
```

Please reason about the problem, then provide a complete Python solution. Put your final solution inside a ```python ... ``` code block.""".strip()


# Prompt functions  (each takes a single dataset row)

def math_prompt_fn(line, model=None):
    return MATH_QUERY_TEMPLATE.format(Question=line["problem"])


def code_prompt_fn(line, model=None):
    """Competitive-programming style code generation. Expects: problem, optional starter_code."""
    starter = (line.get("starter_code") or "").strip() if isinstance(line, dict) else ""
    if starter:
        return CODE_QUERY_TEMPLATE_WITH_STARTER.format(
            Question=line["problem"], StarterCode=starter
        )
    return CODE_QUERY_TEMPLATE.format(Question=line["problem"])


def mcq_prompt_fn(line, model=None):
    """Generic MCQ prompt. Expects normalized fields: problem, choices, choice_labels."""
    labels = list(line["choice_labels"])
    body = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, line["choices"]))
    letters = _format_letter_list(labels)
    return (
        f"{line['problem']}\n\n{body}\n\n"
        f"Please select the correct answer ({letters}), and put your final answer within \\boxed{{}}."
    )


# Registry
# Each entry:  task_name -> (hf_repo, subset, split, prompt_fn)

TRAIN_TASKS = ('gsm8k', 'deepscaler', 'codeforces-cots', 'logiqa', 'openbookqa', 'qasc')
TEST_TASKS = ('gsm8k-test', 'math500', 'aime_24_25', 'livecodebench', 'commonsenseqa', 'bbh')

TRAIN_DATASETS = {
    "gsm8k": ("openai/gsm8k",                           "main",    "train", math_prompt_fn),
    "deepscaler": ("agentica-org/DeepScaleR-Preview-Dataset", "default", "train", math_prompt_fn),
    "codeforces-cots": ("open-r1/codeforces-cots",                 "solutions_decontaminated", "train", code_prompt_fn),
    "logiqa": ("logicreasoning/logi_glue", "logiQA",   "train", mcq_prompt_fn),
    "openbookqa": ("allenai/openbookqa",  "main",          "train", mcq_prompt_fn),
    "qasc": ("allenai/qasc",        None,            "train", mcq_prompt_fn),
}

TEST_DATASETS = {
    "gsm8k-test": ("openai/gsm8k",                           "main",    "test",  math_prompt_fn),
    "math500": ("HuggingFaceH4/MATH-500",                 None,      "test",  math_prompt_fn),
    "aime_24_25": ("Maxwell-Jia/AIME_2024",                  "_COMBINED_AIME_24_25_", "train", math_prompt_fn),
    "livecodebench": ("livecodebench/code_generation",          None,        "test",  code_prompt_fn),
    "commonsenseqa": ("tau/commonsense_qa",  None,            "validation", mcq_prompt_fn),
    "bbh": ("lighteval/bbh",                          "_AGGREGATE_BBH_", "train", mcq_prompt_fn),
}

DATASET_REGISTRY = {**TRAIN_DATASETS, **TEST_DATASETS}


# Datasets that still ship a loading script upstream. datasets>=4.0 refuses those
# ("Dataset scripts are no longer supported"), so we pin the Hub's auto-converted
# parquet branch instead. None/absent -> default revision.
DATASET_REVISION = {}


# Task names that need MCQ normalization (problem/choices/choice_labels/answer schema)
MCQ_TASKS = {"openbookqa", "qasc", "logiqa", "commonsenseqa", "bbh"}


def normalize_mcq_row(task_name, ds):
    """Add normalized columns (problem, choices, choice_labels, answer, index) to a HF Dataset."""
    n = len(ds)
    if task_name == "openbookqa":
        problem = ds["question_stem"]
        choices = [ex["text"] for ex in ds["choices"]]
        choice_labels = [ex["label"] for ex in ds["choices"]]
        answers = ds["answerKey"]
        solution = [None] * n
    elif task_name == "commonsenseqa":
        # tau/commonsense_qa: 5-way MC with ConceptNet-derived choices.
        # Same schema as openbookqa, but the question field is called `question`.
        problem = ds["question"]
        choices = [ex["text"] for ex in ds["choices"]]
        choice_labels = [ex["label"] for ex in ds["choices"]]
        answers = ds["answerKey"]
        solution = [None] * n
    elif task_name == "qasc":
        problem = ds["question"]
        choices = [ex["text"] for ex in ds["choices"]]
        choice_labels = [ex["label"] for ex in ds["choices"]]
        answers = ds["answerKey"]
        # Use combined fact as a free-text rationale (useful as teacher signal)
        solution = ds["combinedfact"] if "combinedfact" in ds.column_names else [None] * n
    elif task_name == "logiqa":
        # logicreasoning/logi_glue config logiQA: has context + question + choices(list) + answer_choice(int)
        problem = [f"{ctx}\n\n{q}".strip() for ctx, q in zip(ds["context"], ds["question"])]
        choices = ds["choices"]
        choice_labels = [["A", "B", "C", "D"][: len(c)] for c in choices]
        answers = [["A", "B", "C", "D"][int(a)] for a in ds["answer_choice"]]
        solution = [None] * n
    else:
        raise ValueError(f"Unknown MCQ task: {task_name}")

    # Drop any existing columns we're about to overwrite, then add normalized ones.
    drop_cols = [c for c in ("problem", "choices", "choice_labels", "answer", "solution", "index")
                 if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    ds = ds.add_column("problem", problem)
    ds = ds.add_column("choices", choices)
    ds = ds.add_column("choice_labels", choice_labels)
    ds = ds.add_column("answer", answers)
    ds = ds.add_column("solution", solution)
    ds = ds.add_column("index", list(range(n)))
    return ds
