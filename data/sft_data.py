from utils.dataset_registry import TRAIN_TASKS, TEST_TASKS
from typing import Dict, List, Any, Optional
import torch
import re
import json
import os
import random
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
from glob import glob
from utils.train_utils import is_main_process


from utils.dataset_registry import DATASET_REGISTRY, DATASET_REVISION, MCQ_TASKS, normalize_mcq_row, META_INFO_DIR, to_chat_prompt


def build_hypernet_query_text(dataset_name, row, fallback: str = "") -> str:
    """Training text fed to the hypernet's text encoder: the raw `problem` field, without
    prompt boilerplate ('Please reason step by step', 'reads from stdin', 'Please
    select the correct answer') or the MCQ options block. The text encoder then
    conditions on the content of the problem rather than on template tokens that
    repeat across a task.

    For MCQ rows this returns the question stem only. Task identity is already
    carried by `task_ids`, and the hypernet does not need the option set to
    produce a useful task-conditioned bias.
    """
    del dataset_name  # unused
    if not isinstance(row, dict):
        return fallback or ""
    return row.get("problem") or fallback or ""


def build_rollout_query_text(dataset_name, row) -> str:
    """Match the pre-c357252 evaluation inputs used by the released checkpoints.

    Legacy math tasks used the raw problem; MCQ, AIME 24/25, and code tasks
    included their task instructions (and MCQ choices) before chat templating.
    """
    if dataset_name in {'gsm8k-test', 'math500'}:
        return row['problem']
    return DATASET_REGISTRY[dataset_name][3](row)


class HyperNetDataset(torch.utils.data.Dataset):
    """
    Base class for hypernet datasets.
    """
    def __init__(self):
        super().__init__()
        pass
        
    def _load_base_dataset(self, dataset_name: str) -> datasets.Dataset:
        if dataset_name not in TRAIN_TASKS:
            raise ValueError(f"Unsupported training dataset: {dataset_name}")
        repo, subset, split, _ = DATASET_REGISTRY[dataset_name]
        ds = load_dataset(repo, subset, split=split,
                          revision=DATASET_REVISION.get(dataset_name))
        if dataset_name == 'gsm8k':
            ds = ds.rename_column("question", "problem")
            ds = ds.rename_column("answer", "solution")
            def _map(batch):
                cots = batch["solution"]
                answer = [re.match(r"^\s*####\s*(.+?)\s*$", cot.strip().splitlines()[-1]).group(1).strip() for cot in cots]
                return {"answer": answer}
            ds = ds.map(_map, batched=True, num_proc=1)
        elif dataset_name in MCQ_TASKS:
            # problem / choices / choice_labels / answer; the options block is added by
            # the registry's mcq_prompt_fn when the LM prompt is built (mirrors HFRolloutDataset).
            ds = normalize_mcq_row(dataset_name, ds)
        elif dataset_name == 'codeforces-cots':
            # Build a 'problem' from the Codeforces statement fields (mirrors HFRolloutDataset._map_cf).
            def _cf(batch):
                problems = []
                n = len(batch["description"])
                for i in range(n):
                    title = (batch.get("title") or [""] * n)[i] or ""
                    desc = batch["description"][i] or ""
                    inp = batch["input_format"][i] or ""
                    outp = batch["output_format"][i] or ""
                    note = (batch.get("note") or [None] * n)[i] or ""
                    examples = (batch.get("examples") or [None] * n)[i] or []
                    ex_text = ""
                    for j, ex in enumerate(examples):
                        ex_text += f"\nExample {j+1}:\nInput:\n{ex.get('input','')}\nOutput:\n{ex.get('output','')}\n"
                    parts = [f"# {title}", "", desc, "", f"Input format: {inp}", f"Output format: {outp}"]
                    if note:
                        parts.append(f"\nNote: {note}")
                    if ex_text:
                        parts.append(ex_text)
                    problems.append("\n".join(parts).strip())
                return {"problem": problems}
            ds = ds.map(_cf, batched=True, num_proc=1)
        elif 'problem' not in ds.column_names:
            raise ValueError(
                f"[Dataset] base dataset '{dataset_name}' has no 'problem' column and no "
                f"normalization branch in _load_base_dataset (cols={ds.column_names[:8]}...). "
                f"Add one to train on it.")
        if is_main_process():
            print(f"[Dataset] Loaded base dataset {dataset_name} with {len(ds)} examples.")
        return ds

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        train_index = self.train_ids[idx]
        solutions = self.ps_pairs[train_index]
        solution_text = random.choice(solutions) if len(solutions) > 1 else solutions[0]
        d_idx, row_idx = train_index
        dataset_name = self.dataset_names[d_idx]
        row = self.base_datasets[dataset_name][row_idx]

        # LM prompt: the task's registry prompt through the chat template, i.e. the exact
        # prompt used to generate this CoT at extraction and the prompt evaluation uses.
        prompt_fn = DATASET_REGISTRY[dataset_name][3]
        prompt = to_chat_prompt(self.base_tokenizer, prompt_fn(row), enable_thinking=True)
        prompt_ids = self.base_tokenizer(prompt, add_special_tokens=False, truncation=False,
                                         return_attention_mask=True)["input_ids"]

        # Target: the post-</think> solution only. The thinking trace is never a target,
        # since the hypernet bias stands in for it.
        sol_ids = self.base_tokenizer(solution_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids = prompt_ids + sol_ids
        attention_mask = [1] * len(input_ids)

        # Teacher forcing on solution tokens only
        labels = [-100] * (len(prompt_ids) - 1) + sol_ids + [self.base_tokenizer.eos_token_id]

        # Training uses raw problem text. HFRolloutDataset separately preserves
        # the legacy evaluation query format for checkpoint reproducibility.
        query = build_hypernet_query_text(dataset_name, row, fallback="")
        enc_query = self.query_tokenizer(
            query,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "query_input_ids": enc_query["input_ids"],
            "query_attention_mask": enc_query["attention_mask"],
            "query_text": query,
            "train_index": train_index,
        }


class HyperNetTrainDataset(HyperNetDataset):
    """Training set: (prompt, correct CoT answer) pairs built from the HF base datasets
    and the correct-row JSONLs in the public distillation layout.

    Each item pairs a problem with one of its correct post-thinking answers (chosen at
    random per access), the registry prompt for the base LLM and the raw problem text
    for the hypernet.
    """

    def __init__(
        self,
        dataset_names: List[str],
        solution_paths: Dict[str, str],
        base_tokenizer: AutoTokenizer,
        query_tokenizer: AutoTokenizer,
    ):
        self.base_tokenizer = base_tokenizer
        self.query_tokenizer = query_tokenizer
        self.dataset_names = list(dataset_names)
        self.base_datasets = {dataset_name: self._load_base_dataset(dataset_name) for dataset_name in dataset_names}
        self.ps_pairs = self._load_solutions_from_cots(dataset_names, solution_paths)
        self.train_ids = list(self.ps_pairs.keys())
        self.train_ids = self._balance_by_dataset(self.train_ids)

    def _balance_by_dataset(self, items):
        """Deterministically upsample every dataset to the largest dataset size."""
        groups = {}
        for item in items:
            groups.setdefault(item[0], []).append(item)
        if len(groups) <= 1:
            return items
        target = max(len(group) for group in groups.values())
        rng = random.Random(0)
        balanced = []
        for dataset_id in sorted(groups):
            group = groups[dataset_id]
            balanced.extend(group * (target // len(group))
                            + rng.sample(group, target % len(group)))
        if is_main_process():
            print(f"[Dataset] Balanced {len(groups)} datasets to {target} examples each.")
        return balanced

    def _load_solutions_from_cots(self, dataset_names: List[str], sol_paths: Dict[str, str]) -> Dict[Any, List[str]]:
        """Collect {(dataset_idx, example_idx): [solution, ...]} from the per-dataset
        JSONL globs written at extraction (preprocess/extract_cot_sharded.py)."""
        ps_pairs = {}
        n_solutions = 0

        for d_idx, dset_name in enumerate(dataset_names):
            all_paths = glob(sol_paths[dset_name])
            # Only published correct-solution files; diagnostic raw/incorrect
            # files and incomplete shard files must never enter training.
            corr_paths = sorted(p for p in all_paths
                                if re.fullmatch(r"outputs_n\d+_seed\d+\.jsonl",
                                                os.path.basename(p)))
            if not corr_paths:
                if is_main_process():
                    print(f"[Dataset] {dset_name}: no files match {sol_paths[dset_name]}; skipping.")
                continue
            ds_sol = load_dataset('json', data_files={'train': corr_paths})['train']
            if dset_name == 'deepscaler':
                if os.path.exists(os.path.join(META_INFO_DIR, 'deepscaler_train_ids.json')):
                    with open(os.path.join(META_INFO_DIR, 'deepscaler_train_ids.json'), 'r') as f:
                        train_ids = set(json.load(f))
                    if is_main_process():
                        print(f"[Dataset] Loaded {len(train_ids)} DeepScaler train IDs from meta_info/deepscaler_train_ids.json")
                else:
                    train_ids = torch.randperm(40315)[:39000].tolist()
                    with open(os.path.join(META_INFO_DIR, 'deepscaler_train_ids.json'), 'w') as f:
                        json.dump(list(train_ids), f)
                    if is_main_process():
                        print(f"[Dataset] Created and Saved DeepScaler {len(train_ids)} train IDs to meta_info/deepscaler_train_ids.json")

                ds_sol = ds_sol.filter(lambda ex: int(ex['index']) in train_ids)
                if is_main_process():
                    unique_ids = set(int(ex['index']) for ex in ds_sol)
                    print(f"[Dataset] Filtered DeepScaler CoT solutions to only train IDs, {len(unique_ids)} train examples remain.")

            if is_main_process():
                print(f"[Dataset] Loading {len(ds_sol)} training solutions from {dset_name} at {sol_paths[dset_name]}")

            base = self.base_datasets[dset_name]
            n_base = len(base)
            n_skipped = 0
            for ex in ds_sol:
                idx = ex['index']
                if not (0 <= idx < n_base):
                    n_skipped += 1
                    continue
                ps_pairs.setdefault((d_idx, idx), []).append(ex['solution'])
                n_solutions += 1
            if n_skipped and is_main_process():
                print(f"[Dataset] {dset_name}: skipped {n_skipped} solutions whose index "
                      f"was out of range for the base dataset ({n_base} rows).")

        if is_main_process():
            print(f"[Dataset] Extracted solutions for {len(ps_pairs)} examples, {n_solutions} solutions from {', '.join(dataset_names)} in total.")

        return ps_pairs


class HFRolloutDataset(torch.utils.data.Dataset):
    """Test-time rollout dataset that loads queries directly from Hugging Face datasets.

    Works for every task in DATASET_REGISTRY: math, MCQ, code.

    Each example dict keeps the normalized dataset row and adds:
      query_text: the legacy evaluation input for the hypernet text encoder:
                  raw problem for older math tasks, full task query otherwise.
                  The LM prompt is built separately by the registry prompt_fn.
      answer:     ground-truth answer (letter for MCQ, scalar or expression for
                  math, None for code, which is scored by execution).
      index:      dataset row index.
      n_choices:  number of options (0 for non-MCQ tasks; >0 enables letter-match scoring).
    """

    def __init__(self, dataset_name: str, n_tasks: int = -1):
        from utils.dataset_registry import DATASET_REGISTRY, MCQ_TASKS, normalize_mcq_row

        if dataset_name not in TEST_TASKS:
            raise ValueError(f"Unsupported test dataset: {dataset_name}")
        repo, subset, split, _ = DATASET_REGISTRY[dataset_name]
        # Special multi-source / aggregate loaders
        if subset == "_AGGREGATE_BBH_":
            from datasets import concatenate_datasets
            from utils.dataset_registry import BBH_PER_SUBTASK_CAP, get_bbh_configs
            cfgs = get_bbh_configs(repo)   # local meta_info cache; no per-run tree-listing API call
            parts = []
            for cfg in cfgs:
                d = load_dataset(repo, cfg, split=split)
                if BBH_PER_SUBTASK_CAP and len(d) > BBH_PER_SUBTASK_CAP:
                    d = d.select(range(BBH_PER_SUBTASK_CAP))
                d = d.add_column("subtask", [cfg] * len(d))
                parts.append(d)
            ds = concatenate_datasets(parts)
        elif subset == "_COMBINED_AIME_24_25_":
            from datasets import concatenate_datasets
            d1 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
            d1 = d1.rename_column("Problem", "problem").rename_column("Answer", "answer")
            d1 = d1.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True)
            d1 = d1.select_columns(["problem", "answer"])
            d2 = load_dataset("MathArena/aime_2025", split="train")
            d2 = d2.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True)
            d2 = d2.select_columns(["problem", "answer"])
            ds = concatenate_datasets([d1, d2])
        else:
            ds = load_dataset(repo, subset, split=split,
                              revision=DATASET_REVISION.get(dataset_name))

        # Per-task normalization (mirrors extract_cot_sharded.py)
        if dataset_name == 'gsm8k-test':
            ds = ds.rename_column("question", "problem")
            ds = ds.rename_column("answer", "solution")
            def _map(batch):
                cots = batch["solution"]
                return {"answer": [re.match(r"^\s*####\s*(.+?)\s*$", c.strip().splitlines()[-1]).group(1).strip() for c in cots]}
            ds = ds.map(_map, batched=True, num_proc=1)
            n_choices = 0
        elif dataset_name == 'bbh':
            def _map_bbh(batch):
                problems, choices_list, labels_list, answers = [], [], [], []
                n = len(batch["input"])
                for i in range(n):
                    problems.append(batch["input"][i])
                    ch = list(batch["choices"][i] or [])
                    lbls = [chr(ord('A') + j) for j in range(len(ch))]
                    choices_list.append(ch); labels_list.append(lbls)
                    tgt = int(batch["target_idx"][i])
                    answers.append(lbls[tgt] if 0 <= tgt < len(ch) else "")
                return {"problem": problems, "choices": choices_list,
                        "choice_labels": labels_list, "answer": answers}
            ds = ds.map(_map_bbh, batched=True, num_proc=1,
                        remove_columns=[c for c in ds.column_names if c != "subtask"])
            n_choices = 7  # max across subtasks
        elif dataset_name == 'aime_24_25':
            ds = ds.map(lambda b: {"answer": [str(a) for a in b["answer"]]}, batched=True, num_proc=1)
            n_choices = 0
        elif dataset_name in MCQ_TASKS:
            ds = normalize_mcq_row(dataset_name, ds)
            n_choices = len(ds[0]["choice_labels"])
        elif dataset_name == 'math500':
            n_choices = 0
        elif dataset_name == 'livecodebench':
            def _map_lcb(batch):
                problems = []
                n = len(batch["question_content"])
                for i in range(n):
                    title = (batch.get("question_title") or [""] * n)[i] or ""
                    qc = batch["question_content"][i] or ""
                    starter = (batch.get("starter_code") or [""] * n)[i] or ""
                    parts = [f"# {title}", "", qc]
                    if starter.strip():
                        parts += ["", "Use this starter code:", "```python", starter, "```"]
                    problems.append("\n".join(parts).strip())
                return {"problem": problems, "answer": [None] * n}
            ds = ds.map(_map_lcb, batched=True, num_proc=1,
                        remove_columns=[c for c in ds.column_names if c not in ("question_id", "platform", "difficulty")])
            n_choices = 0
        else:
            raise ValueError(f"Unsupported test dataset: {dataset_name}")

        # Preserve dataset indices for code-test lookup before slicing.
        if 'index' not in ds.column_names:
            ds = ds.add_column('index', list(range(len(ds))))

        # Subsample
        if n_tasks > 0 and n_tasks < len(ds):
            sel = list(range(n_tasks))
            ds = ds.select(sel)

        # Materialize examples. Each keeps the full normalized row so build_lm_prompt
        # can apply the task's registry prompt_fn. `query_text` is the hypernet
        # text-encoder input in the legacy evaluation format (see build_rollout_query_text).
        # The LM never reads query_text when dataset_name is set.
        examples = []
        for i, row in enumerate(ds):
            entry = dict(row)
            entry["dataset_name"] = dataset_name
            # BBH aggregates subtasks with different option counts (2..7); use the
            # row's actual choice count so the letter parser accepts exactly the
            # offered labels (the prompt already lists only those). Other MCQ tasks
            # have a fixed, correct n_choices.
            if dataset_name == 'bbh' and row.get("choice_labels"):
                entry["n_choices"] = len(row["choice_labels"])
            else:
                entry["n_choices"] = n_choices
            entry["index"] = row.get("index", i)
            entry["query_text"] = build_rollout_query_text(dataset_name, entry)
            examples.append(entry)

        self.examples = examples
        self.dataset_name = dataset_name
        self.n_choices = n_choices
        if is_main_process():
            print(f"[Dataset] HF rollout '{dataset_name}': {len(examples)} examples, "
                  f"n_choices={n_choices}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



class CausalLMWithMaskedPromptsCollator:
    """
    Truncates and right-pads the LM inputs and the hypernet query inputs of a batch
    (lengths rounded up to a multiple of `mul`), then stacks them into tensors
    together with the task and dataset ids.
    """

    def __init__(
        self,
        base_pad_token_id: int,
        query_pad_token_id: int,
        pad_to_multiple_of: int = 1,
        max_length: Optional[int] = None,
        truncation_side: str = "right",         # for main LM inputs
        debug_mode: bool = False,
    ):
        self.base_pad_token_id = base_pad_token_id
        self.query_pad_token_id = query_pad_token_id
        self.mul = pad_to_multiple_of
        self.max_length = max_length
        self.truncation_side = truncation_side
        self.debug_mode = debug_mode

        assert truncation_side in ["left", "right"]

    def _truncate(self, seq, max_len, side="right"):
        if max_len is None or len(seq) <= max_len:
            return seq
        if side == "left":
            return seq[-max_len:]
        else:
            return seq[:max_len]

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        processed = []

        for b in batch:
            # main LM inputs
            input_ids = self._truncate(b["input_ids"], self.max_length, self.truncation_side)
            attention_mask = self._truncate(b["attention_mask"], self.max_length, self.truncation_side)
            labels = self._truncate(b["labels"], self.max_length, self.truncation_side)

            # debug mode: left-pad to the full max_length
            if self.debug_mode and self.max_length and len(input_ids) < self.max_length:
                input_ids = [input_ids[0]] * (self.max_length - len(input_ids)) + input_ids
                attention_mask = [0] * (self.max_length - len(attention_mask)) + attention_mask
                labels = [labels[0]] * (self.max_length - len(labels)) + labels

            # query inputs for hypernet
            query_input_ids = b["query_input_ids"]
            query_attention_mask = b["query_attention_mask"]

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "query_input_ids": query_input_ids,
                "query_attention_mask": query_attention_mask,
                "query_text": b["query_text"],
                "train_index": b["train_index"],
            })

        # max lengths after truncation
        L = max(len(b["input_ids"]) for b in processed)
        Lq = max(len(b["query_input_ids"]) for b in processed)

        if self.mul:
            r = L % self.mul
            if r:
                L += (self.mul - r)
            rq = Lq % self.mul
            if rq:
                Lq += (self.mul - rq)

        def pad(seq, pad_val, target_len):
            return seq + [pad_val] * (target_len - len(seq))

        input_ids = torch.tensor(
            [pad(b["input_ids"], self.base_pad_token_id, L) for b in processed],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [pad(b["attention_mask"], 0, L) for b in processed],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [pad(b["labels"], -100, L) for b in processed],
            dtype=torch.long,
        )

        query_input_ids = torch.tensor(
            [pad(b["query_input_ids"], self.query_pad_token_id, Lq) for b in processed],
            dtype=torch.long,
        )
        query_attention_mask = torch.tensor(
            [pad(b["query_attention_mask"], 0, Lq) for b in processed],
            dtype=torch.long,
        )

        task_ids = torch.tensor(
            [b["train_index"][1] if isinstance(b["train_index"], tuple) else b["train_index"] for b in processed],
            dtype=torch.long,
        )

        # Dataset index (train_index[0]), carried separately for per-dataset
        # loss logging. 0 when train_index is a bare int (single-dataset run).
        dataset_ids = torch.tensor(
            [b["train_index"][0] if isinstance(b["train_index"], tuple) else 0 for b in processed],
            dtype=torch.long,
        )

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "query_text": [b["query_text"] for b in processed],
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "labels": labels,
        }
        return out
    

def load_train_dataset(args):
    # setup tokenizers
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    query_tokenizer = AutoTokenizer.from_pretrained(
        args.text_encoder_name, use_fast=True, trust_remote_code=True
    )
    if query_tokenizer.pad_token is None:
        query_tokenizer.pad_token = query_tokenizer.eos_token
        query_tokenizer.pad_token_id = query_tokenizer.eos_token_id

    train_dataset = HyperNetTrainDataset(
        dataset_names=args.train_datasets,
        solution_paths=args.cot_solution_multi_paths,
        base_tokenizer=base_tokenizer,
        query_tokenizer=query_tokenizer,
    )

    return train_dataset


def setup_data_for_training(args):
    """Load all correct CoTs with the dataset's default equal balancing."""
    train_dataset = load_train_dataset(args)
    train_collator = CausalLMWithMaskedPromptsCollator(
        base_pad_token_id=train_dataset.base_tokenizer.pad_token_id,
        query_pad_token_id=train_dataset.query_tokenizer.pad_token_id,
        debug_mode=args.debug_mode,
        max_length=args.max_train_length,
    )
    return train_dataset, train_collator


def setup_data_for_evaluation(args):
    """Load one of the six public test datasets, then slice and shard it."""
    registry_name = args.task
    ds = HFRolloutDataset(registry_name, n_tasks=-1)
    print(f"[Eval] HFRolloutDataset '{registry_name}' loaded: {len(ds)} examples")

    # task_offset / end_offset / n_tasks slicing
    if getattr(args, 'n_tasks', None) is not None or getattr(args, 'end_offset', None) is not None:
        offset = getattr(args, 'task_offset', 0) or 0
        if args.end_offset is not None:
            args.n_tasks = args.end_offset - offset
        keep = set(range(offset, offset + args.n_tasks))
        ds.examples = [ex for ex in ds.examples if int(ex['index']) in keep]
        print(f"[Eval] After offset/n_tasks filter: {len(ds)} examples (offset={offset}, n_tasks={args.n_tasks})")

    # Shard examples evenly (floor plus remainder).
    if getattr(args, 'num_shards', 1) > 1:
        assert 0 <= args.shard_id < args.num_shards, \
            f"shard_id must be in [0, {args.num_shards-1}], got {args.shard_id}"
        all_indices = sorted(set(int(ex["index"]) for ex in ds.examples))
        total = len(all_indices)
        base_, extra = total // args.num_shards, total % args.num_shards
        if args.shard_id < extra:
            shard_start = args.shard_id * (base_ + 1)
            shard_end = shard_start + (base_ + 1)
        else:
            shard_start = extra * (base_ + 1) + (args.shard_id - extra) * base_
            shard_end = shard_start + base_
        if shard_start >= shard_end:
            print(f"[Shard] shard_id {args.shard_id}/{args.num_shards} empty; nothing to do.")
            return
        shard_idx_set = set(all_indices[shard_start:shard_end])
        before = len(ds.examples)
        ds.examples = [ex for ex in ds.examples if int(ex['index']) in shard_idx_set]
        print(f"[Shard] num_shards={args.num_shards}, shard_id={args.shard_id} -> "
              f"{len(ds)} / {before} examples (indices {min(shard_idx_set)}..{max(shard_idx_set)})")
    else:
        print("[Shard] Sharding disabled (num_shards=1).")

    assert len(ds) > 0, "No examples to evaluate after filtering and sharding."
    return ds
