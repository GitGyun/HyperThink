from __future__ import annotations
import os
from collections import defaultdict
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from transformers import Trainer
from transformers import TrainerCallback
from transformers.trainer_callback import ProgressCallback


from train.callback import GlobalStepAverageLossCallback, QuietProgressCallback, HypernetCheckpointCallback
from utils.train_utils import is_main_process


# Trainer for context distillation


class _SyncStateCallback(TrainerCallback):
    """Sync state with current args after resume loads old state."""
    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        state.max_steps = args.max_steps
        return control


class ContextDistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_losses()
        self.add_callback(_SyncStateCallback())
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    def _reset_losses(self):
        self.losses = []
        self.vq_losses = []
        self.ce_losses = []
        self.usage_losses = []
        self.ce_losses_by_dataset = defaultdict(list)
        self.vq_max_usages = []
        self.vq_hard_perplexities = []

    def _compute_vq_stats(self, outputs):
        vq_index = outputs.get('vq_index', None)
        if vq_index is not None:
            B, Ntok = vq_index.shape
            module = getattr(self.model, "module", self.model)
            K = module.hypernet.param_decoder.num_codes

            max_usages = []
            hard_perps = []

            for i in range(Ntok):
                hist = torch.bincount(vq_index[:, i], minlength=K).float()
                p = hist / hist.sum().clamp_min(1.0)
                max_usages.append((p.max().item() * 100.0))

                nz = p[p > 0]
                ent = -(nz * nz.log()).sum()
                hard_perps.append(torch.exp(ent).item())

            self.vq_max_usages.append(float(sum(max_usages) / len(max_usages)))
            self.vq_hard_perplexities.append(float(sum(hard_perps) / len(hard_perps)))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return self._compute_sft_loss(model, inputs, return_outputs, **kwargs)

    def _compute_sft_loss(self, model, inputs, return_outputs=False, **kwargs):
        module = getattr(model, "module", model)  # unwrap DDP

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            query_input_ids=inputs["query_input_ids"],
            query_attention_mask=inputs["query_attention_mask"],
            query_task_ids=inputs["task_ids"],
            query_text=inputs.get("query_text"),
            get_vq_index=True,
        )
        logits = outputs['outputs'].logits  # [B, L, V]

        labels = inputs['labels']  # [B, L]

        loss = 0

        # Teacher-forcing CE loss
        loss_ce = self.loss_function(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        loss = loss + self.ce_weight * loss_ce
        if model.training:
            self.ce_losses.append(loss_ce.item())

        # Per-dataset CE breakdown (logging only): token-weighted mean within each
        # dataset, computed once and reused. dataset_ids = train_index[0] (from the
        # collator). Logged under train_aux/.
        dataset_names = getattr(self, 'train_dataset_names', None)
        dataset_ids = inputs.get("dataset_ids", None) if isinstance(inputs, dict) else None
        log_by_dataset = dataset_ids is not None and dataset_names and len(dataset_names) > 1
        if model.training and log_by_dataset:
            with torch.no_grad():
                B, L, V = logits.shape
                tok_ce = F.cross_entropy(
                    logits.view(-1, V), labels.view(-1),
                    ignore_index=-100, reduction='none',
                ).view(B, L)
                valid = (labels != -100).to(tok_ce.dtype)
                ex_ce = (tok_ce * valid).sum(dim=1)   # [B] summed token loss per example
                ex_tok = valid.sum(dim=1)             # [B] valid token count per example
                for d in dataset_ids.unique().tolist():
                    m = dataset_ids == d
                    n_tok = ex_tok[m].sum()
                    if n_tok > 0:
                        name = dataset_names[d] if d < len(dataset_names) else str(d)
                        self.ce_losses_by_dataset[name].append((ex_ce[m].sum() / n_tok).item())

        # VQ loss (skip if no hypernet)
        if outputs.get('vq_loss', None) is not None:
            vq_loss = outputs['vq_loss']
            loss = loss + self.vq_weight * vq_loss
            if model.training:
                self.vq_losses.append(vq_loss.item())
                self._compute_vq_stats(outputs)

        # VQ usage loss (skip if no hypernet)
        if outputs.get('usage_loss', None) is not None:
            usage_loss = outputs['usage_loss']
            loss = loss + self.vq_usage_weight * usage_loss
            if model.training:
                self.usage_losses.append(usage_loss.item())

        self.losses.append(loss.item())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if is_main_process():
            # Average the buffered metrics over the logging window
            custom_metrics = {}
            if self.losses:
                custom_metrics["loss"] = sum(self.losses) / len(self.losses)
            if self.vq_losses:
                custom_metrics["vq_loss"] = sum(self.vq_losses) / len(self.vq_losses)
            if self.ce_losses:
                custom_metrics["ce_loss"] = sum(self.ce_losses) / len(self.ce_losses)
            if self.usage_losses:
                custom_metrics["usage_loss"] = sum(self.usage_losses) / len(self.usage_losses)
            # Per-dataset CE: keep train_aux/ as the uppermost category. HF's wandb
            # integration (rewrite_logs) prepends "train/" to every key in `logs`,
            # so routing these through `logs` would yield train/train_aux/...; log
            # them straight to wandb at the same step instead.
            aux_metrics = {f"train_aux/ce_loss_{name}": sum(vals) / len(vals)
                           for name, vals in self.ce_losses_by_dataset.items() if vals}
            if aux_metrics:
                try:
                    if wandb.run is not None:
                        wandb.log(aux_metrics, step=int(self.state.global_step or 0))
                except Exception:
                    pass
            if self.vq_max_usages:
                custom_metrics["vq_code_max_usage"] = sum(self.vq_max_usages) / len(self.vq_max_usages)
            if self.vq_hard_perplexities:
                custom_metrics["vq_hard_perplexity"] = sum(self.vq_hard_perplexities) / len(self.vq_hard_perplexities)

            # Merge them into the standard logs so they share a W&B step
            logs.update(custom_metrics)

            # Clear buffers for the next window
        self._reset_losses()

        # The parent log sends everything to W&B
        super().log(logs, *args, **kwargs)


def setup_trainer(model, args, training_args, train_dataset, train_collator):
    callbacks = [
        GlobalStepAverageLossCallback(out_dir=training_args.output_dir),
        HypernetCheckpointCallback(out_root=training_args.output_dir, save_every=args.save_every),
    ]

    trainer = ContextDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        callbacks=callbacks,
    )
    trainer.vq_weight = args.vq_weight
    trainer.ce_weight = args.ce_weight
    trainer.vq_usage_weight = args.vq_usage_weight
    # Ordered dataset names (index == train_index[0]) for per-dataset CE logging.
    trainer.train_dataset_names = list(args.train_datasets)
    # Also store on the model module so checkpoint callbacks can access it
    module = getattr(model, "module", model)
    module._train_args = vars(args)

    # Replace default progress bar
    for cb in list(trainer.callback_handler.callbacks):
        if isinstance(cb, ProgressCallback):
            trainer.remove_callback(cb.__class__)
    trainer.add_callback(QuietProgressCallback(desc="[Training] "))

    return trainer
