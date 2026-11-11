from transformers import TrainerCallback
from transformers.trainer_callback import ProgressCallback
import wandb
import threading
import json
import os
from collections import defaultdict
import torch


class GlobalStepAverageLossCallback(TrainerCallback):
    """
    Aggregate training `loss` by `state.global_step` within this process.
    For each global_step k, it logs:
      - avg_loss_by_global_step: running average over all loss logs at step k
      - n_updates_by_global_step: number of contributions at step k
    Optionally persists to a JSON file so a later Trainer continues the same tally.
    """
    _lock = threading.Lock()
    _sum_by_step = defaultdict(float)    # {int step: float sum_loss}
    _cnt_by_step = defaultdict(int)      # {int step: int n}

    def __init__(self, out_dir: str | None = None, json_name: str = "global_step_loss_map.json"):
        super().__init__()
        self.out_dir = out_dir
        self.json_path = os.path.join(out_dir, json_name) if out_dir else None
        self._loaded = False

    # persistence helpers
    def _load_if_any(self):
        if self._loaded or not self.json_path or not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
            with self._lock:
                for k, v in data.get("sum_by_step", {}).items():
                    self._sum_by_step[int(k)] += float(v)
                for k, v in data.get("cnt_by_step", {}).items():
                    self._cnt_by_step[int(k)] += int(v)
            self._loaded = True
        except Exception:
            pass

    def _persist(self):
        if not self.json_path:
            return
        try:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            payload = {
                "sum_by_step": {str(k): v for k, v in self._sum_by_step.items()},
                "cnt_by_step": {str(k): v for k, v in self._cnt_by_step.items()},
            }
            tmp = self.json_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.json_path)
        except Exception:
            pass

    # HF hooks
    def on_train_begin(self, args, state, control, **kwargs):
        # only main proc reads from disk
        if getattr(self, "trainer", None) and self.trainer.is_world_process_zero():
            self._load_if_any()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only aggregate TRAINING loss logs. Ignore eval/test logs.
        if logs is None: return
        if wandb.run is None: return
        if "loss" not in logs:  return

        step = int(state.global_step)  # local/global within this Trainer
        loss_val = float(logs["loss"])

        with self._lock:
            self._sum_by_step[step] += loss_val
            self._cnt_by_step[step] += 1
            avg = self._sum_by_step[step] / max(1, self._cnt_by_step[step])

        # main proc re-logs the aggregated metric at the same step index
        wandb.log({
            "train/avg_loss": avg,
            "train/n_updates": self._cnt_by_step[step],
        })
        self._persist()


class QuietProgressCallback(ProgressCallback):
    """
    Keeps tqdm, suppresses the dict writes.
    Optionally shows a tiny postfix instead of the full logs.
    """
    def __init__(self, desc=''):
        super().__init__()
        self.desc = desc

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            # customize tqdm format
            self.training_bar.bar_format = "{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar:40}"
            self.training_bar.leave = False  # don't keep the bar after done
            self.training_bar.desc = self.desc
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not (state.is_world_process_zero and self.training_bar is not None):
            return

        if logs:
            keys = ["loss"]
            postfix = "  ".join(f"{k}:{logs[k]:.4g}" for k in keys if k in logs)
            if postfix:
                try:
                    self.training_bar.set_postfix_str(postfix)  # no extra line
                except Exception:
                    pass  # be defensive
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.training_bar is not None and state.global_step is not None:
            # move bar to current global step
            self.training_bar.n = int(state.global_step)
            self.training_bar.refresh()

    # (Optional) avoid noisy epoch end summaries
    def on_epoch_end(self, *args, **kwargs):
        return


class HypernetCheckpointCallback(TrainerCallback):
    """
    Save only the hypernet weights every `save_every` steps.
    We do not call Trainer.save_model (which triggers safetensors on the full model).
    """

    def __init__(self, out_root: str, save_every: int = 10000):
        self.out_root = out_root
        self.save_every = save_every

    def on_step_end(self, args, state, control, **kwargs):
        # Only rank 0 / single-process saves
        if not ((getattr(args, "local_rank", -1) in (-1, 0)) and state.global_step > 0):
            return control

        if state.global_step % self.save_every != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        # unwrap DDP / FSDP
        module = getattr(model, "module", model)

        # SFTWrapper(base_model, hypernet)
        hypernet = getattr(module, "hypernet", None)
        if hypernet is None:
            return control

        save_dir = os.path.join(self.out_root, f"step_{state.global_step:06d}")
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, "hypernet.pt")
        save_dict = {"state_dict": hypernet.state_dict()}
        # Save training config for reproducibility
        if hasattr(module, '_train_args'):
            save_dict["model_args"] = module._train_args
        torch.save(save_dict, ckpt_path)

        return control
