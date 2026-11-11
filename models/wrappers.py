from __future__ import annotations
import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers import AutoModelForCausalLM

from models.hypernet import ParamHyperNetModel
from utils.train_utils import is_main_process


# SFT wrapper and the bias-injected base model
class SFTWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, hypernet: nn.Module):
        super().__init__()
        self.base_model = base_model           # BiasInjectedQwenWrapper
        self.hypernet = hypernet
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, query_input_ids, query_attention_mask, query_task_ids,
                query_text=None, get_vq_index=False, **kwargs):
        # get bias vector from hypernet
        out_dict = self.hypernet(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            task_ids=query_task_ids,
            query_text=query_text,
            get_vq_index=get_vq_index,
        )
        bias_vec = out_dict["pred_delta"]

        # pass to base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bias_vec=bias_vec,
            **kwargs,
        )
        out_dict['outputs'] = outputs

        return out_dict

    def generate(self, input_ids, attention_mask, query_input_ids, query_attention_mask, query_task_ids, query_text=None, get_vq_index=False, **kwargs):
        # get bias vector from hypernet
        hp_out = self.hypernet(input_ids=query_input_ids, attention_mask=query_attention_mask, task_ids=query_task_ids, query_text=query_text, get_vq_index=get_vq_index)
        param_delta = hp_out["pred_delta"]      # (B, D_param)

        # pass to base model
        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bias_vec=param_delta,
            **kwargs,
        )

        if get_vq_index:
            return outputs, hp_out.get("vq_index", None)
        else:
            return outputs


class BiasInjectedQwenWrapper(nn.Module):
    """
    Wrap a base LLM (Qwen3, SmolLM3, or Olmo-3) and allow per-example bias injection
    on selected projection modules (e.g., q_proj, v_proj, up_proj, gate_proj) in
    selected layers (e.g., the last eight).
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        bias_layers: List[int],
        bias_target_modules: List[str],
    ):
        super().__init__()
        self.base_model = base_model
        self.bias_layers = sorted(set(bias_layers))
        self.bias_target_modules = set(bias_target_modules)

        # key: (layer_idx, module_name) -> Tensor [B, out_features]
        self._current_bias: Optional[Dict[Tuple[int, str], torch.Tensor]] = None

        # list of ((layer_idx, module_name), out_features)
        self.bias_specs: List[Tuple[Tuple[int, str], int]] = []
        self.total_bias_dim: int = 0

        # Keep hooks so they don’t get GC’d
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._inspect_and_register_hooks()

    @property
    def input_splits(self) -> list[int]:
        return [out_dim for (_, out_dim) in self.bias_specs]

    def _inspect_and_register_hooks(self):
        transformer = self.base_model.model
        layers = transformer.layers

        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.bias_layers:
                continue

            # Attention projections
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for name in ["q_proj", "v_proj", "k_proj", "o_proj"]:
                    if name in self.bias_target_modules and hasattr(attn, name):
                        mod = getattr(attn, name)
                        if isinstance(mod, nn.Linear):
                            key = (layer_idx, name)
                            out_dim = mod.out_features
                            self.bias_specs.append((key, out_dim))
                            self.total_bias_dim += out_dim
                            h = mod.register_forward_hook(self._make_bias_hook(key))
                            self._hooks.append(h)

            # MLP projections
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for name in ["up_proj", "gate_proj", "down_proj"]:
                    if name in self.bias_target_modules and hasattr(mlp, name):
                        mod = getattr(mlp, name)
                        if isinstance(mod, nn.Linear):
                            key = (layer_idx, name)
                            out_dim = mod.out_features
                            self.bias_specs.append((key, out_dim))
                            self.total_bias_dim += out_dim
                            h = mod.register_forward_hook(self._make_bias_hook(key))
                            self._hooks.append(h)

        if self.total_bias_dim == 0:
            raise RuntimeError(
                "No target modules found for requested bias_layers / bias_target_modules."
            )

        # deterministic order
        self.bias_specs.sort(key=lambda x: (x[0][0], x[0][1]))

    def _make_bias_hook(self, key: Tuple[int, str]):
        def hook(module, inputs, output):
            if self._current_bias is None:
                return output
            bias = self._current_bias.get(key, None)
            if bias is None:
                return output

            b = bias.to(device=output.device, dtype=output.dtype)
            if b.dim() == 2 and output.dim() >= 2:
                for _ in range(output.dim() - 2):
                    b = b.unsqueeze(1)
            elif b.dim() == 1 and output.dim() >= 1:
                b = b.view(1, *([1] * (output.dim() - 1)), -1)

            return output + b

        return hook

    def _set_bias_from_vec(self, bias_vec: torch.Tensor):
        if bias_vec is None:
            self._current_bias = None
            return

        assert bias_vec.dim() == 2, f"bias_vec must be [B, D_total], got {bias_vec.shape}"
        B, D = bias_vec.shape
        if D != self.total_bias_dim:
            raise ValueError(f"bias_vec second dim {D} != expected total_bias_dim {self.total_bias_dim}")

        offset = 0
        bias_dict: Dict[Tuple[int, str], torch.Tensor] = {}
        for (layer_idx, name), out_dim in self.bias_specs:
            chunk = bias_vec[:, offset:offset + out_dim]   # [B, out_dim]
            offset += out_dim
            bias_dict[(layer_idx, name)] = chunk
        self._current_bias = bias_dict

    def forward(
        self,
        *args,
        bias_vec: Optional[torch.Tensor] = None,
        bias_dict: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
        **kwargs,
    ):
        if bias_vec is not None and bias_dict is not None:
            raise ValueError("Provide either bias_vec or bias_dict, not both.")

        if bias_vec is not None:
            self._set_bias_from_vec(bias_vec)
        else:
            self._current_bias = bias_dict

        try:
            outputs = self.base_model(*args, **kwargs)
        finally:
            self._current_bias = None

        return outputs

    def generate(
        self,
        *args,
        bias_vec: Optional[torch.Tensor] = None,
        bias_dict: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
        **kwargs,
    ):
        if bias_vec is not None and bias_dict is not None:
            raise ValueError("Provide either bias_vec or bias_dict, not both.")

        if bias_vec is not None:
            self._set_bias_from_vec(bias_vec)
        else:
            self._current_bias = bias_dict

        try:
            outputs = self.base_model.generate(*args, **kwargs)
        finally:
            self._current_bias = None

        return outputs


class LoRAInjectedQwenWrapper(nn.Module):
    """
    Wrap a base LLM (Qwen3, SmolLM3, or Olmo-3) and allow per-example LoRA injection
    on selected projection modules (e.g., q_proj, v_proj, up_proj, gate_proj) in
    selected layers (e.g., the last eight).

    Instead of adding bias vectors, this injects low-rank adapters:
        output += (input @ A) @ B
    where A: (in_features, rank), B: (rank, out_features)
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        lora_layers: List[int],
        lora_target_modules: List[str],
        lora_rank: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = sorted(set(lora_layers))
        self.lora_target_modules = set(lora_target_modules)
        self.lora_rank = lora_rank

        # key: (layer_idx, module_name) -> (A: Tensor, B: Tensor)
        self._current_lora: Optional[Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]] = None

        # list of ((layer_idx, module_name), in_features, out_features)
        self.lora_specs: List[Tuple[Tuple[int, str], int, int]] = []
        self.total_lora_dim: int = 0

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._inspect_and_register_hooks()

    @property
    def input_splits(self) -> list[int]:
        """Returns param count per module: in_f * rank + rank * out_f"""
        return [in_f * self.lora_rank + self.lora_rank * out_f
                for (_, in_f, out_f) in self.lora_specs]

    def _inspect_and_register_hooks(self):
        transformer = self.base_model.model
        layers = transformer.layers

        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.lora_layers:
                continue

            # Attention projections
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for name in ["q_proj", "v_proj", "k_proj", "o_proj"]:
                    if name in self.lora_target_modules and hasattr(attn, name):
                        mod = getattr(attn, name)
                        if isinstance(mod, nn.Linear):
                            key = (layer_idx, name)
                            in_dim, out_dim = mod.in_features, mod.out_features
                            param_count = in_dim * self.lora_rank + self.lora_rank * out_dim
                            self.lora_specs.append((key, in_dim, out_dim))
                            self.total_lora_dim += param_count
                            h = mod.register_forward_hook(self._make_lora_hook(key))
                            self._hooks.append(h)

            # MLP projections
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for name in ["up_proj", "gate_proj", "down_proj"]:
                    if name in self.lora_target_modules and hasattr(mlp, name):
                        mod = getattr(mlp, name)
                        if isinstance(mod, nn.Linear):
                            key = (layer_idx, name)
                            in_dim, out_dim = mod.in_features, mod.out_features
                            param_count = in_dim * self.lora_rank + self.lora_rank * out_dim
                            self.lora_specs.append((key, in_dim, out_dim))
                            self.total_lora_dim += param_count
                            h = mod.register_forward_hook(self._make_lora_hook(key))
                            self._hooks.append(h)

        if self.total_lora_dim == 0:
            raise RuntimeError(
                "No target modules found for requested lora_layers / lora_target_modules."
            )

        # deterministic order
        self.lora_specs.sort(key=lambda x: (x[0][0], x[0][1]))

    def _make_lora_hook(self, key: Tuple[int, str]):
        def hook(module, inputs, output):
            if self._current_lora is None:
                return output
            lora = self._current_lora.get(key, None)
            if lora is None:
                return output

            A, B = lora  # A: (B, in_f, r), B: (B, r, out_f)
            x = inputs[0]  # (batch, seq, in_features)
            assert len(inputs) == 1 and isinstance(inputs, Tuple)
            assert x.shape[-1] == A.shape[1], f"Input dim {x.shape[-1]} != A dim {A.shape[1]}"
            A = A.to(device=output.device, dtype=output.dtype)
            B = B.to(device=output.device, dtype=output.dtype)

            # Compute (x @ A) @ B with proper batching
            # x: (B, S, in_f), A: (B, in_f, r), B: (B, r, out_f)
            delta = (x @ A) @ B  # (B, S, out_f)
            return output + delta

        return hook

    def _set_lora_from_vec(self, lora_vec: torch.Tensor):
        if lora_vec is None:
            self._current_lora = None
            return

        assert lora_vec.dim() == 2, f"lora_vec must be [B, D_total], got {lora_vec.shape}"
        B, D = lora_vec.shape
        if D != self.total_lora_dim:
            raise ValueError(f"lora_vec second dim {D} != expected total_lora_dim {self.total_lora_dim}")

        offset = 0
        lora_dict: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        for (layer_idx, name), in_f, out_f in self.lora_specs:
            a_size = in_f * self.lora_rank
            b_size = self.lora_rank * out_f

            A_flat = lora_vec[:, offset:offset + a_size]
            offset += a_size
            B_flat = lora_vec[:, offset:offset + b_size]
            offset += b_size

            A = A_flat.view(B, in_f, self.lora_rank)
            B_mat = B_flat.view(B, self.lora_rank, out_f)
            lora_dict[(layer_idx, name)] = (A, B_mat)

        self._current_lora = lora_dict

    def forward(
        self,
        *args,
        bias_vec: Optional[torch.Tensor] = None,  # flattened per-example LoRA A/B params (same kwarg SFTWrapper uses)
        **kwargs,
    ):
        self._set_lora_from_vec(bias_vec)

        try:
            outputs = self.base_model(*args, **kwargs)
        finally:
            self._current_lora = None

        return outputs

    def generate(
        self,
        *args,
        bias_vec: Optional[torch.Tensor] = None,  # flattened per-example LoRA A/B params (same kwarg SFTWrapper uses)
        **kwargs,
    ):
        self._set_lora_from_vec(bias_vec)

        try:
            outputs = self.base_model.generate(*args, **kwargs)
        finally:
            self._current_lora = None

        return outputs


def load_model(args, hypernet_state=None):
    # setup hypernetwork on base model
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        use_cache=False,
    )

    if args.use_lora_injection:
        base_model = LoRAInjectedQwenWrapper(
            base_model=base,
            lora_layers=[int(layer) for layer in args.bias_layers_to_transform],
            lora_target_modules=list(args.bias_target_modules),
            lora_rank=args.lora_rank,
        )
    else:
        base_model = BiasInjectedQwenWrapper(
            base_model=base,
            bias_layers=[int(layer) for layer in args.bias_layers_to_transform],
            bias_target_modules=list(args.bias_target_modules),
        )

    hypernet = ParamHyperNetModel(
        text_encoder_name=args.text_encoder_name,
        param_splits=base_model.input_splits,
        param_hidden=args.param_hidden,
        dit_layers=args.dit_layers,
        use_text_embedding=args.use_text_embedding,
        num_codes=args.num_codes,
        text_embed_dim=args.text_embed_dim,
        code_dim=args.code_dim,
        vq_beta=getattr(args, 'vq_beta', 0.25),
        vq_gumbel=getattr(args, 'vq_gumbel', True),
        vq_soft=getattr(args, 'vq_soft', True),
        vq_usage_weight=getattr(args, 'vq_usage_weight', 0.0),
        torch_dtype=getattr(args, 'torch_dtype', torch.bfloat16),
        use_lora_injection=getattr(args, 'use_lora_injection', False),
    )
    # Convert entire hypernet to correct dtype before loading state dict
    hypernet = hypernet.to(getattr(args, 'torch_dtype', torch.bfloat16))
    if hypernet_state is not None:
        hypernet.load_state_dict(hypernet_state, strict=True)
    if is_main_process():
        n_params = 0
        for p in hypernet.parameters():
            if p.requires_grad:
                n_params += p.numel()
        print(f"[Model] Hypernetwork loaded, trainable params: {n_params/1024/1024:.2f}M")
        if args.use_lora_injection:
            print(f"[Model] Total LoRA params predicted per query: {base_model.total_lora_dim:,} (rank={args.lora_rank})")
        else:
            print(f"[Model] Total bias params predicted per query: {base_model.total_bias_dim:,}")

    model = SFTWrapper(base_model, hypernet)
    return model
