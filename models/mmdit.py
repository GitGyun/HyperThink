# Adapted from https://github.com/lucidrains/mmdit (MIT License, Copyright (c) 2024 Phil Wang).
from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

from hyper_connections import (
    HyperConnections,
    Residual
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)


class Modality1AttendTo01(Module):
    """
    Update modality 1 only:
      q = modality1
      k,v = concat(modality0, modality1)

    Returns:
      out0: zeros_like(x0)  (so residual keeps modality 0 unchanged)
      out1: attention update for modality 1
    """
    def __init__(
        self,
        *,
        dim0: int,
        dim1: int,
        dim_head: int = 64,
        heads: int = 8,
        qk_rmsnorm: bool = False,
        flash: bool = False,
        softclamp: bool = False,
        softclamp_value: float = 50.,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        dim_inner = dim_head * heads

        # q from modality 1
        self.to_q1 = nn.Linear(dim1, dim_inner, bias=False)

        # k,v from modality 0 and modality 1 (separate projections to handle different dims)
        self.to_kv0 = nn.Linear(dim0, dim_inner * 2, bias=False)
        self.to_kv1 = nn.Linear(dim1, dim_inner * 2, bias=False)

        self.split_q  = Rearrange('b n (h d) -> b h n d', h=heads)
        self.split_kv = Rearrange('b n (kv h d) -> kv b h n d', kv=2, h=heads)

        self.attend = Attend(
            flash=flash,
            softclamp_logits=softclamp,
            logit_softclamp_value=softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out1 = nn.Linear(dim_inner, dim1, bias=False)

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_rms = MultiHeadRMSNorm(dim_head, heads=heads)
            self.k_rms = MultiHeadRMSNorm(dim_head, heads=heads)

        self.register_buffer('dummy', torch.tensor(0), persistent=False)

    def forward(
        self,
        x0: Tensor,                      # (b, n0, dim0)
        x1: Tensor,                      # (b, n1, dim1)
        mask0: Tensor | None = None,     # (b, n0) bool, True=keep
        mask1: Tensor | None = None,     # (b, n1) bool, True=keep
    ):
        device = self.dummy.device

        if mask0 is None:
            mask0 = torch.ones(x0.shape[:2], device=device, dtype=torch.bool)
        if mask1 is None:
            mask1 = torch.ones(x1.shape[:2], device=device, dtype=torch.bool)

        # q from modality 1
        q1 = self.split_q(self.to_q1(x1))  # (b, h, n1, d)

        # kv from modality 0
        kv0 = self.split_kv(self.to_kv0(x0))  # (2, b, h, n0, d)
        k0, v0 = kv0

        # kv from modality 1
        kv1 = self.split_kv(self.to_kv1(x1))  # (2, b, h, n1, d)
        k1, v1 = kv1

        if self.qk_rmsnorm:
            q1 = self.q_rms(q1)
            # normalize keys after concat, but rmsnorm is per-tensor; normalize each then concat
            k0 = self.k_rms(k0)
            k1 = self.k_rms(k1)

        # concat KV along sequence dimension
        k = torch.cat([k0, k1], dim=2)            # (b, h, n0+n1, d)
        v = torch.cat([v0, v1], dim=2)            # (b, h, n0+n1, d)
        kv_mask = torch.cat([mask0, mask1], dim=1) # (b, n0+n1)

        out1, *_ = self.attend(q1, k, v, mask=kv_mask)  # (b, h, n1, d)
        out1 = self.to_out1(self.merge_heads(out1))     # (b, n1, dim1)

        out0 = torch.zeros_like(x0)  # “modality 0 does not attend anywhere”
        return out0, out1

# class

class AdaptiveLayerNorm(Module):
    """Pre-norm LayerNorm used by every MMDiT branch (checkpoint keys `*.ln.weight`)."""
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        x,
    ):
        return self.ln(x)

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_modalities: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        softclamp = False,
        softclamp_value = 50.,
        num_residual_streams = 1,
        cross_attention: bool = False,
        ff_kwargs: dict = dict(),
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # residuals / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.attn_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])
        self.ff_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim) for dim in dim_modalities]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.cross_attention = cross_attention
        if cross_attention:
            assert len(dim_modalities) == 2, 'cross attention only supports 2 modalities currently'
            self.joint_attn = Modality1AttendTo01(
                dim0 = dim_modalities[0],
                dim1 = dim_modalities[1],
                dim_head = dim_head,
                heads = heads,
                qk_rmsnorm = qk_rmsnorm,
                flash = flash_attn,
                softclamp = softclamp,
                softclamp_value = softclamp_value,
            )
        else:
            self.joint_attn = JointAttention(
                dim_inputs = dim_modalities,
                dim_head = dim_head,
                heads = heads,
                qk_rmsnorm = qk_rmsnorm,
                flash = flash_attn,
                softclamp = softclamp,
                softclamp_value = softclamp_value,
            )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim) for dim in dim_modalities]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: tuple[Tensor, ...],
        modality_masks: tuple[Tensor | None, ...] | None = None,
    ):
        assert len(modality_tokens) == self.num_modalities

        # attention layernorms

        modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.attn_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

        # attention

        if self.cross_attention:
            assert len(modality_tokens) == 2, 'cross attention only supports 2 modalities currently'
            x0, x1 = modality_tokens
            m0 = None if modality_masks is None else modality_masks[0]
            m1 = None if modality_masks is None else modality_masks[1]

            out0, out1 = self.joint_attn(x0, x1, mask0=m0, mask1=m1)
            modality_tokens = (out0, out1)
        else:
            modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # add attention residual

        modality_tokens = [add_attn_residual(tokens) for add_attn_residual, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

        # handle feedforward adaptive layernorm

        modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.ff_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # add feedforward residual

        modality_tokens = [add_residual_fn(tokens) for add_residual_fn, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

        # returns

        return modality_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities,
        num_residual_streams = 4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        blocks = [MMDiTBlock(dim_modalities = dim_modalities, num_residual_streams = num_residual_streams, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: tuple[Tensor, ...],
        modality_masks: tuple[Tensor | None, ...] | None = None,
    ):

        modality_tokens = [self.expand_streams(modality) for modality in modality_tokens]

        for block in self.blocks:
            modality_tokens = block(
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        modality_tokens = [self.reduce_streams(modality) for modality in modality_tokens]

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        return tuple(modality_tokens)