import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class VectorQuantizer(nn.Module):
    """Vector quantizer with Gumbel, straight-through soft, or nearest-code assignment.

    Gumbel selection is stochastic at both train and eval time.
    """
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        beta: float = 0.25,
        gumbel_temp: float = 1.0,
        usage_weight: float = 0.0,     # perplexity regularizer weight
        usage_eps: float = 1e-8,
        gumbel: bool = True,
        soft: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.gumbel_temp = gumbel_temp
        self.usage_weight = usage_weight
        self.usage_eps = usage_eps
        self.gumbel = gumbel
        self.soft = soft

        self.embeddings = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0 / math.sqrt(code_dim))

    def _compute_dist(self, z: torch.Tensor) -> torch.Tensor:
        z_sq = (z ** 2).sum(dim=1, keepdim=True)                          # [B,1]
        e_sq = (self.embeddings.weight ** 2).sum(dim=1, keepdim=True).T   # [1,K]
        z_e = z @ self.embeddings.weight.T                                # [B,K]
        return z_sq - 2 * z_e + e_sq

    def forward(
        self,
        z: torch.Tensor,
        return_vq_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        assert z.dim() == 2
        B, C = z.shape
        assert C == self.code_dim

        dist_mat = self._compute_dist(z)  # [B,K], differentiable

        if self.gumbel:
            probs = F.gumbel_softmax(-dist_mat, tau=self.gumbel_temp,
                                     hard=(not self.soft), dim=1)
            z_q = probs @ self.embeddings.weight
            codes = probs.detach().argmax(dim=1)
        elif self.soft:
            probs = (-dist_mat).softmax(dim=1)
            codes = probs.detach().argmax(dim=1)
            z_q_hard = self.embeddings(codes)
            z_q_soft = probs @ self.embeddings.weight
            z_q = z_q_soft + (z_q_hard - z_q_soft).detach()
        else:
            codes = dist_mat.detach().argmin(dim=1)
            z_q = self.embeddings(codes)
            # Match the legacy nearest-code usage regularizer temperature.
            probs = (-dist_mat / 0.2).softmax(dim=1)

        # VQ loss
        if return_vq_loss:
            commit_loss = F.mse_loss(z, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z.detach())
            vq_loss = codebook_loss + self.beta * commit_loss

            # usage / perplexity regularizer (differentiable)
            if self.training and self.usage_weight > 0.0:
                hard = F.one_hot(codes, num_classes=self.num_codes).to(dtype=probs.dtype, device=probs.device)  # [B,K]
                probs_st = hard + probs - probs.detach()  # [B,K]
                q = probs_st.mean(dim=0)  # [K] (forward == hard histogram / B)
                # KL(q || Unif) = sum q * log(q*K)
                usage_loss = (q * torch.log(q * self.num_codes + self.usage_eps)).sum()
            else:
                usage_loss = None
        else:
            vq_loss = None
            usage_loss = None

        if not self.soft and not self.gumbel:
            z_q = z + (z_q - z).detach()
        return z_q, vq_loss, usage_loss, codes


class VQParamDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,           # H (mmDiT hidden size)
        param_splits: List[int],   # [d_0, d_1, ..., d_{N-1}]
        num_codes: int = 256,
        code_dim: int = 128,
        beta: float = 0.25,
        usage_weight: float = 0.0,     # perplexity regularizer weight
        zero_init: bool = True,    # If False, use default init (needed for LoRA to have gradient flow)
        gumbel: bool = True,
        soft: bool = True,
    ):
        """
        For each token i:
          - proj_down_i:  H -> code_dim
          - VQ_i:         code_dim -> quantized code_dim
          - proj_up_i:    code_dim -> d_i
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.param_splits = param_splits
        self.num_tokens = len(param_splits)

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.zero_init = zero_init

        # Per-token subspace + VQ + decoder
        self.proj_down = nn.ModuleList(
            [nn.Linear(hidden_dim, code_dim) for _ in range(self.num_tokens)]
        )
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(num_codes, code_dim, beta, usage_weight=usage_weight,
                             gumbel=gumbel, soft=soft)
             for _ in range(self.num_tokens)]
        )
        self.proj_up = nn.ModuleList(
            [nn.Linear(code_dim, d_i) for d_i in param_splits]
        )

        if self.zero_init:
            # Zero init is safe for bias injection (gradient = 1)
            # but causes dead gradients for LoRA (gradient depends on A*B)
            for proj_up in self.proj_up:
                nn.init.zeros_(proj_up.weight)
                nn.init.zeros_(proj_up.bias)

    def forward(
        self,
        token_states: torch.Tensor,
        return_vq_loss: bool = False,
        get_vq_index: bool = False,
    ):
        """
        token_states: (B, N, H)
        returns:
            param_vec: (B, sum(param_splits))
            vq_loss: scalar tensor
        """
        B, N, H = token_states.shape
        assert N == self.num_tokens, f"N={N}, expected {self.num_tokens}"
        assert H == self.hidden_dim

        chunks = []
        vq_losses = []
        usage_losses = []
        codes = []

        # loop over tokens/layers
        for i in range(N):
            # token i state: (B, H)
            h_i = token_states[:, i, :]                        # (B, H)
            z_i = self.proj_down[i](h_i)                       # (B, code_dim)

            z_q_i, vq_loss_i, usage_loss_i, codes_i = self.quantizers[i](z_i, return_vq_loss=return_vq_loss)      # (B, code_dim), scalar
            if return_vq_loss:
                vq_losses.append(vq_loss_i)
                if usage_loss_i is not None:
                    usage_losses.append(usage_loss_i)
            if get_vq_index:
                codes.append(codes_i)

            chunk_i = self.proj_up[i](z_q_i)                   # (B, d_i)
            chunks.append(chunk_i)

        param_vec = torch.cat(chunks, dim=-1)                  # (B, D)
        vq_loss = torch.stack(vq_losses).mean() if return_vq_loss else None  # average over layers
        usage_loss = torch.stack(usage_losses).mean() if return_vq_loss and usage_losses else None
        codes = torch.stack(codes, dim=1) if get_vq_index else None  # (B, N)

        out_dict = {
            'pred_delta': param_vec,
            'vq_loss': vq_loss,
            'usage_loss': usage_loss,
            'vq_index': codes,
        }

        return out_dict
