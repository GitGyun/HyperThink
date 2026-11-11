from typing import Optional

import torch
import torch.nn as nn

from models.mmdit import MMDiT
from models.vq_decoder import VQParamDecoder
from models.text_encoder import TextEncoder


class ParamHyperNetModel(nn.Module):
    """HyperThink hypernetwork. Maps a query to parameter deltas (biases, or LoRA
    factors) for the frozen LLM through three parts: the Text Encoder
    (`text_encoder`), the Bias Encoder (`mmdit`), and the VQ Decoder (`param_decoder`).
    """
    def __init__(
        self,
        text_encoder_name,
        param_splits,
        param_hidden: int = 128,
        dit_layers: int = 3,
        use_text_embedding: bool = False,
        num_codes: int = 256,
        text_embed_dim: int = 128,
        code_dim: int = 512,
        vq_beta: float = 0.25,
        vq_usage_weight: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_lora_injection: bool = False,  # If True, disable zero_init to allow gradient flow through LoRA A*B
        vq_gumbel: bool = True,
        vq_soft: bool = True,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype

        # text encoder (frozen backbone; only the embedding projection trains)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            use_text_embedding=use_text_embedding,
            embed_dim=text_embed_dim,
        ).eval()
        self.text_dim = self.text_encoder.dim
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.embed_proj.parameters():
            p.requires_grad = True

        # param tokens
        self.param_dim = param_hidden
        self.param_tokens = nn.Parameter(torch.empty(len(param_splits), self.param_dim))
        nn.init.normal_(self.param_tokens, mean=0.0, std=0.02)

        decoder_hidden = 512  # VQ decoder input width
        # MMDiT backbone
        self.mmdit = MMDiT(
            depth=dit_layers,
            dim_modalities=(self.text_dim, self.param_dim),
            qk_rmsnorm=True,
            flash_attn=True,
            cross_attention=False,
        )
        if decoder_hidden != param_hidden:
            self.param_proj = nn.Linear(param_hidden, decoder_hidden)
        else:
            self.param_proj = nn.Identity()

        # param decoder
        code_dim = code_dim if code_dim > 0 else param_hidden
        self.param_decoder = VQParamDecoder(
            hidden_dim=decoder_hidden,
            param_splits=param_splits,
            code_dim=code_dim,
            num_codes=num_codes,
            beta=vq_beta,
            gumbel=vq_gumbel,
            soft=vq_soft,
            usage_weight=vq_usage_weight,
            zero_init=(not use_lora_injection),  # LoRA needs non-zero init for gradient flow
        )

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_text: Optional[list] = None,
        get_vq_index: bool = False,
        **kwargs,
    ):
        """Predict parameter deltas for a batch of queries.

        Args:
            input_ids: (B, T) text-encoder token ids of the queries.
            attention_mask: (B, T) attention mask for input_ids.
            get_vq_index: also return the selected code of every parameter token.
        Returns:
            dict with `pred_delta` (B, D) flattened parameter deltas, `vq_loss` and
            `usage_loss` (training only), `vq_index` (B, P) when requested, and
            `backbone_output` (B, P, H) before quantization.
        """
        # MMDiT backbone: use text encoder
        text_tokens = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # (B, T, H_text)
        B = text_tokens.size(0)

        param_tokens = self.param_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, P, H_param)

        if self.text_encoder.use_text_embedding:
            # one pooled token -> text_tokens.size(1) == 1
            text_mask = torch.ones(B, text_tokens.size(1), device=text_tokens.device, dtype=torch.bool)
        else:
            text_mask = attention_mask.bool()
        param_mask = torch.ones(B, param_tokens.size(1), device=param_tokens.device, dtype=torch.bool)

        _, out_param = self.mmdit(
            modality_tokens=(text_tokens, param_tokens),
            modality_masks=(text_mask, param_mask),
        )
        out_param = self.param_proj(out_param)

        # decode each parameter token into its slice of the parameter deltas
        out_dict = self.param_decoder(
            out_param,
            return_vq_loss=self.training,
            get_vq_index=get_vq_index,
        )
        out_dict['backbone_output'] = out_param  # (B, N, H) pre-VQ

        return out_dict
