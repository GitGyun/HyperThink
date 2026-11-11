
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel


class TextEncoder(nn.Module):
    """Frozen HF encoder (final-layer hidden states) + a trainable linear projection."""
    def __init__(self, model_name: str, use_text_embedding: bool = False, embed_dim: int = -1):
        super().__init__()
        self.enc_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.enc_cfg,
            use_safetensors=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.dim = getattr(self.enc_cfg, "d_model", None) or getattr(self.enc_cfg, "hidden_size")
        self.embed_dim = embed_dim
        if embed_dim > 0:
            self.embed_proj = nn.Linear(self.dim, embed_dim)
            self.dim = embed_dim
        else:
            self.embed_proj = nn.Identity()

        self.use_text_embedding = use_text_embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state                                   # (B, T, H)
        if self.use_text_embedding:
            hidden = self.last_token_pool(hidden, attention_mask).unsqueeze(1)   # (B, 1, H)
        return self.embed_proj(hidden)
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
