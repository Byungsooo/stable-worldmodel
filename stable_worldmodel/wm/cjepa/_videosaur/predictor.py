"""Slot-dynamics predictor, vendored from martius-lab/videosaur (MIT License).

Source: videosaur/modules/networks.py — ``Attention``, ``TransformerEncoderLayer``,
``TransformerEncoder``, trimmed to the single self-attention-only, no-mask,
no-memory, no-dropout configuration used by VideoSAUR's PushT config
(``predictor: name: networks.TransformerEncoder, dim: 128, n_blocks: 1,
n_heads: 4``). This is VideoSAUR's own lightweight slot-to-slot dynamics
model, applied *inside* the recurrent temporal loop (see video.py's
``LatentProcessor``) to propagate slot state from one frame to the next —
distinct from, and upstream of, C-JEPA's own bidirectional predictor.

Kept parameter names (``self_attn.qkv``, ``self_attn.out_proj``, ``linear1``,
``linear2``, ``norm1``, ``norm2``) identical to upstream so the released
checkpoint's ``state_dict`` loads without remapping.
"""

import torch
from torch import nn


class Attention(nn.Module):
    """Self-attention with a combined QKV projection (adapted from timm's ViT)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('`dim` must be divisible by `num_heads`')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Pre-norm transformer encoder layer with custom (combined-QKV) self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, qkv_bias: bool = True):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = Attention(d_model, nhead, qkv_bias=qkv_bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src
        x = x + self.self_attn(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, n_blocks: int, n_heads: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(dim, n_heads, hidden_dim) for _ in range(n_blocks)]
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        for block in self.blocks:
            x = block(x)
        return x
