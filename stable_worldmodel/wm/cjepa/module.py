import torch
import torch.nn as nn

from stable_worldmodel.wm.lewm.module import Attention, Embedder, FeedForward


class BidirectionalBlock(nn.Module):
    """Transformer block with full (non-causal) bidirectional attention."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.mlp(self.norm2(x))
        return x


class BidirectionalTransformer(nn.Module):
    """Bidirectional transformer for joint masked slot prediction (paper predictor f)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [BidirectionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        """x: (B, L, D) where L = T_total * N_total"""
        for block in self.layers:
            x = block(x)
        return self.norm(x)


class TemporalPosEmb(nn.Module):
    """Learnable temporal positional embedding e_tau (Eq. 3 in paper).

    No positional encoding along the entity dimension, consistent with
    permutation-equivariant slot representations (Wu et al., 2023).
    """

    def __init__(self, max_len, dim):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, t):
        """t: LongTensor of shape (...) with timestep indices"""
        return self.emb(t)


# Re-export Embedder so callers can import from one place
__all__ = ['BidirectionalTransformer', 'TemporalPosEmb', 'Embedder']
