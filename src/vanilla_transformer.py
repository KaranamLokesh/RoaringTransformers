"""
Vanilla Transformer for language modeling baseline.
Standard dense attention masks, no Roaring Bitmap optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)  # broadcast (1,1,L,L)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class VanillaTransformer(nn.Module):
    """Standard Transformer language model with dense causal + sliding window mask."""

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4,
                 ff_dim=512, max_seq_len=2048, dropout=0.1, window_size=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_dense_mask(self, seq_len, device):
        """Create a causal mask, optionally with a sliding window."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = 0 if self.window_size is None else max(0, i - self.window_size)
            mask[i, start:i + 1] = 0.0
        return mask

    def forward(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        attn_mask = self._create_dense_mask(L, device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_mask_memory_bytes(self, seq_len):
        """Return the memory footprint of the dense attention mask in bytes."""
        return seq_len * seq_len * 4  # float32
