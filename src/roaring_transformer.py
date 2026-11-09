"""
Roaring Transformer — full Transformer language model that integrates
Roaring Bitmaps for attention mask management, gradient compression,
and weight pruning.

Same architecture as VanillaTransformer but with three Roaring components:
1. Attention mask stored as a Roaring Bitmap on the host, realized to a dense
   tensor once per sequence length and cached (accelerator memory is unchanged)
2. Top-K gradient index encoding via Roaring (compress_gradients() reports
   payload sizes; it does not replace the optimizer's gradient exchange)
3. Post-warmup magnitude pruning with surviving indices tracked in Roaring
   Bitmaps (pruned weights are zeroed; compute stays dense)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .roaring_attention import RoaringAttentionMask
from .roaring_gradient import RoaringGradientCompressor
from .roaring_pruning import RoaringPruningManager


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
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
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


class RoaringTransformer(nn.Module):
    """
    Transformer language model with Roaring Bitmap optimizations.

    Key differences from VanillaTransformer:
    - Attention mask is stored as Roaring Bitmaps and realized on-demand
    - Gradient compression uses Roaring Bitmaps for Top-K index encoding
    - Weight pruning masks are tracked via Roaring Bitmaps
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4,
                 ff_dim=512, max_seq_len=2048, dropout=0.1, window_size=128,
                 grad_sparsity=0.01, prune_ratio=0.5, prune_warmup=100):
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
        self.lm_head.weight = self.token_emb.weight

        # Roaring components
        self.grad_compressor = RoaringGradientCompressor(sparsity=grad_sparsity)
        self.pruning_manager = RoaringPruningManager(
            prune_ratio=prune_ratio, warmup_steps=prune_warmup
        )

        # Cache for realized masks (cleared when seq_len changes)
        self._mask_cache = {}

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_roaring_mask(self, seq_len, device):
        """Get or create a Roaring-backed attention mask, realized to dense."""
        if seq_len not in self._mask_cache:
            roaring_mask = RoaringAttentionMask(
                seq_len, window_size=self.window_size, causal=True
            )
            realized = roaring_mask.realize_dense(device)
            self._mask_cache[seq_len] = {
                'roaring': roaring_mask,
                'realized': realized,
            }
        cached = self._mask_cache[seq_len]
        if cached['realized'].device != device:
            cached['realized'] = cached['realized'].to(device)
        return cached

    def forward(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        mask_info = self._get_roaring_mask(L, device)
        attn_mask = mask_info['realized']

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_mask_memory_bytes(self, seq_len):
        """Return both dense equivalent and Roaring compressed mask sizes."""
        mask_info = self._get_roaring_mask(seq_len, 'cpu')
        roaring = mask_info['roaring']
        return {
            'dense_bytes': roaring.get_dense_size_bytes(),
            'roaring_bytes': roaring.get_compressed_size_bytes(),
            'compression_ratio': roaring.compression_ratio(),
        }

    def compress_gradients(self):
        """Compress all gradients using Roaring Bitmaps. Returns stats."""
        stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                s = self.grad_compressor.get_compression_stats(param.grad)
                stats[name] = s
        return stats

    def prune_step(self):
        """Advance the pruning manager by one step."""
        self.pruning_manager.step(self)
        if self.pruning_manager.is_pruned:
            self.pruning_manager.apply_masks(self)
