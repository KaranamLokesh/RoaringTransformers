"""
Roaring Bitmap-backed sparse attention mask management.
Stores attention patterns as compressed Roaring Bitmaps on CPU,
realizes them into GPU-compatible formats on demand.
"""

import torch
import numpy as np
from pyroaring import BitMap
import time


class RoaringAttentionMask:
    """
    Manages sparse attention masks using Roaring Bitmaps.

    Instead of storing a dense (seq_len x seq_len) float mask in VRAM,
    this stores a list of Roaring Bitmaps (one per row) on the CPU.
    Each bitmap contains the column indices that the row is allowed to attend to.

    The mask is "realized" (converted to a dense or sparse tensor) only when
    needed for the forward pass, then discarded.
    """

    def __init__(self, seq_len, window_size=None, causal=True):
        self.seq_len = seq_len
        self.window_size = window_size
        self.causal = causal
        self.row_bitmaps = []
        self._build_bitmaps()

    def _build_bitmaps(self):
        """Build per-row Roaring Bitmaps for the attention pattern."""
        self.row_bitmaps = []
        for i in range(self.seq_len):
            bm = BitMap()
            if self.causal:
                start = 0 if self.window_size is None else max(0, i - self.window_size)
                end = i + 1
            else:
                if self.window_size is not None:
                    start = max(0, i - self.window_size)
                    end = min(self.seq_len, i + self.window_size + 1)
                else:
                    start = 0
                    end = self.seq_len
            bm.add_range(start, end)
            self.row_bitmaps.append(bm)

    def realize_dense(self, device='cpu'):
        """Convert Roaring Bitmaps to a dense float attention mask.

        Returns a (seq_len, seq_len) tensor where allowed positions are 0.0
        and blocked positions are -inf.
        """
        mask = torch.full(
            (self.seq_len, self.seq_len), float('-inf'),
            dtype=torch.float32, device='cpu'
        )
        for i, bm in enumerate(self.row_bitmaps):
            indices = torch.tensor(list(bm), dtype=torch.long)
            mask[i, indices] = 0.0
        return mask.to(device)

    def realize_sparse_csr(self, device='cpu'):
        """Convert Roaring Bitmaps to a sparse CSR tensor.

        Returns a sparse CSR tensor with 0.0 at allowed positions.
        This is more memory-efficient for very long sequences.
        """
        row_indices = []
        col_indices = []
        for i, bm in enumerate(self.row_bitmaps):
            cols = list(bm)
            row_indices.extend([i] * len(cols))
            col_indices.extend(cols)

        values = torch.zeros(len(row_indices), dtype=torch.float32)
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        sparse = torch.sparse_coo_tensor(
            indices, values, (self.seq_len, self.seq_len)
        )
        return sparse.to_sparse_csr().to(device)

    def get_compressed_size_bytes(self):
        """Return the total serialized size of all Roaring Bitmaps in bytes."""
        return sum(len(bm.serialize()) for bm in self.row_bitmaps)

    def get_dense_size_bytes(self):
        """Return the equivalent dense mask size in bytes (float32)."""
        return self.seq_len * self.seq_len * 4

    def compression_ratio(self):
        """Return how much smaller the Roaring representation is."""
        return self.get_dense_size_bytes() / self.get_compressed_size_bytes()

    def combine_masks(self, other):
        """Combine two Roaring masks using bitwise AND (intersection).

        This is useful for combining causal + padding masks, or
        local window + global sink tokens.
        """
        assert self.seq_len == other.seq_len
        combined = RoaringAttentionMask.__new__(RoaringAttentionMask)
        combined.seq_len = self.seq_len
        combined.window_size = self.window_size
        combined.causal = self.causal
        combined.row_bitmaps = [
            a & b for a, b in zip(self.row_bitmaps, other.row_bitmaps)
        ]
        return combined

    def add_global_tokens(self, global_indices):
        """Add global 'sink' tokens that every position can attend to.

        This implements the Longformer-style global attention pattern.
        """
        global_bm = BitMap(global_indices)
        for bm in self.row_bitmaps:
            bm |= global_bm

    def __repr__(self):
        dense_mb = self.get_dense_size_bytes() / (1024 * 1024)
        roaring_kb = self.get_compressed_size_bytes() / 1024
        return (
            f"RoaringAttentionMask(seq_len={self.seq_len}, "
            f"window={self.window_size}, causal={self.causal}, "
            f"dense={dense_mb:.1f}MB, roaring={roaring_kb:.1f}KB, "
            f"compression={self.compression_ratio():.1f}x)"
        )


def benchmark_mask_creation(seq_len, window_size=128):
    """Benchmark Roaring vs dense mask creation time and memory."""
    # Dense
    t0 = time.time()
    dense_mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        start = max(0, i - window_size)
        dense_mask[i, start:i + 1] = 0.0
    dense_time = time.time() - t0
    dense_mem = dense_mask.element_size() * dense_mask.nelement()

    # Roaring
    t0 = time.time()
    roaring = RoaringAttentionMask(seq_len, window_size=window_size, causal=True)
    roaring_time = time.time() - t0
    roaring_mem = roaring.get_compressed_size_bytes()

    return {
        'seq_len': seq_len,
        'window_size': window_size,
        'dense_time_s': dense_time,
        'roaring_time_s': roaring_time,
        'dense_mem_bytes': dense_mem,
        'roaring_mem_bytes': roaring_mem,
        'compression_ratio': dense_mem / roaring_mem,
    }
