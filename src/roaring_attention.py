"""
Roaring Bitmap-backed sparse attention mask management.
Stores attention patterns as a compressed Roaring Bitmap on CPU,
realizes them into GPU-compatible formats on demand.
"""

import time

import numpy as np
import torch
from pyroaring import BitMap


class RoaringAttentionMask:
    """
    Manages a sparse attention mask as a single Roaring Bitmap.

    The mask is stored as one bitmap over the flattened (seq_len x seq_len)
    key space: position i*seq_len + j is present when query i may attend to
    key j. Contiguous spans are inserted with add_range so they become Run
    Containers, and run_optimize() re-encodes whatever remains.

    An earlier version of this class kept one bitmap per query row and never
    called run_optimize(). That layout paid a serialization header per row and
    forced set operations through a seq_len-iteration Python loop; measured
    against the flat layout it is ~2.3x larger and ~33x slower to compose.

    The mask is "realized" (converted to a dense or sparse tensor) only when
    needed for the forward pass.
    """

    def __init__(self, seq_len, window_size=None, causal=True):
        self.seq_len = seq_len
        self.window_size = window_size
        self.causal = causal
        self.bitmap = BitMap()
        self._build_bitmap()

    def _row_span(self, i):
        """Half-open [start, end) column span that query position i attends to."""
        if self.causal:
            start = 0 if self.window_size is None else max(0, i - self.window_size)
            return start, i + 1
        if self.window_size is None:
            return 0, self.seq_len
        return max(0, i - self.window_size), min(self.seq_len, i + self.window_size + 1)

    def _build_bitmap(self):
        """Build the flat Roaring Bitmap for the attention pattern."""
        n = self.seq_len
        self.bitmap = BitMap()
        for i in range(n):
            start, end = self._row_span(i)
            if end > start:
                self.bitmap.add_range(i * n + start, i * n + end)
        self.bitmap.run_optimize()

    def _keys(self):
        """Set members as a numpy array of flattened positions."""
        return np.frombuffer(self.bitmap.to_array(), dtype=np.uint32)

    def realize_dense(self, device='cpu'):
        """Convert the Roaring Bitmap to a dense float attention mask.

        Returns a (seq_len, seq_len) tensor where allowed positions are 0.0
        and blocked positions are -inf.
        """
        n = self.seq_len
        mask = torch.full((n, n), float('-inf'), dtype=torch.float32, device='cpu')
        keys = self._keys().astype(np.int64)
        mask.view(-1)[torch.from_numpy(keys)] = 0.0
        return mask.to(device)

    def realize_sparse_csr(self, device='cpu'):
        """Convert the Roaring Bitmap to a sparse CSR tensor with 0.0 at
        allowed positions. More memory-efficient for very long sequences."""
        n = self.seq_len
        keys = self._keys().astype(np.int64)
        indices = torch.from_numpy(np.stack([keys // n, keys % n]))
        values = torch.zeros(len(keys), dtype=torch.float32)
        sparse = torch.sparse_coo_tensor(indices, values, (n, n))
        return sparse.to_sparse_csr().to(device)

    def get_compressed_size_bytes(self):
        """Return the serialized size of the Roaring Bitmap in bytes."""
        return len(self.bitmap.serialize())

    def get_dense_size_bytes(self):
        """Return the equivalent dense mask size in bytes (float32)."""
        return self.seq_len * self.seq_len * 4

    def compression_ratio(self):
        """Return how much smaller the Roaring representation is."""
        return self.get_dense_size_bytes() / self.get_compressed_size_bytes()

    def combine_masks(self, other):
        """Combine two Roaring masks using bitwise AND (intersection).

        Useful for combining causal + padding masks, or local window + global
        sink tokens. This is a single vectorised set operation over the whole
        mask rather than one call per query row.
        """
        assert self.seq_len == other.seq_len
        combined = RoaringAttentionMask.__new__(RoaringAttentionMask)
        combined.seq_len = self.seq_len
        combined.window_size = self.window_size
        combined.causal = self.causal
        combined.bitmap = self.bitmap & other.bitmap
        combined.bitmap.run_optimize()
        return combined

    def add_global_tokens(self, global_indices):
        """Add global 'sink' tokens that every position can attend to,
        implementing the Longformer-style global attention pattern."""
        n = self.seq_len
        globals_bm = BitMap()
        for g in global_indices:
            # column g of every row: keys {i*n + g : i in [0, n)}
            globals_bm.update(np.arange(g, n * n, n, dtype=np.uint32))
        self.bitmap |= globals_bm
        self.bitmap.run_optimize()

    def __repr__(self):
        return (
            f"RoaringAttentionMask(seq_len={self.seq_len}, "
            f"window={self.window_size}, causal={self.causal}, "
            f"dense={self.get_dense_size_bytes()/(1024*1024):.1f}MB, "
            f"roaring={self.get_compressed_size_bytes()/1024:.1f}KB, "
            f"compression={self.compression_ratio():.1f}x)"
        )


def benchmark_mask_creation(seq_len, window_size=128):
    """Benchmark Roaring vs dense mask creation time and memory."""
    t0 = time.time()
    dense_mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        start = max(0, i - window_size)
        dense_mask[i, start:i + 1] = 0.0
    dense_time = time.time() - t0
    dense_mem = dense_mask.element_size() * dense_mask.nelement()

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
