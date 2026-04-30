"""
Roaring Bitmap-based gradient compression for distributed training.

In distributed training (DDP/FSDP), the All-Reduce step sends gradients
between GPUs. Top-K sparsification reduces communication by only sending
the most important gradients. Roaring Bitmaps compress the index set of
those top-K gradients far better than raw integer arrays — especially
when gradients are clustered in specific layers or heads.
"""

import torch
import numpy as np
from pyroaring import BitMap
import time


class RoaringGradientCompressor:
    """
    Compresses gradient tensors using Top-K sparsification with
    Roaring Bitmap index encoding.

    Standard approach: send (indices_array, values_array)
    Roaring approach: send (roaring_serialized_indices, values_array)

    The Roaring serialization is typically 2-5x smaller than a raw
    uint32 index array when gradients are clustered.
    """

    def __init__(self, sparsity=0.01):
        """
        Args:
            sparsity: Fraction of gradients to keep (e.g., 0.01 = top 1%).
        """
        self.sparsity = sparsity
        self.residuals = {}  # Error feedback for gradient accumulation

    def compress(self, gradient_tensor, layer_name=None):
        """Compress a gradient tensor using Top-K + Roaring Bitmaps.

        Args:
            gradient_tensor: The gradient to compress.
            layer_name: Optional name for error feedback tracking.

        Returns:
            compressed_indices: Serialized Roaring Bitmap bytes
            values: The Top-K gradient values (dense tensor)
            original_shape: Shape for reconstruction
        """
        original_shape = gradient_tensor.shape
        flat_grad = gradient_tensor.detach().view(-1).float()

        # Error feedback: add residual from previous step
        if layer_name and layer_name in self.residuals:
            flat_grad = flat_grad + self.residuals[layer_name]

        k = max(1, int(self.sparsity * flat_grad.numel()))
        topk_vals, topk_indices = torch.topk(torch.abs(flat_grad), k)

        # Store residual (error feedback)
        if layer_name:
            residual = flat_grad.clone()
            residual[topk_indices] = 0.0
            self.residuals[layer_name] = residual

        # Pack indices into Roaring Bitmap
        indices_np = topk_indices.cpu().numpy().astype(np.uint32)
        rb = BitMap(indices_np)
        compressed_indices = rb.serialize()

        values = flat_grad[topk_indices]

        return compressed_indices, values, original_shape

    def decompress(self, compressed_indices, values, original_shape):
        """Decompress a Roaring-compressed gradient back to a full tensor.

        Args:
            compressed_indices: Serialized Roaring Bitmap bytes
            values: The sparse gradient values
            original_shape: Original tensor shape

        Returns:
            Reconstructed gradient tensor
        """
        rb = BitMap.deserialize(compressed_indices)
        indices = torch.from_numpy(np.array(list(rb), dtype=np.int64))

        grad = torch.zeros(int(np.prod(original_shape)), dtype=values.dtype)
        grad[indices] = values.cpu()
        return grad.view(original_shape)

    def get_compression_stats(self, gradient_tensor):
        """Compare Roaring vs raw array compression for a gradient."""
        flat_grad = gradient_tensor.detach().view(-1).float()
        k = max(1, int(self.sparsity * flat_grad.numel()))
        topk_vals, topk_indices = torch.topk(torch.abs(flat_grad), k)

        indices_np = topk_indices.cpu().numpy().astype(np.uint32)

        # Raw array size
        raw_index_bytes = indices_np.nbytes

        # Roaring size
        rb = BitMap(indices_np)
        roaring_bytes = len(rb.serialize())

        # Values size (same for both)
        values_bytes = k * 4  # float32

        return {
            'num_params': flat_grad.numel(),
            'top_k': k,
            'raw_index_bytes': raw_index_bytes,
            'roaring_index_bytes': roaring_bytes,
            'values_bytes': values_bytes,
            'raw_total_bytes': raw_index_bytes + values_bytes,
            'roaring_total_bytes': roaring_bytes + values_bytes,
            'index_compression_ratio': raw_index_bytes / max(1, roaring_bytes),
            'total_compression_ratio': (raw_index_bytes + values_bytes) / max(1, roaring_bytes + values_bytes),
        }


def benchmark_gradient_compression(param_count=1_000_000, sparsity=0.01):
    """Benchmark Roaring vs raw gradient compression."""
    grad = torch.randn(param_count)
    compressor = RoaringGradientCompressor(sparsity=sparsity)

    # Roaring compression
    t0 = time.time()
    compressed, values, shape = compressor.compress(grad)
    roaring_compress_time = time.time() - t0

    t0 = time.time()
    restored = compressor.decompress(compressed, values, shape)
    roaring_decompress_time = time.time() - t0

    stats = compressor.get_compression_stats(grad)
    stats['roaring_compress_time_s'] = roaring_compress_time
    stats['roaring_decompress_time_s'] = roaring_decompress_time

    return stats
