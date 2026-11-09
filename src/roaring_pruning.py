"""
Roaring Bitmap-based weight pruning.

After a warm-up phase, identifies the smallest-magnitude weights and stores
the indices of the surviving weights as Roaring Bitmaps. apply_masks() zeroes
the pruned weights in place; the forward pass is still a dense matmul, so
this tracks sparsity but does not reduce FLOPs by itself.
"""

import torch
import numpy as np
from pyroaring import BitMap


class RoaringPruningManager:
    """
    Manages weight pruning masks using Roaring Bitmaps.

    For each parameter, stores a Roaring Bitmap containing the flat indices
    of "alive" (non-pruned) weights. This is memory-efficient when the
    pruning ratio is high (e.g., 90% pruned = only 10% of indices stored).
    """

    def __init__(self, prune_ratio=0.5, warmup_steps=100):
        """
        Args:
            prune_ratio: Fraction of weights to prune (by magnitude).
            warmup_steps: Number of training steps before pruning kicks in.
        """
        self.prune_ratio = prune_ratio
        self.warmup_steps = warmup_steps
        self.masks = {}  # name -> BitMap of alive indices
        self.is_pruned = False
        self.step_count = 0

    def step(self, model):
        """Call after each optimizer step to potentially trigger pruning."""
        self.step_count += 1
        if self.step_count == self.warmup_steps and not self.is_pruned:
            self.compute_masks(model)

    def compute_masks(self, model):
        """Compute pruning masks for all parameters based on weight magnitude."""
        self.masks = {}
        for name, param in model.named_parameters():
            if param.dim() < 2:  # Skip biases and layer norms
                continue

            flat = param.data.view(-1).abs()
            k = max(1, int((1 - self.prune_ratio) * flat.numel()))
            _, alive_indices = torch.topk(flat, k)

            indices_np = alive_indices.cpu().numpy().astype(np.uint32)
            self.masks[name] = BitMap(indices_np)

        self.is_pruned = True

    def apply_masks(self, model):
        """Apply pruning masks to zero out pruned weights."""
        if not self.is_pruned:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.masks:
                    continue
                flat = param.data.view(-1)
                alive = self.masks[name]
                # Create a zero tensor, copy only alive values
                pruned = torch.zeros_like(flat)
                alive_idx = torch.from_numpy(
                    np.array(list(alive), dtype=np.int64)
                ).to(param.device)
                pruned[alive_idx] = flat[alive_idx]
                param.data = pruned.view(param.shape)

    def get_stats(self):
        """Return pruning statistics."""
        if not self.masks:
            return {'is_pruned': False}

        total_params = 0
        alive_params = 0
        roaring_bytes = 0

        for name, bm in self.masks.items():
            alive = len(bm)
            # Estimate total from prune_ratio
            total = int(alive / (1 - self.prune_ratio))
            total_params += total
            alive_params += alive
            roaring_bytes += len(bm.serialize())

        raw_mask_bytes = total_params  # 1 byte per bool in a naive mask
        return {
            'is_pruned': True,
            'total_params': total_params,
            'alive_params': alive_params,
            'pruned_ratio': 1 - (alive_params / max(1, total_params)),
            'roaring_mask_bytes': roaring_bytes,
            'naive_mask_bytes': raw_mask_bytes,
            'mask_compression_ratio': raw_mask_bytes / max(1, roaring_bytes),
        }
