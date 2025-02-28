from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


def compute_cache_sizes(
    n_layers: int, max_seq_len: int, sliding_window: Optional[int] | Optional[List[int]]
) -> List[int]:
    """Compute cache sizes based on sliding window or max sequence length."""
    if sliding_window is None:
        return [max_seq_len] * n_layers
    elif isinstance(sliding_window, int):
        return [sliding_window] * n_layers
    elif isinstance(sliding_window, list):
        if n_layers % len(sliding_window) != 0:
            raise ValueError(f"n_layers must be divisible by len(sliding_window), but got {n_layers} % {len(sliding_window)}")
        num_repeats = n_layers // len(sliding_window)
        return num_repeats * [w if w is not None else max_seq_len for w in sliding_window]
    else:
        raise TypeError(f"Invalid type for sliding_window: {type(sliding_window)}")


@dataclass
class CacheInputMetadata:
    """
    Holds metadata about cache inputs such as positions, cache masks, and sequence lengths.
    """
    positions: torch.Tensor
    to_cache_mask: torch.Tensor
    cached_elements: torch.Tensor
    cache_positions: torch.Tensor
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_tensor_lists(l1: List[torch.Tensor], l2: List[torch.Tensor]) -> List[torch.Tensor]:
    """Interleave two lists of tensors."""
    if len(l1) != len(l2):
        raise ValueError(f"Lists must have the same length, but got {len(l1)} != {len(l2)}")
    return [tensor for pair in zip(l1, l2) for tensor in pair]


def rotate_cache(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    """
    Rotate cache tensor along the sequence dimension based on sequence length.
    """
    assert cache.ndim == 3, f"Expected 3D tensor, got {cache.ndim}-dimensional tensor"
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    """
    Provides a view into the cache for a particular layer, including the ability to update and interleave keys and values.
    """
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: CacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """Update the cache with new keys and values."""
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave cached keys and values with new input keys and values.
        Returns interleaved tensors.
        """
        assert xk.ndim == xv.ndim == 3, "xk and xv must have 3 dimensions"
        assert xk.shape == xv.shape, "xk and xv must have the same shape"

        if all(s == 0 for s in self.metadata.seqlens):
            # No cache to interleave
            return xk, xv

        xk_split = torch.split(xk, self.metadata.seqlens)
        xv_split = torch.split(xv, self.metadata.seqlens)

        # Rotate cache for each sequence length
        rotated_cache_k = [rotate_cache(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        rotated_cache_v = [rotate_cache(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        # Interleave cache and new inputs
        interleaved_k = interleave_tensor_lists(rotated_cache_k, list(xk_split))
        interleaved_v = interleave_tensor_lists(rotated_cache_v, list(xv_split))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        """Returns the maximum sequence length for this cache."""
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        """Returns the cached keys."""
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        """Returns the cached values."""
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        """Returns whether the cache is in prefill mode."""
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        """Returns the attention mask."""
        return self.metadata.mask


class BufferCache:
    """
    A buffer cache for handling variable length sequences with a rectangular allocation.
    """
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] | Optional[List[int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        self.cache_sizes = compute_cache_sizes(n_layers, max_seq_len, sliding_window)
        assert len(self.cache_sizes) == n_layers, f"Expected {n_layers} cache sizes, but got {len(self.cache_sizes)}"

        self.cache_k = {}
        self.cache_v = {}
        for i, cache_size in enumerate(self.cache_sizes):
            self.cache_k[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))
            self.cache_v[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))

        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        """Retrieve the cache view for a given layer."""
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self) -> None:
        """Reset the cache."""
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        """Initialize the sequence lengths for the cache."""
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        """Returns the device the cache is allocated on."""
        return self.cache_k[0].device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        """Move the cache to a different device and dtype."""
        for i in range(self.n_layers):
            self.cache_k[i] = self.cache_k[i].to(device=device, dtype=dtype)
            self.cache_v[i] = self.cache_v[i].to(device=device, dtype=dtype)
        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        """Update the sequence lengths for the cache."""
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> List[CacheInputMetadata]:
        """Generate the cache input metadata based on sequence lengths."""
        metadata = []
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        for cache_size in self.cache_sizes:
            metadata.append(self._get_input_metadata_layer(cache_size, seqlens))
        return metadata

    def _get_input_metadata_layer(self, cache_size: int, seqlens: List[int]) -> CacheInputMetadata:
        """Generate metadata for a single layer."""
        masks = [[x >= seqlen - cache_size for x in range(seqlen)] for seqlen in seqlens]
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        cached_elements = torch.tensor([i - cache_size for i in range(seqlen) if i >= cache_size], device=self.device, dtype=torch.long)
        positions = torch.tensor([i for i in range(seqlen)], device=self.device, dtype=torch.long)
        cache_positions = torch.tensor([i - cache_size for i in range(seqlen) if i >= cache_size], device=self.device, dtype=torch.long)
        prefill = [seqlen < cache_size for seqlen in seqlens]
        mask = BlockDiagonalMask(seqlens)
        return CacheInputMetadata(positions, to_cache_mask, cached_elements, cache_positions, prefill, mask, seqlens)
