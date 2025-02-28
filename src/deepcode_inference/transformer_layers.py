from functools import partial
from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from deepcode_inference.args import LoraArgs
from deepcode_inference.cache import CacheView
from deepcode_inference.lora import LoRALinear
from deepcode_inference.moe import MoeArgs, MoeLayer
from deepcode_inference.rope import apply_rotary_emb


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Repeats the key and value tensors along the specified dimension.

    Args:
        keys: The key tensor.
        values: The value tensor.
        repeats: Number of times to repeat the tensors.
        dim: The dimension along which to repeat the tensors.

    Returns:
        Repeated key and value tensors.
    """
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def maybe_lora(lora_args: Optional[LoraArgs]) -> Union[Type[nn.Linear], partial[LoRALinear]]:
    """
    Returns a Linear or LoRALinear module based on whether LoraArgs are provided.

    Args:
        lora_args: Optional LoraArgs configuration.

    Returns:
        A Linear module or a partially applied LoRALinear with the provided arguments.
    """
    if lora_args is None:
        return nn.Linear
    else:
        return partial(LoRALinear, rank=lora_args.rank, scaling=lora_args.scaling)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        lora: Optional[LoraArgs] = None,
    ):
        """
        Initializes the Attention module with the given dimensions and hyperparameters.

        Args:
            dim: Input dimension of the model.
            n_heads: Number of attention heads.
            head_dim: The dimension of each attention head.
            n_kv_heads: Number of key-value heads.
            lora: Optional LoraArgs configuration for low-rank adaptation.
        """
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5

        MaybeLora = maybe_lora(lora)
        self.wq = MaybeLora(dim, n_heads * head_dim, bias=False)
        self.wk = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wv = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wo = MaybeLora(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        """
        Performs the attention mechanism on the input tensor.

        Args:
            x: The input tensor of shape (sequence length, batch size, dim).
            freqs_cis: Frequency embeddings for rotary position encoding.
            cache: Optional cache view for key-value pairs.
            mask: Optional block diagonal mask for attention.

        Returns:
            The output tensor after applying attention.
        """
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        # Compute queries, keys, and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)

        # Apply rotary position encoding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Cache handling: store/retrieve key-value pairs
        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)

        # Repeat keys and values for multiple attention heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # Perform memory-efficient attention using xformers
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask if cache is None else cache.mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, lora: Optional[LoraArgs] = None):
        """
        Initializes the FeedForward network.

        Args:
            dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            lora: Optional LoraArgs for low-rank adaptation.
        """
        super().__init__()

        MaybeLora = maybe_lora(lora)
        self.w1 = MaybeLora(dim, hidden_dim, bias=False)
        self.w2 = MaybeLora(hidden_dim, dim, bias=False)
        self.w3 = MaybeLora(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForward network.

        Args:
            x: Input tensor of shape (batch size, seq length, dim).

        Returns:
            Output tensor after applying the feed-forward network.
        """
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Implements RMS Normalization.

        Args:
            dim: The dimension of the input tensor.
            eps: A small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform RMS normalization on the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        lora: Optional[LoraArgs] = None,
        moe: Optional[MoeArgs] = None,
    ):
        """
        Initializes a Transformer Block.

        Args:
            dim: Input dimension of the model.
            hidden_dim: Dimension of the hidden layer in the feed-forward network.
            n_heads: Number of attention heads.
            n_kv_heads: Number of key-value heads.
            head_dim: Dimension of each attention head.
            norm_eps: Small value for normalization.
            lora: Optional LoraArgs configuration for low-rank adaptation.
            moe: Optional MoeArgs configuration for mixture-of-experts.
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            lora=lora,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        if moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora) for _ in range(moe.num_experts)],
                gate=nn.Linear(dim, moe.num_experts, bias=False),
                moe_args=moe,
            )
        else:
            self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Block.

        Args:
            x: Input tensor.
            freqs_cis: Rotary embeddings for position encoding.
            cache: Optional cache view for key-value pairs.
            mask: Optional attention mask.

        Returns:
            Output tensor after passing through attention and feed-forward layers.
        """
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out
