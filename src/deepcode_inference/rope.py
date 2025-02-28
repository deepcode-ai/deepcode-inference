from typing import Tuple

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    Precomputes the frequencies for rotary embeddings.

    Args:
        dim (int): The dimension of the embedding.
        end (int): The maximum sequence length.
        theta (float): A scaling factor for the frequencies.

    Returns:
        torch.Tensor: A tensor of complex frequencies of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings to the query and key tensors.

    Args:
        xq (torch.Tensor): The query tensor of shape (batch_size, seq_len, dim).
        xk (torch.Tensor): The key tensor of shape (batch_size, seq_len, dim).
        freqs_cis (torch.Tensor): The precomputed frequency tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The transformed query and key tensors with rotary embeddings applied.
    """
    # Reshape xq and xk to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Expand freqs_cis to match query/key dimensions
    freqs_cis = freqs_cis[:, None, :]
    
    # Apply rotary embedding to query and key
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    Precomputes 2D frequencies for rotary embeddings for attention mechanisms that use both height and width.

    Args:
        dim (int): The dimension of the embedding.
        height (int): The height of the attention map.
        width (int): The width of the attention map.
        theta (float): A scaling factor for the frequencies.

    Returns:
        torch.Tensor: A 2D complex tensor of shape (height, width, dim // 2) containing the precomputed frequencies.
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    # Precompute frequencies along height and width axes
    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    
    # Combine frequency bases to form the 2D frequency tensor
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)
