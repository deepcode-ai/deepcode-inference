from typing import List, Optional
import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from deepcode_inference.args import VisionEncoderArgs
from deepcode_inference.rope import precompute_freqs_cis_2d
from deepcode_inference.transformer_layers import RMSNorm, TransformerBlock


def generate_position_meshgrid(
    patch_embeddings: List[torch.Tensor],
) -> torch.Tensor:
    """
    Generates a meshgrid for position encoding based on patch embeddings.

    Args:
        patch_embeddings: List of tensor patch embeddings from input images.

    Returns:
        A tensor representing the position coordinates for each patch.
    """
    positions = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(embedding.shape[-2]),
                    torch.arange(embedding.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for embedding in patch_embeddings
        ]
    )
    return positions


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionEncoderArgs):
        """
        Initializes the Vision Transformer model with provided configuration.

        Args:
            config: The configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        
        # Patch convolution layer
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        
        # Pre-normalization layer
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        
        # Transformer blocks
        self.transformer_blocks = VisionTransformerBlocks(config)

        # Frequency encoding initialization
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        """Returns the maximum patches per side based on the image size and patch size."""
        return self.config.image_size // self.config.patch_size

    @property
    def device(self) -> torch.device:
        """Returns the device of the model's parameters."""
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        """
        Computes and caches the frequency terms for rotary position embeddings (ROPE).

        Returns:
            A tensor containing frequency terms.
        """
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.config.hidden_size // self.config.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.config.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer model.

        Args:
            images: A list of N images of shape (C, H, W) to process.

        Returns:
            A tensor of image features for all tokens across images.
        """
        # Extract patch embeddings for each image
        patch_embeddings = [self.patch_conv(img.unsqueeze(0)).squeeze(0) for img in images]

        # Flatten patch embeddings and apply pre-normalization
        patch_embeddings = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeddings], dim=0)
        patch_embeddings = self.ln_pre(patch_embeddings)

        # Generate positional embeddings and compute frequency terms
        positions = generate_position_meshgrid(patch_embeddings).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # Create block diagonal mask for attention mechanism
        mask = BlockDiagonalMask.from_seqlens(
            [p.shape[-2] * p.shape[-1] for p in patch_embeddings]
        )
        
        # Pass through the transformer layers
        transformer_output = self.transformer_blocks(patch_embeddings, mask=mask, freqs_cis=freqs_cis)

        return transformer_output


class VisionLanguageAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        A simple vision-language adapter layer for transforming features.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, output_dim, bias=True)
        self.activation = nn.GELU()
        self.output_layer = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision-language adapter."""
        return self.output_layer(self.activation(self.input_layer(x)))


class VisionTransformerBlocks(nn.Module):
    def __init__(self, config: VisionEncoderArgs):
        """
        Stacks multiple Transformer blocks as part of the Vision Transformer model.

        Args:
            config: Configuration object containing hyperparameters for transformer layers.
        """
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_size,
                    hidden_dim=config.intermediate_size,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_attention_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                    norm_eps=1e-5,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: BlockDiagonalMask, freqs_cis: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through all Transformer blocks."""
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs_cis)
        return x
