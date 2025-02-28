from dataclasses import dataclass
from typing import List, Optional, Union
from simple_parsing.helpers import Serializable

from deepcode_inference.lora import LoraArgs
from deepcode_inference.moe import MoeArgs


@dataclass(frozen=True)
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float = 1e4  # for rope-2D
    image_token_id: int = 10


@dataclass(frozen=True)
class TransformerArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    max_batch_size: int = 0

    rope_theta: Optional[float] = None
    moe: Optional[MoeArgs] = None
    lora: Optional[LoraArgs] = None

    sliding_window: Optional[Union[int, List[int]]] = None
    _sliding_window: Optional[Union[int, List[int]]] = None

    model_type: str = "transformer"
    vision_encoder: Optional[VisionEncoderArgs] = None

    def __post_init__(self) -> None:
        assert self.model_type == "transformer", f"Invalid model type: {self.model_type}"
        assert not (self.sliding_window and self._sliding_window), "Only one of sliding_window or _sliding_window should be set"

        if self.sliding_window is None:
            object.__setattr__(self, "sliding_window", self._sliding_window)


@dataclass(frozen=True)
class MambaArgs(Serializable):
    dim: int
    n_layers: int
    vocab_size: int
    n_groups: int
    rms_norm: bool
    residual_in_fp32: bool
    fused_add_norm: bool
    pad_vocab_size_multiple: int
    tie_embeddings: bool
    model_type: str = "mamba"

    def __post_init__(self) -> None:
        assert self.model_type == "mamba", f"Invalid model type: {self.model_type}"
