from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

from deepcode_inference.cache import BufferCache


class ModelBase(nn.Module, ABC):
    """Abstract base class for all model types. Defines common functionality and abstract methods 
    that need to be implemented by all model subclasses.
    """
    
    def __init__(self) -> None:
        """Initializes the model base class by invoking the parent class constructor."""
        super().__init__()

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Return the model's data type (e.g., torch.float32, torch.float16)."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device (e.g., 'cuda' or 'cpu') on which the model is located."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: Optional[List[int]] = None,  # Optional as it's not supported for now
        cache: Optional[BufferCache] = None,  # Optional as it's not supported for now
    ) -> torch.Tensor:
        """Performs a forward pass through the model.
        
        Args:
            input_ids: Tensor containing the input IDs for the model.
            seqlens: Sequence lengths for each input in the batch (Optional).
            cache: An optional buffer cache for storing intermediate results (Optional).
        
        Returns:
            A tensor containing the model's output.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "ModelBase":
        """Loads a model from a folder containing its saved weights and configuration.
        
        Args:
            folder: Path to the folder containing the model's weights and configuration.
            max_batch_size: Maximum batch size supported by the model.
            num_pipeline_ranks: Number of pipeline ranks for distributed processing.
            device: Device on which to load the model ('cuda' or 'cpu').
            dtype: Data type (optional, defaults to None).
        
        Returns:
            An instance of the model class.
        """
        pass


class MyModel(ModelBase):
    """A concrete implementation of the ModelBase class."""
    
    def __init__(self, model_path: str, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")) -> None:
        """Initialize the model with a path to its saved weights and configurations."""
        super().__init__()
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self._load_model()

    def _load_model(self):
        """Private method to load the model from the specified path."""
        # Example: Loading model logic can go here
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model folder at {self.model_path} does not exist.")
        # Load model logic (this is a placeholder)
        # self.model = torch.load(self.model_path, map_location=self.device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, input_ids: torch.Tensor, seqlens: Optional[List[int]] = None, cache: Optional[BufferCache] = None) -> torch.Tensor:
        """Forward pass implementation for this specific model."""
        # Placeholder for forward pass logic
        return input_ids

    @staticmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "MyModel":
        """Loads a MyModel instance from a folder containing the model weights."""
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Model folder {folder} not found.")
        # Additional checks and logic for loading model can be placed here
        
        # Creating and returning an instance of the model
        model = MyModel(str(folder_path), dtype=dtype or torch.float32, device=device)
        return model
