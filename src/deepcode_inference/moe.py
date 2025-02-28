import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        """
        Args:
            experts (List[nn.Module]): List of expert layers.
            gate (nn.Module): Gate module that determines how experts are selected.
            moe_args (MoeArgs): Arguments for controlling the number of experts.
        """
        super().__init__()
        assert len(experts) > 0, "There should be at least one expert."
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        gate_logits = self.gate(inputs)  # Shape: (batch_size, num_experts)
        
        # Select the top-k experts based on gate logits
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)  # Normalize weights
        
        results = torch.zeros_like(inputs)  # Initialize results tensor
        
        # For each expert, compute its contribution
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)  # Find indices of selected experts
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])  # Aggregate
        return results


class MoeModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int, num_experts_per_tok: int):
        """
        Args:
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality.
            num_experts (int): Number of experts.
            num_experts_per_tok (int): Number of experts to activate per token.
        """
        super().__init__()

        # Initialize experts (e.g., simple linear layers as experts)
        self.experts = [nn.Linear(input_dim, output_dim) for _ in range(num_experts)]
        
        # Initialize gate (e.g., a simple linear layer to select experts)
        self.gate = nn.Linear(input_dim, num_experts)

        # Arguments for MoE layer
        moe_args = MoeArgs(num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

        # Create the MoE layer
        self.moe_layer = MoeLayer(experts=self.experts, gate=self.gate, moe_args=moe_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe_layer(x)

    @staticmethod
    def from_args(args):
        """
        Creates the MoE model from arguments.

        Args:
            args (Namespace): Arguments containing model parameters.

        Returns:
            MoeModel: The created MoE model.
        """
        return MoeModel(input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        num_experts=args.num_experts,
                        num_experts_per_tok=args.num_experts_per_tok)
