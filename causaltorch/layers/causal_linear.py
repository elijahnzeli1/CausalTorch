"""
Linear layer with causal constraints for CausalTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLinear(nn.Module):
    """Linear layer with causal constraints enforced via masking.
    
    This layer ensures that information only flows along the causal graph
    by masking connections that violate the causal structure.
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        adjacency_mask (torch.Tensor): Binary adjacency matrix of the causal graph
            where a 1 at position [i, j] means variable i causes variable j
    """
    def __init__(self, in_features, out_features, adjacency_mask):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('mask', adjacency_mask.T.float())
        
        # Initialize with masked weights
        with torch.no_grad():
            self.weight *= self.mask
    
    def forward(self, x):
        """Forward pass with causal constraints.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with causal constraints applied
        """
        # Enforce mask during forward pass
        return F.linear(x, self.weight * self.mask, self.bias)