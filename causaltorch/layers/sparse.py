"""
CausalTorch Sparse Layers Module
================================

This module contains implementations of sparse neural network layers
that dynamically activate only a small subset of parameters based on the task.
These layers enable more efficient computation while maintaining performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable


class LotteryTicketRouter(nn.Module):
    """Implements dynamic sparse activation based on the Lottery Ticket Hypothesis.
    
    This layer routes inputs through a sparse subnetwork of a larger base model,
    activating only the parameters relevant to the current task. This reduces
    computation while maintaining task performance.
    
    Args:
        base_model (nn.Module): The base model to sparsify
        sparsity (float): Target sparsity level (0.0-1.0), where 1.0 means
            activating only 0% of parameters (highest sparsity)
        task_embedding_dim (int): Dimension of task embedding vectors
    """
    def __init__(
        self,
        base_model: nn.Module,
        sparsity: float = 0.9,
        task_embedding_dim: int = 128
    ):
        super().__init__()
        self.base_model = base_model
        self.sparsity = min(0.99, max(0.0, sparsity))  # Clamp to [0, 0.99]
        
        # Register masks for each parameter
        self.masks = nn.ParameterDict()
        
        # Task embedding to mask generator
        self.mask_generator = nn.ModuleDict()
        
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                # Initialize with ones (all weights active)
                self.masks[name] = nn.Parameter(torch.ones_like(param), requires_grad=False)
                
                # Create mask generator for this parameter
                flat_dim = param.numel()
                hidden_dim = min(2048, max(128, flat_dim // 8))  # Adaptive hidden dim
                
                self.mask_generator[name] = nn.Sequential(
                    nn.Linear(task_embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, flat_dim),
                    nn.Sigmoid()
                )
    
    def update_masks(self, task_embedding: torch.Tensor) -> Dict[str, float]:
        """Update parameter masks based on task embedding.
        
        Args:
            task_embedding (torch.Tensor): Embedding vector for the current task
            
        Returns:
            Dict[str, float]: Dictionary mapping parameter names to sparsity levels
        """
        sparsity_levels = {}
        
        for name, param in self.base_model.named_parameters():
            if name in self.masks:
                # Generate relevance scores for each parameter
                flat_scores = self.mask_generator[name](task_embedding)
                
                # Reshape to parameter shape
                scores = flat_scores.view(*param.shape)
                
                # Determine threshold for keeping top (1-sparsity)% weights
                k = int((1 - self.sparsity) * scores.numel())
                if k < 1:
                    k = 1  # Always keep at least one parameter
                
                # Find the threshold value for top-k
                threshold = torch.topk(scores.view(-1), k).values[-1]
                
                # Create binary mask
                mask = (scores >= threshold).float()
                
                # Update the mask
                self.masks[name].data = mask
                
                # Calculate actual sparsity
                sparsity_level = 1.0 - (mask.sum().item() / mask.numel())
                sparsity_levels[name] = sparsity_level
        
        return sparsity_levels
    
    def forward(self, x: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sparse activation.
        
        Args:
            x (torch.Tensor): Input tensor
            task_embedding (torch.Tensor, optional): Task embedding vector
                If None, uses current masks without updating
                
        Returns:
            torch.Tensor: Output tensor
        """
        # Update masks if task embedding is provided
        if task_embedding is not None:
            self.update_masks(task_embedding)
        
        # Apply masks to base model parameters
        original_values = {}
        
        for name, param in self.base_model.named_parameters():
            if name in self.masks:
                # Store original values
                original_values[name] = param.data.clone()
                
                # Apply mask
                param.data = param.data * self.masks[name]
        
        # Forward pass through base model
        output = self.base_model(x)
        
        # Restore original parameter values
        for name, param in self.base_model.named_parameters():
            if name in original_values:
                param.data = original_values[name]
        
        return output


class SparseLinear(nn.Module):
    """Linear layer with adaptive sparsity based on task context.
    
    This layer implements a more efficient sparse linear transformation
    that dynamically determines which weights to use based on the input.
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool): Whether to include bias
        sparsity (float): Target sparsity level (0.0-1.0)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize weights and optional bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize mask with same shape as weights
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        # Score generator to determine importance of weights
        hidden_dim = max(64, min(512, (in_features + out_features) // 4))
        self.score_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features * out_features),
            nn.Sigmoid()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def update_mask(self, x: torch.Tensor):
        """Update weight mask based on input features.
        
        Args:
            x (torch.Tensor): Input tensor
        """
        # Compute average input features across batch
        avg_features = x.mean(dim=0)
        
        # Generate importance scores for each weight
        scores = self.score_net(avg_features).view(self.out_features, self.in_features)
        
        # Keep top (1-sparsity)% of weights
        k = int((1 - self.sparsity) * scores.numel())
        if k < 1:
            k = 1  # Always keep at least one parameter
        
        # Find threshold for top-k
        flat_scores = scores.view(-1)
        threshold = torch.topk(flat_scores, k).values[-1]
        
        # Create binary mask
        self.mask = (scores >= threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse weights.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Update mask based on current input
        if self.training:
            self.update_mask(x)
        
        # Apply mask to weights
        sparse_weight = self.weight * self.mask
        
        # Compute output
        return F.linear(x, sparse_weight, self.bias)


class CausalStateSpaceModel(nn.Module):
    """Causal State-Space Model (SSM) with O(n) complexity.
    
    This implements a linear time alternative to attention mechanisms
    by using state-space models structured according to causal graphs.
    
    Args:
        state_dim (int): Dimension of hidden state
        input_dim (int): Dimension of input features
        causal_graph (torch.Tensor, optional): Adjacency matrix for causal graph
    """
    def __init__(
        self,
        state_dim: int = 64,
        input_dim: int = 64,
        causal_graph: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Initialize state transition matrix (A)
        if causal_graph is not None:
            # Use causal graph as initialization for A
            A = F.normalize(causal_graph.float(), p=1, dim=1)
            self.A = nn.Parameter(A)
        else:
            # Initialize with identity matrix + small noise
            self.A = nn.Parameter(torch.eye(state_dim) + 0.01 * torch.randn(state_dim, state_dim))
        
        # Input-to-state matrix (B)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        
        # State-to-output matrix (C)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.01)
        
        # Direct input-to-output matrix (D) - optional skip connection
        self.D = nn.Parameter(torch.eye(input_dim) * 0.01)
    
    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Causal state-space forward pass.
        
        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len, input_dim]
            initial_state (torch.Tensor, optional): Initial state
                
        Returns:
            torch.Tensor: Output sequence [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(batch_size, self.state_dim, device=x.device)
        else:
            state = initial_state
        
        # Output container
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            # Get input at this time step
            x_t = x[:, t, :]
            
            # Update state: s_t = A * s_{t-1} + B * x_t
            state = torch.matmul(state, self.A.t()) + torch.matmul(x_t, self.B.t())
            
            # Compute output: y_t = C * s_t + D * x_t
            output = torch.matmul(state, self.C.t()) + torch.matmul(x_t, self.D)
            
            outputs.append(output)
        
        # Stack outputs
        return torch.stack(outputs, dim=1) 