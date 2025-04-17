"""
Causal Modal Fusion Module
========================

This module provides methods for fusing features from different modalities
using causal structures to guide the fusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from causaltorch.layers import CausalLinear
from causaltorch.layers.sparse import LotteryTicketRouter
from causaltorch.rules import CausalRuleSet


class CausalModalFusion(nn.Module):
    """Fusion module that combines modalities using causal constraints.
    
    This module combines features from different modalities (text, image, etc.)
    while respecting causal constraints between them. It supports multiple
    fusion methods including attention-based, sparse, and linear fusion.
    
    Args:
        modality_dims (dict): Dictionary mapping modality names to dimensions
        hidden_dim (int): Dimension of fused representation
        causal_rules (CausalRuleSet): Rules specifying causal relationships
        fusion_method (str): Method for fusion ('attention', 'sparse', or 'linear')
    """
    def __init__(
        self,
        modality_dims,
        hidden_dim=768,
        causal_rules=None,
        fusion_method="attention"
    ):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Create projections for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Create fusion mechanism
        if fusion_method == "attention":
            # Multi-head cross-modal attention
            self.num_heads = 8
            self.head_dim = hidden_dim // self.num_heads
            
            # Query, key, value projections
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Output projection
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Causal biasing
            if causal_rules:
                self.causal_bias = nn.Parameter(
                    torch.zeros(1, len(modality_dims), len(modality_dims))
                )
                
                # Initialize causal bias based on rules
                if hasattr(causal_rules, 'to_adjacency_matrix'):
                    # Extract the direct causal connections
                    adj_matrix, variables = causal_rules.to_adjacency_matrix()
                    
                    # Map variable names to modalities
                    modality_map = {mod: i for i, mod in enumerate(modality_dims.keys())}
                    
                    # Set the bias based on causal connections
                    for rule in causal_rules.rules:
                        cause = rule.cause
                        effect = rule.effect
                        strength = rule.strength
                        
                        if cause in modality_map and effect in modality_map:
                            cause_idx = modality_map[cause]
                            effect_idx = modality_map[effect]
                            # Set bias based on strength
                            self.causal_bias[0, effect_idx, cause_idx] = strength
        
        elif fusion_method == "sparse":
            # Use the LotteryTicketRouter for sparse fusion
            total_dim = sum(modality_dims.values())
            self.router = LotteryTicketRouter(
                total_dim, 
                hidden_dim,
                sparsity=0.8  # 80% sparsity
            )
            
        elif fusion_method == "linear":
            # Simple linear fusion
            total_dim = sum(modality_dims.values())
            
            if causal_rules and hasattr(causal_rules, 'to_adjacency_matrix'):
                # Create adjacency matrix for causal constraints
                adj_matrix, variables = causal_rules.to_adjacency_matrix()
                
                # Convert to torch tensor
                adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
                
                # Resize/pad the adjacency matrix to match dimensions
                num_variables = len(variables)
                
                # CausalLinear will transpose this mask, so proper sizing is important
                padded_adj = torch.zeros((total_dim, hidden_dim), dtype=torch.float)
                
                # Place adjacency matrix in the top-left corner
                if num_variables > 0:
                    # Ensure we don't go out of bounds
                    n_rows = min(num_variables, total_dim)
                    n_cols = min(num_variables, hidden_dim)
                    
                    # Copy the adjacency matrix with correct orientation
                    padded_adj[:n_rows, :n_cols] = adj_tensor.T[:n_rows, :n_cols]
                
                # Use causal linear layer with properly sized mask
                self.fusion_layer = CausalLinear(total_dim, hidden_dim, padded_adj)
            else:
                self.fusion_layer = nn.Linear(total_dim, hidden_dim)
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, modality_features):
        """Fuse features from different modalities.
        
        Args:
            modality_features (dict): Dictionary mapping modality names to feature tensors
                Each tensor should be [batch_size, modality_dim]
            
        Returns:
            torch.Tensor: Fused multimodal representation [batch_size, hidden_dim]
        """
        batch_size = next(iter(modality_features.values())).size(0)
        
        # Project each modality to common dimension
        projected = {}
        for modality, features in modality_features.items():
            if modality in self.projections:
                projected[modality] = self.projections[modality](features)
            else:
                raise ValueError(f"Unknown modality: {modality}")
        
        # Apply fusion based on selected method
        if self.fusion_method == "attention":
            # Stack modalities
            modality_list = list(projected.keys())
            stacked = torch.stack([projected[m] for m in modality_list], dim=1)  # [batch, num_mod, dim]
            
            # Apply attention-based fusion
            q = self.q_proj(stacked).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(stacked).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(stacked).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Add causal bias if available
            if hasattr(self, 'causal_bias'):
                # Expand bias to batch size and heads
                bias = self.causal_bias.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)
                scores = scores + bias
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
            
            # Apply output projection
            fused = self.out_proj(context)
            
            # Average across modalities
            fused = torch.mean(fused, dim=1)
            
        elif self.fusion_method == "sparse":
            # Concatenate features
            features_list = [projected[m] for m in self.modality_dims.keys()]
            concat_features = torch.cat(features_list, dim=1)
            
            # Apply sparse routing
            fused = self.router(concat_features)
            
        elif self.fusion_method == "linear":
            # Concatenate features
            features_list = [projected[m] for m in self.modality_dims.keys()]
            concat_features = torch.cat(features_list, dim=1)
            
            # Apply linear fusion
            fused = self.fusion_layer(concat_features)
        
        # Apply normalization
        fused = self.layer_norm(fused)
        
        return fused 