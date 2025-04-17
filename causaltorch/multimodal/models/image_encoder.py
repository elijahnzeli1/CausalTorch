"""
Image Encoder with Causal Constraints
====================================

This module provides an image encoder that extracts causal features from images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from causaltorch.layers import CausalLinear
from causaltorch.rules import CausalRuleSet


class CausalImageEncoder(nn.Module):
    """An image encoder that extracts causally relevant features.
    
    This leverages pre-trained vision models and causal layers
    to identify causal elements in images.
    
    Args:
        model_name (str): Name of pre-trained vision model to use
        causal_rules (CausalRuleSet): Set of causal rules to enforce
        hidden_dim (int): Dimension of output features
        freeze_base (bool): Whether to freeze base model parameters
    """
    def __init__(
        self,
        model_name="resnet18",
        causal_rules=None,
        hidden_dim=768,
        freeze_base=True
    ):
        super().__init__()
        
        # Load pre-trained vision model
        if model_name == "resnet18":
            # Fix deprecated parameters
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove classification head
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            base_dim = 512
        elif model_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            base_dim = 2048
        elif model_name == "vit_b_16":
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.base_model = base_model
            base_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Create causal projection
        if causal_rules and hasattr(causal_rules, 'to_adjacency_matrix'):
            # Extract adjacency matrix from causal rules
            adj_matrix, variables = causal_rules.to_adjacency_matrix()
            
            # Convert to torch tensor
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
            
            # Resize/pad the adjacency matrix to match dimensions
            num_variables = len(variables)
            
            # CausalLinear will transpose this mask, so proper sizing is important
            padded_adj = torch.zeros((base_dim, hidden_dim), dtype=torch.float)
            
            # Place adjacency matrix in the top-left corner
            if num_variables > 0:
                # Ensure we don't go out of bounds
                n_rows = min(num_variables, base_dim)
                n_cols = min(num_variables, hidden_dim)
                
                # Copy the adjacency matrix with correct orientation
                padded_adj[:n_rows, :n_cols] = adj_tensor.T[:n_rows, :n_cols]
            
            # Use causal linear layer with properly sized mask
            self.projection = CausalLinear(
                base_dim, hidden_dim, padded_adj
            )
        else:
            # Use regular linear layer if no causal rules
            self.projection = nn.Linear(base_dim, hidden_dim)
        
        # Add normalization and activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
        self.hidden_dim = hidden_dim
    
    def forward(self, images):
        """Extract causal features from images.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Encoded causal features [batch_size, hidden_dim]
        """
        # Encode with base model
        base_features = self.base_model(images)
        
        # Flatten if needed (for ResNet)
        if len(base_features.shape) > 2:
            base_features = base_features.view(base_features.size(0), -1)
        
        # Apply causal projection
        causal_features = self.projection(base_features)
        
        # Apply normalization and activation
        outputs = self.activation(self.norm(causal_features))
        
        return outputs
    
    def extract_causal_concepts(self, images, concept_labels=None):
        """Extract specific causal concepts from images.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            concept_labels (List[str], optional): Names of concepts to extract
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping concept names to extracted values
        """
        # Get base features
        features = self.forward(images)
        
        # If no concept labels provided, return raw features
        if concept_labels is None:
            return {"features": features}
        
        # Initialize output dictionary
        concepts = {}
        
        # Extract each concept (simplified implementation)
        # In a real implementation, we would have concept extractors for each label
        feature_dim = features.size(1)
        concept_dim = feature_dim // max(1, len(concept_labels))
        
        for i, label in enumerate(concept_labels):
            # Extract slice of features for this concept
            start_idx = i * concept_dim
            end_idx = min((i + 1) * concept_dim, feature_dim)
            
            # Apply concept-specific processing
            concept_value = torch.mean(features[:, start_idx:end_idx], dim=1)
            concepts[label] = concept_value
        
        return concepts
