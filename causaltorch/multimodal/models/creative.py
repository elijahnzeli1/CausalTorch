"""
Creative Components for Multimodal Causal Models
=============================================

This module provides components for creative generation and counterfactual reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualDreamer(nn.Module):
    """Module for generating counterfactual representations through causal interventions.
    
    This module takes a representation and applies causal interventions to generate
    counterfactual versions that respect the causal structure.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of transformation layers
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create transformation layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Input projection if dimensions differ
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Variable-specific intervention projections
        self.intervention_projections = nn.ModuleDict({
            'text_weather': nn.Linear(1, hidden_dim),
            'text_objects': nn.Linear(1, hidden_dim),
            'text_style': nn.Linear(1, hidden_dim),
            'img_weather': nn.Linear(1, hidden_dim),
            'img_objects': nn.Linear(1, hidden_dim),
            'img_style': nn.Linear(1, hidden_dim)
        })
        
    def forward(self, x):
        """Forward pass without intervention."""
        x = self.input_proj(x)
        
        # Apply transformation layers
        for layer in self.layers:
            # Residual connection
            residual = x
            x = layer(x)
            x = x + residual
            
        return x
    
    def intervene(self, x, interventions):
        """Apply causal interventions to generate counterfactual representations.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            interventions (dict): Dictionary mapping variable names to intervention values
            
        Returns:
            torch.Tensor: Counterfactual features [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        x = self.input_proj(x)
        
        # Apply interventions
        for var_name, value in interventions.items():
            if var_name in self.intervention_projections:
                # Create intervention tensor
                intervention = torch.tensor([[value]], device=x.device).expand(batch_size, 1)
                
                # Project intervention
                intervention_effect = self.intervention_projections[var_name](intervention)
                
                # Apply intervention through additive update
                x = x + intervention_effect
        
        # Apply transformation layers
        for layer in self.layers:
            # Residual connection
            residual = x
            x = layer(x)
            x = x + residual
            
        return x
    
    def imagine_multiple(self, x, intervention_list):
        """Generate multiple counterfactual samples with different interventions.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            intervention_list (list): List of intervention dictionaries
            
        Returns:
            list: List of counterfactual samples
        """
        return [self.intervene(x, intervention) for intervention in intervention_list]
    
    def add_intervention_variable(self, var_name):
        """Add a new intervention variable to the model.
        
        Args:
            var_name (str): Name of the new variable
        """
        if var_name not in self.intervention_projections:
            self.intervention_projections[var_name] = nn.Linear(1, self.hidden_dim)
            print(f"Added intervention variable: {var_name}")
    
    def explain_intervention(self, intervention, detailed=False):
        """Generate an explanation of the applied intervention.
        
        Args:
            intervention (dict): The intervention that was applied
            detailed (bool): Whether to provide a detailed explanation
            
        Returns:
            str: Explanation of the intervention
        """
        explanations = []
        
        for var_name, value in intervention.items():
            # Extract modality and concept from variable name
            parts = var_name.split('_')
            if len(parts) >= 2:
                modality, concept = parts[0], '_'.join(parts[1:])
                
                # Interpret the value
                value_interp = "increased" if value > 0 else "decreased"
                if abs(value) > 0.7:
                    value_interp = f"strongly {value_interp}"
                
                # Create explanation
                explanation = f"The {concept} in the {modality} was {value_interp}"
                explanations.append(explanation)
            else:
                explanations.append(f"Variable {var_name} was set to {value}")
        
        if detailed and explanations:
            # Add details about expected effects
            explanations.append("\nExpected effects:")
            for var_name in intervention:
                if var_name in self.intervention_projections:
                    # Get weight matrix norm as proxy for importance
                    weight = self.intervention_projections[var_name].weight
                    importance = weight.abs().mean().item()
                    explanations.append(f"- {var_name}: influence strength = {importance:.2f}")
        
        return "\n".join(explanations) if explanations else "No intervention applied"