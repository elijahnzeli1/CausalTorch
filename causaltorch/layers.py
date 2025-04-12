"""
CausalTorch Layers Module
=========================

This module contains neural network layers that enforce causal relationships
during training and inference.
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


class CausalAttentionLayer(nn.Module):
    """Attention layer that enforces causal rules in text generation.
    
    This layer modifies attention scores to bias the model toward 
    generating text that follows causal rules.
    
    Args:
        causal_rules (dict): Dictionary mapping causes to effects with strengths
    """
    def __init__(self, causal_rules):
        super().__init__()
        self.rules = causal_rules
    
    def forward(self, attention_scores, input_text):
        """Apply causal rules to attention scores.
        
        Args:
            attention_scores (torch.Tensor): Original attention scores
            input_text (str): Input text to check for causes
            
        Returns:
            torch.Tensor: Modified attention scores biased by causal rules
        """
        # We assume a tokenizer is available in the parent model
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer not found. Please set self.tokenizer in the parent model.")
        
        for cause, effect_info in self.rules.items():
            if cause in input_text:
                effect_ids = tokenizer.encode(effect_info["effect"], add_special_tokens=False)
                for token_id in effect_ids:
                    attention_scores[..., token_id] += effect_info["strength"]
        
        return attention_scores


class TemporalCausalConv(nn.Module):
    """Convolutional layer that enforces temporal causal relationships.
    
    This layer is designed for video generation, ensuring that causes 
    lead to effects across frames with proper temporal offsets.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        causal_rules (dict): Temporal causal rules
    """
    def __init__(self, in_channels, out_channels, kernel_size, causal_rules):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.causal_rules = causal_rules
    
    def forward(self, x, metadata):
        """Apply temporal causal rules during convolution.
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, frames, height, width]
            metadata (dict): Frame metadata containing causal variables
            
        Returns:
            torch.Tensor: Output with causal effects applied
        """
        # Apply standard convolution
        output = self.conv(x)
        
        # Apply causal rules
        for cause, effect_info in self.causal_rules.items():
            if cause in metadata:
                # Get effect parameters
                effect = effect_info["effect"]
                intensity = effect_info.get("intensity", 1.0)
                offset = effect_info.get("temporal_offset", 0)
                duration = effect_info.get("duration", 1)
                
                # Apply the effect
                cause_value = metadata[cause]
                
                # Example: add dust under horse hooves
                if cause == "hoof_contact" and effect == "dust":
                    dust_intensity = cause_value * intensity
                    # Add dust effect to specific regions in future frames
                    if offset < x.size(2):  # Check if offset is within frame range
                        # Apply dust to ground area
                        output[:, :, offset:offset+duration, 50:60, 20:30] += dust_intensity.unsqueeze(1)
        
        return output


class CausalSymbolicLayer(nn.Module):
    """Layer that enforces symbolic causal rules in the latent space.
    
    This layer directly modifies latent variables according to causal rules,
    ensuring that dependent variables change when their causes are modified.
    
    Args:
        causal_rules (dict, optional): Dictionary of causal rules
    """
    def __init__(self, causal_rules=None):
        super().__init__()
        self.causal_rules = causal_rules or {}
    
    def forward(self, z):
        """Apply causal constraints to latent variables.
        
        Args:
            z (torch.Tensor): Latent tensor where dimensions represent causal variables
            
        Returns:
            torch.Tensor: Modified latent tensor with causal constraints applied
        """
        # Example: enforce "rain → wet ground" rule
        if "rain" in self.causal_rules:
            rain_idx = 0  # Assume rain is the first dimension
            wet_ground_idx = 1  # Assume ground wetness is the second dimension
            
            # Get rule parameters
            rule = self.causal_rules["rain"]
            strength = rule.get("strength", 0.9)
            threshold = rule.get("threshold", 0.5)
            
            # Apply rule: if rain > threshold, ground should be wet
            rain_intensity = z[:, rain_idx]
            ground_wetness = torch.sigmoid((rain_intensity - threshold) * 10) * strength
            z[:, wet_ground_idx] = ground_wetness
        
        return z 