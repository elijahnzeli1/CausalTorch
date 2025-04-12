"""Temporal convolutional layer with causal constraints for CausalTorch."""

import torch
import torch.nn as nn


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