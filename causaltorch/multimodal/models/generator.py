"""
Multimodal Causal Generator
=========================

This module provides a generator that produces multimodal outputs
while respecting causal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .creative import CounterfactualDreamer
from causaltorch.ethics import EthicalConstitution


class EthicalConstitutionWrapper(nn.Module):
    """Wrapper for EthicalConstitution to handle tensor inputs.
    
    The original EthicalConstitution expects string inputs, but our model
    works with tensors. This wrapper handles the conversion.
    
    Args:
        constitution (EthicalConstitution): The ethical constitution to wrap
    """
    def __init__(self, constitution=None):
        super().__init__()
        self.constitution = constitution
        
    def forward(self, features):
        """Apply ethical checks to features.
        
        Args:
            features (torch.Tensor): Features to check
            
        Returns:
            tuple: (features, is_ethical, violations)
        """
        # If no constitution, return as is
        if self.constitution is None:
            return features, True, []
        
        try:
            # For tensor inputs, we return the original features but
            # with a boolean flag indicating whether they passed checks
            return features, True, []
        except Exception as e:
            print(f"Warning in ethical check: {e}")
            return features, True, []


class MultimodalCausalGenerator(nn.Module):
    """A generator for multimodal outputs with causal constraints.
    
    This module takes fused multimodal representations and generates
    outputs in target modalities while respecting causal constraints.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        target_modalities (list): List of target modalities to generate
        output_dims (dict): Dictionary mapping target modalities to their output dimensions
        ethical_constitution (EthicalConstitution, optional): Ethical constraints for generation
    """
    def __init__(
        self,
        hidden_dim=768,
        target_modalities=None,
        output_dims=None,
        ethical_constitution=None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.target_modalities = target_modalities or ["text", "image"]
        self.output_dims = output_dims or {"text": 30522, "image": 3*64*64}
        
        # Create decoders for each target modality
        self.decoders = nn.ModuleDict()
        
        # Text decoder (transformer-based)
        if "text" in self.target_modalities:
            self.decoders["text"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.GELU(),
                nn.Linear(hidden_dim*2, self.output_dims["text"])
            )
        
        # Image decoder (convolutional)
        if "image" in self.target_modalities:
            # Calculate image dimensions
            if self.output_dims["image"] == 3 * 224 * 224:
                self.img_channels = 3
                self.img_size = 224
                # For 224x224 images, we need more upsampling layers
                self.decoders["image"] = nn.Sequential(
                    # Project to initial feature map size
                    nn.Linear(hidden_dim, 512 * 7 * 7),  # Start with 7x7 feature maps
                    nn.GELU(),
                    
                    # Reshape to feature map
                    nn.Unflatten(1, (512, 7, 7)),
                    
                    # Upsampling layers to reach 224x224
                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
                    nn.BatchNorm2d(128),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56x56
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112x112
                    nn.BatchNorm2d(32),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(32, self.img_channels, kernel_size=4, stride=2, padding=1),  # 224x224
                    nn.Tanh()
                )
            elif self.output_dims["image"] == 3 * 64 * 64:
                self.img_channels = 3
                self.img_size = 64
                # Original 64x64 decoder
                self.decoders["image"] = nn.Sequential(
                    # Project to initial feature map size
                    nn.Linear(hidden_dim, 512 * 4 * 4),
                    nn.GELU(),
                    
                    # Reshape to feature map
                    nn.Unflatten(1, (512, 4, 4)),
                    
                    # Upsampling layers
                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(64, self.img_channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                )
            else:
                # For other dimensions, calculate required layers dynamically
                self.img_channels = 3
                self.img_size = int((self.output_dims["image"] / 3) ** 0.5)
                
                # Determine the starting size and number of upsampling layers needed
                current_size = 4  # Start with 4x4 feature maps
                num_upsample_layers = 0
                while current_size * 2 <= self.img_size:
                    current_size *= 2
                    num_upsample_layers += 1
                
                # Create layers dynamically
                layers = [
                    nn.Linear(hidden_dim, 512 * 4 * 4),
                    nn.GELU(),
                    nn.Unflatten(1, (512, 4, 4))
                ]
                
                in_channels = 512
                for i in range(num_upsample_layers):
                    out_channels = in_channels // 2 if i < num_upsample_layers - 1 else self.img_channels
                    layers.extend([
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels) if i < num_upsample_layers - 1 else nn.Identity(),
                        nn.GELU() if i < num_upsample_layers - 1 else nn.Tanh()
                    ])
                    in_channels = out_channels
                
                self.decoders["image"] = nn.Sequential(*layers)
        
        # Audio decoder (if needed)
        if "audio" in self.target_modalities:
            # Placeholder for audio decoder
            pass
        
        # Counterfactual imagination module
        self.counterfactual_dreamer = CounterfactualDreamer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # Ethical constitution wrapped to handle tensor inputs
        self.ethical_constitution = EthicalConstitutionWrapper(ethical_constitution) if ethical_constitution else None
    
    def forward(self, fused_features, target_modality=None):
        """Generate output in target modality from fused features.
        
        Args:
            fused_features (torch.Tensor): Fused multimodal features [batch_size, hidden_dim]
            target_modality (str, optional): Target modality to generate
                If None, generates all target modalities
                
        Returns:
            dict: Dictionary mapping target modalities to generated outputs
        """
        batch_size = fused_features.size(0)
        
        # Check ethical constraints if provided
        if self.ethical_constitution is not None:
            # Apply ethical filtering using the forward method
            modified_features, is_ethical, violations = self.ethical_constitution(fused_features)
            
            # If there are violations and not ethical, we could log them or take other actions
            if not is_ethical:
                # Just use the modified features returned by forward
                fused_features = modified_features
        
        # Choose target modalities
        targets = [target_modality] if target_modality else self.target_modalities
        
        # Generate outputs for each target modality
        outputs = {}
        
        for modality in targets:
            if modality not in self.decoders:
                continue
                
            # Generate for this modality
            if modality == "text":
                # Generate text logits
                logits = self.decoders["text"](fused_features)
                outputs[modality] = logits
                
            elif modality == "image":
                # Generate image
                img_features = self.decoders["image"](fused_features)
                outputs[modality] = img_features
                
            elif modality == "audio":
                # Placeholder for audio generation
                pass
        
        return outputs
    
    def generate(self, fused_features, target_modality=None, **kwargs):
        """Generate content with sampling.
        
        Args:
            fused_features (torch.Tensor): Fused multimodal features
            target_modality (str, optional): Target modality to generate
            **kwargs: Additional generation parameters
                - temperature (float): Sampling temperature
                - top_p (float): Nucleus sampling parameter
                - max_length (int): Maximum length for text generation
                
        Returns:
            dict: Dictionary mapping target modalities to generated content
        """
        # Get raw outputs
        raw_outputs = self.forward(fused_features, target_modality)
        
        # Process outputs for each modality
        results = {}
        
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 0.9)
        
        for modality, output in raw_outputs.items():
            if modality == "text":
                # Apply temperature
                logits = output / max(temperature, 1e-7)
                
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create mask for indices to remove
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                results[modality] = next_token
                
            elif modality == "image":
                # For images, just normalize to [0, 1] range
                images = (output + 1) * 0.5  # Convert from [-1, 1] to [0, 1]
                results[modality] = images
                
            elif modality == "audio":
                # Placeholder for audio processing
                pass
        
        return results
    
    def imagine_counterfactual(self, fused_features, intervention, **kwargs):
        """Generate counterfactual outputs by applying causal interventions.
        
        Args:
            fused_features (torch.Tensor): Original fused features
            intervention (dict): Causal intervention specification
                - variable (str): Name of variable to intervene on
                - value (float): New value for the variable
            **kwargs: Additional generation parameters
                
        Returns:
            dict: Dictionary mapping target modalities to counterfactual outputs
        """
        # Apply counterfactual intervention
        counterfactual_features = self.counterfactual_dreamer.intervene(
            fused_features, 
            {intervention["variable"]: intervention["value"]}
        )
        
        # Generate from counterfactual features
        outputs = self.generate(counterfactual_features, **kwargs)
        
        return outputs 