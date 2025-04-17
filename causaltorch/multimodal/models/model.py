"""
Multimodal Causal Model
=====================

This module provides a complete multimodal model that integrates
all the causal components.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..utils.metrics import calculate_multimodal_causal_consistency


class MultimodalCausalModel(nn.Module):
    """A complete multimodal causal model that combines all components.
    
    This model handles:
    - Text encoding with causal awareness
    - Image encoding with causal awareness
    - Modality fusion
    - Generation with ethical constraints
    
    Args:
        text_encoder (nn.Module, optional): Pre-initialized text encoder
        image_encoder (nn.Module, optional): Pre-initialized image encoder
        fusion (nn.Module, optional): Pre-initialized fusion module
        generator (nn.Module, optional): Pre-initialized generator
        text_criterion (nn.Module, optional): Loss function for text
        image_criterion (nn.Module, optional): Loss function for images
        text_encoder_name (str): Name of text encoder if not pre-initialized
        image_encoder_name (str): Name of image encoder if not pre-initialized
        hidden_dim (int): Dimension of hidden layers
        fusion_method (str): Method for fusion ('attention', 'linear', etc.)
    """
    
    def __init__(
        self,
        text_encoder=None,
        image_encoder=None,
        fusion=None,
        generator=None,
        text_criterion=None,
        image_criterion=None,
        text_encoder_name="bert-base-uncased",
        image_encoder_name="resnet18",
        hidden_dim=768,
        fusion_method="attention"
    ):
        """Initialize the model."""
        super().__init__()
        
        # Use provided components or create new ones
        if text_encoder is not None and image_encoder is not None and fusion is not None and generator is not None:
            self.text_encoder = text_encoder
            self.image_encoder = image_encoder
            self.fusion = fusion
            self.generator = generator
            self.text_criterion = text_criterion or nn.CrossEntropyLoss()
            self.image_criterion = image_criterion or nn.MSELoss()
        else:
            print(f"Creating new model components with parameters:")
            print(f"  - text_encoder_name: {text_encoder_name}")
            print(f"  - image_encoder_name: {image_encoder_name}")
            print(f"  - hidden_dim: {hidden_dim}")
            print(f"  - fusion_method: {fusion_method}")
            
            # Import here to avoid circular imports
            from .text_encoder import CausalTextEncoder
            from .image_encoder import CausalImageEncoder
            from .fusion import CausalModalFusion
            from .generator import MultimodalCausalGenerator
            
            # Create basic components
            self.text_encoder = CausalTextEncoder(
                model_name=text_encoder_name,
                hidden_dim=hidden_dim
            )
            
            self.image_encoder = CausalImageEncoder(
                model_name=image_encoder_name,
                hidden_dim=hidden_dim
            )
            
            self.fusion = CausalModalFusion(
                modality_dims={
                    "text": hidden_dim,
                    "image": hidden_dim
                },
                hidden_dim=hidden_dim,
                fusion_method=fusion_method
            )
            
            self.generator = MultimodalCausalGenerator(
                hidden_dim=hidden_dim
            )
            
            self.text_criterion = nn.CrossEntropyLoss()
            self.image_criterion = nn.MSELoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        causal_graphs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for text
            images: Image tensors
            causal_graphs: Causal graph annotations
            
        Returns:
            Dictionary containing:
            - text_loss: Loss for text generation
            - image_loss: Loss for image generation
            - causal_loss: Loss for causal consistency
        """
        # Encode text
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Encode images
        image_features = self.image_encoder(images)
        
        # Prepare modality features
        modality_features = {
            "text": self.text_encoder.encode_pooled(
                input_ids=input_ids,
                attention_mask=attention_mask
            ) if hasattr(self.text_encoder, 'encode_pooled') else text_features,
            "image": image_features
        }
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Generate outputs
        outputs = self.generator(fused_features)
        
        # Calculate losses
        losses = {}
        
        # Text reconstruction loss
        if "text" in outputs:
            text_logits = outputs["text"]
            
            # Check the dimensionality of the logits
            if len(text_logits.shape) == 2:
                # If 2D (batch_size, vocab_size), use as is
                losses["text_loss"] = self.text_criterion(
                    text_logits,
                    input_ids[:, 0]  # Use first token as target for simplicity
                )
            elif len(text_logits.shape) == 3:
                # If 3D (batch_size, seq_len, vocab_size), apply sequence processing
                text_targets = input_ids[:, 1:].contiguous()
                text_logits = text_logits[:, :-1, :].contiguous()
                losses["text_loss"] = self.text_criterion(
                    text_logits.view(-1, text_logits.size(-1)),
                    text_targets.view(-1)
                )
            else:
                # Unexpected shape, set placeholder loss
                losses["text_loss"] = torch.tensor(0.0, device=text_logits.device)
                print(f"Warning: Unexpected text_logits shape: {text_logits.shape}")
        
        # Image reconstruction loss
        if "image" in outputs:
            image_pred = outputs["image"]
            
            # Check for size mismatch between prediction and target
            if image_pred.shape != images.shape:
                print(f"Warning: Image size mismatch - prediction: {image_pred.shape}, target: {images.shape}")
                
                # If dimensions are different, resize the prediction to match target
                if len(image_pred.shape) == 4 and len(images.shape) == 4:
                    # Get target dimensions
                    _, _, target_h, target_w = images.shape
                    
                    # Only resize if needed
                    if image_pred.shape[2] != target_h or image_pred.shape[3] != target_w:
                        try:
                            # Use interpolate for resizing
                            image_pred = torch.nn.functional.interpolate(
                                image_pred, 
                                size=(target_h, target_w),
                                mode='bilinear',
                                align_corners=False
                            )
                            print(f"Resized prediction to match target: {image_pred.shape}")
                        except Exception as e:
                            print(f"Error resizing image: {e}")
            
            # Calculate the loss
            losses["image_loss"] = self.image_criterion(image_pred, images)
        
        # Causal consistency loss
        causal_consistency = calculate_multimodal_causal_consistency(
            modality_features["text"],
            modality_features["image"],
            causal_graphs,
            outputs
        )
        losses["causal_loss"] = 1.0 - causal_consistency
        
        return losses
    
    def generate_text(self, images, **kwargs):
        """Generate text based on input images.
        
        Args:
            images: Input image tensors
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Encode images
        image_features = self.image_encoder(images)
        
        # Create modality features dict (with empty text)
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Create dummy text embeddings if needed
        dummy_text = torch.zeros(batch_size, self.fusion.hidden_dim, device=device)
        
        modality_features = {
            "text": dummy_text,
            "image": image_features
        }
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Generate text
        outputs = self.generator.generate(fused_features, target_modality="text", **kwargs)
        
        return outputs["text"]
    
    def generate_image(self, text_ids, attention_mask=None, **kwargs):
        """Generate images based on input text.
        
        Args:
            text_ids: Input text token IDs
            attention_mask: Attention mask for text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images
        """
        # Encode text
        text_features = self.text_encoder.encode_pooled(
            input_ids=text_ids,
            attention_mask=attention_mask
        )
        
        # Create modality features dict (with empty image)
        batch_size = text_features.size(0)
        device = text_features.device
        
        # Create dummy image embeddings if needed
        dummy_image = torch.zeros(batch_size, self.fusion.hidden_dim, device=device)
        
        modality_features = {
            "text": text_features,
            "image": dummy_image
        }
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Generate image
        outputs = self.generator.generate(fused_features, target_modality="image", **kwargs)
        
        return outputs["image"]
    
    def generate_counterfactual(self, input_ids, attention_mask, images, intervention, **kwargs):
        """Generate counterfactual outputs by applying interventions.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            images: Input images
            intervention (dict): Causal intervention to apply
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with counterfactual outputs for each modality
        """
        # Encode inputs
        text_features = self.text_encoder.encode_pooled(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_features = self.image_encoder(images)
        
        # Create modality features
        modality_features = {
            "text": text_features,
            "image": image_features
        }
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Generate counterfactual
        outputs = self.generator.imagine_counterfactual(
            fused_features,
            intervention,
            **kwargs
        )
        
        return outputs 