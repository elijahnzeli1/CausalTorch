"""
CausalTorch Models Module
=========================

This module contains pre-built causal neuro-symbolic generative models
for text, image, and video generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

try:
    import pytorch_lightning as pl # type: ignore
except ImportError:
    # Optional dependencies
    pass

from causaltorch.layers import CausalAttentionLayer, CausalSymbolicLayer, TemporalCausalConv


class CausalPositionalEncoding(nn.Module):
    """Positional encoding with causal constraints for text generation."""
    
    def __init__(self, d_model: int, max_seq_length: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class CausalTransformerBlock(nn.Module):
    """Causal Transformer block with integrated causal reasoning."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal_rules: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Standard multi-head attention (we'll integrate causal logic separately)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        
        # Causal symbolic reasoning layer
        self.causal_symbolic = CausalSymbolicLayer(causal_rules)
        
        # Store causal rules for attention modification
        self.causal_rules = causal_rules
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with causal attention and symbolic reasoning."""
        # Create causal mask for self-attention (prevent looking ahead)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Self-attention with causal mask and residual connection
        attn_output, attn_weights = self.self_attention(
            x, x, x, 
            attn_mask=causal_mask,
            need_weights=True
        )
        x = self.attention_norm(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        # Apply causal symbolic reasoning
        x = self.causal_symbolic(x)
        
        return x


class cnsg(nn.Module):
    """Causal Neuro-Symbolic Generator for text generation.
    
    This model implements a native CausalTorch text generation architecture
    with integrated causal reasoning, without relying on external models.
    
    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        d_ff (int): Feed-forward network dimension
        max_seq_length (int): Maximum sequence length
        causal_rules (dict): Dictionary of causal rules to enforce
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 768, n_heads: int = 12, 
                 n_layers: int = 12, d_ff: int = 3072, max_seq_length: int = 1024, 
                 causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.causal_rules = causal_rules or {}
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = CausalPositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers with causal reasoning
        self.transformer_layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, self.causal_rules)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Causal reasoning for generation constraints
        self.generation_causal_layer = CausalSymbolicLayer(self.causal_rules)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with causal constraints.
        
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_length]
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Target token IDs for training
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs including logits and loss
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers with causal reasoning
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Apply generation-level causal constraints
        x = self.generation_causal_layer(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs["loss"] = loss
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                causal_constraints: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Generate text with causal constraints.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [1, seq_length]
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            causal_constraints (dict, optional): Additional causal constraints
            
        Returns:
            torch.Tensor: Generated token sequence
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs["logits"]
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply causal constraints to logits if specified
                if causal_constraints:
                    next_token_logits = self._apply_causal_constraints(
                        next_token_logits, generated, causal_constraints
                    )
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for stop conditions (e.g., EOS token)
                if next_token.item() == 50256:  # EOS token for GPT-2 vocab
                    break
        
        return generated
    
    def _apply_causal_constraints(self, logits: torch.Tensor, context: torch.Tensor, 
                                constraints: Dict[str, Any]) -> torch.Tensor:
        """Apply causal constraints to generation logits."""
        # This method can be extended to implement specific causal constraints
        # For now, we apply a simple penalty/boost based on causal rules
        
        if "forbidden_words" in constraints:
            for word_id in constraints["forbidden_words"]:
                logits[:, word_id] = float('-inf')
        
        if "encouraged_words" in constraints:
            for word_id in constraints["encouraged_words"]:
                logits[:, word_id] += 2.0  # Boost probability
        
        return logits
    
    def get_causal_attention_patterns(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract causal attention patterns for analysis."""
        attention_patterns = {}
        
        with torch.no_grad():
            x = self.token_embeddings(input_ids)
            x = self.positional_encoding(x)
            
            for i, layer in enumerate(self.transformer_layers):
                # Extract attention weights from causal attention layer
                if hasattr(layer.causal_attention, 'attention_weights'):
                    attention_patterns[f'layer_{i}'] = layer.causal_attention.attention_weights
        
        return attention_patterns


class CausalVisionPatchEmbedding(nn.Module):
    """Patch embedding with causal constraints for vision transformers."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.causal_rules = causal_rules or {}
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection
        self.patch_projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Causal constraints on patch relationships
        self.causal_patch_layer = CausalSymbolicLayer(self.causal_rules)
        
        # Positional embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings with causal constraints."""
        batch_size = x.shape[0]
        
        # Extract patches and project to embedding dimension
        patches = self.patch_projection(x)  # [B, embed_dim, H/P, W/P]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Apply causal constraints to patch relationships
        patches = self.causal_patch_layer(patches)
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        patches = torch.cat([class_tokens, patches], dim=1)
        
        # Add positional embeddings
        patches += self.position_embeddings
        
        return patches


class CausalVisionTransformerBlock(nn.Module):
    """Vision Transformer block with causal reasoning."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal_rules = causal_rules or {}
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
        # Causal reasoning layer
        self.causal_reasoning = CausalSymbolicLayer(self.causal_rules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal attention and reasoning."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = self.mlp_norm(x + mlp_output)
        
        # Apply causal reasoning
        x = self.causal_reasoning(x)
        
        return x


class CausalVisionTransformer(nn.Module):
    """Causal Vision Transformer for image classification and analysis.
    
    This model implements a Vision Transformer with integrated causal reasoning
    for understanding visual relationships and hierarchical image features.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Size of image patches
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        causal_rules (dict): Causal rules for visual reasoning
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, 
                 causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.causal_rules = causal_rules or {}
        
        # Patch embedding
        self.patch_embedding = CausalVisionPatchEmbedding(
            img_size, patch_size, in_channels, embed_dim, causal_rules
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CausalVisionTransformerBlock(embed_dim, num_heads, mlp_ratio, causal_rules)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classification_head = nn.Linear(embed_dim, num_classes)
        
        # Causal analysis head for understanding visual relationships
        self.causal_analysis_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, len(causal_rules) if causal_rules else 10)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_causal_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with optional causal analysis.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            return_causal_analysis (bool): Whether to return causal analysis
            
        Returns:
            Dict containing logits and optional causal analysis
        """
        # Convert to patch embeddings
        x = self.patch_embedding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Extract class token for classification
        class_token = x[:, 0]
        
        # Classification
        logits = self.classification_head(class_token)
        
        outputs = {"logits": logits}
        
        # Causal analysis if requested
        if return_causal_analysis:
            causal_scores = self.causal_analysis_head(class_token)
            outputs["causal_analysis"] = causal_scores
            
            # Extract patch tokens for spatial causal analysis
            patch_tokens = x[:, 1:]  # Remove class token
            outputs["patch_embeddings"] = patch_tokens
        
        return outputs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations without classification."""
        x = self.patch_embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        return x[:, 0]  # Return class token
    
    def analyze_visual_causality(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze causal relationships in visual features."""
        features = self.patch_embedding(x)
        
        causal_maps = {}
        
        for i, block in enumerate(self.transformer_blocks):
            features = block(features)
            
            # Extract causal reasoning from this layer
            if hasattr(block.causal_reasoning, 'causal_rules'):
                causal_maps[f'layer_{i}'] = features
        
        return causal_maps


class CausalObjectDetector(nn.Module):
    """Causal Object Detector with reasoning about object relationships.
    
    This model detects objects while understanding causal relationships
    between objects, spatial arrangements, and contextual dependencies.
    """
    
    def __init__(self, backbone_dim: int = 768, num_classes: int = 80, 
                 max_objects: int = 100, causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.causal_rules = causal_rules or {}
        
        # Vision backbone (can be replaced with any backbone)
        self.backbone = CausalVisionTransformer(
            num_classes=backbone_dim,  # Use as feature extractor
            causal_rules=causal_rules
        )
        
        # Object detection heads
        self.bbox_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, 4)  # x, y, w, h
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, num_classes)
        )
        
        # Causal relationship detector
        self.relationship_detector = nn.Sequential(
            nn.Linear(backbone_dim * 2, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, len(causal_rules) if causal_rules else 10)
        )
        
        # Object interaction causal layer
        self.object_causal_layer = CausalSymbolicLayer(causal_rules)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect objects with causal relationship analysis."""
        # Extract features
        backbone_output = self.backbone(x, return_causal_analysis=True)
        features = backbone_output["patch_embeddings"]  # [B, num_patches, embed_dim]
        
        batch_size, num_patches, embed_dim = features.shape
        
        # Apply causal reasoning to object features
        features = self.object_causal_layer(features)
        
        # Object detection for each patch
        bboxes = self.bbox_head(features)  # [B, num_patches, 4]
        class_logits = self.classification_head(features)  # [B, num_patches, num_classes]
        
        # Analyze relationships between objects
        relationships = []
        for i in range(min(num_patches, self.max_objects)):
            for j in range(i + 1, min(num_patches, self.max_objects)):
                # Concatenate features of object pairs
                pair_features = torch.cat([features[:, i], features[:, j]], dim=-1)
                relationship_score = self.relationship_detector(pair_features)
                relationships.append(relationship_score)
        
        if relationships:
            relationships = torch.stack(relationships, dim=1)
        else:
            relationships = torch.zeros(batch_size, 1, len(self.causal_rules) if self.causal_rules else 10)
        
        return {
            "bboxes": bboxes,
            "class_logits": class_logits,
            "relationships": relationships,
            "features": features
        }


class CausalSegmentationModel(nn.Module):
    """Causal Semantic Segmentation Model.
    
    Performs pixel-level segmentation while understanding causal relationships
    between different semantic regions and their contextual dependencies.
    """
    
    def __init__(self, backbone_dim: int = 768, num_classes: int = 21, 
                 causal_rules: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes
        self.causal_rules = causal_rules or {}
        
        # Vision backbone
        self.backbone = CausalVisionTransformer(
            patch_size=8,  # Smaller patches for dense prediction
            embed_dim=backbone_dim,
            causal_rules=causal_rules
        )
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(backbone_dim, backbone_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(backbone_dim // 2, backbone_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(backbone_dim // 4, num_classes, kernel_size=4, stride=2, padding=1)
        )
        
        # Causal region relationship analyzer
        self.region_causal_analyzer = CausalSymbolicLayer(causal_rules)
        
        # Context reasoning head
        self.context_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, len(causal_rules) if causal_rules else 10)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform segmentation with causal reasoning."""
        batch_size, channels, height, width = x.shape
        
        # Extract features
        backbone_output = self.backbone(x, return_causal_analysis=True)
        patch_features = backbone_output["patch_embeddings"]
        
        # Calculate patch grid dimensions
        patch_size = self.backbone.patch_size
        patch_h = height // patch_size
        patch_w = width // patch_size
        
        # Reshape patch features to spatial grid
        patch_features = patch_features.view(batch_size, patch_h, patch_w, self.backbone_dim)
        patch_features = patch_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply causal reasoning to spatial features
        spatial_features = patch_features.view(batch_size, self.backbone_dim, -1).permute(0, 2, 1)
        spatial_features = self.region_causal_analyzer(spatial_features)
        spatial_features = spatial_features.permute(0, 2, 1).view(batch_size, self.backbone_dim, patch_h, patch_w)
        
        # Decode to segmentation map
        segmentation_logits = self.decoder(spatial_features)
        
        # Context analysis
        global_context = patch_features.mean(dim=[2, 3])  # Global average pooling
        context_analysis = self.context_head(global_context)
        
        return {
            "segmentation_logits": segmentation_logits,
            "context_analysis": context_analysis,
            "spatial_features": spatial_features
        }


class CNSGNet(nn.Module):
    """Causal Neuro-Symbolic Generative Network for image generation.
    
    This model implements a VAE/GAN with causal structure in the latent space.
    
    Args:
        latent_dim (int): Dimension of the latent space
        causal_rules (dict, optional): Dictionary of causal rules
        img_size (int, optional): Size of generated images
    """
    def __init__(self, latent_dim=3, causal_rules=None, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and variance for VAE
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Causal layer to enforce relationships in latent space
        self.causal_layer = CausalSymbolicLayer(causal_rules)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            tuple: (mu, log_var) parameters of latent distribution
        """
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Generated image
        """
        # Apply causal constraints to latent vector
        z = self.causal_layer(z)
        
        # Decode to image
        h = self.decoder(z)
        return h.view(h.size(0), 1, self.img_size, self.img_size)
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            tuple: (reconstructed_x, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
    
    def generate(self, rain_intensity=None, num_samples=1):
        """Generate images with causal constraints.
        
        Args:
            rain_intensity (float, optional): Rain intensity value (0-1)
            num_samples (int, optional): Number of images to generate
            
        Returns:
            torch.Tensor: Generated images
        """
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_samples, self.latent_dim)
            
            # If rain_intensity is specified, set the rain dimension
            if rain_intensity is not None:
                z[:, 0] = rain_intensity
            
            # Generate images with causal constraints
            images = self.decode(z)
            return images


class CNSG_VideoGenerator(nn.Module):
    """Causal Neuro-Symbolic Video Generator.
    
    This model generates temporally consistent video with causal constraints
    between frames.
    
    Args:
        frame_size (tuple): Height and width of video frames
        latent_dim (int): Dimension of the latent space
        causal_rules (dict, optional): Dictionary of temporal causal rules
    """
    def __init__(self, frame_size=(64, 64), latent_dim=16, causal_rules=None):
        super().__init__()
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        height, width = frame_size
        
        # Frame generator network
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 3, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Latent dynamics network (predicts next latent state)
        self.latent_encoder = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        
        # Temporal causal layer
        self.temporal_causal = TemporalCausalConv(3, 3, kernel_size=3, causal_rules=causal_rules or {})
    
    def forward(self, initial_frame, initial_latent, seq_length=24, metadata=None):
        """Generate a video sequence with causal temporal constraints.
        
        Args:
            initial_frame (torch.Tensor): Starting frame [batch, 3, H, W]
            initial_latent (torch.Tensor): Initial latent state [batch, latent_dim]
            seq_length (int): Number of frames to generate
            metadata (dict, optional): Frame metadata for causal rules
            
        Returns:
            torch.Tensor: Generated video sequence [batch, seq_length, 3, H, W]
        """
        batch_size = initial_frame.size(0)
        device = initial_frame.device
        frames = [initial_frame]
        latent = initial_latent
        
        # Generate frames sequentially
        for t in range(seq_length - 1):
            # Get previous frame
            prev_frame = frames[-1]
            
            # Update latent state
            latent_input = latent.unsqueeze(1)  # Add sequence dimension
            latent_output, _ = self.latent_encoder(latent_input)
            latent = latent_output.squeeze(1)  # Remove sequence dimension
            
            # Generate next frame
            # Expand latent to match frame spatial dimensions
            h, w = prev_frame.shape[-2:]
            latent_expanded = latent.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            gen_input = torch.cat([prev_frame, latent_expanded], dim=1)
            next_frame = self.generator(gen_input)
            
            # Apply temporal causal effects
            if metadata is not None:
                # Get metadata for current frame
                frame_metadata = {k: v[t] if isinstance(v, list) else v for k, v in metadata.items()}
                next_frame = self.temporal_causal(next_frame.unsqueeze(2), frame_metadata).squeeze(2)
            
            frames.append(next_frame)
        
        # Stack frames to make video
        video = torch.stack(frames, dim=1)  # [batch, seq_length, 3, H, W]
        return video
    
    def generate_battle_scene(self, num_frames=24):
        """Generate a battle scene with horses, arrows, and causal effects.
        
        Args:
            num_frames (int): Number of frames to generate
            
        Returns:
            torch.Tensor: Generated battle video
        """
        # Create initial inputs
        batch_size = 1
        initial_frame = torch.randn(batch_size, 3, self.frame_size[0], self.frame_size[1])
        initial_latent = torch.zeros(batch_size, self.latent_dim)
        
        # Set up metadata with causal events
        metadata = {
            # Hoof contacts ground at specific frames
            "hoof_contact": [1.0 if i % 6 == 0 else 0.0 for i in range(num_frames)],
            
            # Arrow hits at frame 10
            "arrow_hit": [1.0 if i == 10 else 0.0 for i in range(num_frames)]
        }
        
        # Generate video
        return self.forward(initial_frame, initial_latent, seq_length=num_frames, metadata=metadata) 