"""
Text Encoder with Causal Constraints
===================================

This module provides a text encoder that extracts causal features from text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .causal_attention import CausalAttentionLayer
from causaltorch.rules import CausalRuleSet


class CausalTextEncoder(nn.Module):
    """A text encoder that extracts causally relevant features.
    
    This leverages pre-trained language models and causal attention
    to identify causal relationships in text.
    
    Args:
        model_name (str): Name of pre-trained model to use as base encoder
        causal_rules (CausalRuleSet): Set of causal rules to enforce in attention
        hidden_dim (int): Dimension of the output features
        freeze_base (bool): Whether to freeze the base model parameters
    """
    def __init__(
        self,
        model_name="bert-base-uncased",
        causal_rules=None,
        hidden_dim=768,
        freeze_base=True
    ):
        super().__init__()
        
        # Load pre-trained text model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get base model output dimension
        base_dim = self.base_model.config.hidden_size
        
        # Add causal attention layer
        rules_dict = {}
        if causal_rules and hasattr(causal_rules, 'rules'):
            rules_dict = {rule.cause: {"effect": rule.effect, "strength": rule.strength} 
                      for rule in causal_rules.rules}
        self.causal_attn = CausalAttentionLayer(rules_dict, attention_dim=base_dim)
        
        # Pass the tokenizer to the causal attention layer
        self.causal_attn.tokenizer = self.tokenizer
        
        # Add projection to target dimension if needed
        self.projection = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ) if base_dim != hidden_dim else nn.Identity()
        
        self.hidden_dim = hidden_dim
    
    def forward(self, text=None, input_ids=None, attention_mask=None):
        """Extract causal features from text.
        
        Args:
            text (List[str], optional): List of text inputs
            input_ids (torch.Tensor, optional): Pre-tokenized input IDs
            attention_mask (torch.Tensor, optional): Attention mask for input_ids
            
        Returns:
            torch.Tensor: Encoded causal features [batch_size, seq_len, hidden_dim]
        """
        # Tokenize text if raw text is provided
        if text is not None and input_ids is None:
            tokenized = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].to(self.base_model.device)
            attention_mask = tokenized["attention_mask"].to(self.base_model.device)
        
        # Encode with base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract hidden states
        hidden_states = base_outputs.last_hidden_state
        
        # Apply causal attention
        causal_states, _ = self.causal_attn(hidden_states, attention_mask)
        
        # Project to target dimension
        outputs = self.projection(causal_states)
        
        return outputs
    
    def encode_pooled(self, text=None, input_ids=None, attention_mask=None):
        """Get a single pooled vector representation for each text.
        
        Args:
            text (List[str], optional): List of text inputs
            input_ids (torch.Tensor, optional): Pre-tokenized input IDs
            attention_mask (torch.Tensor, optional): Attention mask for input_ids
            
        Returns:
            torch.Tensor: Pooled encoded features [batch_size, hidden_dim]
        """
        # Get sequence output
        sequence_output = self.forward(text, input_ids, attention_mask)
        
        # Use attention mask if available for proper mean pooling
        if attention_mask is not None:
            # Expand mask for proper broadcasting
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(sequence_output)
            
            # Apply mask and compute mean over sequence length
            # We need to sum and divide by sum of mask to handle variable sequence lengths
            sum_embeddings = torch.sum(sequence_output * expanded_mask, dim=1)
            sum_mask = torch.sum(expanded_mask, dim=1)
            
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean pooling if no mask is provided
            pooled_output = torch.mean(sequence_output, dim=1)
        
        return pooled_output 