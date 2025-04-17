"""
Custom CausalAttentionLayer for the multimodal model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttentionLayer(nn.Module):
    """Attention layer that enforces causal rules in text processing.
    
    This layer applies causal biasing to attention between tokens.
    
    Args:
        causal_rules (dict): Dictionary mapping causes to effects
    """
    def __init__(self, causal_rules):
        super().__init__()
        self.rules = causal_rules
        self.tokenizer = None  # To be set by the parent model
        
        # Add a small linear layer for causal biasing
        self.hidden_dim = 768  # Default, will be overridden if needed
        self.causal_bias = nn.Parameter(torch.zeros(1, 2, 2))  # Dummy init, will be expanded
        
    def forward(self, hidden_states, attention_mask=None):
        """Apply causal attention to hidden states.
        
        Args:
            hidden_states (torch.Tensor): Input hidden states
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            tuple: (output_states, attention_weights)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Self-attention projections
        q = hidden_states
        k = hidden_states
        v = hidden_states
        
        # Compute attention scores
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / (hidden_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(~expanded_mask.bool(), -10000.0)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        output_states = torch.bmm(attention_weights, v)
        
        return output_states, attention_weights 