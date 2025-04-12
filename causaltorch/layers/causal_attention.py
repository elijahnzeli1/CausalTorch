"""Attention layer with causal constraints for CausalTorch."""

import torch
import torch.nn as nn


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