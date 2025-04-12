"""
CausalTorch: Causal Neural-Symbolic Generative Networks
======================================================

A PyTorch library for building generative models with causal constraints.

Key components:
- CNSG models for text, image, and video generation with causal rules
- Graph-based causal rule definition and visualization
- Specialized neural layers for enforcing causal constraints 
- Metrics for evaluating causal fidelity and consistency
"""

__version__ = "0.1.0"

from ..models import CNSG_GPT2, CNSGNet, CNSG_VideoGenerator
from ..rules import CausalRule, CausalRuleSet, load_default_rules
from ..layers import CausalLinear, CausalAttentionLayer, CausalSymbolicLayer
from ..metrics import calculate_cfs, temporal_consistency, novelty_index

__all__ = [
    "CNSG_GPT2", "CNSGNet", "CNSG_VideoGenerator",
    "CausalRule", "CausalRuleSet", "load_default_rules", 
    "CausalLinear", "CausalAttentionLayer", "CausalSymbolicLayer",
    "calculate_cfs", "temporal_consistency", "novelty_index",
]