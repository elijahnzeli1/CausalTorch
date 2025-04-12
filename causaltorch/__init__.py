"""
CausalTorch: A PyTorch extension for causal deep learning.
"""

__version__ = '0.1.0'

# Import core components
from .rules import CausalRule, CausalRuleSet, load_default_rules
from .layers import CausalSymbolicLayer, CausalAttentionLayer, CausalLinear, TemporalCausalConv

# Import models - using the alias defined in models/__init__.py
from .models import CNSGImageGenerator, CNSGTextGenerator, CNSG_VideoGenerator

__all__ = [
    'CausalRule',
    'CausalRuleSet',
    'load_default_rules',
    'CausalSymbolicLayer',
    'CausalAttentionLayer',
    'CausalLinear',
    'TemporalCausalConv',
    'CNSGImageGenerator',
    'CNSGTextGenerator',
    'CNSG_VideoGenerator'
]