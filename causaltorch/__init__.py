"""
CausalTorch: A PyTorch extension for causal deep learning.

This package provides tools and models for incorporating causal relationships
into deep learning models using PyTorch.
"""

__version__ = '0.1.0'

# Import core components
from .rules import CausalRule, CausalRuleSet, load_default_rules
from .layers import CausalSymbolicLayer

# Import models
from .models import CNSGImageGenerator, CNSG_GPT2 as CNSGTextGenerator, CNSG_VideoGenerator

__all__ = [
    'CausalRule',
    'CausalRuleSet',
    'load_default_rules',
    'CausalSymbolicLayer',
    'CNSGImageGenerator',
    'CNSGTextGenerator',  # This will work because of the alias above
    'CNSG_VideoGenerator'
]