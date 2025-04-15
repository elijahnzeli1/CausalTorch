"""
CausalTorch Rules Package
========================

This package provides tools for defining and operating with causal relationships,
implementing causal inference algorithms, and analyzing causal graphs.
"""

# Import from the engine module
from .engine import CausalRule, CausalRuleSet, load_default_rules

# Import from the core module
from .core import CausalInference, CausalGraphAnalysis

__all__ = [
    # Engine components
    'CausalRule', 
    'CausalRuleSet', 
    'load_default_rules',
    
    # Core components
    'CausalInference',
    'CausalGraphAnalysis'
]