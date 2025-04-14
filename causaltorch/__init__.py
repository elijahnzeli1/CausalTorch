"""
CausalTorch: Causal Neural-Symbolic Generative Networks
======================================================

A PyTorch library for building generative models with causal constraints.

Key components:
- Native causal transformer models for text, image, and video generation
- Graph-based causal rule definition and visualization
- Specialized neural layers for enforcing causal constraints
- Ethics by architecture - built-in ethical constraints
- Meta-learning for dynamic architecture generation
- Federated learning with causal knowledge sharing
- Creative computation through counterfactual reasoning
"""

__version__ = "2.0.0"

# Export core models
from .models import (
    CausalTransformer,
    CausalLanguageModel,
    SelfEvolvingTextGenerator,
    FewShotCausalTransformer,
    MultimodalCausalTransformer,
    CounterfactualCausalTransformer,
    CNSGImageGenerator,
    CNSG_VideoGenerator,
    # Legacy aliases
    CNSG_GPT2,
    CNSGTextGenerator,
    CNSGNet
)

# Export causal rules
from .rules import CausalRule, CausalRuleSet, load_default_rules

# Export layers
from .layers import CausalLinear, CausalAttentionLayer, CausalSymbolicLayer

# Export ethics components
from .ethics.constitution import EthicalConstitution, EthicalRule, load_default_ethical_rules, EthicalLoss

# Export federated learning components
from .federated.dao import CausalDAO, FederatedClient

# Export creative computation components
from .creative.dreamer import CounterfactualDreamer, CausalIntervention, CreativeMetrics, NoveltySearch

# Export metrics
from .metrics import calculate_cfs, temporal_consistency, novelty_index

__all__ = [
    # Models
    "CausalTransformer",
    "CausalLanguageModel",
    "SelfEvolvingTextGenerator",
    "FewShotCausalTransformer",
    "MultimodalCausalTransformer",
    "CounterfactualCausalTransformer",
    "CNSGImageGenerator",
    "CNSG_VideoGenerator",
    # Legacy aliases
    "CNSG_GPT2",
    "CNSGTextGenerator",
    "CNSGNet",
    # Rules
    "CausalRule", 
    "CausalRuleSet", 
    "load_default_rules",
    # Layers
    "CausalLinear",
    "CausalAttentionLayer",
    "CausalSymbolicLayer",
    # Ethics
    "EthicalConstitution", 
    "EthicalRule",
    "load_default_ethical_rules",
    "EthicalLoss",
    # Federated Learning
    "CausalDAO",
    "FederatedClient",
    # Creative Computation
    "CausalIntervention",
    "CounterfactualDreamer",
    "CreativeMetrics",
    "NoveltySearch",
    # Metrics
    "calculate_cfs",
    "temporal_consistency",
    "novelty_index"
]