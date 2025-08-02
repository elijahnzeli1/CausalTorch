"""
CausalTorch: Causal Neural-Symbolic Generative Networks
======================================================

A PyTorch library for building generative models with causal constraints.
Built on core architecture principles:

1. PyTorch Foundation → CausalTorch Core → Specialized Modules
2. Standalone operation for from-scratch and fine-tuning scenarios  
3. Integrated causal reasoning engine as central hub
4. Intervention API, Counterfactual Engine, and Causal Regularization

Key components:
- CausalTorch Core: Foundation for all causal AI models
- From-Scratch Model Building: Build causal models from ground up
- Pre-trained Model Fine-tuning: Add causality to existing models
- Causal Reasoning Engine: Central hub for causal inference
- Intervention API: Perform causal interventions (do-calculus)
- Counterfactual Engine: Generate "what-if" scenarios
- Causal Regularization: Training stability and consistency
"""

__version__ = "2.0.1"

# Core Architecture Components (Primary Interface)
from .core_architecture import (
    CausalTorchCore,
    CausalReasoningEngine,
    FromScratchModelBuilder,
    PretrainedModelFineTuner,
    InterventionAPI,
    CounterfactualEngine,
    CausalRegularization,
    CausalAdapter
)

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
    cnsg,
    CNSGTextGenerator,
    CNSGNet
)

# Export causal rules
from .rules import CausalRule, CausalRuleSet, load_default_rules

# Export layers
from .layers import CausalLinear, CausalAttentionLayer, CausalSymbolicLayer

# Export ethics components
# Make this consistent with the ethics package's __init__.py
from .ethics import (
    EthicalConstitution,
    EthicalRule,
    EthicalTextFilter,
    EthicalLoss,
    load_default_ethical_rules
)

# Export federated learning components
from .federated.dao import CausalDAO, FederatedClient

# Export creative computation components
# Creative computation components  
from .creative import (
    CausalIntervention,
    CounterfactualDreamer, 
    CreativeMetrics,
    NoveltySearch
)

# Training infrastructure
from .training import CausalTrainer, CausalFineTuner

# MLOps platform
from .mlops import CausalMLOps, init_mlops, MLOpsTrainer

# Export metrics
from .metrics import calculate_cfs, temporal_consistency, novelty_index

__all__ = [
    # Core Architecture (Primary Interface)
    "CausalTorchCore",
    "CausalReasoningEngine", 
    "FromScratchModelBuilder",
    "PretrainedModelFineTuner",
    "InterventionAPI",
    "CounterfactualEngine",
    "CausalRegularization",
    "CausalAdapter",
    # Models (Built on Core)
    "CausalTransformer",
    "CausalLanguageModel",
    "SelfEvolvingTextGenerator",
    "FewShotCausalTransformer",
    "MultimodalCausalTransformer",
    "CounterfactualCausalTransformer",
    "CNSGImageGenerator",
    "CNSG_VideoGenerator",
    # Legacy aliases
    "cnsg",
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
    "EthicalTextFilter",
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
    # Training Infrastructure
    "CausalTrainer",
    "CausalFineTuner", 
    # MLOps Platform
    "CausalMLOps",
    "init_mlops",
    "MLOpsTrainer",
    # Metrics
    "calculate_cfs",
    "temporal_consistency",
    "novelty_index"
]