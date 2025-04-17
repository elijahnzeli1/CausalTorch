"""
CausalTorch: Causality-infused Deep Learning Library
===================================================

CausalTorch provides tools for integrating causal inference and reasoning with
deep learning models. It enables researchers and practitioners to build
models that respect causal relationships in data, perform interventions,
and generate counterfactual examples.

Main Components:
- `models`: Causal deep learning models with native causal reasoning
- `rules`: Tooling for specifying and enforcing causal rules
- `ethics`: Ethical constraints and governance for causal AI
- `federated`: Distributed causal learning components
- `creative`: Tools for causal creativity and counterfactual imagination
- `multimodal`: Multimodal causal learning across text, images, and more
"""

# Import submodules
from causaltorch import models
from causaltorch import rules
from causaltorch import ethics
from causaltorch import utils
from causaltorch import federated
from causaltorch import creative
from causaltorch import multimodal
from causaltorch import visualization
from causaltorch import layers

# Version information
__version__ = '2.0.2'

# Define what gets imported with "from causaltorch import *"
__all__ = [
    'models',
    'rules',
    'ethics',
    'utils',
    'federated',
    'creative',
    'multimodal',
    'visualization',
    'layers',
    '__version__'
]