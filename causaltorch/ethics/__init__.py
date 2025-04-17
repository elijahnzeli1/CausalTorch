"""
CausalTorch Ethics Package
=========================

This package implements ethical constraints and safeguards for AI models,
ensuring that generated outputs adhere to ethical principles.
"""

# Import directly from the constitution module
from .constitution import (
    EthicalConstitution,
    EthicalRule,
    EthicalTextFilter,
    EthicalLoss,
    load_default_ethical_rules
)

__all__ = [
    'EthicalConstitution',
    'EthicalRule',
    'EthicalTextFilter',
    'EthicalLoss',
    'load_default_ethical_rules'
] 