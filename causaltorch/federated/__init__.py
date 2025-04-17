"""
CausalTorch Federated Learning Package
=====================================

This package implements decentralized learning approaches for CausalTorch,
enabling distributed training while preserving causal knowledge.
"""

from .dao import CausalDAO, FederatedClient

__all__ = [
    'CausalDAO',
    'FederatedClient'
] 