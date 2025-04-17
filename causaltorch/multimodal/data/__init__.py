"""
Multimodal Causal Data
=====================

This package provides data handling components for multimodal causal learning,
including datasets, preprocessing, and data augmentation utilities.
"""

# Import data components
from .dataset import MultimodalCausalDataset

# Define exported components
__all__ = [
    'MultimodalCausalDataset'
] 