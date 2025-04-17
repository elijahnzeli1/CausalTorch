"""
CausalTorch Visualization Package
================================

Visualization tools for causal graphs, model evaluation, and metrics.

This package provides plotting and visualization utilities for:
- Causal graphs and relationships
- Attention weights with causal highlighting
- Latent space traversals for causal variables
- Evaluation metrics and comparisons
"""

__version__ = "2.0.2"

# Import plot functions
from .plot import (
    plot_causal_graph,
    visualize_attention,
    plot_latent_traversal,
    plot_causal_intervention
)

# Import metrics visualization functions
from .metrics import (
    plot_cfs_comparison,
    plot_causal_effects_heatmap,
    plot_temporal_consistency,
    plot_novelty_distribution
)

__all__ = [
    # Plot functions
    "plot_causal_graph",
    "visualize_attention",
    "plot_latent_traversal",
    "plot_causal_intervention",
    
    # Metrics visualization
    "plot_cfs_comparison",
    "plot_causal_effects_heatmap",
    "plot_temporal_consistency",
    "plot_novelty_distribution"
]