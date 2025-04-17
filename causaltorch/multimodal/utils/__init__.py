"""
Multimodal Causal Utilities
==========================

This package provides utilities for working with multimodal causal models,
including visualization, metrics, training, and causal graph manipulation.
"""

# Import main utility components
from .metrics import (
    calculate_multimodal_causal_consistency,
    evaluate_counterfactual_quality,
    evaluate_model_causal_fidelity
)

from .causal_utils import (
    MultimodalCausalGraph,
    calculate_causal_consistency,
    apply_intervention,
    generate_synthetic_causal_graph
)

from .visualization import (
    visualize_causal_graph,
    visualize_attention_weights,
    visualize_counterfactual_comparison,
    display_multimodal_outputs,
    create_web_visualization
)

from .training import (
    MultimodalTrainer,
    prepare_dataloaders,
    load_checkpoint,
    GradualWarmupScheduler,
    log_hyperparameters
)

__all__ = [
    # Metrics
    'calculate_multimodal_causal_consistency',
    'evaluate_counterfactual_quality',
    'evaluate_model_causal_fidelity',
    
    # Causal utilities
    'MultimodalCausalGraph',
    'calculate_causal_consistency',
    'apply_intervention',
    'generate_synthetic_causal_graph',
    
    # Visualization
    'visualize_causal_graph',
    'visualize_attention_weights',
    'visualize_counterfactual_comparison',
    'display_multimodal_outputs',
    'create_web_visualization',
    
    # Training
    'MultimodalTrainer',
    'prepare_dataloaders',
    'load_checkpoint',
    'GradualWarmupScheduler',
    'log_hyperparameters'
] 