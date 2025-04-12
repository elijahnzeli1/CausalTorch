"""
Visualization utilities for model evaluation and causal metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
import seaborn as sns # type: ignore


def plot_cfs_comparison(
    model_names: List[str],
    cfs_scores: List[float],
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    title: str = "Causal Fidelity Score Comparison",
) -> plt.Figure:
    """Plot a comparison of Causal Fidelity Scores between models.
    
    Args:
        model_names: List of model names
        cfs_scores: List of CFS scores (0-1 range)
        figsize: Figure size (width, height)
        colors: Optional list of colors for bars
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if len(model_names) != len(cfs_scores):
        raise ValueError("Length of model_names and cfs_scores must match")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        # Repeat colors if needed
        colors = (colors * ((len(model_names) // len(colors)) + 1))[:len(model_names)]
    
    # Create bar chart
    bars = ax.bar(model_names, cfs_scores, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Set labels and title
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Causal Fidelity Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Add a dashed horizontal line at y=1.0 (perfect score)
    ax.axhline(y=1.0, linestyle='--', alpha=0.3, color='gray')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_causal_effects_heatmap(
    causes: List[str],
    effects: List[str],
    effect_strengths: List[List[float]],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd",
    title: str = "Causal Effect Strengths",
) -> plt.Figure:
    """Create a heatmap visualizing the strengths of causal effects.
    
    Args:
        causes: List of cause variable names
        effects: List of effect variable names
        effect_strengths: 2D list of effect strengths
        figsize: Figure size (width, height)
        cmap: Colormap name
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if len(causes) != len(effect_strengths):
        raise ValueError("Length of causes must match first dimension of effect_strengths")
    
    if len(effects) != len(effect_strengths[0]):
        raise ValueError("Length of effects must match second dimension of effect_strengths")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        effect_strengths,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=effects,
        yticklabels=causes,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Effects", fontsize=12)
    ax.set_ylabel("Causes", fontsize=12)
    
    plt.tight_layout()
    
    return fig


def plot_temporal_consistency(
    frame_consistency_scores: List[float],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot temporal consistency scores across video frames.
    
    Args:
        frame_consistency_scores: List of consistency scores between consecutive frames
        labels: Optional list of frame labels
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(frame_consistency_scores))
    ax.plot(x, frame_consistency_scores, 'o-', linewidth=2, markersize=8, color='#3498db')
    
    # Fill area under the curve
    ax.fill_between(x, 0, frame_consistency_scores, alpha=0.2, color='#3498db')
    
    # Set x-tick labels if provided
    if labels:
        if len(labels) == len(frame_consistency_scores) + 1:
            # If labels represent frame pairs
            pair_labels = [f"{labels[i]}-{labels[i+1]}" for i in range(len(labels)-1)]
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Set labels and title
    ax.set_xlabel('Frame Pairs', fontsize=12)
    ax.set_ylabel('Temporal Consistency Score', fontsize=12)
    ax.set_title('Temporal Consistency Across Video Sequence', fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add a dashed horizontal line at y=1.0 (perfect consistency)
    ax.axhline(y=1.0, linestyle='--', alpha=0.3, color='gray')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_novelty_distribution(
    generated_samples: torch.Tensor,
    training_samples: torch.Tensor,
    feature_extractor: callable,
    figsize: Tuple[int, int] = (10, 6),
    n_bins: int = 30,
) -> plt.Figure:
    """Plot the distribution of novelty scores for generated samples.
    
    Args:
        generated_samples: Tensor of generated samples
        training_samples: Tensor of training samples
        feature_extractor: Function to extract features from samples
        figsize: Figure size (width, height)
        n_bins: Number of histogram bins
        
    Returns:
        matplotlib Figure object
    """
    # Extract features
    with torch.no_grad():
        gen_features = feature_extractor(generated_samples).cpu().numpy()
        train_features = feature_extractor(training_samples).cpu().numpy()
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = []
    for gen in gen_features:
        # Get minimum distance to any training sample (closest neighbor)
        min_dist = cosine_distances([gen], train_features).min()
        distances.append(min_dist)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    ax.hist(distances, bins=n_bins, alpha=0.7, color='#3498db')
    
    # Add vertical line for mean
    mean_dist = np.mean(distances)
    ax.axvline(mean_dist, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean Novelty: {mean_dist:.3f}')
    
    # Set labels and title
    ax.set_xlabel('Novelty Score (Distance to Nearest Training Sample)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Novelty Scores for Generated Samples', fontsize=14)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig