"""
Visualization utilities for causal graphs and attention weights.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns


def plot_causal_graph(
    graph: nx.DiGraph,
    figsize: Tuple[int, int] = (10, 8),
    show_weights: bool = True,
    node_color: str = 'skyblue',
    edge_color: str = 'gray',
    title: str = "Causal Graph",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualize a causal graph with customizable styling.
    
    Args:
        graph: NetworkX DiGraph representing the causal structure
        figsize: Figure size (width, height)
        show_weights: Whether to display edge weights
        node_color: Color for graph nodes
        edge_color: Color for graph edges
        title: Plot title
        ax: Optional matplotlib Axes object to draw on
        
    Returns:
        matplotlib Figure object
    """
    close_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        close_fig = True
    
    # Create position layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=2000, 
        node_color=node_color,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        arrowsize=20, 
        width=2,
        edge_color=edge_color,
        ax=ax
    )
    
    # Draw edge labels if requested
    if show_weights:
        edge_labels = {(u, v): f"{d.get('weight', 0.0):.2f}" 
                      for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=10,
            ax=ax
        )
    
    # Draw node labels
    nx.draw_networkx_labels(
        graph, pos,
        font_size=12,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    
    if close_fig:
        return fig
    return ax.figure


def visualize_attention(
    attention_weights: torch.Tensor,
    tokens: List[str],
    cause_indices: Optional[List[int]] = None,
    effect_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Visualize attention weights with highlighted causal relationships.
    
    Args:
        attention_weights: Tensor of attention weights [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        cause_indices: Optional list of indices for tokens that are causes
        effect_indices: Optional list of indices for tokens that are effects
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Average across batch and heads
    if len(attention_weights.shape) == 4:
        attention = attention_weights.mean(dim=(0, 1)).cpu().numpy()
    else:
        attention = attention_weights.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention,
        square=True,
        cmap="Blues",
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax
    )
    
    # Highlight cause-effect relationships if provided
    if cause_indices and effect_indices:
        # Add rectangles around cause-effect attention cells
        for c_idx in cause_indices:
            for e_idx in effect_indices:
                ax.add_patch(plt.Rectangle(
                    (e_idx, c_idx),
                    1, 1,
                    fill=False,
                    edgecolor='red',
                    lw=2
                ))
    
    ax.set_title("Attention Weights with Causal Relationships", fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_latent_traversal(
    model: torch.nn.Module,
    latent_dim: int,
    cause_idx: int,
    effect_idx: int,
    num_steps: int = 10,
    figsize: Tuple[int, int] = (12, 3),
) -> plt.Figure:
    """Visualize latent space traversal to show causal effects.
    
    Args:
        model: Generative model with decode method
        latent_dim: Dimension of latent space
        cause_idx: Index of causal variable in latent space
        effect_idx: Index of effect variable in latent space
        num_steps: Number of steps in traversal
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Create figure
    fig, axs = plt.subplots(1, num_steps, figsize=figsize)
    
    # Create latent traversal
    with torch.no_grad():
        for i, val in enumerate(np.linspace(0, 1, num_steps)):
            # Create latent vector
            z = torch.zeros(1, latent_dim)
            z[0, cause_idx] = val
            
            # Generate image
            img = model.decode(z)
            
            # Display
            if isinstance(img, torch.Tensor):
                if img.shape[1] == 1:  # Grayscale
                    axs[i].imshow(img[0, 0].cpu().numpy(), cmap='gray')
                else:  # RGB
                    img_np = img[0].permute(1, 2, 0).cpu().numpy()
                    # Clip to valid range
                    img_np = np.clip(img_np, 0, 1)
                    axs[i].imshow(img_np)
            
            axs[i].set_title(f"{val:.1f}")
            axs[i].axis('off')
    
    plt.suptitle(f"Traversal of Causal Variable {cause_idx}", fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_causal_intervention(
    model: torch.nn.Module,
    base_latent: torch.Tensor,
    intervention_dim: int,
    values: List[float],
    figsize: Tuple[int, int] = (12, 3),
) -> plt.Figure:
    """Visualize the effect of causal interventions.
    
    Args:
        model: Generative model with decode method
        base_latent: Base latent vector to modify
        intervention_dim: Dimension to intervene on
        values: List of values to set for the intervened dimension
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Create figure
    num_interventions = len(values)
    fig, axs = plt.subplots(1, num_interventions + 1, figsize=figsize)
    
    with torch.no_grad():
        # First, show baseline
        base_img = model.decode(base_latent)
        if isinstance(base_img, torch.Tensor):
            if base_img.shape[1] == 1:  # Grayscale
                axs[0].imshow(base_img[0, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                img_np = base_img[0].permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                axs[0].imshow(img_np)
        axs[0].set_title("Baseline")
        axs[0].axis('off')
        
        # Then show interventions
        for i, val in enumerate(values):
            # Clone and modify latent
            z = base_latent.clone()
            z[0, intervention_dim] = val
            
            # Generate image
            img = model.decode(z)
            
            # Display
            if isinstance(img, torch.Tensor):
                if img.shape[1] == 1:  # Grayscale
                    axs[i+1].imshow(img[0, 0].cpu().numpy(), cmap='gray')
                else:  # RGB
                    img_np = img[0].permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    axs[i+1].imshow(img_np)
            
            axs[i+1].set_title(f"do({intervention_dim})={val:.1f}")
            axs[i+1].axis('off')
    
    plt.suptitle(f"Causal Intervention on Variable {intervention_dim}", fontsize=14)
    plt.tight_layout()
    
    return fig