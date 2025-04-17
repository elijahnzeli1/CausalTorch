"""
Visualization Utilities for Multimodal Causal Models
===================================================

This module provides visualization tools for multimodal causal models,
including causal graph visualization, attention map visualization,
and counterfactual comparison visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple, Union, Optional


def visualize_causal_graph(
    causal_rules,
    modality_markers=True,
    highlight_interventions=None,
    node_size=1000,
    figsize=(10, 8),
    save_path=None
):
    """Visualize a causal graph based on causal rules.
    
    Args:
        causal_rules (dict): Dictionary of causal rules
        modality_markers (bool): Whether to use different colors for different modalities
        highlight_interventions (dict): Dictionary of interventions to highlight
        node_size (int): Size of nodes in the graph
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Standardize causal rules if they're in a non-standard format
    from causaltorch.multimodal.utils.metrics import _standardize_causal_graph, _extract_effect_info
    standardized_rules = _standardize_causal_graph(causal_rules)
    
    # Add nodes and edges
    for cause, effects in standardized_rules.items():
        # If cause is a string representation of a complex object, clean it up for display
        if isinstance(cause, str) and cause.startswith("<") and "object at" in cause:
            display_cause = cause.split()[0].strip("<>")
        else:
            display_cause = str(cause)
            
        # Add cause node if it doesn't exist
        if not G.has_node(display_cause):
            G.add_node(display_cause)
        
        # Add effect nodes and edges
        for effect_data in effects:
            effect, strength = _extract_effect_info(effect_data)
            
            # Clean up effect name for display
            if isinstance(effect, str) and effect.startswith("<") and "object at" in effect:
                display_effect = effect.split()[0].strip("<>")
            else:
                display_effect = str(effect)
                
            # Add effect node if it doesn't exist
            if not G.has_node(display_effect):
                G.add_node(display_effect)
                
            # Add edge with weight attribute
            G.add_edge(display_cause, display_effect, weight=strength)
    
    # Set up the figure
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    
    # Define node colors based on modality
    if modality_markers:
        from causaltorch.multimodal.utils.metrics import _get_variable_modality
        
        node_colors = []
        for node in G.nodes():
            modality = _get_variable_modality(node)
            if modality == "text":
                node_colors.append("skyblue")
            elif modality == "image":
                node_colors.append("salmon")
            elif modality == "audio":
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightgray")
    else:
        node_colors = ["lightblue" for _ in G.nodes()]
        
    # Highlight intervention nodes if specified
    if highlight_interventions:
        for i, node in enumerate(G.nodes()):
            for intervention_var, _ in highlight_interventions.items():
                if str(node) == str(intervention_var):
                    node_colors[i] = "yellow"
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.8,
        edgecolors='black'
    )
    
    # Draw edges with width based on weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        alpha=0.6,
        edge_color='gray',
        arrowsize=20
    )
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add a legend
    if modality_markers:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Text'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=15, label='Image'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Audio'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, label='Other')
        ]
        if highlight_interventions:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Intervention')
            )
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.title('Causal Graph Visualization')
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()


def visualize_attention_weights(
    attention_weights,
    source_tokens=None,
    target_tokens=None,
    figsize=(10, 8),
    cmap='viridis',
    save_path=None
):
    """Visualize attention weights between source and target.
    
    Args:
        attention_weights (torch.Tensor): Attention weights [source_len, target_len]
        source_tokens (list): List of source tokens
        target_tokens (list): List of target tokens
        figsize (tuple): Figure size
        cmap (str): Colormap for heatmap
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Create x and y ticks
    if source_tokens is None:
        source_tokens = [f"S{i}" for i in range(attention_weights.shape[0])]
    if target_tokens is None:
        target_tokens = [f"T{i}" for i in range(attention_weights.shape[1])]
    
    # Plot heatmap
    plt.imshow(attention_weights, cmap=cmap, aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    # Set labels
    plt.xticks(range(len(target_tokens)), target_tokens, rotation=90)
    plt.yticks(range(len(source_tokens)), source_tokens)
    
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.title('Attention Weights')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()


def visualize_counterfactual_comparison(
    original,
    counterfactual,
    intervention,
    figsize=(15, 10),
    save_path=None
):
    """Create a visual comparison between original and counterfactual outputs.
    
    Args:
        original (dict): Original outputs with 'text' and/or 'image' keys
        counterfactual (dict): Counterfactual outputs
        intervention (dict): The intervention that was applied
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplots based on modalities present
    modalities = set(original.keys()).union(counterfactual.keys())
    
    # Title describing the intervention
    intervention_str = f"Intervention: {intervention['variable']} = {intervention['value']}"
    plt.suptitle(intervention_str, fontsize=16)
    
    # Set up row layouts
    num_rows = len(modalities)
    row = 1
    
    # Process each modality
    for modality in sorted(modalities):
        if modality == "text":
            # Handle text modality
            has_original = modality in original
            has_counterfactual = modality in counterfactual
            
            if has_original and has_counterfactual:
                ax = plt.subplot(num_rows, 2, (row - 1) * 2 + 1)
                ax.text(0.5, 0.5, original[modality], 
                        wrap=True, horizontalalignment='center', 
                        verticalalignment='center', fontsize=12)
                ax.set_title("Original Text")
                ax.axis('off')
                
                ax = plt.subplot(num_rows, 2, (row - 1) * 2 + 2)
                ax.text(0.5, 0.5, counterfactual[modality], 
                        wrap=True, horizontalalignment='center', 
                        verticalalignment='center', fontsize=12)
                ax.set_title("Counterfactual Text")
                ax.axis('off')
            elif has_original:
                ax = plt.subplot(num_rows, 1, row)
                ax.text(0.5, 0.5, original[modality], 
                        wrap=True, horizontalalignment='center', 
                        verticalalignment='center', fontsize=12)
                ax.set_title("Original Text (No Counterfactual)")
                ax.axis('off')
            elif has_counterfactual:
                ax = plt.subplot(num_rows, 1, row)
                ax.text(0.5, 0.5, counterfactual[modality], 
                        wrap=True, horizontalalignment='center', 
                        verticalalignment='center', fontsize=12)
                ax.set_title("Counterfactual Text (No Original)")
                ax.axis('off')
                
        elif modality == "image":
            # Handle image modality
            has_original = modality in original
            has_counterfactual = modality in counterfactual
            
            if has_original and has_counterfactual:
                # Original image
                ax = plt.subplot(num_rows, 2, (row - 1) * 2 + 1)
                img = _tensor_to_image(original[modality])
                ax.imshow(img)
                ax.set_title("Original Image")
                ax.axis('off')
                
                # Counterfactual image
                ax = plt.subplot(num_rows, 2, (row - 1) * 2 + 2)
                img = _tensor_to_image(counterfactual[modality])
                ax.imshow(img)
                ax.set_title("Counterfactual Image")
                ax.axis('off')
            elif has_original:
                ax = plt.subplot(num_rows, 1, row)
                img = _tensor_to_image(original[modality])
                ax.imshow(img)
                ax.set_title("Original Image (No Counterfactual)")
                ax.axis('off')
            elif has_counterfactual:
                ax = plt.subplot(num_rows, 1, row)
                img = _tensor_to_image(counterfactual[modality])
                ax.imshow(img)
                ax.set_title("Counterfactual Image (No Original)")
                ax.axis('off')
                
        elif modality == "audio":
            # Placeholder for future audio visualization
            ax = plt.subplot(num_rows, 1, row)
            ax.text(0.5, 0.5, "Audio visualization not implemented yet", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"{modality.capitalize()} Modality")
            ax.axis('off')
            
        else:
            # Generic handler for other modalities
            ax = plt.subplot(num_rows, 1, row)
            ax.text(0.5, 0.5, f"{modality} representation not visualizable", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"{modality.capitalize()} Modality")
            ax.axis('off')
        
        row += 1
    
    # Add a separator line between rows
    for i in range(1, num_rows):
        fig.add_subplot(num_rows, 1, i)
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def _tensor_to_image(tensor):
    """Convert a tensor to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Image tensor
        
    Returns:
        PIL.Image: The converted image
    """
    if isinstance(tensor, torch.Tensor):
        # Ensure tensor is on CPU and detached from computation graph
        tensor = tensor.detach().cpu()
        
        # Handle different tensor shapes
        if tensor.dim() == 4 and tensor.size(0) == 1:  # [1, C, H, W]
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:  # [C, H, W]
            if tensor.shape[0] == 1:  # Grayscale
                tensor = tensor.repeat(3, 1, 1)  # Convert to RGB
            
            # Normalize if needed
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
                
            # Rearrange to [H, W, C] for plotting
            img = tensor.permute(1, 2, 0).numpy()
            
            # Clip values to valid range
            img = np.clip(img, 0, 1)
            
            return img
        else:
            # If not a standard image tensor, return a placeholder
            return np.zeros((100, 100, 3))
    elif isinstance(tensor, np.ndarray):
        if tensor.ndim == 3:
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # [C, H, W]
                # Transpose to [H, W, C]
                tensor = np.transpose(tensor, (1, 2, 0))
            
            # Ensure RGB
            if tensor.shape[2] == 1:
                tensor = np.repeat(tensor, 3, axis=2)
                
            # Normalize if needed
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
                
            return np.clip(tensor, 0, 1)
        else:
            return tensor
    else:
        # If not a tensor or array, return a placeholder
        return np.zeros((100, 100, 3))


def create_web_visualization(
    model_outputs,
    causal_rules,
    interventions=None,
    include_graph=True,
    include_counterfactuals=True
):
    """Create interactive HTML visualization for web display.
    
    Args:
        model_outputs (dict): Model outputs for different modalities
        causal_rules (dict): Causal rules
        interventions (list): List of interventions
        include_graph (bool): Whether to include causal graph visualization
        include_counterfactuals (bool): Whether to include counterfactual visualization
        
    Returns:
        str: HTML string for display
    """
    html_parts = [
        """<!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Causal Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .flex-item { flex: 1; min-width: 300px; }
                h1, h2, h3 { color: #333; }
                img { max-width: 100%; height: auto; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
                .caption { font-style: italic; color: #666; margin-top: 5px; }
                button { background-color: #4CAF50; color: white; padding: 10px 15px; 
                        border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
                button:hover { background-color: #45a049; }
                .intervention-panel { background-color: #f0f8ff; padding: 15px; margin-top: 15px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multimodal Causal Model Visualization</h1>
        """
    ]
    
    # Add causal graph visualization if requested
    if include_graph and causal_rules:
        try:
            # Create causal graph figure
            fig = visualize_causal_graph(causal_rules, highlight_interventions=interventions)
            
            # Convert figure to base64 image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            html_parts.append(
                f"""
                <div class="section">
                    <h2>Causal Graph</h2>
                    <div class="flex-container">
                        <div class="flex-item">
                            <img src="data:image/png;base64,{img_str}" alt="Causal Graph">
                            <div class="caption">Visualization of causal relationships between variables.</div>
                        </div>
                    </div>
                </div>
                """
            )
        except Exception as e:
            html_parts.append(
                f"""
                <div class="section">
                    <h2>Causal Graph</h2>
                    <p>Error generating causal graph visualization: {str(e)}</p>
                </div>
                """
            )
    
    # Add model outputs visualization
    html_parts.append('<div class="section"><h2>Model Outputs</h2><div class="flex-container">')
    
    # Process each modality
    for modality, output in model_outputs.items():
        if modality == "text":
            # Handle text output
            html_parts.append(
                f"""
                <div class="flex-item">
                    <h3>Text Output</h3>
                    <pre>{output}</pre>
                </div>
                """
            )
        elif modality == "image" and isinstance(output, (torch.Tensor, np.ndarray)):
            try:
                # Convert image tensor to base64
                fig = plt.figure(figsize=(5, 5))
                plt.imshow(_tensor_to_image(output))
                plt.axis('off')
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                html_parts.append(
                    f"""
                    <div class="flex-item">
                        <h3>Image Output</h3>
                        <img src="data:image/png;base64,{img_str}" alt="Model Generated Image">
                    </div>
                    """
                )
            except Exception as e:
                html_parts.append(
                    f"""
                    <div class="flex-item">
                        <h3>Image Output</h3>
                        <p>Error rendering image: {str(e)}</p>
                    </div>
                    """
                )
        else:
            # Generic handler for other modalities
            html_parts.append(
                f"""
                <div class="flex-item">
                    <h3>{modality.capitalize()} Output</h3>
                    <p>Output type: {type(output).__name__}</p>
                    <pre>{str(output)[:500]}{'...' if len(str(output)) > 500 else ''}</pre>
                </div>
                """
            )
    
    html_parts.append('</div></div>')  # Close model outputs section
    
    # Add counterfactual visualizations if requested
    if include_counterfactuals and interventions:
        html_parts.append('<div class="section"><h2>Counterfactual Scenarios</h2>')
        
        # Add interactive buttons for each intervention
        html_parts.append('<div class="intervention-buttons">')
        html_parts.append('<p>Select an intervention to visualize:</p>')
        
        for i, intervention in enumerate(interventions):
            var_name = intervention.get('variable', f'Intervention {i+1}')
            var_value = intervention.get('value', '')
            button_text = f"{var_name} = {var_value}"
            html_parts.append(f'<button onclick="showIntervention({i})">{button_text}</button>')
        
        html_parts.append('</div>')
        
        # Add placeholders for each intervention visualization
        for i, intervention in enumerate(interventions):
            display = 'block' if i == 0 else 'none'
            html_parts.append(f'<div id="intervention-{i}" class="intervention-panel" style="display: {display};">')
            
            # Add intervention details
            var_name = intervention.get('variable', 'Unknown')
            var_value = intervention.get('value', 'Unknown')
            html_parts.append(f'<h3>Intervention: {var_name} = {var_value}</h3>')
            
            # Add placeholder for counterfactual visualization
            html_parts.append('<p>Counterfactual visualization would be displayed here.</p>')
            html_parts.append(f'<pre>{intervention}</pre>')
            
            html_parts.append('</div>')
        
        # Add JavaScript for interaction
        html_parts.append(
            """
            <script>
                function showIntervention(index) {
                    // Hide all intervention panels
                    const panels = document.querySelectorAll('.intervention-panel');
                    panels.forEach(panel => {
                        panel.style.display = 'none';
                    });
                    
                    // Show the selected panel
                    document.getElementById(`intervention-${index}`).style.display = 'block';
                }
            </script>
            """
        )
        
        html_parts.append('</div>')  # Close counterfactuals section
    
    # Close the HTML document
    html_parts.append('</div></body></html>')
    
    return '\n'.join(html_parts)


def display_multimodal_outputs(
    text_output,
    image_output,
    figsize=(12, 6),
    max_text_len=500,
    save_path=None
):
    """Display text and image outputs side by side.
    
    Args:
        text_output (str): Text output from the model
        image_output (torch.Tensor or numpy.ndarray): Image output
        figsize (tuple): Figure size
        max_text_len (int): Maximum text length to display
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display image
    if image_output is not None:
        try:
            img = _tensor_to_image(image_output)
            ax1.imshow(img)
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error displaying image:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
    else:
        ax1.text(0.5, 0.5, "No image output", 
                horizontalalignment='center', verticalalignment='center')
    
    ax1.set_title("Image Output")
    ax1.axis('off')
    
    # Display text
    if text_output:
        # Truncate if needed
        display_text = text_output
        if len(text_output) > max_text_len:
            display_text = text_output[:max_text_len] + "..."
        
        ax2.text(0.05, 0.95, display_text, 
                verticalalignment='top', wrap=True, 
                fontsize=12)
    else:
        ax2.text(0.5, 0.5, "No text output", 
                horizontalalignment='center', verticalalignment='center')
    
    ax2.set_title("Text Output")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig 