"""
CausalTorch Metrics Module
==========================

This module contains evaluation metrics for causal generative models,
including the Causal Fidelity Score (CFS) for measuring adherence to causal rules.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Callable


def calculate_cfs(model: torch.nn.Module,
                  test_cases: List[Tuple],
                  effect_checker: Callable = None) -> float:
    """Calculate the Causal Fidelity Score for a model.
    
    The CFS measures how often the model's outputs adhere to causal rules
    when given inputs containing specific causal factors.
    
    Args:
        model: The causal generative model to evaluate
        test_cases: List of (input, expected_effect) tuples
        effect_checker: Optional function to check if effect is present
            If None, will use simple substring matching
            
    Returns:
        float: Causal Fidelity Score (0-1, higher is better)
    """
    correct = 0
    total = len(test_cases)
    
    # Default effect checker for text
    def default_effect_checker(output, expected_effect):
        if isinstance(output, str) and isinstance(expected_effect, str):
            return expected_effect.lower() in output.lower()
    
    # Use the provided effect_checker or the default one
    if effect_checker is None:
        effect_checker = default_effect_checker
    
    for input_data, expected_effect in test_cases:
        # Generate output
        with torch.no_grad():
            output = model.generate(input_data)
        
        # Check if effect is present in output
        if effect_checker(output, expected_effect):
            correct += 1
    
    return correct / total


def calculate_image_cfs(model: torch.nn.Module,
                        causal_rules: Dict,
                        num_samples: int = 50) -> float:
    """Calculate the Causal Fidelity Score for an image generation model.
    
    Tests if generated images adhere to causal rules like "rain → wet ground".
    
    Args:
        model: The image generation model
        causal_rules: Dictionary of causal rules to check
        num_samples: Number of test samples to generate
        
    Returns:
        float: Causal Fidelity Score (0-1, higher is better)
    """
    correct = 0
    total = 0
    
    # Generate images with varying rain intensities
    rain_intensities = np.linspace(0, 1, num_samples)
    
    for rain in rain_intensities:
        # Generate image
        with torch.no_grad():
            image = model.generate(rain_intensity=rain)
        
        # For the rain → wet ground rule
        if "rain" in causal_rules:
            total += 1
            
            # Calculate ground wetness (assume ground is in bottom rows)
            ground_wetness = 1.0 - image[0, 0, -8:, :].mean().item()  # Invert: darker = wetter
            
            # Check if causal rule is satisfied
            threshold = causal_rules["rain"].get("threshold", 0.5)
            if (rain > threshold and ground_wetness > 0.5) or (rain <= threshold and ground_wetness <= 0.7):
                correct += 1
    
    return correct / total if total > 0 else 0.0


def temporal_consistency(frames: torch.Tensor) -> float:
    """Measure frame-to-frame consistency in a video sequence.
    
    Higher values indicate smoother transitions between frames.
    
    Args:
        frames: Video tensor [batch, frames, channels, height, width]
        
    Returns:
        float: Temporal consistency score (0-1, higher is better)
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError("frames must be a torch.Tensor")
    
    if frames.dim() < 5:  # [batch, frames, channels, height, width]
        raise ValueError("frames must have 5 dimensions")
    
    # Calculate frame differences
    frame_diffs = torch.abs(frames[:, 1:] - frames[:, :-1]).mean(dim=(2, 3, 4))
    
    # Normalize to 0-1 range and invert (higher = more consistent)
    max_diff = frames.max() - frames.min()
    if max_diff > 0:
        consistency = 1.0 - (frame_diffs / max_diff).mean().item()
    else:
        consistency = 1.0
    
    return consistency


def novelty_index(generated: torch.Tensor,
                  training_data: torch.Tensor,
                  similarity_fn: Callable = None) -> float:
    """Calculate how novel the generated samples are compared to training data.
    
    Higher values indicate outputs that are more different from training data.
    
    Args:
        generated: Generated samples
        training_data: Training samples
        similarity_fn: Function to calculate similarity between samples
            If None, uses cosine similarity
            
    Returns:
        float: Novelty index (0-1, higher = more novel)
    """
    if not isinstance(generated, torch.Tensor) or not isinstance(training_data, torch.Tensor):
        raise TypeError("Both generated and training_data must be torch.Tensor")
    
    # Default similarity function
    if similarity_fn is None:
        def similarity_fn(a, b):
            # Flatten tensors
            a_flat = a.view(a.size(0), -1)
            b_flat = b.view(b.size(0), -1)
            
            # Normalize
            a_norm = torch.nn.functional.normalize(a_flat, p=2, dim=1)
            b_norm = torch.nn.functional.normalize(b_flat, p=2, dim=1)
            
            # Calculate cosine similarity
            sim_matrix = torch.mm(a_norm, b_norm.t())
            return sim_matrix.max(dim=1)[0]
    
    # Calculate similarities between generated and training samples
    similarities = similarity_fn(generated, training_data)
    
    # Novelty = 1 - similarity
    return 1.0 - similarities.mean().item()


def causal_intervention_score(model: torch.nn.Module,
                              intervention_fn: Callable,
                              expected_effect_fn: Callable,
                              num_samples: int = 10) -> float:
    """Measure how well a model responds to causal interventions.
    
    Tests if intervening on a cause produces the expected effect.
    
    Args:
        model: The causal model to evaluate
        intervention_fn: Function that performs interventions on latent variables
        expected_effect_fn: Function that checks if the effect matches expectations
        num_samples: Number of test interventions
        
    Returns:
        float: Intervention score (0-1, higher is better)
    """
    correct = 0
    
    for _ in range(num_samples):
        # Generate a random latent vector
        z = torch.randn(1, model.latent_dim)
        
        # Generate output without intervention
        with torch.no_grad():
            base_output = model.decode(z.clone())
        
        # Apply intervention to latent variables
        z_intervention = intervention_fn(z.clone())
        
        # Generate output with intervention
        with torch.no_grad():
            intervention_output = model.decode(z_intervention)
        
        # Check if effect matches expectations
        if expected_effect_fn(base_output, intervention_output, z, z_intervention):
            correct += 1
    
    return correct / num_samples 