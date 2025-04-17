"""
Metrics for Multimodal Causal Models
==================================

This module provides metrics for evaluating multimodal causal generative models.
"""

import torch
import numpy as np
from causaltorch.utils.metrics import calculate_causal_fidelity_score


def calculate_multimodal_causal_consistency(
    text_features,
    image_features,
    causal_graphs,
    model_outputs
):
    """Calculate causal consistency across modalities.
    
    This measures how well the model maintains causal relationships
    between different modalities.
    
    Args:
        text_features (torch.Tensor): Text features [batch_size, dim]
        image_features (torch.Tensor): Image features [batch_size, dim]
        causal_graphs: Causal relationships (can be dict, list, or custom object)
        model_outputs (dict): Dictionary of model outputs for different modalities
        
    Returns:
        float: Causal consistency score (0-1, higher is better)
    """
    # Validate inputs
    if not all(isinstance(m, torch.Tensor) for m in [text_features, image_features]):
        print(f"Warning: features should be tensors. Returning default consistency score.")
        return 0.5
    
    if not model_outputs or not isinstance(model_outputs, dict):
        print(f"Warning: model_outputs should be a non-empty dictionary. Returning default consistency score.")
        return 0.5
    
    # Convert causal_graphs to a standardized format if needed
    causal_rules = _standardize_causal_graph(causal_graphs)
    
    batch_size = text_features.size(0)
    consistency_scores = []
    
    # Check each causal rule
    for cause_key, effects in causal_rules.items():
        # Extract cause variable
        cause_var = _extract_cause_var(cause_key, effects)
            
        # Process each effect
        for effect in effects:
            # Extract effect variable and strength
            effect_var, strength = _extract_effect_info(effect)
            
            # Determine which modality contains the cause and effect
            cause_modality = _get_variable_modality(cause_var)
            effect_modality = _get_variable_modality(effect_var)
            
            # Skip if modality not present
            if cause_modality not in model_outputs or effect_modality not in model_outputs:
                continue
            
            # Extract cause representation
            if cause_modality == "text":
                cause_repr = text_features
            else:
                cause_repr = image_features
            
            # Extract effect representation
            effect_repr = model_outputs[effect_modality]
            
            # Calculate correlation between cause and effect
            # This is a simplified measure and can be replaced with more sophisticated metrics
            try:
                # Flatten effect representation if needed
                if effect_repr.dim() > 2:
                    effect_repr = effect_repr.view(batch_size, -1)
                
                # Calculate consistency score using cosine similarity
                # between cause representation and effect representation
                norm_cause = torch.norm(cause_repr, dim=1, keepdim=True)
                norm_effect = torch.norm(effect_repr, dim=1, keepdim=True)
                
                # Avoid division by zero
                valid_samples = (norm_cause > 1e-8) & (norm_effect > 1e-8)
                if valid_samples.sum() == 0:
                    continue
                
                # Calculate cosine similarity for valid samples
                cos_sim = torch.zeros(batch_size, device=cause_repr.device)
                valid_indices = valid_samples.squeeze().nonzero(as_tuple=True)[0]
                
                if len(valid_indices) > 0:
                    valid_cause = cause_repr[valid_indices]
                    valid_effect = effect_repr[valid_indices]
                    valid_norm_cause = norm_cause[valid_indices]
                    valid_norm_effect = norm_effect[valid_indices]
                    
                    valid_cos_sim = (valid_cause * valid_effect).sum(dim=1) / (valid_norm_cause * valid_norm_effect).squeeze()
                    cos_sim[valid_indices] = valid_cos_sim
                
                # Scale by causal strength and convert to [0, 1] range
                consistency = (cos_sim * strength + 1) / 2
                
                # Average across batch
                avg_consistency = consistency.mean().item()
                consistency_scores.append(avg_consistency)
                
            except Exception as e:
                print(f"Error calculating consistency for {cause_var} -> {effect_var}: {e}")
    
    # Return average consistency across all rules, or default if no scores
    if consistency_scores:
        return sum(consistency_scores) / len(consistency_scores)
    else:
        return 0.5


def _standardize_causal_graph(causal_graphs):
    """Convert various causal graph formats to a standardized dictionary.
    
    Args:
        causal_graphs: Causal graph in various formats
        
    Returns:
        dict: Standardized causal rules
    """
    causal_rules = {}
    
    # Handle dictionary format
    if isinstance(causal_graphs, dict):
        # Special case for 'causal_relations' key
        if 'causal_relations' in causal_graphs:
            causal_relations = causal_graphs['causal_relations']
            
            if isinstance(causal_relations, list):
                # Process list of relations
                for relation in causal_relations:
                    if isinstance(relation, dict) and 'cause' in relation and 'effect' in relation:
                        cause = relation['cause']
                        effect = relation['effect']
                        strength = relation.get('strength', 0.8)
                        
                        # Convert unhashable types to string
                        cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                        
                        if cause_key not in causal_rules:
                            causal_rules[cause_key] = []
                        causal_rules[cause_key].append({
                            "effect": effect,
                            "strength": strength,
                            "original_cause": cause
                        })
            elif isinstance(causal_relations, dict):
                # Process dictionary of cause-effect pairs
                for cause, effects in causal_relations.items():
                    cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                    
                    causal_rules[cause_key] = []
                    if isinstance(effects, list):
                        for effect in effects:
                            causal_rules[cause_key].append({
                                "effect": effect,
                                "strength": 0.8,
                                "original_cause": cause
                            })
                    else:
                        causal_rules[cause_key].append({
                            "effect": effects,
                            "strength": 0.8,
                            "original_cause": cause
                        })
        else:
            # Direct dictionary
            for cause, effects in causal_graphs.items():
                cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                causal_rules[cause_key] = effects
    
    # Handle CausalRuleSet-like objects
    elif hasattr(causal_graphs, 'rules'):
        if isinstance(causal_graphs.rules, list):
            # Convert list of rule objects
            for rule in causal_graphs.rules:
                if hasattr(rule, 'cause') and hasattr(rule, 'effect'):
                    cause = rule.cause
                    cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                    
                    if cause_key not in causal_rules:
                        causal_rules[cause_key] = []
                    causal_rules[cause_key].append({
                        "effect": rule.effect,
                        "strength": getattr(rule, 'strength', 0.8),
                        "original_cause": cause
                    })
        else:
            # Dictionary of rules
            for cause, effects in causal_graphs.rules.items():
                cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                causal_rules[cause_key] = effects
    
    # Handle list format
    elif isinstance(causal_graphs, list):
        for rule in causal_graphs:
            if isinstance(rule, dict) and 'cause' in rule and 'effect' in rule:
                cause = rule['cause']
                cause_key = str(cause) if not isinstance(cause, (str, int, float, bool, tuple)) else cause
                
                if cause_key not in causal_rules:
                    causal_rules[cause_key] = []
                causal_rules[cause_key].append({
                    "effect": rule['effect'],
                    "strength": rule.get('strength', 0.8),
                    "original_cause": cause
                })
    
    return causal_rules


def _extract_cause_var(cause_key, effects):
    """Extract the original cause variable from effects if available."""
    if isinstance(effects, list) and effects and isinstance(effects[0], dict):
        if "original_cause" in effects[0]:
            return effects[0]["original_cause"]
    return cause_key


def _extract_effect_info(effect):
    """Extract effect variable and strength from effect data."""
    if isinstance(effect, dict) and "effect" in effect:
        effect_var = effect["effect"]
        strength = effect.get("strength", 0.8)
    elif hasattr(effect, 'effect'):
        effect_var = effect.effect
        strength = getattr(effect, 'strength', 0.8)
    else:
        effect_var = effect
        strength = 0.8
    return effect_var, strength


def _get_variable_modality(variable_name):
    """Determine which modality a variable belongs to based on naming convention.
    
    Args:
        variable_name (str): Name of the variable
    
    Returns:
        str: Modality name ('text', 'image', or 'unknown')
    """
    var_str = str(variable_name).lower()
    
    # Check for text indicators
    if var_str.startswith('text_') or var_str.startswith('t_') or 'text' in var_str or 'word' in var_str:
        return "text"
    
    # Check for image indicators
    if var_str.startswith('img_') or var_str.startswith('image_') or 'visual' in var_str:
        return "image"
    
    # Check for audio indicators (for future expansion)
    if var_str.startswith('audio_') or var_str.startswith('sound_'):
        return "audio"
    
    # Default to unknown modality
    return "unknown"


def evaluate_counterfactual_quality(
    original_outputs,
    counterfactual_outputs,
    intervention,
    causal_rules
):
    """Evaluate the quality of counterfactual outputs.
    
    Args:
        original_outputs (dict): Original model outputs
        counterfactual_outputs (dict): Counterfactual outputs after intervention
        intervention (dict): The intervention that was applied
        causal_rules (dict): Causal rules defining relationships
        
    Returns:
        dict: Quality metrics including:
            - intervention_effect: How strongly the intervention affected the target
            - counterfactual_plausibility: How plausible the counterfactual is
            - minimum_change: How minimal the changes are (higher is better)
    """
    # Calculate intervention effect
    intervention_effect = _calculate_intervention_effect(
        original_outputs,
        counterfactual_outputs,
        intervention["variable"],
        intervention["value"],
        causal_rules
    )
    
    # Calculate counterfactual plausibility
    counterfactual_plausibility = _calculate_counterfactual_plausibility(
        counterfactual_outputs,
        causal_rules
    )
    
    # Calculate minimum change
    minimum_change = _calculate_minimum_change(
        original_outputs,
        counterfactual_outputs,
        intervention["variable"]
    )
    
    return {
        "intervention_effect": intervention_effect,
        "counterfactual_plausibility": counterfactual_plausibility,
        "minimum_change": minimum_change,
        "overall_quality": (intervention_effect + counterfactual_plausibility + minimum_change) / 3
    }


def _calculate_intervention_effect(original, counterfactual, intervention_var, intervention_val, causal_rules):
    """Calculate how strongly the intervention affected the output.
    
    Args:
        original (dict): Original outputs
        counterfactual (dict): Counterfactual outputs
        intervention_var (str): Variable that was intervened on
        intervention_val (float): Value set for the intervention
        causal_rules (dict): Causal rules
        
    Returns:
        float: Intervention effect score (0-1)
    """
    # This is a simplified implementation - in practice you'd use more sophisticated
    # methods to measure the effect of intervention
    effect_score = 0.0
    num_effects = 0
    
    # Get the modality of the intervention
    intervention_modality = _get_variable_modality(intervention_var)
    
    # If we have the intervened modality in both outputs
    if intervention_modality in original and intervention_modality in counterfactual:
        # Calculate difference between original and counterfactual
        orig_tensor = original[intervention_modality]
        cf_tensor = counterfactual[intervention_modality]
        
        # If tensors are different shapes, skip this comparison
        if orig_tensor.shape != cf_tensor.shape:
            return 0.5  # Default middle score
        
        # Calculate normalized difference
        diff = torch.norm(cf_tensor - orig_tensor) / torch.norm(orig_tensor)
        
        # Scale to [0, 1] with sigmoid
        import math
        effect_score = 1 / (1 + math.exp(-diff.item() * 5))
        num_effects = 1
    
    # If no effects were measured, return default score
    if num_effects == 0:
        return 0.5
    
    return effect_score


def _calculate_counterfactual_plausibility(counterfactual, causal_rules):
    """Calculate how plausible the counterfactual is according to causal rules.
    
    Args:
        counterfactual (dict): Counterfactual outputs
        causal_rules (dict): Causal rules
        
    Returns:
        float: Plausibility score (0-1)
    """
    # Simplified implementation returning a reasonable default
    # In a full implementation, you would check if the counterfactual
    # respects known causal relationships
    return 0.8


def _calculate_minimum_change(original, counterfactual, intervention_var):
    """Calculate how minimal the changes are (higher score is better).
    
    Args:
        original (dict): Original outputs
        counterfactual (dict): Counterfactual outputs
        intervention_var (str): Variable that was intervened on
        
    Returns:
        float: Minimum change score (0-1)
    """
    # Calculate average change across all modalities
    changes = []
    
    # Get intervention modality
    intervention_modality = _get_variable_modality(intervention_var)
    
    # Check each modality
    for modality in set(original.keys()).intersection(counterfactual.keys()):
        # Skip the directly intervened modality
        if modality == intervention_modality:
            continue
            
        # Get tensors
        orig_tensor = original[modality].detach()
        cf_tensor = counterfactual[modality].detach()
        
        # Ensure same shape
        if orig_tensor.shape != cf_tensor.shape:
            continue
        
        # Calculate normalized difference
        try:
            if torch.norm(orig_tensor) > 1e-8:
                diff = torch.norm(cf_tensor - orig_tensor) / torch.norm(orig_tensor)
                
                # Convert to a [0, 1] score where higher means less change
                score = torch.exp(-diff * 5).item()
                changes.append(score)
        except Exception as e:
            print(f"Error calculating change for {modality}: {e}")
    
    # Return average, or default if no changes measured
    if changes:
        return sum(changes) / len(changes)
    else:
        return 0.5


def evaluate_model_causal_fidelity(
    model_outputs,
    ground_truth,
    causal_rules,
    modalities=["text", "image"]
):
    """Evaluate how well a model's outputs follow known causal rules.
    
    Args:
        model_outputs (dict): Model outputs for different modalities
        ground_truth (dict): Ground truth data
        causal_rules (dict): Causal rules
        modalities (list): List of modalities to evaluate
        
    Returns:
        dict: Fidelity scores for each modality and overall
    """
    fidelity_scores = {}
    
    # Check each modality
    for modality in modalities:
        if modality not in model_outputs or modality not in ground_truth:
            fidelity_scores[f"{modality}_fidelity"] = 0.0
            continue
        
        # Get outputs and ground truth for this modality
        outputs = model_outputs[modality]
        truth = ground_truth[modality]
        
        # Calculate causal fidelity using existing metric
        try:
            fidelity = calculate_causal_fidelity_score(
                outputs.detach().cpu().numpy(), 
                truth.detach().cpu().numpy(), 
                causal_rules
            )
            fidelity_scores[f"{modality}_fidelity"] = fidelity
        except Exception as e:
            print(f"Error calculating {modality} fidelity: {e}")
            fidelity_scores[f"{modality}_fidelity"] = 0.0
    
    # Calculate overall fidelity
    valid_scores = [score for score in fidelity_scores.values() if score > 0]
    if valid_scores:
        fidelity_scores["overall_fidelity"] = sum(valid_scores) / len(valid_scores)
    else:
        fidelity_scores["overall_fidelity"] = 0.0
    
    return fidelity_scores 