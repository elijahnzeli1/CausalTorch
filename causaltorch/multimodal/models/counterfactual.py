"""
Counterfactual Dreamer Module
=============================

This module provides the CounterfactualDreamer class, which enables
generation of counterfactual samples by applying causal interventions
to a base generative model.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional, Tuple, Any

class CounterfactualDreamer(nn.Module):
    """
    CounterfactualDreamer enables generation of counterfactual samples
    by applying interventions to generative models.
    
    This class wraps any generative model and allows for:
    1. Defining "what if" scenarios through causal interventions
    2. Generating counterfactual samples based on those interventions
    3. Explaining the interventions and their expected effects
    """
    
    def __init__(
        self,
        base_generator: nn.Module,
        rules: Any = None,
        latent_dim: int = 128,
        modality_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the CounterfactualDreamer.
        
        Args:
            base_generator: Base generative model to use
            rules: Causal ruleset or graph defining relationships
            latent_dim: Dimension of latent space
            modality_map: Mapping from variable names to modalities
        """
        super().__init__()
        self.base_generator = base_generator
        self.rules = rules
        self.latent_dim = latent_dim
        self.modality_map = modality_map or {}
        self.interventions = []
        
    def imagine(
        self, 
        interventions: Optional[List[Dict]] = None,
        num_samples: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual samples based on interventions.
        
        Args:
            interventions: List of intervention specifications
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for the base generator
            
        Returns:
            Dictionary containing generated samples
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Store interventions for explanation
        self.interventions = interventions or []
        
        # Generate base latents
        latents = torch.randn(num_samples, self.latent_dim)
        
        # Apply interventions to latents if applicable
        if interventions:
            for intervention in interventions:
                latents = self._apply_intervention_to_latent(latents, intervention)
        
        # Generate using the base generator
        with torch.no_grad():
            outputs = self.base_generator(latents, **kwargs)
            
        return outputs
    
    def _apply_intervention_to_latent(
        self, 
        latents: torch.Tensor, 
        intervention: Dict
    ) -> torch.Tensor:
        """
        Apply an intervention to latent vectors.
        
        Args:
            latents: Latent vectors to intervene on
            intervention: Intervention specification
            
        Returns:
            Modified latent vectors
        """
        var = intervention.get('variable', '')
        value = intervention.get('value', 0.0)
        strength = intervention.get('strength', 1.0)
        
        # Simple demonstration implementation - in practice, this would be more sophisticated
        # and based on the learned causal structure in latent space
        if isinstance(var, str) and var:
            var_hash = hash(var) % self.latent_dim
            original = latents[:, var_hash].clone()
            latents[:, var_hash] = (1 - strength) * original + strength * value
            
        return latents
    
    def explain_interventions(self) -> str:
        """
        Generate an explanation of the applied interventions.
        
        Returns:
            String explanation of interventions
        """
        if not self.interventions:
            return "No interventions applied. Generated samples represent the base distribution."
        
        explanations = []
        for i, intervention in enumerate(self.interventions):
            var = intervention.get('variable', 'unknown')
            value = intervention.get('value', 'N/A')
            strength = intervention.get('strength', 1.0)
            desc = intervention.get('description', f"Setting {var} to {value}")
            
            explanations.append(f"Intervention {i+1}: {desc} (strength: {strength:.2f})")
            
            # If we have causal rules, explain expected effects
            if self.rules is not None:
                try:
                    # This assumes rules has a get_descendants method
                    descendants = self.rules.get_descendants(var)
                    if descendants:
                        explanations.append(f"  Expected effects on: {', '.join(descendants)}")
                except (AttributeError, TypeError):
                    pass
                    
        return "\n".join(explanations)
    
    def compare_to_baseline(
        self, 
        baseline_samples: Dict[str, torch.Tensor],
        counterfactual_samples: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compare counterfactual samples to baseline samples.
        
        Args:
            baseline_samples: Baseline samples (no intervention)
            counterfactual_samples: Samples with interventions
            
        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}
        
        # Implement comparison logic for each modality/variable
        for key in counterfactual_samples:
            if key in baseline_samples:
                b_tensor = baseline_samples[key]
                c_tensor = counterfactual_samples[key]
                
                if isinstance(b_tensor, torch.Tensor) and isinstance(c_tensor, torch.Tensor):
                    # Calculate difference metrics
                    if b_tensor.numel() > 0 and c_tensor.numel() > 0:
                        # Reshape if needed
                        if b_tensor.shape != c_tensor.shape:
                            if b_tensor.numel() == c_tensor.numel():
                                c_tensor = c_tensor.reshape(b_tensor.shape)
                        
                        # Calculate MSE difference
                        metrics[f"{key}_mse"] = nn.functional.mse_loss(b_tensor, c_tensor).item()
                        
                        # Calculate cosine similarity if multi-dimensional
                        if b_tensor.dim() > 1 or b_tensor.numel() > 10:
                            b_flat = b_tensor.reshape(-1)
                            c_flat = c_tensor.reshape(-1)
                            
                            b_norm = torch.norm(b_flat)
                            c_norm = torch.norm(c_flat)
                            
                            if b_norm > 0 and c_norm > 0:
                                cos_sim = torch.dot(b_flat, c_flat) / (b_norm * c_norm)
                                metrics[f"{key}_cosine_sim"] = cos_sim.item()
        
        return metrics 