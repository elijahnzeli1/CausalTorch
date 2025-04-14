"""
CausalTorch Creative Module
=========================

This module implements creative generation capabilities for CausalTorch,
enabling the production of novel content through causal perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import networkx as nx
import random
import copy

from ..rules import CausalRuleSet, CausalRule


class CausalIntervention:
    """Represents a causal intervention on a graph.
    
    A causal intervention is a do(X=x) operation that sets a variable
    to a specific value, breaking its dependencies on parent nodes.
    
    Args:
        variable (str): The variable to intervene on
        value (Any): The value to set the variable to
        strength (float): The strength of the intervention (0-1)
        description (str, optional): Human-readable description
    """
    def __init__(
        self,
        variable: str,
        value: Any,
        strength: float = 1.0,
        description: Optional[str] = None
    ):
        self.variable = variable
        self.value = value
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.description = description or f"do({variable}={value})"
    
    def __str__(self) -> str:
        return self.description
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation
        """
        return {
            "variable": self.variable,
            "value": self.value,
            "strength": self.strength,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CausalIntervention':
        """Create from dictionary.
        
        Args:
            data (Dict): Dictionary data
            
        Returns:
            CausalIntervention: New intervention instance
        """
        return cls(
            variable=data["variable"],
            value=data["value"],
            strength=data.get("strength", 1.0),
            description=data.get("description")
        )


class CounterfactualDreamer(nn.Module):
    """Generates novel concepts by perturbing causal graphs.
    
    This module enables creative generation by exploring "what if" 
    scenarios through causal interventions. It modifies causal 
    relationships to produce novel but coherent outputs.
    
    Args:
        base_generator (nn.Module): Base generative model
        rules (CausalRuleSet): Causal ruleset defining relationships
        latent_dim (int): Dimension of latent space
    """
    def __init__(
        self,
        base_generator: nn.Module,
        rules: CausalRuleSet,
        latent_dim: int = 128
    ):
        super().__init__()
        self.base_generator = base_generator
        self.rules = copy.deepcopy(rules)
        self.latent_dim = latent_dim
        
        # MLP for mapping causal variables to latent space
        self.latent_mapper = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Track interventions applied
        self.applied_interventions = []
    
    def intervene(
        self,
        causal_graph: CausalRuleSet,
        intervention: CausalIntervention
    ) -> CausalRuleSet:
        """Apply a causal intervention to a graph.
        
        Args:
            causal_graph (CausalRuleSet): Original causal graph
            intervention (CausalIntervention): Intervention to apply
            
        Returns:
            CausalRuleSet: Modified causal graph
        """
        # Create a copy to avoid modifying the original
        new_graph = copy.deepcopy(causal_graph)
        
        # Get the variable to intervene on
        variable = intervention.variable
        
        # For do(X=x), remove all incoming edges to X
        # This simulates setting X to a fixed value regardless of its causes
        for cause in new_graph.get_all_causes():
            for rule in list(new_graph.get_rules_for_cause(cause)):
                if rule.effect == variable:
                    # Remove causal connection to intervened variable
                    new_graph.remove_rule(cause, variable)
        
        # Record the intervention
        self.applied_interventions.append(intervention)
        
        return new_graph
    
    def random_intervention(self) -> CausalIntervention:
        """Generate a random intervention on the causal graph.
        
        Returns:
            CausalIntervention: Random intervention
        """
        # Get all variables (causes and effects)
        all_variables = set(self.rules.get_all_causes()).union(
            set(self.rules.get_all_effects())
        )
        
        if not all_variables:
            raise ValueError("Causal graph has no variables to intervene on")
        
        # Choose a random variable
        variable = random.choice(list(all_variables))
        
        # Generate a random value
        # For simplicity, we use a random float between -1 and 1
        # In a real implementation, this would be type-appropriate for the variable
        value = np.random.uniform(-1, 1)
        
        # Generate random strength
        strength = np.random.uniform(0.5, 1.0)
        
        return CausalIntervention(
            variable=variable,
            value=value,
            strength=strength,
            description=f"Random intervention: do({variable}={value:.2f})"
        )
    
    def _get_variable_mapping(self) -> Dict[str, int]:
        """Map causal variables to indices in latent space.
        
        Returns:
            Dict[str, int]: Mapping from variable names to indices
        """
        # Get all variables (causes and effects)
        all_variables = set(self.rules.get_all_causes()).union(
            set(self.rules.get_all_effects())
        )
        
        # Create mapping (alphabetical order for determinism)
        sorted_variables = sorted(all_variables)
        mapping = {var: i % self.latent_dim for i, var in enumerate(sorted_variables)}
        
        return mapping
    
    def _encode_interventions(self, interventions: List[CausalIntervention]) -> torch.Tensor:
        """Encode interventions into a latent vector.
        
        Args:
            interventions (List[CausalIntervention]): List of interventions
            
        Returns:
            torch.Tensor: Latent vector encoding the interventions
        """
        # Get variable mapping
        var_mapping = self._get_variable_mapping()
        
        # Initialize latent vector
        z = torch.zeros(self.latent_dim)
        
        # Apply each intervention to the latent vector
        for intervention in interventions:
            if intervention.variable in var_mapping:
                idx = var_mapping[intervention.variable]
                z[idx] = intervention.value * intervention.strength
        
        return z
    
    def imagine(
        self,
        interventions: Optional[List[CausalIntervention]] = None,
        num_samples: int = 1,
        random_seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate novel content through causal interventions.
        
        Args:
            interventions (List[CausalIntervention], optional): List of interventions
                If None, a random intervention will be applied
            num_samples (int): Number of samples to generate
            random_seed (int, optional): Random seed for reproducibility
            
        Returns:
            torch.Tensor: Generated content
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # If no interventions provided, create a random one
        if interventions is None:
            interventions = [self.random_intervention()]
        
        # Keep track of applied interventions
        self.applied_interventions = interventions
        
        # Encode interventions into latent space
        z = self._encode_interventions(interventions)
        
        # Expand for multiple samples
        z_batch = z.unsqueeze(0).expand(num_samples, -1)
        
        # Map to full latent space
        z_mapped = self.latent_mapper(z_batch)
        
        # Generate content
        with torch.no_grad():
            outputs = self.base_generator.decode(z_mapped)
        
        return outputs
    
    def explain_interventions(self) -> List[str]:
        """Explain the interventions that were applied.
        
        Returns:
            List[str]: List of intervention descriptions
        """
        return [str(intervention) for intervention in self.applied_interventions]


class NoveltySearch(nn.Module):
    """Searches for novel solutions through divergent exploration.
    
    This module implements novelty search, which rewards solutions based
    on how different they are from previously discovered solutions,
    rather than on a fixed objective function.
    
    Args:
        base_model (nn.Module): Base model to evolve
        behavior_fn (Callable): Function to extract behavior characteristics
        population_size (int): Size of population
        num_generations (int): Number of generations
        mutation_rate (float): Probability of parameter mutation
    """
    def __init__(
        self,
        base_model: nn.Module,
        behavior_fn: Callable,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.behavior_fn = behavior_fn
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        
        # Archive of behaviors
        self.archive = []
        
        # Current population
        self.population = self._init_population()
    
    def _init_population(self) -> List[Dict[str, torch.Tensor]]:
        """Initialize population with random variations of base model.
        
        Returns:
            List[Dict[str, torch.Tensor]]: Population of model parameters
        """
        population = []
        
        # Get base model parameters
        base_params = {name: param.clone() for name, param in self.base_model.named_parameters()}
        
        # Add base model to population
        population.append(base_params)
        
        # Create rest of population with mutations
        for _ in range(self.population_size - 1):
            new_member = {}
            for name, param in base_params.items():
                # Add random noise to parameters
                noise = torch.randn_like(param) * self.mutation_rate
                new_member[name] = param + noise
            
            population.append(new_member)
        
        return population
    
    def _mutate(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Mutate parameters with random noise.
        
        Args:
            params (Dict[str, torch.Tensor]): Parameters to mutate
            
        Returns:
            Dict[str, torch.Tensor]: Mutated parameters
        """
        mutated = {}
        for name, param in params.items():
            # Only mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                # Gaussian noise scaled by parameter magnitude
                noise = torch.randn_like(param) * param.std() * 0.1
                mutated[name] = param + noise
            else:
                mutated[name] = param.clone()
        
        return mutated
    
    def _calculate_novelty(self, behavior: np.ndarray, k: int = 15) -> float:
        """Calculate novelty as average distance to k-nearest neighbors.
        
        Args:
            behavior (np.ndarray): Behavior vector
            k (int): Number of nearest neighbors to consider
            
        Returns:
            float: Novelty score
        """
        if not self.archive:
            return float('inf')  # First behavior is maximally novel
        
        # Calculate distances to all archive behaviors
        distances = []
        for archived_behavior in self.archive:
            dist = np.linalg.norm(behavior - archived_behavior)
            distances.append(dist)
        
        # Sort distances and take average of k nearest
        distances.sort()
        k_nearest = min(k, len(distances))
        avg_dist = np.mean(distances[:k_nearest])
        
        return avg_dist
    
    def _select_parents(
        self,
        novelty_scores: List[float],
        num_parents: int
    ) -> List[int]:
        """Select parents based on novelty scores.
        
        Args:
            novelty_scores (List[float]): Novelty scores for population
            num_parents (int): Number of parents to select
            
        Returns:
            List[int]: Indices of selected parents
        """
        # Convert to numpy array
        scores = np.array(novelty_scores)
        
        # Normalize scores to probabilities
        if scores.max() > scores.min():
            probs = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            probs = np.ones_like(scores) / len(scores)
        
        # Select parents with probability proportional to novelty
        parents = np.random.choice(
            len(scores),
            size=num_parents,
            replace=True,
            p=probs / probs.sum()
        )
        
        return parents.tolist()
    
    def _crossover(
        self,
        parent1: Dict[str, torch.Tensor],
        parent2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create child by crossing over parameters from parents.
        
        Args:
            parent1 (Dict[str, torch.Tensor]): First parent parameters
            parent2 (Dict[str, torch.Tensor]): Second parent parameters
            
        Returns:
            Dict[str, torch.Tensor]: Child parameters
        """
        child = {}
        for name, param1 in parent1.items():
            param2 = parent2[name]
            
            # Random interpolation between parents
            alpha = random.random()
            child[name] = alpha * param1 + (1 - alpha) * param2
        
        return child
    
    def run_search(self) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """Run the novelty search algorithm.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], List[float]]: (best_params, history)
                - best_params: Parameters of most novel individual
                - history: History of best novelty scores
        """
        history = []
        best_novelty = 0
        best_params = None
        
        for generation in range(self.num_generations):
            # Calculate behaviors and novelty for current population
            behaviors = []
            novelty_scores = []
            
            for individual in self.population:
                # Apply parameters to base model
                with torch.no_grad():
                    for name, param in self.base_model.named_parameters():
                        if name in individual:
                            param.copy_(individual[name])
                
                # Generate behavior with current model
                with torch.no_grad():
                    behavior = self.behavior_fn(self.base_model)
                
                # Calculate novelty
                novelty = self._calculate_novelty(behavior)
                
                behaviors.append(behavior)
                novelty_scores.append(novelty)
                
                # Check if most novel so far
                if novelty > best_novelty:
                    best_novelty = novelty
                    best_params = {name: param.clone() for name, param in individual.items()}
            
            # Add most novel to archive
            best_idx = np.argmax(novelty_scores)
            self.archive.append(behaviors[best_idx])
            
            # Record best novelty this generation
            history.append(max(novelty_scores))
            
            # Create next generation through selection, crossover, and mutation
            next_population = []
            
            # Keep the best individual (elitism)
            next_population.append(self.population[best_idx])
            
            # Fill rest of population with children
            while len(next_population) < self.population_size:
                # Select parents
                parent_indices = self._select_parents(novelty_scores, 2)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                
                # Mutate child
                child = self._mutate(child)
                
                # Add to next generation
                next_population.append(child)
            
            # Update population
            self.population = next_population
        
        # Return best individual and history
        return best_params, history


class CreativeMetrics:
    """Metrics for evaluating creative outputs.
    
    This class provides methods to measure various aspects of creativity,
    such as novelty, diversity, and coherence.
    """
    
    @staticmethod
    def novelty_score(
        output: torch.Tensor,
        reference_outputs: List[torch.Tensor],
        similarity_fn: Optional[Callable] = None
    ) -> float:
        """Calculate novelty as distance from reference outputs.
        
        Args:
            output (torch.Tensor): Generated output
            reference_outputs (List[torch.Tensor]): Reference outputs
            similarity_fn (Callable, optional): Function to calculate similarity
            
        Returns:
            float: Novelty score (higher = more novel)
        """
        if not reference_outputs:
            return 1.0  # Maximum novelty if no references
        
        # Default similarity function (cosine similarity)
        if similarity_fn is None:
            def similarity_fn(a, b):
                a_flat = a.view(-1)
                b_flat = b.view(-1)
                return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
        
        # Calculate similarities to all references
        similarities = [similarity_fn(output, ref) for ref in reference_outputs]
        
        # Novelty = 1 - max similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        return novelty
    
    @staticmethod
    def diversity_score(outputs: List[torch.Tensor], similarity_fn: Optional[Callable] = None) -> float:
        """Calculate diversity of a set of outputs.
        
        Args:
            outputs (List[torch.Tensor]): Set of outputs
            similarity_fn (Callable, optional): Function to calculate similarity
            
        Returns:
            float: Diversity score (higher = more diverse)
        """
        if len(outputs) < 2:
            return 0.0  # No diversity with fewer than 2 outputs
        
        # Default similarity function (cosine similarity)
        if similarity_fn is None:
            def similarity_fn(a, b):
                a_flat = a.view(-1)
                b_flat = b.view(-1)
                return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                similarities.append(similarity_fn(outputs[i], outputs[j]))
        
        # Diversity = 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    @staticmethod
    def causal_coherence(
        output: torch.Tensor,
        causal_rules: CausalRuleSet,
        consistency_fn: Callable
    ) -> float:
        """Measure adherence to causal constraints.
        
        Args:
            output (torch.Tensor): Generated output
            causal_rules (CausalRuleSet): Causal rules
            consistency_fn (Callable): Function that checks if output 
                adheres to given causal rule
            
        Returns:
            float: Coherence score (higher = more coherent)
        """
        if not causal_rules.rules:
            return 1.0  # Maximum coherence if no rules
        
        # Check each causal rule
        coherence_scores = []
        
        for cause in causal_rules.get_all_causes():
            for rule in causal_rules.get_rules_for_cause(cause):
                # Check if output adheres to this causal rule
                score = consistency_fn(output, cause, rule.effect)
                coherence_scores.append(score)
        
        # Average coherence across all rules
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        
        return avg_coherence 