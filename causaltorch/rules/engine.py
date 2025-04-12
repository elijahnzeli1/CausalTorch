"""
CausalTorch Rule Engine

This module implements the core functionality for the rule-based causal inference system.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set


class CausalRule:
    """
    A representation of a causal relationship between two variables.
    
    Attributes:
        cause (str): The cause variable
        effect (str): The effect variable
        strength (float): The strength of the causal effect [-1.0 to 1.0]
        conditions (Dict): Optional conditions for when this rule applies
    """
    
    def __init__(self, cause: str, effect: str, strength: float, conditions: Optional[Dict] = None):
        """
        Initialize a causal rule.
        
        Args:
            cause: The variable that causes the effect
            effect: The variable affected by the cause
            strength: Strength of the causal relationship [-1.0 to 1.0]
            conditions: Optional conditions for when this rule applies
        """
        self.cause = cause
        self.effect = effect
        self.strength = max(-1.0, min(1.0, strength))  # Clamp between -1.0 and 1.0
        self.conditions = conditions or {}
        
    def applies(self, context: Dict) -> bool:
        """
        Check if this rule applies given the context.
        
        Args:
            context: Dictionary of context variables
            
        Returns:
            True if the rule applies in this context
        """
        for key, required_value in self.conditions.items():
            if key not in context or context[key] != required_value:
                return False
        return True
    
    def __repr__(self) -> str:
        """String representation of the rule."""
        cond_str = f" [when {self.conditions}]" if self.conditions else ""
        return f"{self.cause} -> {self.effect} ({self.strength:.2f}){cond_str}"


class CausalRuleSet:
    """
    A collection of causal rules that can be used to modify tensors based on causal relationships.
    """
    
    def __init__(self):
        """Initialize an empty rule set."""
        self.rules: List[CausalRule] = []
        self.variables: Set[str] = set()
    
    def add_rule(self, cause: str, effect: str, strength: float, conditions: Optional[Dict] = None) -> None:
        """
        Add a rule to the rule set.
        
        Args:
            cause: The variable that causes the effect
            effect: The variable affected by the cause
            strength: Strength of the causal relationship [-1.0 to 1.0]
            conditions: Optional conditions for when this rule applies
        """
        rule = CausalRule(cause, effect, strength, conditions)
        self.rules.append(rule)
        self.variables.add(cause)
        self.variables.add(effect)
    
    def remove_rule(self, cause: str, effect: str) -> bool:
        """
        Remove a rule from the rule set.
        
        Args:
            cause: The cause variable
            effect: The effect variable
            
        Returns:
            True if a rule was removed, False otherwise
        """
        initial_length = len(self.rules)
        self.rules = [r for r in self.rules if not (r.cause == cause and r.effect == effect)]
        return len(self.rules) < initial_length
    
    def get_rules_for_effect(self, effect: str) -> List[CausalRule]:
        """
        Get all rules that affect the given variable.
        
        Args:
            effect: The effect variable to look for
            
        Returns:
            List of rules affecting this variable
        """
        return [r for r in self.rules if r.effect == effect]
    
    def get_rules_for_cause(self, cause: str) -> List[CausalRule]:
        """
        Get all rules that have the given variable as a cause.
        
        Args:
            cause: The cause variable to look for
            
        Returns:
            List of rules with this variable as a cause
        """
        return [r for r in self.rules if r.cause == cause]
    
    def get_causes(self, variable: str) -> List[str]:
        """
        Get all variables that directly cause the given variable.
        
        Args:
            variable: The variable whose causes we want to find
            
        Returns:
            List of causes
        """
        return [r.cause for r in self.rules if r.effect == variable]
    
    def get_effects(self, variable: str) -> List[str]:
        """
        Get all variables directly affected by the given variable.
        
        Args:
            variable: The variable whose effects we want to find
            
        Returns:
            List of effects
        """
        return [r.effect for r in self.rules if r.cause == variable]
    
    def apply_rules(self, tensor: torch.Tensor, context: Dict = None) -> torch.Tensor:
        """
        Apply the causal rules to modify a tensor.
        
        Args:
            tensor: The input tensor to modify
            context: Optional context for rule conditions
            
        Returns:
            Modified tensor
        """
        context = context or {}
        modified = tensor.clone()
        
        # Apply all applicable rules
        for rule in self.rules:
            if rule.applies(context):
                # Here we'll make a simple modification based on the rule's strength
                # In a real system, this would be more sophisticated
                effect_factor = 1.0 + rule.strength * 0.1
                modified = modified * effect_factor
        
        return modified
    
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Convert the rule set to an adjacency matrix representation.
        
        Returns:
            Tuple of (adjacency_matrix, variable_names)
        """
        variables = sorted(list(self.variables))
        var_to_idx = {var: i for i, var in enumerate(variables)}
        n = len(variables)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n, n))
        
        for rule in self.rules:
            i = var_to_idx[rule.cause]
            j = var_to_idx[rule.effect]
            adj_matrix[i, j] = rule.strength
        
        return adj_matrix, variables
    
    def __repr__(self) -> str:
        """String representation of the rule set."""
        return f"CausalRuleSet({len(self.rules)} rules, {len(self.variables)} variables)"


def load_default_rules() -> CausalRuleSet:
    """Load a set of default causal rules for common scenarios.
    
    Returns:
        CausalRuleSet: A set of default causal rules
    """
    ruleset = CausalRuleSet()
    
    # Weather-related rules
    ruleset.add_rule("rain", "wet_ground", 0.9)
    ruleset.add_rule("sun", "dry_ground", 0.8)
    ruleset.add_rule("lightning", "thunder", 1.0)
    
    # Physical rules
    ruleset.add_rule("push", "move", 0.7)
    ruleset.add_rule("heat", "temperature_rise", 0.8)
    
    return ruleset