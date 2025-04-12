"""
CausalTorch Rules Module
========================

This module provides utilities for defining, validating, and managing causal rules
that drive the generation process in Causal Neuro-Symbolic Generative Networks.
"""

import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class CausalRule:
    """Represents a single causal rule with a cause, effect, and parameters.
    
    Args:
        cause (str): The causal variable/event
        effect (str): The effect variable/event
        strength (float): Strength of the causal relationship (0-1)
        temporal_offset (int, optional): Time delay before effect occurs (for video)
        duration (int, optional): Duration of the effect (for video)
        threshold (float, optional): Threshold for cause to trigger effect
    """
    def __init__(self, cause: str, effect: str, strength: float = 0.9,
                 temporal_offset: int = 0, duration: int = 1,
                 threshold: float = 0.5):
        self.cause = cause
        self.effect = effect
        self.strength = max(0.0, min(1.0, strength))  # Clamp to 0-1
        self.temporal_offset = temporal_offset
        self.duration = max(1, duration)  # At least 1 frame
        self.threshold = threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization."""
        return {
            "effect": self.effect,
            "strength": self.strength,
            "temporal_offset": self.temporal_offset,
            "duration": self.duration,
            "threshold": self.threshold
        }
    
    @classmethod
    def from_dict(cls, cause: str, effect_dict: Dict) -> 'CausalRule':
        """Create a CausalRule from a cause string and effect dictionary."""
        return cls(
            cause=cause,
            effect=effect_dict["effect"],
            strength=effect_dict.get("strength", 0.9),
            temporal_offset=effect_dict.get("temporal_offset", 0),
            duration=effect_dict.get("duration", 1),
            threshold=effect_dict.get("threshold", 0.5)
        )
    
    def __str__(self) -> str:
        return f"{self.cause} → {self.effect} (strength: {self.strength:.2f})"


class CausalRuleSet:
    """A collection of causal rules with validation and graph representation.
    
    This class manages a set of causal rules, validates them for consistency,
    and provides utilities for visualization and serialization.
    
    Args:
        rules (Dict, optional): Dictionary mapping causes to effect dictionaries
    """
    def __init__(self, rules: Optional[Dict] = None):
        self.rules = {}
        self.graph = nx.DiGraph()
        
        if rules:
            self.add_rules_from_dict(rules)
    
    def add_rule(self, rule: CausalRule) -> None:
        """Add a single causal rule."""
        if rule.cause not in self.rules:
            self.rules[rule.cause] = []
        
        # Check for duplicates
        for existing_rule in self.rules[rule.cause]:
            if existing_rule.effect == rule.effect:
                self.rules[rule.cause].remove(existing_rule)
                break
        
        self.rules[rule.cause].append(rule)
        
        # Update the graph
        self.graph.add_edge(rule.cause, rule.effect, 
                           weight=rule.strength,
                           temporal_offset=rule.temporal_offset)
    
    def add_rules_from_dict(self, rules_dict: Dict) -> None:
        """Add multiple rules from a dictionary."""
        for cause, effect_dict_or_list in rules_dict.items():
            if isinstance(effect_dict_or_list, list):
                for effect_dict in effect_dict_or_list:
                    rule = CausalRule.from_dict(cause, effect_dict)
                    self.add_rule(rule)
            else:
                rule = CausalRule.from_dict(cause, effect_dict_or_list)
                self.add_rule(rule)
    
    def remove_rule(self, cause: str, effect: str) -> bool:
        """Remove a rule by cause and effect."""
        if cause in self.rules:
            for rule in self.rules[cause]:
                if rule.effect == effect:
                    self.rules[cause].remove(rule)
                    self.graph.remove_edge(cause, effect)
                    
                    # Clean up empty causes
                    if not self.rules[cause]:
                        del self.rules[cause]
                    
                    return True
        return False
    
    def get_rules_for_cause(self, cause: str) -> List[CausalRule]:
        """Get all rules for a specific cause."""
        return self.rules.get(cause, [])
    
    def get_all_causes(self) -> List[str]:
        """Get all causes in the ruleset."""
        return list(self.rules.keys())
    
    def get_all_effects(self) -> List[str]:
        """Get all effects in the ruleset."""
        effects = set()
        for rules in self.rules.values():
            for rule in rules:
                effects.add(rule.effect)
        return list(effects)
    
    def is_acyclic(self) -> bool:
        """Check if the causal graph is acyclic (no loops)."""
        try:
            nx.find_cycle(self.graph)
            return False
        except nx.NetworkXNoCycle:
            return True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate the ruleset for consistency and acyclicity."""
        if not self.is_acyclic():
            return False, "Causal graph contains cycles"
        
        # Check for other consistency issues
        for cause, rules in self.rules.items():
            effects = [rule.effect for rule in rules]
            if len(effects) != len(set(effects)):
                return False, f"Duplicate effects for cause '{cause}'"
        
        return True, "Valid"
    
    def to_dict(self) -> Dict:
        """Convert the ruleset to a dictionary for serialization."""
        result = {}
        for cause, rules in self.rules.items():
            if len(rules) == 1:
                result[cause] = rules[0].to_dict()
            else:
                result[cause] = [rule.to_dict() for rule in rules]
        return result
    
    def to_json(self, filepath: str) -> None:
        """Save the ruleset to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'CausalRuleSet':
        """Load a ruleset from a JSON file."""
        with open(filepath, 'r') as f:
            rules_dict = json.load(f)
        return cls(rules_dict)
    
    def visualize(self, figsize=(10, 8), show_weights=True) -> None:
        """Visualize the causal graph."""
        from .visualization import plot_causal_graph
        
        # Use the more sophisticated visualization function
        plot_causal_graph(
            graph=self.graph,
            figsize=figsize,
            show_weights=show_weights,
            title="Causal Graph"
        )
        plt.show()
    
    def adjacency_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """Get the adjacency matrix representation of the causal graph.
        
        Returns:
            tuple: (node_labels, adjacency_matrix)
        """
        # Get all nodes (causes and effects)
        nodes = list(self.graph.nodes())
        nodes.sort()  # For consistency
        
        # Create matrix
        n = len(nodes)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Fill matrix with edge weights
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if self.graph.has_edge(source, target):
                    matrix[i][j] = self.graph.get_edge_data(source, target)['weight']
        
        return nodes, matrix


def load_default_rules() -> CausalRuleSet:
    """Load a set of default causal rules for common scenarios."""
    default_rules = {
        "rain": {
            "effect": "ground_wet",
            "strength": 0.9
        },
        "fire": {
            "effect": "smoke",
            "strength": 0.8
        },
        "cold": [
            {
                "effect": "ice",
                "strength": 0.7
            },
            {
                "effect": "frost",
                "strength": 0.6
            }
        ],
        "arrow_hit": {
            "effect": "soldier_fall",
            "strength": 0.9,
            "temporal_offset": 3
        },
        "explosion": {
            "effect": "smoke_cloud",
            "strength": 0.95,
            "duration": 10
        },
        "gene_mutation_X": {
            "effect": "neuropathy",
            "strength": 0.95
        }
    }
    
    return CausalRuleSet(default_rules)