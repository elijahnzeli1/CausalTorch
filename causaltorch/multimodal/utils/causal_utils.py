"""
Causal Utilities for Multimodal Models
======================================

This module provides utilities for working with causal graphs in multimodal contexts,
including causal graph construction, manipulation, and intervention functions.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Set, Any
import copy
import json


class MultimodalCausalGraph:
    """Class representing a causal graph for multimodal data.
    
    This class provides methods for constructing, manipulating, and
    querying causal relationships across different modalities.
    """
    
    def __init__(self, relations=None):
        """Initialize a multimodal causal graph.
        
        Args:
            relations (list or dict): Initial causal relations
                If list: List of dicts with 'cause', 'effect', and optional 'strength'
                If dict: Keys are causes, values are effects or lists of effects
        """
        self.relations = []
        self.modality_map = {}  # Maps variables to their modalities
        self._graph = None  # Lazily built NetworkX graph
        
        if relations:
            self.add_relations(relations)
    
    def add_relations(self, relations):
        """Add causal relations to the graph.
        
        Args:
            relations (list or dict): Causal relations to add
        """
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, dict) and 'cause' in relation and 'effect' in relation:
                    cause = relation['cause']
                    effect = relation['effect']
                    strength = relation.get('strength', 0.8)
                    modality_cause = relation.get('modality_cause', self._infer_modality(cause))
                    modality_effect = relation.get('modality_effect', self._infer_modality(effect))
                    
                    self.add_relation(cause, effect, strength, modality_cause, modality_effect)
        
        elif isinstance(relations, dict):
            for cause, effects in relations.items():
                if isinstance(effects, list):
                    for effect in effects:
                        if isinstance(effect, dict) and 'effect' in effect:
                            self.add_relation(
                                cause, 
                                effect['effect'], 
                                effect.get('strength', 0.8),
                                effect.get('modality_cause', self._infer_modality(cause)),
                                effect.get('modality_effect', self._infer_modality(effect['effect']))
                            )
                        else:
                            self.add_relation(
                                cause, 
                                effect, 
                                0.8,  # Default strength
                                self._infer_modality(cause),
                                self._infer_modality(effect)
                            )
                else:
                    self.add_relation(
                        cause, 
                        effects, 
                        0.8,  # Default strength
                        self._infer_modality(cause),
                        self._infer_modality(effects)
                    )
    
    def add_relation(self, cause, effect, strength=0.8, modality_cause=None, modality_effect=None):
        """Add a single causal relation to the graph.
        
        Args:
            cause: The cause variable
            effect: The effect variable
            strength (float): Strength of the causal relationship
            modality_cause (str): Modality of the cause variable
            modality_effect (str): Modality of the effect variable
        """
        # Infer modalities if not provided
        if modality_cause is None:
            modality_cause = self._infer_modality(cause)
        
        if modality_effect is None:
            modality_effect = self._infer_modality(effect)
        
        # Add to relations
        self.relations.append({
            'cause': cause,
            'effect': effect,
            'strength': strength
        })
        
        # Update modality map
        self.modality_map[str(cause)] = modality_cause
        self.modality_map[str(effect)] = modality_effect
        
        # Reset cached graph
        self._graph = None
    
    def _infer_modality(self, variable):
        """Infer the modality of a variable based on naming conventions.
        
        Args:
            variable: The variable to infer modality for
            
        Returns:
            str: Inferred modality ('text', 'image', or 'unknown')
        """
        var_str = str(variable).lower()
        
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
    
    def get_graph(self):
        """Get a NetworkX directed graph representation.
        
        Returns:
            networkx.DiGraph: Directed graph representation
        """
        if self._graph is None:
            self._build_graph()
        
        return self._graph
    
    def _build_graph(self):
        """Build a NetworkX graph from relations."""
        G = nx.DiGraph()
        
        for relation in self.relations:
            cause = str(relation['cause'])
            effect = str(relation['effect'])
            strength = relation['strength']
            
            if not G.has_node(cause):
                G.add_node(cause, modality=self.modality_map.get(cause, 'unknown'))
            
            if not G.has_node(effect):
                G.add_node(effect, modality=self.modality_map.get(effect, 'unknown'))
            
            G.add_edge(cause, effect, weight=strength)
        
        self._graph = G
    
    def find_paths(self, source, target):
        """Find all paths from source to target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            list: List of paths (each path is a list of nodes)
        """
        G = self.get_graph()
        source_str = str(source)
        target_str = str(target)
        
        if not G.has_node(source_str) or not G.has_node(target_str):
            return []
        
        try:
            return list(nx.all_simple_paths(G, source_str, target_str))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_children(self, node):
        """Get all direct children of a node.
        
        Args:
            node: The node to get children for
            
        Returns:
            list: List of child nodes
        """
        G = self.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return []
        
        return list(G.successors(node_str))
    
    def get_parents(self, node):
        """Get all direct parents of a node.
        
        Args:
            node: The node to get parents for
            
        Returns:
            list: List of parent nodes
        """
        G = self.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return []
        
        return list(G.predecessors(node_str))
    
    def get_ancestors(self, node):
        """Get all ancestors of a node.
        
        Args:
            node: The node to get ancestors for
            
        Returns:
            set: Set of ancestor nodes
        """
        G = self.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return set()
        
        ancestors = set()
        queue = list(G.predecessors(node_str))
        
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend([p for p in G.predecessors(current) if p not in ancestors])
        
        return ancestors
    
    def get_descendants(self, node):
        """Get all descendants of a node.
        
        Args:
            node: The node to get descendants for
            
        Returns:
            set: Set of descendant nodes
        """
        G = self.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return set()
        
        descendants = set()
        queue = list(G.successors(node_str))
        
        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend([s for s in G.successors(current) if s not in descendants])
        
        return descendants
    
    def get_markov_blanket(self, node):
        """Get the Markov blanket of a node (parents, children, and children's parents).
        
        Args:
            node: The node to get the Markov blanket for
            
        Returns:
            set: Set of nodes in the Markov blanket
        """
        G = self.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return set()
        
        blanket = set()
        
        # Add parents
        blanket.update(G.predecessors(node_str))
        
        # Add children
        children = list(G.successors(node_str))
        blanket.update(children)
        
        # Add children's other parents
        for child in children:
            blanket.update([p for p in G.predecessors(child) if p != node_str])
        
        return blanket
    
    def perform_intervention(self, node, value):
        """Create a new graph with an intervention on a node.
        
        This implements do(X=x) operation in causal inference.
        
        Args:
            node: The node to intervene on
            value: The value to set the node to
            
        Returns:
            MultimodalCausalGraph: New graph with the intervention
        """
        new_graph = copy.deepcopy(self)
        G = new_graph.get_graph()
        node_str = str(node)
        
        if not G.has_node(node_str):
            return new_graph
        
        # Remove all incoming edges to the intervened node
        parents = list(G.predecessors(node_str))
        for parent in parents:
            G.remove_edge(parent, node_str)
        
        # Add intervention information
        G.nodes[node_str]['intervened'] = True
        G.nodes[node_str]['intervention_value'] = value
        
        # Update the internal graph
        new_graph._graph = G
        
        return new_graph
    
    def check_d_separation(self, X, Y, Z=None):
        """Check if X and Y are d-separated given Z.
        
        Args:
            X: First set of nodes
            Y: Second set of nodes
            Z: Conditioning set of nodes (optional)
            
        Returns:
            bool: True if X and Y are d-separated given Z
        """
        G = self.get_graph()
        
        # Convert to sets of strings
        X_set = {str(x) for x in X} if isinstance(X, (list, set, tuple)) else {str(X)}
        Y_set = {str(y) for y in Y} if isinstance(Y, (list, set, tuple)) else {str(Y)}
        Z_set = {str(z) for z in Z} if Z and isinstance(Z, (list, set, tuple)) else ({str(Z)} if Z else set())
        
        # Check if any nodes don't exist in the graph
        for node_set in [X_set, Y_set, Z_set]:
            for node in node_set:
                if not G.has_node(node):
                    return False
        
        # Use NetworkX's d-separation test
        return nx.d_separated(G, X_set, Y_set, Z_set)
    
    def get_modality_subgraph(self, modality):
        """Get a subgraph containing only nodes of a specific modality.
        
        Args:
            modality (str): The modality to filter by
            
        Returns:
            networkx.DiGraph: Subgraph with only nodes of the specified modality
        """
        G = self.get_graph()
        
        # Filter nodes by modality
        nodes = [n for n, attr in G.nodes(data=True) 
                if attr.get('modality') == modality]
        
        return G.subgraph(nodes)
    
    def get_cross_modal_edges(self):
        """Get all edges that connect different modalities.
        
        Returns:
            list: List of (source, target, attributes) tuples for cross-modal edges
        """
        G = self.get_graph()
        cross_modal_edges = []
        
        for source, target, attrs in G.edges(data=True):
            source_modality = G.nodes[source].get('modality', 'unknown')
            target_modality = G.nodes[target].get('modality', 'unknown')
            
            if source_modality != target_modality:
                cross_modal_edges.append((source, target, attrs))
        
        return cross_modal_edges
    
    def get_relation_strength(self, cause, effect):
        """Get the strength of a causal relation.
        
        Args:
            cause: The cause variable
            effect: The effect variable
            
        Returns:
            float: Strength of the causal relation, or 0.0 if not found
        """
        G = self.get_graph()
        cause_str = str(cause)
        effect_str = str(effect)
        
        if G.has_edge(cause_str, effect_str):
            return G.edges[cause_str, effect_str].get('weight', 0.0)
        else:
            return 0.0
    
    def to_dict(self):
        """Convert the causal graph to a dictionary representation.
        
        Returns:
            dict: Dictionary representation of the causal graph
        """
        return {
            'causal_relations': self.relations,
            'modality_map': self.modality_map
        }
    
    def to_json(self, indent=2):
        """Convert the causal graph to a JSON string.
        
        Args:
            indent (int): Indentation for JSON formatting
            
        Returns:
            str: JSON string representation of the causal graph
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data):
        """Create a causal graph from a dictionary representation.
        
        Args:
            data (dict): Dictionary representation of a causal graph
            
        Returns:
            MultimodalCausalGraph: New causal graph instance
        """
        graph = cls()
        
        if 'causal_relations' in data:
            graph.add_relations(data['causal_relations'])
        
        if 'modality_map' in data:
            graph.modality_map.update(data['modality_map'])
        
        return graph
    
    @classmethod
    def from_json(cls, json_str):
        """Create a causal graph from a JSON string.
        
        Args:
            json_str (str): JSON string representation of a causal graph
            
        Returns:
            MultimodalCausalGraph: New causal graph instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


def calculate_causal_consistency(predictions, ground_truth, causal_graph):
    """Calculate causal consistency between predictions and ground truth.
    
    Measures how well the predictions follow the causal relationships
    specified in the causal graph compared to ground truth.
    
    Note: This is a general-purpose function for assessing causal consistency.
    For multimodal-specific evaluations, consider using calculate_multimodal_causal_consistency()
    which is optimized for text-image relationships.
    
    Args:
        predictions (dict): Predicted outputs for different variables
        ground_truth (dict): Ground truth values for different variables
        causal_graph: Causal graph specifying relationships
        
    Returns:
        float: Consistency score between 0 and 1
    """
    if isinstance(causal_graph, dict):
        graph = MultimodalCausalGraph.from_dict(causal_graph)
    elif isinstance(causal_graph, MultimodalCausalGraph):
        graph = causal_graph
    else:
        graph = MultimodalCausalGraph(causal_graph)
    
    # Convert string keys to str type
    str_predictions = {str(k): v for k, v in predictions.items()}
    str_ground_truth = {str(k): v for k, v in ground_truth.items()}
    
    # Get causal relationships
    G = graph.get_graph()
    total_score = 0.0
    count = 0
    
    # Check each causal relationship
    for cause, effect, attrs in G.edges(data=True):
        # Skip if we don't have predictions or ground truth for both variables
        if cause not in str_predictions or effect not in str_predictions:
            continue
        if cause not in str_ground_truth or effect not in str_ground_truth:
            continue
        
        # Get the values
        pred_cause = str_predictions[cause]
        pred_effect = str_predictions[effect]
        true_cause = str_ground_truth[cause]
        true_effect = str_ground_truth[effect]
        
        # Calculate consistency for this relationship
        cause_consistency = _calculate_variable_consistency(pred_cause, true_cause)
        effect_consistency = _calculate_variable_consistency(pred_effect, true_effect)
        
        # If cause is predicted correctly and effect is predicted correctly,
        # the causal relationship is maintained
        relation_strength = attrs.get('weight', 0.8)
        relation_consistency = (cause_consistency * effect_consistency) ** (1 / relation_strength)
        
        total_score += relation_consistency
        count += 1
    
    if count == 0:
        return 0.5  # Default score when no relationships can be evaluated
    
    return total_score / count


def _calculate_variable_consistency(prediction, ground_truth):
    """Calculate consistency between prediction and ground truth for a single variable.
    
    Args:
        prediction: Predicted value
        ground_truth: Ground truth value
        
    Returns:
        float: Consistency score between 0 and 1
    """
    # Handle different types of data
    if isinstance(prediction, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
        # Tensor case
        if prediction.shape != ground_truth.shape:
            # If shapes don't match, reshape if possible
            if prediction.numel() == ground_truth.numel():
                prediction = prediction.reshape(ground_truth.shape)
            else:
                # Can't directly compare, use a simple heuristic
                return 0.5
        
        # Calculate cosine similarity for high-dimensional data
        if prediction.dim() > 1 or prediction.numel() > 10:
            pred_flat = prediction.reshape(-1)
            truth_flat = ground_truth.reshape(-1)
            
            # Normalize
            pred_norm = torch.norm(pred_flat)
            truth_norm = torch.norm(truth_flat)
            
            if pred_norm > 0 and truth_norm > 0:
                cos_sim = torch.dot(pred_flat, truth_flat) / (pred_norm * truth_norm)
                return (cos_sim.item() + 1) / 2  # Scale to [0, 1]
        
        # For low-dimensional data, use L2 distance
        error = torch.nn.functional.mse_loss(prediction, ground_truth)
        consistency = torch.exp(-error).item()
        return consistency
    
    elif isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
        # NumPy case
        if prediction.shape != ground_truth.shape:
            if prediction.size == ground_truth.size:
                prediction = prediction.reshape(ground_truth.shape)
            else:
                return 0.5
        
        # Similar logic to the tensor case
        if prediction.ndim > 1 or prediction.size > 10:
            pred_flat = prediction.flatten()
            truth_flat = ground_truth.flatten()
            
            pred_norm = np.linalg.norm(pred_flat)
            truth_norm = np.linalg.norm(truth_flat)
            
            if pred_norm > 0 and truth_norm > 0:
                cos_sim = np.dot(pred_flat, truth_flat) / (pred_norm * truth_norm)
                return (cos_sim + 1) / 2
        
        error = np.mean((prediction - ground_truth) ** 2)
        consistency = np.exp(-error)
        return consistency
    
    elif isinstance(prediction, (int, float)) and isinstance(ground_truth, (int, float)):
        # Numeric case
        error = abs(prediction - ground_truth)
        max_val = max(abs(prediction), abs(ground_truth))
        if max_val > 0:
            rel_error = error / max_val
        else:
            rel_error = error
        
        return np.exp(-rel_error)
    
    elif isinstance(prediction, str) and isinstance(ground_truth, str):
        # String case
        if prediction == ground_truth:
            return 1.0
        
        # Simple string similarity based on common characters
        common_chars = set(prediction) & set(ground_truth)
        total_chars = set(prediction) | set(ground_truth)
        
        if total_chars:
            return len(common_chars) / len(total_chars)
        else:
            return 0.0
    
    # Default case
    return 0.5


def apply_intervention(tensor_dict, intervention, modality_map=None):
    """Apply an intervention to a dictionary of tensors.
    
    Args:
        tensor_dict (dict): Dictionary of tensors
        intervention (dict): Intervention specification with 'variable' and 'value'
        modality_map (dict): Map from variable names to modalities
        
    Returns:
        dict: Dictionary with intervention applied
    """
    if not isinstance(intervention, dict) or 'variable' not in intervention:
        return tensor_dict
    
    variable = intervention['variable']
    value = intervention.get('value', 1.0)
    
    # Copy the tensor dict to avoid modifying the original
    result = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) 
             for k, v in tensor_dict.items()}
    
    # If the variable is directly in the dict, modify it
    str_variable = str(variable)
    if str_variable in result:
        if isinstance(result[str_variable], torch.Tensor):
            # For tensors, we need to ensure the value has the right shape
            if isinstance(value, (int, float)):
                result[str_variable].fill_(value)
            elif isinstance(value, torch.Tensor) and value.shape == result[str_variable].shape:
                result[str_variable].copy_(value)
        else:
            # For non-tensors, just set the value
            result[str_variable] = value
        
        return result
    
    # If not directly in dict, check if we can map to a modality
    if modality_map is None:
        # Try to infer modality
        if isinstance(variable, str):
            var_str = variable.lower()
            if 'text' in var_str or 'word' in var_str:
                modality = 'text'
            elif 'image' in var_str or 'visual' in var_str:
                modality = 'image'
            else:
                return result  # Can't determine modality
        else:
            return result  # Can't determine modality
    else:
        # Use provided modality map
        modality = modality_map.get(str_variable)
        if modality is None:
            return result  # Variable not in modality map
    
    # Apply intervention based on modality
    if modality == 'text' and 'text' in result:
        if isinstance(result['text'], torch.Tensor):
            # For tensors, we need to ensure the value has the right shape
            if isinstance(value, (int, float)):
                result['text'].fill_(value)
            elif isinstance(value, torch.Tensor) and value.shape == result['text'].shape:
                result['text'].copy_(value)
        else:
            # For non-tensors, just set the value
            result['text'] = value
    
    elif modality == 'image' and 'image' in result:
        if isinstance(result['image'], torch.Tensor):
            # For tensors, we need to ensure the value has the right shape
            if isinstance(value, (int, float)):
                result['image'].fill_(value)
            elif isinstance(value, torch.Tensor) and value.shape == result['image'].shape:
                result['image'].copy_(value)
        else:
            # For non-tensors, just set the value
            result['image'] = value
    
    return result


def generate_synthetic_causal_graph(num_nodes=10, edge_probability=0.3, 
                                   modalities=None, seed=None):
    """Generate a synthetic causal graph for testing.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        edge_probability (float): Probability of an edge between any two nodes
        modalities (list): List of modalities to assign to nodes
        seed (int): Random seed for reproducibility
        
    Returns:
        MultimodalCausalGraph: Generated causal graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default modalities
    if modalities is None:
        modalities = ['text', 'image']
    
    # Create a random DAG
    # We use the Erdos-Renyi algorithm but ensure acyclicity
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # Fill the upper triangular part with random edges
    # This ensures acyclicity since we only have i->j where i<j
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < edge_probability:
                adj_matrix[i, j] = 1
    
    # Create node names and assign modalities
    nodes = []
    modality_map = {}
    
    for i in range(num_nodes):
        # Assign a modality
        modality = np.random.choice(modalities)
        
        # Create a name based on modality
        if modality == 'text':
            name = f"text_var_{i}"
        elif modality == 'image':
            name = f"image_var_{i}"
        else:
            name = f"{modality}_var_{i}"
        
        nodes.append(name)
        modality_map[name] = modality
    
    # Create the causal relations
    relations = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                # Add a relation with random strength
                strength = np.random.uniform(0.5, 1.0)
                relations.append({
                    'cause': nodes[i],
                    'effect': nodes[j],
                    'strength': strength
                })
    
    # Create the causal graph
    graph = MultimodalCausalGraph(relations)
    
    # Update the modality map
    for node, modality in modality_map.items():
        graph.modality_map[node] = modality
    
    return graph 