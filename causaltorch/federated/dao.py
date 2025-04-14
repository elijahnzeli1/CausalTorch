"""
CausalTorch Federated Learning Module
====================================

This module implements decentralized learning approaches for CausalTorch,
enabling distributed training while preserving causal knowledge.
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import copy
import hashlib
import json
import time

from ..rules import CausalRuleSet, CausalRule


class CausalDAO:
    """Decentralized Autonomous Organization for federated causal learning.
    
    This class enables distributed training of causal models across multiple
    clients without sharing raw data. Instead, local causal graphs and model
    updates are shared and merged through a consensus mechanism.
    
    Args:
        initial_graph (CausalRuleSet, optional): Initial global causal graph
        consensus_threshold (float): Minimum proportion of nodes that must agree
            for a causal relationship to be accepted (0.0-1.0)
        model_aggregation (str): Method for aggregating models from clients
            - 'fedavg': Weighted average based on data size (FedAvg algorithm)
            - 'fedprox': FedAvg with proximity constraint
    """
    def __init__(
        self,
        initial_graph: Optional[CausalRuleSet] = None,
        consensus_threshold: float = 0.6,
        model_aggregation: str = 'fedavg'
    ):
        self.global_causal_graph = initial_graph or CausalRuleSet()
        self.consensus_threshold = consensus_threshold
        self.model_aggregation = model_aggregation
        
        # Store client models and their weights
        self.client_models = {}
        self.client_weights = {}
        
        # Store client causal graphs
        self.client_graphs = {}
        
        # Voting records for causal relationships
        self.causal_votes = {}
        
        # Maintain a global model (will be initialized later)
        self.global_model = None
        
        # Block history for auditability
        self.blocks = []
        
        # Blockchain-like structure for tracking updates
        self.chain = []
        self._add_block("genesis", "genesis", {"graph": self._hash_graph(self.global_causal_graph)})
    
    def register_client(self, client_id: str, data_size: int):
        """Register a new client with the DAO.
        
        Args:
            client_id (str): Unique identifier for the client
            data_size (int): Size of client's local dataset
        """
        if client_id not in self.client_weights:
            self.client_weights[client_id] = data_size
    
    def update_local_graph(
        self,
        client_id: str,
        local_graph: CausalRuleSet,
        signature: Optional[str] = None
    ) -> bool:
        """Update a client's local causal graph and process votes.
        
        Args:
            client_id (str): Client identifier
            local_graph (CausalRuleSet): Client's local causal graph
            signature (str, optional): Cryptographic signature for authentication
            
        Returns:
            bool: True if the update was accepted
        """
        # Validate the graph is acyclic
        is_valid, message = local_graph.validate()
        if not is_valid:
            print(f"Rejected graph from {client_id}: {message}")
            return False
        
        # Store the client's graph
        self.client_graphs[client_id] = local_graph
        
        # Process votes for causal relationships
        self._process_votes(client_id, local_graph)
        
        # Update the global graph if needed
        self._update_global_graph()
        
        # Add a block to the history
        self._add_block(client_id, 'graph_update', {
            'timestamp': self._get_timestamp(),
            'graph_hash': self._hash_graph(local_graph)
        })
        
        return True
    
    def _process_votes(self, client_id: str, local_graph: CausalRuleSet):
        """Process votes from a client's local graph.
        
        Args:
            client_id (str): Client identifier
            local_graph (CausalRuleSet): Client's local causal graph
        """
        # Get all causal relationships from the local graph
        for cause in local_graph.get_all_causes():
            for rule in local_graph.get_rules_for_cause(cause):
                # Create a unique key for this causal relationship
                rel_key = f"{cause}->{rule.effect}"
                
                # Initialize vote tracking if needed
                if rel_key not in self.causal_votes:
                    self.causal_votes[rel_key] = {
                        'votes': {},
                        'strengths': []
                    }
                
                # Record vote and causal strength
                self.causal_votes[rel_key]['votes'][client_id] = True
                self.causal_votes[rel_key]['strengths'].append(rule.strength)
    
    def _update_global_graph(self):
        """Update the global causal graph based on consensus."""
        # Create a new empty graph
        new_global_graph = CausalRuleSet()
        
        # Count total clients
        total_clients = len(self.client_graphs)
        if total_clients == 0:
            return
        
        # Check each relationship for consensus
        for rel_key, vote_data in self.causal_votes.items():
            # Calculate consensus ratio
            vote_count = len(vote_data['votes'])
            consensus_ratio = vote_count / total_clients
            
            # If consensus threshold is met, add to global graph
            if consensus_ratio >= self.consensus_threshold:
                # Parse the relationship key
                cause, effect = rel_key.split("->")
                
                # Calculate average strength
                avg_strength = np.mean(vote_data['strengths'])
                
                # Create and add the rule
                rule = CausalRule(
                    cause=cause,
                    effect=effect,
                    strength=float(avg_strength)
                )
                new_global_graph.add_rule(rule)
        
        # Update the global graph
        self.global_causal_graph = new_global_graph
    
    def update_local_model(
        self,
        client_id: str,
        model_update: Dict[str, torch.Tensor],
        data_size: Optional[int] = None,
        signature: Optional[str] = None
    ) -> bool:
        """Update a client's local model.
        
        Args:
            client_id (str): Client identifier
            model_update (Dict[str, torch.Tensor]): Model parameter updates
            data_size (int, optional): Size of client's local dataset
            signature (str, optional): Cryptographic signature for authentication
            
        Returns:
            bool: True if the update was accepted
        """
        # Validate client
        if client_id not in self.client_weights:
            print(f"Rejected: Unknown client {client_id}")
            return False
        
        # Update client's weight if data_size is provided
        if data_size is not None:
            self.client_weights[client_id] = data_size
        
        # Store client's model parameters
        self.client_models[client_id] = model_update
        
        # Add a block to the history
        self._add_block(client_id, 'model_update', {
            'timestamp': self._get_timestamp(),
            'parameters_updated': len(model_update)
        })
        
        return True
    
    def aggregate_models(self) -> Dict[str, torch.Tensor]:
        """Aggregate client models into a global model.
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated model parameters
        """
        if not self.client_models:
            return None
        
        # Get all parameter names from first client
        first_client = list(self.client_models.keys())[0]
        param_names = list(self.client_models[first_client].keys())
        
        # Calculate total weight
        total_weight = sum(self.client_weights.values())
        
        # Initialize global model
        global_model = {}
        
        # Aggregate parameters
        for param_name in param_names:
            # FedAvg: weighted average based on data size
            if self.model_aggregation == 'fedavg':
                global_model[param_name] = sum(
                    self.client_models[client][param_name] * self.client_weights[client] / total_weight
                    for client in self.client_models
                )
            # Add other aggregation methods here if needed
        
        self.global_model = global_model
        return global_model
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get the current global model.
        
        Returns:
            Dict[str, torch.Tensor]: Global model parameters
        """
        if self.global_model is None:
            self.global_model = self.aggregate_models()
        return self.global_model
    
    def get_global_graph(self) -> CausalRuleSet:
        """Get the current global causal graph.
        
        Returns:
            CausalRuleSet: Global causal graph
        """
        return self.global_causal_graph
    
    def _get_timestamp(self) -> int:
        """Get current timestamp.
        
        Returns:
            int: Current timestamp
        """
        return int(time.time())
    
    def _hash_graph(self, graph: CausalRuleSet) -> str:
        """Create a hash of a graph for verification.
        
        Args:
            graph (CausalRuleSet): Graph to hash
            
        Returns:
            str: Hash value
        """
        # Convert graph to dict, then to JSON string
        graph_str = json.dumps(graph.to_dict(), sort_keys=True)
        
        # Create hash
        return hashlib.sha256(graph_str.encode()).hexdigest()
    
    def _add_block(self, client_id: str, block_type: str, data: Dict):
        """Add a block to the block history.
        
        Args:
            client_id (str): Client that created the block
            block_type (str): Type of block ('graph_update' or 'model_update')
            data (Dict): Block data
        """
        # Get hash of previous block
        prev_hash = self._hash_block(self.blocks[-1]) if self.blocks else '0' * 64
        
        # Create block
        block = {
            'index': len(self.blocks),
            'timestamp': self._get_timestamp(),
            'client_id': client_id,
            'type': block_type,
            'data': data,
            'previous_hash': prev_hash
        }
        
        # Add block hash
        block['hash'] = self._hash_block(block)
        
        # Add to chain
        self.chain.append(block)
    
    def _hash_block(self, block: Dict) -> str:
        """Create a hash of a block for verification.
        
        Args:
            block (Dict): Block to hash
            
        Returns:
            str: Hash value
        """
        # Create a copy without the hash field
        block_copy = copy.deepcopy(block)
        if 'hash' in block_copy:
            del block_copy['hash']
        
        # Convert to JSON string
        block_str = json.dumps(block_copy, sort_keys=True)
        
        # Create hash
        return hashlib.sha256(block_str.encode()).hexdigest()


class FederatedClient:
    """Client for federated causal learning.
    
    This class represents a client in the federated learning system,
    managing local training and communication with the CausalDAO.
    
    Args:
        client_id (str): Unique identifier for the client
        model (nn.Module): Local model
        data_size (int): Size of local dataset
        local_causal_graph (CausalRuleSet, optional): Local causal graph
    """
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        data_size: int,
        local_causal_graph: Optional[CausalRuleSet] = None
    ):
        self.client_id = client_id
        self.model = model
        self.data_size = data_size
        self.local_causal_graph = local_causal_graph or CausalRuleSet()
        
        # Store initial parameters
        self.initial_params = self._get_model_parameters()
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 1
    ) -> Dict[str, float]:
        """Train the local model on local data.
        
        Args:
            dataloader: Local data loader
            optimizer: Optimization algorithm
            loss_fn: Loss function
            epochs: Number of training epochs
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        metrics = {'loss': 0.0}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                # Extract inputs and labels
                inputs, labels = batch
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get the model update (difference from initial parameters).
        
        Returns:
            Model parameter updates
        """
        # Get current parameters
        current_params = self._get_model_parameters()
        
        # Calculate differences
        updates = {}
        for name, param in current_params.items():
            if name in self.initial_params:
                updates[name] = param - self.initial_params[name]
        
        return updates
    
    def update_model(self, global_params: Dict[str, torch.Tensor]):
        """Update local model with global parameters.
        
        Args:
            global_params (Dict[str, torch.Tensor]): Global model parameters
        """
        current_params = self._get_model_parameters()
        
        # Update parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    param.copy_(global_params[name])
        
        # Update initial parameters
        self.initial_params = self._get_model_parameters()
    
    def discover_causal_graph(
        self,
        dataloader: torch.utils.data.DataLoader,
        significance_threshold: float = 0.05
    ) -> CausalRuleSet:
        """Discover causal relationships from local data.
        
        Args:
            dataloader: Local data loader
            significance_threshold: P-value threshold for causal discovery
            
        Returns:
            CausalRuleSet: Discovered causal graph
        """
        # This is a placeholder for actual causal discovery
        # In a real implementation, you would use methods like:
        # - PC algorithm
        # - FCI algorithm
        # - GES algorithm
        # - NOTEARS
        # - etc.
        
        # For now, we'll return the existing local graph
        # In a real implementation, this would discover causal
        # relationships from the data
        
        return self.local_causal_graph
    
    def update_local_graph(self, global_graph: CausalRuleSet, merge_method: str = 'union'):
        """Update local causal graph with global graph.
        
        Args:
            global_graph (CausalRuleSet): Global causal graph
            merge_method (str): Method for merging graphs
                - 'union': Include relationships from both graphs
                - 'intersection': Only include relationships in both graphs
                - 'global': Replace local with global graph
        """
        if merge_method == 'global':
            # Simply replace local graph with global
            self.local_causal_graph = copy.deepcopy(global_graph)
        
        elif merge_method == 'intersection':
            # Only keep relationships that exist in both graphs
            new_graph = CausalRuleSet()
            
            for cause in global_graph.get_all_causes():
                for global_rule in global_graph.get_rules_for_cause(cause):
                    # Check if this relationship exists in local graph
                    local_rules = self.local_causal_graph.get_rules_for_cause(cause)
                    for local_rule in local_rules:
                        if local_rule.effect == global_rule.effect:
                            # Calculate average strength
                            avg_strength = (local_rule.strength + global_rule.strength) / 2
                            
                            # Create new rule with average strength
                            new_rule = CausalRule(
                                cause=cause,
                                effect=global_rule.effect,
                                strength=avg_strength
                            )
                            
                            new_graph.add_rule(new_rule)
            
            self.local_causal_graph = new_graph
        
        elif merge_method == 'union':
            # Include relationships from both graphs
            new_graph = copy.deepcopy(self.local_causal_graph)
            
            for cause in global_graph.get_all_causes():
                for global_rule in global_graph.get_rules_for_cause(cause):
                    # Check if this relationship exists in local graph
                    exists = False
                    local_rules = self.local_causal_graph.get_rules_for_cause(cause)
                    
                    for local_rule in local_rules:
                        if local_rule.effect == global_rule.effect:
                            exists = True
                            break
                    
                    # Add if not exists
                    if not exists:
                        new_rule = CausalRule(
                            cause=cause,
                            effect=global_rule.effect,
                            strength=global_rule.strength
                        )
                        new_graph.add_rule(new_rule)
            
            self.local_causal_graph = new_graph 
    
    def _get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters as dictionary.
        
        Returns:
            Model parameters
        """
        return {name: param.clone().detach() for name, param in self.model.named_parameters()} 