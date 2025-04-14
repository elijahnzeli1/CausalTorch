"""
CausalTorch Meta-Learning Module
================================

This module contains components for implementing causal meta-learning,
including HyperNetworks that can dynamically generate task-specific
neural architectures based on causal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..rules import CausalRuleSet


class CausalHyperNetwork(nn.Module):
    """A meta-learning network that generates task-specific neural architectures
    based on causal graphs.
    
    This HyperNetwork takes a causal graph as input and outputs parameters for
    a task-specific neural network. It learns to map causal relationships to
    optimized network architectures.
    
    Args:
        input_dim (int): Dimension of the input features
        output_dim (int): Dimension of the output features
        hidden_dim (int): Dimension of hidden layers in generated networks
        meta_hidden_dim (int): Dimension of hidden layers in the meta-learner
        num_layers (int): Number of layers in generated networks
        activation (str): Activation function for generated networks
    """
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        hidden_dim: int = 256,
        meta_hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Dictionary of activation functions
        self.activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU()
        }
        self.activation = self.activations.get(activation, nn.ReLU())
        
        # Calculate total number of parameters needed for the generated network
        # First layer: input_dim × hidden_dim + hidden_dim (weights + biases)
        # Hidden layers: hidden_dim × hidden_dim + hidden_dim
        # Last layer: hidden_dim × output_dim + output_dim
        total_params = (input_dim * hidden_dim + hidden_dim)
        for _ in range(num_layers - 2):
            total_params += (hidden_dim * hidden_dim + hidden_dim)
        total_params += (hidden_dim * output_dim + output_dim)
        
        # Meta-network to process causal graph
        self.graph_encoder = nn.Sequential(
            nn.Linear(input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.ReLU(),
        )
        
        # Parameter generators for each layer
        self.weight_generators = nn.ModuleList()
        self.bias_generators = nn.ModuleList()
        
        # First layer
        self.weight_generators.append(
            nn.Linear(meta_hidden_dim, input_dim * hidden_dim)
        )
        self.bias_generators.append(
            nn.Linear(meta_hidden_dim, hidden_dim)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.weight_generators.append(
                nn.Linear(meta_hidden_dim, hidden_dim * hidden_dim)
            )
            self.bias_generators.append(
                nn.Linear(meta_hidden_dim, hidden_dim)
            )
        
        # Output layer
        self.weight_generators.append(
            nn.Linear(meta_hidden_dim, hidden_dim * output_dim)
        )
        self.bias_generators.append(
            nn.Linear(meta_hidden_dim, output_dim)
        )
    
    def forward(self, causal_graph: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Generate network parameters based on causal graph.
        
        Args:
            causal_graph (torch.Tensor): Adjacency matrix representation
                of the causal graph, shape [batch_size, nodes, nodes]
                
        Returns:
            Dict: Dictionary of weight and bias parameters for generated network
        """
        batch_size = causal_graph.size(0)
        
        # Flatten the adjacency matrix
        flattened_graph = causal_graph.view(batch_size, -1)
        
        # Make sure input dimension matches
        if flattened_graph.size(1) != self.input_dim:
            # If dimensions don't match, pad or truncate
            if flattened_graph.size(1) < self.input_dim:
                padding = torch.zeros(batch_size, self.input_dim - flattened_graph.size(1),
                                     device=flattened_graph.device)
                flattened_graph = torch.cat([flattened_graph, padding], dim=1)
            else:
                flattened_graph = flattened_graph[:, :self.input_dim]
        
        # Encode the graph structure
        encoded_graph = self.graph_encoder(flattened_graph)
        
        # Generate weights and biases for each layer
        params = {}
        
        for i, (w_gen, b_gen) in enumerate(zip(self.weight_generators, self.bias_generators)):
            if i == 0:
                # First layer
                w = w_gen(encoded_graph).view(batch_size, self.hidden_dim, self.input_dim)
                b = b_gen(encoded_graph)
            elif i == self.num_layers - 1:
                # Last layer
                w = w_gen(encoded_graph).view(batch_size, self.output_dim, self.hidden_dim)
                b = b_gen(encoded_graph)
            else:
                # Hidden layers
                w = w_gen(encoded_graph).view(batch_size, self.hidden_dim, self.hidden_dim)
                b = b_gen(encoded_graph)
            
            params[f'weight_{i}'] = w
            params[f'bias_{i}'] = b
        
        return params
    
    def generate_architecture(self, causal_graph: torch.Tensor) -> nn.Module:
        """Generate a task-specific neural network based on causal graph.
        
        Args:
            causal_graph (torch.Tensor): Adjacency matrix representation
                of the causal graph
                
        Returns:
            nn.Module: Generated neural network
        """
        params = self.forward(causal_graph)
        
        # Create a dynamic network using the generated parameters
        class DynamicNetwork(nn.Module):
            def __init__(self, params, num_layers, activation):
                super().__init__()
                self.params = params
                self.num_layers = num_layers
                self.activation = activation
            
            def forward(self, x):
                batch_size = self.params['weight_0'].size(0)
                
                # Handle the case of multiple samples with batch size of 1
                if x.size(0) > 1 and batch_size == 1:
                    # Expand parameters to match input batch size
                    expanded_params = {}
                    for k, v in self.params.items():
                        expanded_params[k] = v.expand(x.size(0), *v.shape[1:])
                    params = expanded_params
                else:
                    params = self.params
                
                output = x
                
                for i in range(self.num_layers):
                    # Apply weights and biases
                    w = params[f'weight_{i}']
                    b = params[f'bias_{i}']
                    
                    # For each sample in the batch, apply its specific weights
                    outputs = []
                    for j in range(x.size(0)):
                        sample = output[j:j+1]
                        sample_w = w[min(j, w.size(0)-1)]
                        sample_b = b[min(j, b.size(0)-1)]
                        
                        # Apply linear transformation
                        outputs.append(F.linear(sample, sample_w, sample_b))
                    
                    output = torch.cat(outputs, dim=0)
                    
                    # Apply activation except on the last layer
                    if i < self.num_layers - 1:
                        output = self.activation(output)
                
                return output
        
        return DynamicNetwork(params, self.num_layers, self.activation)


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning implementation for causal tasks.
    
    This class implements MAML (Finn et al., 2017) adapted for causal learning tasks.
    It learns a meta-model that can quickly adapt to new causal tasks with few examples.
    
    Args:
        model (nn.Module): Base model to meta-train
        inner_lr (float): Learning rate for task-specific (inner) adaptation
        meta_lr (float): Learning rate for meta-update
        num_inner_steps (int): Number of gradient steps for task adaptation
    """
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor, loss_fn: Callable) -> nn.Module:
        """Adapt the model to a new task using the support set.
        
        Args:
            support_x (torch.Tensor): Support set inputs
            support_y (torch.Tensor): Support set targets
            loss_fn (Callable): Loss function for task adaptation
            
        Returns:
            nn.Module: Adapted model for the specific task
        """
        # Create a clone of the model for task-specific adaptation
        adapted_params = [p.clone().requires_grad_(True) for p in self.model.parameters()]
        adapted_model = type(self.model)(*self.model.init_args)
        
        # Update the adapted model's parameters for inner loop optimization
        for _ in range(self.num_inner_steps):
            # Forward pass with adapted parameters
            with torch.enable_grad():
                predictions = adapted_model(support_x, params=adapted_params)
                loss = loss_fn(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            
            # Update adapted parameters
            adapted_params = [p - self.inner_lr * g for p, g in zip(adapted_params, grads)]
        
        # Return the adapted model
        return adapted_model
    
    def meta_train(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: Callable
    ) -> float:
        """Perform meta-training on a batch of tasks.
        
        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function for both adaptation and meta-update
            
        Returns:
            float: Average meta-loss across tasks
        """
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in task_batch:
            # Adapt the model to the current task
            adapted_model = self.adapt(support_x, support_y, loss_fn)
            
            # Evaluate on query set
            predictions = adapted_model(query_x)
            task_loss = loss_fn(predictions, query_y)
            
            meta_loss += task_loss
        
        # Average meta-loss
        meta_loss = meta_loss / len(task_batch)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() 