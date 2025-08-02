"""
CausalTorch Core Architecture Analysis & Implementation
======================================================

This document outlines how CausalTorch implements the core architecture principles
shown in the provided diagram, ensuring standalone operation for both from-scratch
model building and fine-tuning scenarios.

Architecture Principles (from diagram):
1. PyTorch → CausalTorch Core → Specialized Modules
2. Causal Reasoning Engine as the central hub
3. Standalone capabilities for all use cases
4. Integrated intervention, counterfactual, and regularization APIs

Current Implementation Status:
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod


class CausalTorchCore(nn.Module):
    """
    Core CausalTorch module that provides the foundational causal reasoning
    capabilities. All other modules inherit from or integrate with this core.
    
    This implements the central "CausalTorch Core" from the architecture diagram.
    """
    
    def __init__(self, causal_config: Optional[Dict] = None):
        super().__init__()
        self.causal_config = causal_config or {}
        self.causal_reasoning_engine = CausalReasoningEngine(self.causal_config)
        self.intervention_api = InterventionAPI(self.causal_reasoning_engine)
        self.counterfactual_engine = CounterfactualEngine(self.causal_reasoning_engine)
        self.causal_regularization = CausalRegularization(self.causal_reasoning_engine)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Base forward pass with causal reasoning."""
        # Apply causal reasoning to input
        causal_context = self.causal_reasoning_engine(x, **kwargs)
        return causal_context
    
    def enable_intervention_mode(self, interventions: Dict[str, Any]):
        """Enable intervention API for counterfactual reasoning."""
        return self.intervention_api.apply_interventions(interventions)
    
    def compute_causal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with causal regularization."""
        return self.causal_regularization.regularized_loss(predictions, targets)


class CausalReasoningEngine(nn.Module):
    """
    Central causal reasoning engine that powers all CausalTorch functionality.
    
    This is the core component from the architecture diagram that connects
    to all other specialized modules.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 512)
        
        # Input projection to ensure proper dimensions
        self.input_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.causal_graph = self._build_causal_graph()
        self.reasoning_layers = nn.ModuleList([
            CausalReasoningLayer(self.hidden_dim)
            for _ in range(config.get('num_reasoning_layers', 3))
        ])
        
    def _build_causal_graph(self) -> 'CausalGraph':
        """Build the causal graph from configuration."""
        return CausalGraph(self.config.get('causal_rules', []))
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Apply causal reasoning to input tensor."""
        context = context or {}
        
        # Handle different input dimensions
        original_shape = x.shape
        if x.dim() == 2:
            # (batch, features) -> project to hidden_dim if needed
            if x.shape[-1] != self.hidden_dim:
                # Need to project to correct dimension
                if not hasattr(self, '_dynamic_projection') or self._dynamic_projection.in_features != x.shape[-1]:
                    self._dynamic_projection = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
                x = self._dynamic_projection(x)
            # Add sequence dimension: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # (batch, seq, features) -> ensure correct feature dimension
            if x.shape[-1] != self.hidden_dim:
                if not hasattr(self, '_dynamic_projection') or self._dynamic_projection.in_features != x.shape[-1]:
                    self._dynamic_projection = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
                batch_size, seq_len, _ = x.shape
                x = self._dynamic_projection(x.view(-1, x.shape[-1])).view(batch_size, seq_len, self.hidden_dim)
        
        # Apply causal reasoning layers
        causal_features = x
        for layer in self.reasoning_layers:
            causal_features = layer(causal_features, self.causal_graph)
        
        # Return to original shape if needed
        if len(original_shape) == 2 and causal_features.dim() == 3:
            causal_features = causal_features.squeeze(1)
        
        return {
            'causal_features': causal_features,
            'causal_graph_state': self.causal_graph.get_state() if hasattr(self.causal_graph, 'get_state') else {},
            'reasoning_confidence': self._compute_confidence(causal_features)
        }
    
    def _compute_confidence(self, features: torch.Tensor) -> torch.Tensor:
        """Compute confidence in causal reasoning."""
        return torch.sigmoid(features.norm(dim=-1))


class CausalReasoningLayer(nn.Module):
    """Individual layer for causal reasoning."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.causal_attention = CausalAttention(hidden_dim)
        self.causal_mlp = CausalMLP(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, causal_graph=None) -> torch.Tensor:
        """Apply causal reasoning with graph constraints."""
        # Causal attention
        attn_out = self.causal_attention(x, causal_graph)
        x = self.norm1(x + attn_out)
        
        # Causal MLP
        mlp_out = self.causal_mlp(x, causal_graph)
        x = self.norm2(x + mlp_out)
        
        return x


class CausalAttention(nn.Module):
    """Attention mechanism with causal constraints."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, causal_graph=None) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # Add sequence dimension for 2D inputs (batch, hidden_dim) -> (batch, 1, hidden_dim)
            x = x.unsqueeze(1)
            added_seq_dim = True
        else:
            added_seq_dim = False
            
        batch_size, seq_len, _ = x.shape
        
        # Compute attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Create causal mask (lower triangular) only if sequence length > 1
        if seq_len > 1:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            # Expand mask to match scores dimensions [batch, heads, seq, seq]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)
        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_proj(attn_out)
        
        # Remove added sequence dimension if we added it
        if added_seq_dim:
            output = output.squeeze(1)
            
        return output


class CausalMLP(nn.Module):
    """MLP with causal constraints."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, causal_graph=None) -> torch.Tensor:
        """Apply MLP with causal regularization."""
        x = self.fc1(x)
        x = self.activation(x)
        
        # Apply causal constraints if graph is provided
        if causal_graph is not None and hasattr(causal_graph, 'apply_constraints'):
            x = causal_graph.apply_constraints(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class InterventionAPI:
    """
    API for performing causal interventions (do-calculus).
    
    Implements the "Intervention API" component from the architecture diagram.
    """
    
    def __init__(self, reasoning_engine: CausalReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.active_interventions = {}
    
    def apply_interventions(self, interventions: Dict[str, Any]) -> 'InterventionContext':
        """Apply causal interventions."""
        self.active_interventions.update(interventions)
        return InterventionContext(self, interventions)
    
    def remove_interventions(self, variable_names: List[str]):
        """Remove specific interventions."""
        for name in variable_names:
            self.active_interventions.pop(name, None)
    
    def clear_interventions(self):
        """Clear all interventions."""
        self.active_interventions.clear()


class CounterfactualEngine:
    """
    Engine for counterfactual reasoning and generation.
    
    Implements the "Counterfactual Engine" component from the architecture diagram.
    """
    
    def __init__(self, reasoning_engine: CausalReasoningEngine):
        self.reasoning_engine = reasoning_engine
    
    def generate_counterfactual(self, input_data: torch.Tensor, 
                              interventions: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate counterfactual examples."""
        # Store original state
        original_output = self.reasoning_engine(input_data)
        
        # Apply interventions
        with self.reasoning_engine.causal_graph.intervention_context(interventions):
            counterfactual_output = self.reasoning_engine(input_data)
        
        return {
            'original': original_output['causal_features'],
            'counterfactual': counterfactual_output['causal_features'],
            'intervention_effect': counterfactual_output['causal_features'] - original_output['causal_features']
        }


class CausalRegularization:
    """
    Causal regularization for training stability.
    
    Implements the "Causal Regularization" component from the architecture diagram.
    """
    
    def __init__(self, reasoning_engine: CausalReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.regularization_strength = 0.1
    
    def regularized_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with causal regularization."""
        base_loss = nn.functional.mse_loss(predictions, targets)
        
        # Add causal consistency regularization
        causal_consistency = self._compute_causal_consistency()
        
        # Add causal strength regularization
        causal_strength = self._compute_causal_strength_penalty()
        
        total_loss = base_loss + self.regularization_strength * (causal_consistency + causal_strength)
        return total_loss
    
    def _compute_causal_consistency(self) -> torch.Tensor:
        """Compute causal consistency penalty."""
        # Placeholder - would implement graph consistency checks
        return torch.tensor(0.0)
    
    def _compute_causal_strength_penalty(self) -> torch.Tensor:
        """Compute penalty for overly strong causal effects."""
        # Placeholder - would implement strength regularization
        return torch.tensor(0.0)


class CausalGraph:
    """Represents and manages the causal graph structure."""
    
    def __init__(self, rules):
        self.rules = rules
        self.adjacency_matrix = self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build adjacency matrix from rules."""
        # Simplified implementation
        size = len(self.rules.variables) if hasattr(self.rules, 'variables') else 10
        return torch.eye(size)
    
    def get_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Get causal attention mask."""
        # For now, return causal mask (lower triangular)
        return torch.tril(torch.ones(seq_len, seq_len))
    
    def apply_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal constraints to tensor."""
        # Placeholder - would apply graph-based constraints
        return x
    
    def get_state(self) -> Dict:
        """Get current graph state."""
        return {"num_nodes": self.adjacency_matrix.shape[0]}
    
    def intervention_context(self, interventions: Dict):
        """Context manager for applying interventions."""
        return InterventionContext(self, interventions)


class InterventionContext:
    """Context manager for causal interventions."""
    
    def __init__(self, graph_or_api, interventions: Dict):
        self.target = graph_or_api
        self.interventions = interventions
        self.original_state = None
    
    def __enter__(self):
        # Store original state and apply interventions
        self.original_state = getattr(self.target, 'state', None)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state is not None:
            pass  # Would restore state


# Specialized model classes that inherit from CausalTorchCore

class FromScratchModelBuilder(CausalTorchCore):
    """
    Implements the "From-Scratch Model Building" capability.
    
    Provides tools for building causal AI models from scratch.
    """
    
    def __init__(self, model_config: Dict):
        super().__init__(model_config.get('causal_config', {}))
        self.model_config = model_config
        self.architecture_generator = self._create_architecture_generator()
    
    def _create_architecture_generator(self):
        """Create architecture generator based on causal constraints."""
        return CausalArchitectureGenerator(self.causal_reasoning_engine)
    
    def build_model(self, task_type: str, **kwargs) -> nn.Module:
        """Build a model from scratch for the specified task."""
        if task_type == "text_generation":
            return self._build_text_model(**kwargs)
        elif task_type == "image_generation":
            return self._build_image_model(**kwargs)
        elif task_type == "classification":
            return self._build_classifier(**kwargs)
        elif task_type == "regression":
            return self._build_regressor(**kwargs)
        elif task_type == "reinforcement_learning":
            return self._build_rl_agent(**kwargs)
        elif task_type == "policy_network":
            return self._build_policy_network(**kwargs)
        elif task_type == "value_network":
            return self._build_value_network(**kwargs)
        elif task_type == "q_network":
            return self._build_q_network(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _build_text_model(self, vocab_size: int, max_length: int, **kwargs) -> nn.Module:
        """Build causal text generation model."""
        from causaltorch.models.text_model import CausalTextGenerator
        return CausalTextGenerator(
            vocab_size=vocab_size,
            max_length=max_length,
            causal_core=self,
            **kwargs
        )
    
    def _build_image_model(self, **kwargs) -> nn.Module:
        """Build causal image generation model."""
        from causaltorch.models.image_model import CNSGImageGenerator
        return CNSGImageGenerator(causal_core=self, **kwargs)
    
    def _build_classifier(self, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
        """Build causal classification model."""
        return CausalClassifier(input_dim, num_classes, self)
    
    def _build_regressor(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build causal regression model."""
        return CausalRegressor(input_dim, output_dim, self)
    
    def _build_rl_agent(self, state_dim: int, action_dim: int, agent_type: str = "dqn", **kwargs) -> nn.Module:
        """Build causal reinforcement learning agent."""
        return CausalRLAgent(state_dim, action_dim, agent_type, self, **kwargs)
    
    def _build_policy_network(self, state_dim: int, action_dim: int, **kwargs) -> nn.Module:
        """Build causal policy network for RL."""
        return CausalPolicyNetwork(state_dim, action_dim, self, **kwargs)
    
    def _build_value_network(self, state_dim: int, **kwargs) -> nn.Module:
        """Build causal value network for RL."""
        return CausalValueNetwork(state_dim, self, **kwargs)
    
    def _build_q_network(self, state_dim: int, action_dim: int, **kwargs) -> nn.Module:
        """Build causal Q-network for RL."""
        return CausalQNetwork(state_dim, action_dim, self, **kwargs)


class PretrainedModelFineTuner(CausalTorchCore):
    """
    Implements the "Pre-trained Model Fine-tuning" capability.
    
    Provides tools for fine-tuning existing models with causal constraints.
    """
    
    def __init__(self, pretrained_model: nn.Module, causal_config: Dict):
        super().__init__(causal_config)
        self.pretrained_model = pretrained_model
        self.adapter_layers = self._create_causal_adapters()
    
    def _create_causal_adapters(self) -> nn.ModuleDict:
        """Create causal adapter layers."""
        adapters = nn.ModuleDict()
        
        # Add causal adapters to different layers of pretrained model
        for name, module in self.pretrained_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name:  # Ensure name is not empty
                # Replace dots and other invalid characters with underscores for module names
                clean_name = name.replace('.', '_').replace('-', '_').replace(' ', '_')
                if clean_name and clean_name != '':  # Double check name is valid
                    adapters[clean_name] = CausalAdapter(module, self.causal_reasoning_engine)
        
        # If no adapters were created, create a default one
        if len(adapters) == 0:
            adapters['default_adapter'] = CausalAdapter(
                nn.Linear(1, 1), self.causal_reasoning_engine
            )
        
        return adapters
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with causal fine-tuning."""
        # Apply causal reasoning first
        causal_context = super().forward(x, **kwargs)
        
        # Apply adapted pretrained model
        output = self._forward_with_adapters(x, causal_context)
        
        return output
    
    def _forward_with_adapters(self, x: torch.Tensor, causal_context: Dict) -> torch.Tensor:
        """Forward pass through pretrained model with causal adapters."""
        # This would hook into the pretrained model's forward pass
        # and apply causal adapters at appropriate layers
        return self.pretrained_model(x)


class CausalAdapter(nn.Module):
    """Adapter layer for adding causal reasoning to pretrained modules."""
    
    def __init__(self, original_module: nn.Module, reasoning_engine: CausalReasoningEngine):
        super().__init__()
        self.original_module = original_module
        self.reasoning_engine = reasoning_engine
        self.causal_projection = nn.Linear(
            original_module.out_features if hasattr(original_module, 'out_features') else 512,
            original_module.out_features if hasattr(original_module, 'out_features') else 512
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal adaptation."""
        # Original computation
        original_out = self.original_module(x)
        
        # Apply causal reasoning
        causal_context = self.reasoning_engine(x)
        causal_features = causal_context['causal_features']
        
        # Combine original and causal features
        adapted_out = original_out + self.causal_projection(causal_features)
        
        return adapted_out


class CausalArchitectureGenerator:
    """Generates neural architectures based on causal constraints."""
    
    def __init__(self, reasoning_engine: CausalReasoningEngine):
        self.reasoning_engine = reasoning_engine
    
    def generate_architecture(self, task_constraints: Dict) -> nn.Module:
        """Generate architecture based on task and causal constraints."""
        # This would implement meta-learning for architecture generation
        # based on causal graph structure
        pass


class CausalClassifier(nn.Module):
    """Classification model with causal reasoning."""
    
    def __init__(self, input_dim: int, num_classes: int, causal_core: CausalTorchCore):
        super().__init__()
        self.causal_core = causal_core
        # Use the causal core's hidden dimension for the classifier input
        hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal reasoning
        causal_context = self.causal_core(x)
        causal_features = causal_context['causal_features']
        
        # Classification
        return self.classifier(causal_features)


class CausalRegressor(nn.Module):
    """Regression model with causal reasoning."""
    
    def __init__(self, input_dim: int, output_dim: int, causal_core: CausalTorchCore):
        super().__init__()
        self.causal_core = causal_core
        # Use the causal core's hidden dimension for the regressor input
        hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal reasoning
        causal_context = self.causal_core(x)
        causal_features = causal_context['causal_features']
        
        # Regression
        return self.regressor(causal_features)


# ============================================================================
# REINFORCEMENT LEARNING MODULE WITH EPISODIC MEMORY
# ============================================================================

class EpisodicMemory:
    """
    Episodic memory for RL agents to store and retrieve past experiences.
    
    Features:
    - Stores state-action-reward-next_state transitions
    - Causal relationship tracking between actions and outcomes
    - Experience prioritization based on causal strength
    - Memory consolidation for long-term storage
    """
    
    def __init__(self, capacity: int = 100000, causal_threshold: float = 0.1):
        self.capacity = capacity
        self.causal_threshold = causal_threshold
        self.memory = []
        self.position = 0
        self.causal_weights = []
        
    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float, 
             next_state: torch.Tensor, done: bool, causal_strength: float = 1.0):
        """Store a transition in episodic memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.causal_weights.append(0.0)
        
        # Store experience with causal metadata
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'causal_strength': causal_strength,
            'timestamp': len(self.memory)
        }
        
        self.memory[self.position] = experience
        self.causal_weights[self.position] = causal_strength
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, use_causal_priority: bool = True) -> List[Dict]:
        """Sample experiences from memory, optionally prioritizing by causal strength."""
        if use_causal_priority and len(self.causal_weights) > 0:
            # Sample based on causal strength priorities
            weights = torch.tensor(self.causal_weights[:len(self.memory)])
            weights = torch.softmax(weights / 0.1, dim=0)  # Temperature scaling
            indices = torch.multinomial(weights, batch_size, replacement=True)
        else:
            # Uniform random sampling
            indices = torch.randint(0, len(self.memory), (batch_size,))
        
        return [self.memory[i] for i in indices]
    
    def get_causal_episodes(self, min_strength: float = None) -> List[Dict]:
        """Retrieve episodes with significant causal relationships."""
        min_strength = min_strength or self.causal_threshold
        return [exp for exp in self.memory if exp and exp['causal_strength'] >= min_strength]
    
    def consolidate_memory(self, forget_ratio: float = 0.1):
        """Consolidate memory by forgetting low-causal-strength experiences."""
        if len(self.memory) < self.capacity:
            return
        
        # Sort by causal strength and keep the most causally significant
        indexed_memory = [(i, exp, weight) for i, (exp, weight) in 
                         enumerate(zip(self.memory, self.causal_weights)) if exp]
        indexed_memory.sort(key=lambda x: x[2], reverse=True)
        
        # Keep top experiences and randomly sample from the rest
        keep_count = int(len(indexed_memory) * (1 - forget_ratio))
        self.memory = [item[1] for item in indexed_memory[:keep_count]]
        self.causal_weights = [item[2] for item in indexed_memory[:keep_count]]
        self.position = len(self.memory) % self.capacity
    
    def __len__(self):
        return len([exp for exp in self.memory if exp is not None])


class CausalRLAgent(nn.Module):
    """
    Complete Reinforcement Learning Agent with Causal Reasoning and Episodic Memory.
    
    Supports multiple RL algorithms:
    - Deep Q-Network (DQN)
    - Policy Gradient (PG)
    - Actor-Critic (A2C)
    - Proximal Policy Optimization (PPO)
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_type: str, 
                 causal_core: CausalTorchCore, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type.lower()
        self.causal_core = causal_core
        self.hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        
        # Configuration
        self.memory_capacity = kwargs.get('memory_capacity', 100000)
        self.batch_size = kwargs.get('batch_size', 64)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.1)
        
        # Episodic Memory
        self.episodic_memory = EpisodicMemory(self.memory_capacity)
        
        # Build agent networks based on type
        self._build_agent_networks()
        
        # Causal intervention tracking for RL
        self.intervention_history = []
        
    def _build_agent_networks(self):
        """Build networks based on agent type."""
        if self.agent_type == "dqn":
            self.q_network = CausalQNetwork(self.state_dim, self.action_dim, self.causal_core)
            self.target_q_network = CausalQNetwork(self.state_dim, self.action_dim, self.causal_core)
            self.update_target_network()
            
        elif self.agent_type == "policy_gradient" or self.agent_type == "pg":
            self.policy_network = CausalPolicyNetwork(self.state_dim, self.action_dim, self.causal_core)
            
        elif self.agent_type == "actor_critic" or self.agent_type == "a2c":
            self.policy_network = CausalPolicyNetwork(self.state_dim, self.action_dim, self.causal_core)
            self.value_network = CausalValueNetwork(self.state_dim, self.causal_core)
            
        elif self.agent_type == "ppo":
            self.policy_network = CausalPolicyNetwork(self.state_dim, self.action_dim, self.causal_core)
            self.value_network = CausalValueNetwork(self.state_dim, self.causal_core)
            self.old_policy_network = CausalPolicyNetwork(self.state_dim, self.action_dim, self.causal_core)
            
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def select_action(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """Select action based on current state using causal reasoning."""
        # Apply causal reasoning to state
        causal_context = self.causal_core(state)
        causal_state = causal_context['causal_features']
        
        if self.agent_type == "dqn":
            return self._select_action_dqn(causal_state, explore)
        elif self.agent_type in ["policy_gradient", "pg", "actor_critic", "a2c", "ppo"]:
            return self._select_action_policy(causal_state, explore)
        else:
            raise ValueError(f"Action selection not implemented for {self.agent_type}")
    
    def _select_action_dqn(self, causal_state: torch.Tensor, explore: bool) -> torch.Tensor:
        """DQN action selection with epsilon-greedy."""
        if explore and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (causal_state.shape[0],))
        else:
            with torch.no_grad():
                q_values = self.q_network(causal_state)
                return q_values.argmax(dim=-1)
    
    def _select_action_policy(self, causal_state: torch.Tensor, explore: bool) -> torch.Tensor:
        """Policy-based action selection."""
        with torch.no_grad():
            action_probs = self.policy_network(causal_state)
            if explore:
                action_dist = torch.distributions.Categorical(action_probs)
                return action_dist.sample()
            else:
                return action_probs.argmax(dim=-1)
    
    def store_experience(self, state: torch.Tensor, action: torch.Tensor, reward: float,
                        next_state: torch.Tensor, done: bool):
        """Store experience in episodic memory with causal strength calculation."""
        # Calculate causal strength between action and reward
        causal_strength = self._compute_causal_strength(state, action, reward, next_state)
        
        # Store in episodic memory
        self.episodic_memory.push(state, action, reward, next_state, done, causal_strength)
    
    def _compute_causal_strength(self, state: torch.Tensor, action: torch.Tensor, 
                               reward: float, next_state: torch.Tensor) -> float:
        """Compute causal strength between action and outcome."""
        # Apply causal reasoning to compute intervention effect
        with torch.no_grad():
            # Baseline state processing
            baseline_context = self.causal_core(state)
            
            # Create action intervention
            action_intervention = {'action': action.float().mean().item()}
            
            # Process with intervention
            with self.causal_core.intervention_api.apply_interventions(action_intervention):
                intervention_context = self.causal_core(state)
            
            # Compute causal strength based on feature differences and reward
            feature_diff = (intervention_context['causal_features'] - 
                          baseline_context['causal_features']).abs().mean()
            
            # Weight by reward magnitude
            causal_strength = feature_diff.item() * abs(reward)
            
        return min(causal_strength, 10.0)  # Clamp to reasonable range
    
    def learn(self) -> Dict[str, float]:
        """Learn from episodic memory using appropriate algorithm."""
        if len(self.episodic_memory) < self.batch_size:
            return {'loss': 0.0}
        
        if self.agent_type == "dqn":
            return self._learn_dqn()
        elif self.agent_type in ["policy_gradient", "pg"]:
            return self._learn_policy_gradient()
        elif self.agent_type in ["actor_critic", "a2c"]:
            return self._learn_actor_critic()
        elif self.agent_type == "ppo":
            return self._learn_ppo()
        else:
            raise ValueError(f"Learning not implemented for {self.agent_type}")
    
    def _learn_dqn(self) -> Dict[str, float]:
        """DQN learning with causal experience replay."""
        experiences = self.episodic_memory.sample(self.batch_size, use_causal_priority=True)
        
        states = torch.stack([exp['state'] for exp in experiences])
        actions = torch.stack([exp['action'] for exp in experiences])
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in experiences])
        dones = torch.tensor([exp['done'] for exp in experiences], dtype=torch.bool)
        
        # Reshape states and actions for proper dimensions
        states = states.view(len(experiences), -1)  # Flatten state to [batch, state_dim]
        next_states = next_states.view(len(experiences), -1)
        actions = actions.view(-1)  # Flatten actions to [batch]
        
        # Apply causal reasoning to states
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with causal regularization
        q_loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
        causal_loss = self.causal_core.compute_causal_loss(current_q_values.squeeze(), target_q_values)
        
        total_loss = q_loss + 0.1 * causal_loss
        
        return {'q_loss': q_loss.item(), 'causal_loss': causal_loss.item(), 'total_loss': total_loss.item()}
    
    def _learn_policy_gradient(self) -> Dict[str, float]:
        """Policy gradient learning with causal rewards."""
        experiences = self.episodic_memory.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in experiences])
        actions = torch.stack([exp['action'] for exp in experiences])
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32)
        
        # Compute policy loss with causal weighting
        action_probs = self.policy_network(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        
        # Weight rewards by causal strength
        causal_weights = torch.tensor([exp['causal_strength'] for exp in experiences])
        weighted_rewards = rewards * causal_weights
        
        policy_loss = -(log_probs.squeeze() * weighted_rewards).mean()
        
        return {'policy_loss': policy_loss.item()}
    
    def _learn_actor_critic(self) -> Dict[str, float]:
        """Actor-Critic learning with causal advantage."""
        experiences = self.episodic_memory.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in experiences])
        actions = torch.stack([exp['action'] for exp in experiences])
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in experiences])
        dones = torch.tensor([exp['done'] for exp in experiences], dtype=torch.bool)
        
        # Compute values and advantages
        values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        
        # Causal advantage computation
        td_targets = rewards + self.gamma * next_values * ~dones
        advantages = td_targets - values
        
        # Actor loss
        action_probs = self.policy_network(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss with causal regularization
        critic_loss = nn.functional.mse_loss(values, td_targets.detach())
        causal_loss = self.causal_core.compute_causal_loss(values, td_targets.detach())
        
        total_critic_loss = critic_loss + 0.1 * causal_loss
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'causal_loss': causal_loss.item(),
            'total_critic_loss': total_critic_loss.item()
        }
    
    def _learn_ppo(self) -> Dict[str, float]:
        """PPO learning with causal clipping."""
        experiences = self.episodic_memory.sample(self.batch_size)
        
        states = torch.stack([exp['state'] for exp in experiences])
        actions = torch.stack([exp['action'] for exp in experiences])
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32)
        
        # Get old action probabilities
        with torch.no_grad():
            old_action_probs = self.old_policy_network(states)
            old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        # Get current action probabilities
        current_action_probs = self.policy_network(states)
        current_log_probs = torch.log(current_action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        # Compute PPO clipped loss with causal weighting
        ratios = torch.exp(current_log_probs - old_log_probs)
        causal_weights = torch.tensor([exp['causal_strength'] for exp in experiences])
        weighted_rewards = rewards * causal_weights
        
        clip_epsilon = 0.2
        clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
        
        policy_loss = -torch.min(ratios * weighted_rewards, clipped_ratios * weighted_rewards).mean()
        
        return {'ppo_loss': policy_loss.item()}
    
    def update_target_network(self):
        """Update target network for DQN."""
        if hasattr(self, 'target_q_network'):
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def save_intervention_episode(self, interventions: Dict[str, Any], episode_reward: float):
        """Save intervention episode for causal analysis."""
        self.intervention_history.append({
            'interventions': interventions,
            'episode_reward': episode_reward,
            'timestamp': len(self.intervention_history)
        })
    
    def get_causal_analysis(self) -> Dict[str, Any]:
        """Analyze causal relationships learned by the agent."""
        if len(self.intervention_history) == 0:
            return {'message': 'No intervention history available'}
        
        # Analyze intervention effects
        intervention_effects = {}
        for episode in self.intervention_history:
            for key, value in episode['interventions'].items():
                if key not in intervention_effects:
                    intervention_effects[key] = []
                intervention_effects[key].append(episode['episode_reward'])
        
        # Compute average effects
        causal_effects = {}
        for key, rewards in intervention_effects.items():
            causal_effects[key] = {
                'mean_reward': sum(rewards) / len(rewards),
                'reward_variance': torch.var(torch.tensor(rewards)).item(),
                'sample_count': len(rewards)
            }
        
        return {
            'causal_effects': causal_effects,
            'total_episodes': len(self.intervention_history),
            'memory_size': len(self.episodic_memory)
        }


class CausalPolicyNetwork(nn.Module):
    """Policy network with causal reasoning for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, causal_core: CausalTorchCore):
        super().__init__()
        self.causal_core = causal_core
        hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        causal_context = self.causal_core(state)
        causal_features = causal_context['causal_features']
        return self.policy_head(causal_features)


class CausalValueNetwork(nn.Module):
    """Value network with causal reasoning for RL agents."""
    
    def __init__(self, state_dim: int, causal_core: CausalTorchCore):
        super().__init__()
        self.causal_core = causal_core
        hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        causal_context = self.causal_core(state)
        causal_features = causal_context['causal_features']
        return self.value_head(causal_features)


class CausalQNetwork(nn.Module):
    """Q-network with causal reasoning for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, causal_core: CausalTorchCore):
        super().__init__()
        self.causal_core = causal_core
        hidden_dim = causal_core.causal_reasoning_engine.hidden_dim
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        causal_context = self.causal_core(state)
        causal_features = causal_context['causal_features']
        return self.q_head(causal_features)


# Example usage demonstrating the architecture
def demonstrate_core_architecture():
    """Demonstrate the core architecture principles."""
    
    print("🏗️ CausalTorch Core Architecture Demonstration")
    print("=" * 60)
    
    # 1. From-Scratch Model Building
    print("\n1. 📝 From-Scratch Model Building")
    model_config = {
        'causal_config': {
            'hidden_dim': 512,
            'num_reasoning_layers': 3,
            'causal_rules': [
                {'cause': 'input', 'effect': 'hidden', 'strength': 0.8},
                {'cause': 'hidden', 'effect': 'output', 'strength': 0.9}
            ]
        }
    }
    
    builder = FromScratchModelBuilder(model_config)
    text_model = builder.build_model('text_generation', vocab_size=1000, max_length=128)
    print(f"✅ Built text model: {type(text_model).__name__}")
    
    # 2. Pre-trained Model Fine-tuning
    print("\n2. 🔧 Pre-trained Model Fine-tuning")
    pretrained = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    finetuner = PretrainedModelFineTuner(pretrained, model_config['causal_config'])
    print(f"✅ Created fine-tuner with {len(finetuner.adapter_layers)} causal adapters")
    
    # 3. Causal Reasoning Engine
    print("\n3. 🧠 Causal Reasoning Engine")
    reasoning_engine = CausalReasoningEngine(model_config['causal_config'])
    test_input = torch.randn(2, 10, 512)
    causal_output = reasoning_engine(test_input)
    print(f"✅ Causal reasoning output keys: {list(causal_output.keys())}")


class CausalGraph:
    """Simple causal graph implementation."""
    
    def __init__(self, causal_rules: List[Dict] = None):
        self.rules = causal_rules or []
        self.nodes = set()
        self.edges = []
        self.active_interventions = {}
        
        # Build graph from rules
        for rule in self.rules:
            cause = rule.get('cause', '')
            effect = rule.get('effect', '')
            strength = rule.get('strength', 1.0)
            
            self.nodes.add(cause)
            self.nodes.add(effect)
            self.edges.append((cause, effect, strength))
    
    def get_state(self) -> Dict:
        """Get current graph state."""
        return {
            'nodes': list(self.nodes),
            'edges': self.edges,
            'num_rules': len(self.rules),
            'active_interventions': self.active_interventions
        }
    
    def apply_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal constraints to tensor."""
        # Simple constraint application - can be enhanced
        return x * 0.9  # Apply small dampening as causal constraint
    
    def intervention_context(self, interventions: Dict[str, Any]):
        """Context manager for applying interventions."""
        return InterventionContext(self, interventions)


class InterventionContext:
    """Context manager for causal interventions."""
    
    def __init__(self, causal_graph: CausalGraph, interventions: Dict[str, Any]):
        self.causal_graph = causal_graph
        self.interventions = interventions
        self.previous_interventions = {}
    
    def __enter__(self):
        # Save previous interventions and apply new ones
        self.previous_interventions = self.causal_graph.active_interventions.copy()
        self.causal_graph.active_interventions.update(self.interventions)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous interventions
        self.causal_graph.active_interventions = self.previous_interventions
