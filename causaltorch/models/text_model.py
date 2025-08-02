"""
Text generation models with causal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from typing import Dict, List, Optional, Tuple, Union

from causaltorch.layers.meta_learning import CausalHyperNetwork

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    # Optional dependencies
    pass

from ..layers import CausalAttentionLayer
from ..rules import CausalRuleSet, CausalRule


class cnsg(nn.Module):
    """Causal Neuro-Symbolic GPT-2 model for text generation.
    
    This model extends GPT-2 with causal attention to enforce logical
    relationships in generated text.
    
    Args:
        pretrained_model_name (str): Name of pretrained GPT-2 model
        causal_rules (dict): Dictionary of causal rules to enforce
    """
    def __init__(self, pretrained_model_name="gpt2", causal_rules=None):
        super().__init__()
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        
        # Add causal attention layer
        self.causal_attn = CausalAttentionLayer(causal_rules or {})
        
        # Set tokenizer (will be initialized by user)
        self.tokenizer = None
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with causal constraints.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Target token IDs
            
        Returns:
            transformers.modeling_outputs.CausalLMOutputWithCrossAttentions:
                Model outputs with modified attention based on causal rules
        """
        # Run GPT-2 with output_attentions=True to get attention matrices
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True
        )
        
        # Get input text
        if self.tokenizer is not None and input_ids is not None:
            input_text = self.tokenizer.decode(input_ids[0])
            
            # Apply causal attention modifications
            if outputs.attentions is not None:
                # For simplicity, we only modify the last layer's attention
                self.causal_attn.tokenizer = self.tokenizer
                modified_attention = self.causal_attn(outputs.attentions[-1], input_text)
                
                # In a full implementation, we would use this modified attention
                # to recompute the final layer's outputs
                
        return outputs
    
    def generate(self, input_ids=None, max_length=None, **kwargs):
        """Generate text with causal constraints.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            max_length (int, optional): Maximum output length
            **kwargs: Additional generation parameters
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        # For now, we use GPT-2's generation method directly
        # In a full implementation, we would modify the generation
        # algorithm to incorporate causal constraints at each step
        return self.gpt2.generate(
            input_ids=input_ids,
            max_length=max_length,
            **kwargs
        )


class CausalTransformer(nn.Module):
    """Native causal transformer architecture with built-in causal reasoning.
    
    This model is designed from the ground up to integrate causal relationships
    at every level of the architecture, without depending on external models.
    
    Args:
        vocab_size (int): Size of vocabulary
        hidden_dim (int, optional): Hidden dimension size. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        causal_rules (Union[CausalRuleSet, Dict, None], optional): Causal rules. Defaults to None.
        sparsity (float, optional): Level of sparse activation (0-1). Defaults to 0.9.
        ethical_constitution (Optional[nn.Module], optional): Ethical constraints. Defaults to None.
    """
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12, 
        causal_rules: Union[CausalRuleSet, Dict, None] = None,
        sparsity: float = 0.9,
        ethical_constitution: Optional[nn.Module] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Convert dict to CausalRuleSet if needed
        if isinstance(causal_rules, dict):
            self.causal_rules = CausalRuleSet()
            for cause, effect_dict in causal_rules.items():
                self.causal_rules.add_rule(
                    CausalRule(cause, effect_dict["effect"], strength=effect_dict.get("strength", 0.5))
                )
        else:
            self.causal_rules = causal_rules or CausalRuleSet()
        
        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(1024, hidden_dim)
        
        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                causal_rules=self.causal_rules,
                sparsity=sparsity
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Ethical constitution
        self.ethical_constitution = ethical_constitution
        
        # Tokenizer (will be set by user)
        self.tokenizer = None
        
        # Causal discovery module
        self.causal_discovery = CausalDiscoveryModule(hidden_dim)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Either logits or (logits, attention_weights)
        """
        # Get embeddings
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Keep track of attention for interpretability
        attentions = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, attn = block(x, attention_mask=attention_mask)
            attentions.append(attn)
        
        # Get logits
        logits = self.output(x)
        
        if return_attention:
            return logits, attentions
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 100, 
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate text with ethical constraints and causal reasoning.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Generated token IDs [batch_size, seq_len+generated]
        """
        self.eval()
        
        # Ensure input_ids has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Maximum generation length
        max_new_tokens = max_length - input_ids.size(1)
        if max_new_tokens <= 0:
            return input_ids
        
        # Generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get predictions
                logits, attentions = self.forward(input_ids, return_attention=True)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Apply ethical checks if constitution is available
                if self.ethical_constitution and self.tokenizer:
                    # Decode the sequence so far to check it
                    current_text = self.tokenizer.decode(input_ids[0])
                    next_token_text = self.tokenizer.decode(next_token[0])
                    combined_text = current_text + next_token_text
                    
                    # Check if the new token would create unethical content
                    _, passed, _ = self.ethical_constitution(combined_text)
                    
                    # If not passed, try alternative tokens
                    if not passed:
                        attempts = 0
                        while not passed and attempts < 5:
                            # Try a different token
                            probs[0, next_token[0, 0]] = 0  # Zero out probability of problematic token
                            probs = probs / probs.sum()  # Renormalize
                            next_token = torch.multinomial(probs, num_samples=1)
                            
                            # Check again
                            next_token_text = self.tokenizer.decode(next_token[0])
                            combined_text = current_text + next_token_text
                            _, passed, _ = self.ethical_constitution(combined_text)
                            attempts += 1
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for end of generation
                if self.tokenizer and next_token[0, 0].item() == self.tokenizer.eos_token_id:
                    break
        
        return input_ids


class CausalTransformerBlock(nn.Module):
    """Transformer block with causal reasoning capabilities.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        causal_rules (CausalRuleSet, optional): Causal rules
        sparsity (float, optional): Level of sparse activation (0-1)
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        causal_rules: Optional[CausalRuleSet] = None, 
        sparsity: float = 0.9
    ):
        super().__init__()
        self.causal_attention = CausalSelfAttention(hidden_dim, num_heads, causal_rules)
        
        # Dynamic sparse feed-forward network
        self.sparse_ffn = SparseFFN(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            sparsity=sparsity
        )
        
        # Layer norms and residual connections
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the block.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            (output, attention_weights)
        """
        # Self-attention with causality
        residual = x
        x = self.ln1(x)
        x_attn, attention = self.causal_attention(x, attention_mask)
        x = residual + x_attn
        
        # Feed-forward
        residual = x
        x = self.ln2(x)
        x = residual + self.sparse_ffn(x)
        
        return x, attention


class CausalSelfAttention(nn.Module):
    """Self-attention with causal biasing based on causal rules.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        causal_rules (CausalRuleSet, optional): Causal rules
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        causal_rules: Optional[CausalRuleSet] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal_rules = causal_rules or CausalRuleSet()
        
        # Query, key, value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Causal graph projections
        if len(self.causal_rules) > 0:
            self.causal_biasing = CausalBiasingNetwork(self.causal_rules, hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the attention layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(~attention_mask, -10000.0)
        
        # Apply causal biasing if rules exist
        if hasattr(self, 'causal_biasing'):
            causal_bias = self.causal_biasing(x)
            causal_bias = causal_bias.view(batch_size, seq_len, self.num_heads, seq_len).transpose(1, 2)
            scores = scores + causal_bias
        
        # Compute attention weights
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        return output, attention


class CausalBiasingNetwork(nn.Module):
    """Network that biases attention scores based on causal rules.
    
    Args:
        causal_rules (CausalRuleSet): Set of causal rules
        hidden_dim (int): Hidden dimension size
    """
    def __init__(self, causal_rules: CausalRuleSet, hidden_dim: int):
        super().__init__()
        self.rules = causal_rules
        
        # Networks to convert hidden states to causal variables
        self.cause_extractors = nn.ModuleDict()
        self.effect_predictors = nn.ModuleDict()
        
        # Initialize extractors and predictors for each rule
        for cause, effect_dict in self.rules.items():
            effect = effect_dict["effect"]
            self.cause_extractors[cause] = nn.Linear(hidden_dim, 1)
            self.effect_predictors[effect] = nn.Linear(1, hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute causal biasing matrix for attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Causal bias matrix [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Initialize bias matrix
        bias = torch.zeros(batch_size, seq_len, seq_len, device=hidden_states.device)
        
        # Extract causal variables and compute bias
        for cause, effect_dict in self.rules.items():
            if cause in self.cause_extractors and effect_dict["effect"] in self.effect_predictors:
                # Extract cause variable from each token
                cause_scores = self.cause_extractors[cause](hidden_states).squeeze(-1)  # [batch, seq_len]
                
                # Compute expected effect based on cause
                expected_effect = cause_scores.unsqueeze(-1) * effect_dict["strength"]  # [batch, seq_len, 1]
                
                # Predict effect embedding
                effect_embedding = self.effect_predictors[effect_dict["effect"]](expected_effect)  # [batch, seq_len, hidden]
                
                # Compute similarity between effect embedding and all token embeddings
                similarity = torch.bmm(effect_embedding, hidden_states.transpose(1, 2))  # [batch, seq_len, seq_len]
                
                # Add to bias matrix
                bias = bias + similarity
        
        return bias


class SparseFFN(nn.Module):
    """Sparse feed-forward network using dynamic routing.
    
    Args:
        hidden_dim (int): Input/output dimension
        intermediate_dim (int): Intermediate dimension
        sparsity (float): Level of sparse activation (0-1)
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, sparsity: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.sparsity = sparsity
        
        # Sparse activation based on Lottery Ticket Hypothesis
        if sparsity > 0:
            self.sparse_router = nn.Linear(hidden_dim, intermediate_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse activation.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Regular FFN path
        h = F.gelu(self.fc1(x))
        
        # Apply sparsity if needed
        if self.sparsity > 0:
            # Generate sparse activation mask
            router_logits = self.sparse_router(x)
            top_k = max(1, int(router_logits.size(-1) * (1 - self.sparsity)))
            _, indices = torch.topk(router_logits, k=top_k, dim=-1)
            
            # Create sparse mask
            mask = torch.zeros_like(router_logits)
            mask.scatter_(-1, indices, 1.0)
            
            # Apply mask
            h = h * mask
        
        return self.fc2(h)


class CausalDiscoveryModule(nn.Module):
    """Module for discovering causal relationships in text.
    
    Args:
        hidden_dim (int): Hidden dimension size
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Networks to extract potential causes and effects
        self.cause_extractor = nn.Linear(hidden_dim, 128)
        self.effect_extractor = nn.Linear(hidden_dim, 128)
        
        # Network to predict causal strength
        self.strength_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Will be set by user
        self.tokenizer = None
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """Discover causal relationships in text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of discovered causal rules
        """
        # Simple pattern-based discovery for this implementation
        discovered_rules = {}
        
        if self.tokenizer is None:
            return discovered_rules
        
        # Decode text
        text = self.tokenizer.decode(input_ids[0])
        
        # Simple pattern matching to discover potential causal relationships
        patterns = [
            (r"(\w+) causes (\w+)", 0.9),
            (r"if (\w+) then (\w+)", 0.8),
            (r"(\w+) leads to (\w+)", 0.7),
            (r"(\w+) results in (\w+)", 0.8),
            (r"because of (\w+), (\w+)", 0.8),
            (r"(\w+) is why (\w+)", 0.7)
        ]
        
        for pattern, default_strength in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for cause, effect in matches:
                # Store discovered rule
                if cause not in discovered_rules:
                    discovered_rules[cause] = {"effect": effect, "strength": default_strength}
        
        return discovered_rules


class CausalLanguageModel(nn.Module):
    """Complete language model with causal reasoning capabilities.
    
    Args:
        vocab_size (int): Size of vocabulary
        hidden_dim (int, optional): Hidden dimension size. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        causal_rules (Union[CausalRuleSet, Dict, None], optional): Causal rules. Defaults to None.
        ethical_constitution (Optional[nn.Module], optional): Ethical constraints. Defaults to None.
    """
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12,
        causal_rules: Union[CausalRuleSet, Dict, None] = None,
        ethical_constitution: Optional[nn.Module] = None
    ):
        super().__init__()
        # Create the transformer
        self.transformer = CausalTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            causal_rules=causal_rules,
            ethical_constitution=ethical_constitution
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]
            
        Returns:
            Model outputs
        """
        outputs = self.transformer(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # Calculate language modeling loss
            shift_logits = outputs[..., :-1, :].contiguous() if isinstance(outputs, tuple) else outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return outputs with loss
        class ModelOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return ModelOutput(loss=loss, logits=outputs)
    
    def generate(self, input_ids=None, max_length=None, **kwargs):
        """Generate text with causal constraints.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum output length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        return self.transformer.generate(
            input_ids=input_ids,
            max_length=max_length,
            **kwargs
        )


class SelfEvolvingTextGenerator(nn.Module):
    """Text generator that can adapt its architecture to different tasks.
    
    Args:
        vocab_size (int): Size of vocabulary
        hypernetwork (Optional[nn.Module], optional): Hypernetwork for adaptation. Defaults to None.
    """
    def __init__(self, vocab_size: int, hypernetwork: Optional[nn.Module] = None):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Hypernetwork for generating task-specific architectures
        self.hypernetwork = hypernetwork or CausalHyperNetwork(
            input_dim=100,  # Graph representation size
            output_dim=768,  # Base hidden dimension
            hidden_dim=128,
            meta_hidden_dim=64
        )
        
        # Tokenizer and base architecture are defined dynamically
        self.causal_transformer = None
        self.current_task_embedding = None
    
    def adapt_to_task(self, task_description=None, causal_graph=None):
        """Evolve the architecture based on the task or causal graph.
        
        Args:
            task_description (Optional[str], optional): Task description. Defaults to None.
            causal_graph (Optional[CausalRuleSet], optional): Causal graph. Defaults to None.
            
        Returns:
            The adapted transformer model
        """
        # Extract or create causal graph from task description
        if causal_graph is None and task_description is not None:
            causal_graph = self.extract_causal_graph(task_description)
        
        # Generate task-specific transformer parameters
        params = self.hypernetwork(causal_graph)
        
        # Create or update the architecture
        if self.causal_transformer is None:
            self.causal_transformer = DynamicCausalTransformer(
                vocab_size=self.vocab_size,
                params=params
            )
        else:
            self.causal_transformer.update_parameters(params)
        
        # Store current task embedding
        self.current_task_embedding = causal_graph
        
        return self.causal_transformer
    
    def extract_causal_graph(self, task_description):
        """Extract causal structure from a text description.
        
        Args:
            task_description (str): Task description
            
        Returns:
            Extracted causal graph as a tensor
        """
        # Simple implementation using patterns
        graph_tensor = torch.zeros(10, 10)  # Simple 10-node graph
        
        # Extract causal relations from description
        patterns = [
            (r"(\w+) causes (\w+)", 0.9),
            (r"if (\w+) then (\w+)", 0.8),
            (r"(\w+) leads to (\w+)", 0.7)
        ]
        
        node_map = {}  # Map words to indices
        next_idx = 0
        
        for pattern, strength in patterns:
            matches = re.findall(pattern, task_description, re.IGNORECASE)
            for cause, effect in matches:
                # Get or assign indices
                if cause not in node_map:
                    if next_idx >= 10:
                        continue  # Skip if too many nodes
                    node_map[cause] = next_idx
                    next_idx += 1
                
                if effect not in node_map:
                    if next_idx >= 10:
                        continue  # Skip if too many nodes
                    node_map[effect] = next_idx
                    next_idx += 1
                
                # Set edge
                cause_idx = node_map[cause]
                effect_idx = node_map[effect]
                graph_tensor[cause_idx, effect_idx] = strength
        
        return graph_tensor.flatten()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Model outputs
        """
        if self.causal_transformer is None:
            raise ValueError("Model must be adapted to a task first using adapt_to_task()")
        return self.causal_transformer(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the adapted model.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Generated token IDs
        """
        if self.causal_transformer is None:
            raise ValueError("Model must be adapted to a task first using adapt_to_task()")
        return self.causal_transformer.generate(*args, **kwargs)


class DynamicCausalTransformer(nn.Module):
    """Transformer with dynamically generated parameters.
    
    Args:
        vocab_size (int): Size of vocabulary
        params (Dict[str, torch.Tensor]): Generated parameters
    """
    def __init__(self, vocab_size: int, params: Dict[str, torch.Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.params = params
        
        # Initialize with given parameters
        self._build_architecture()
    
    def _build_architecture(self):
        """Build architecture from parameters."""
        # For simplicity, assuming parameters define a basic transformer
        hidden_dim = self.params.get('hidden_dim', 768)
        num_layers = self.params.get('num_layers', 6)
        num_heads = self.params.get('num_heads', 12)
        
        # Create standard components
        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(1024, hidden_dim)
        self.output = nn.Linear(hidden_dim, self.vocab_size)
        
        # Create dynamic blocks from parameters
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Use parameters for this layer
            layer_prefix = f'layer_{i}_'
            layer_params = {k.replace(layer_prefix, ''): v 
                           for k, v in self.params.items() 
                           if k.startswith(layer_prefix)}
            
            # Create block with these parameters
            block = DynamicTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                params=layer_params
            )
            self.blocks.append(block)
    
    def update_parameters(self, params: Dict[str, torch.Tensor]):
        """Update architecture with new parameters.
        
        Args:
            params (Dict[str, torch.Tensor]): New parameters
        """
        self.params = params
        # Rebuild architecture with new parameters
        self._build_architecture()
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Model outputs
        """
        # Get embeddings
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Get logits
        logits = self.output(x)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text using the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Generated token IDs [batch_size, seq_len+generated]
        """
        # Basic generation loop (simplified for brevity)
        self.eval()
        
        # Ensure input_ids has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Maximum generation length
        max_new_tokens = max_length - input_ids.size(1)
        if max_new_tokens <= 0:
            return input_ids
        
        # Generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get predictions
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class DynamicTransformerBlock(nn.Module):
    """Transformer block with dynamically generated parameters.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        params (Dict[str, torch.Tensor]): Generated parameters
    """
    def __init__(self, hidden_dim: int, num_heads: int, params: Dict[str, torch.Tensor]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.params = params
        
        # Create standard components - could be replaced with dynamic params
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attention_mask=None):
        """Forward pass through the block.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Standard transformer block operation for simplicity
        # Could be enhanced with dynamic parameter application
        residual = x
        x = self.ln1(x)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = ~attention_mask
        x_attn, _ = self.attn(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1), 
                             key_padding_mask=attn_mask)
        x = residual + x_attn.transpose(0, 1)
        
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        
        return x


class FewShotCausalTransformer(nn.Module):
    """Transformer optimized for few-shot learning of causal patterns.
    
    Args:
        vocab_size (int): Size of vocabulary
        base_model (Optional[nn.Module], optional): Base transformer model. Defaults to None.
        hidden_dim (int, optional): Hidden dimension size. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 12.
    """
    def __init__(
        self, 
        vocab_size: int, 
        base_model: Optional[nn.Module] = None,
        hidden_dim: int = 768, 
        num_layers: int = 12
    ):
        super().__init__()
        # Either use provided model or create a new one
        self.transformer = base_model or CausalTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Learnable example encoder
        self.example_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fusion layer to integrate few-shot examples
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Store examples in memory
        self.examples = []
        self.extracted_rules = CausalRuleSet()
    
    def learn_from_examples(self, examples):
        """Adapt the model to few-shot examples of causal patterns.
        
        Args:
            examples: List of (prompt, completion) pairs
        """
        # Store examples
        self.examples = examples
        
        # Extract causal rules from examples
        for prompt, completion in examples:
            full_text = prompt + " " + completion
            
            # Use simple pattern matching for demonstration
            patterns = [
                (r"(\w+) causes (\w+)", 0.9),
                (r"if (\w+) then (\w+)", 0.8),
                (r"when (\w+), (\w+)", 0.7),
                (r"(\w+) leads to (\w+)", 0.85)
            ]
            
            for pattern, strength in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                for cause, effect in matches:
                    self.extracted_rules.add_rule(cause, effect, strength)
        
        # Update the transformer's causal rules
        self.transformer.causal_rules = self.extracted_rules
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass with few-shot conditioning.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Model outputs
        """
        # Just use the transformer with the learned rules
        return self.transformer(input_ids, attention_mask)
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text conditioned on few-shot examples.
        
        Args:
            prompt: Input prompt text or token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text or token IDs
        """
        # Check if we need to tokenize the prompt
        if isinstance(prompt, str) and hasattr(self.transformer, 'tokenizer'):
            input_ids = self.transformer.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = prompt if isinstance(prompt, torch.Tensor) else torch.tensor([prompt])
        
        # Generate with updated causal rules
        return self.transformer.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )


class CausalHyperNetwork(nn.Module):
    """Meta-network that generates task-specific architectures from causal graphs.
    
    Args:
        input_dim (int): Input dimension (causal graph representation)
        output_dim (int): Output dimension (model hidden size)
        hidden_dim (int, optional): HyperNetwork hidden dimension. Defaults to 128.
        meta_hidden_dim (int, optional): Meta-network hidden dimension. Defaults to 64.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 128, 
        meta_hidden_dim: int = 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.meta_hidden_dim = meta_hidden_dim
        
        # Number of layers in generated networks
        self.num_layers = 6
        
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
        
        # Create generators for different parts of the architecture
        # 1. Self-attention parameters
        self.qkv_generator = nn.Linear(meta_hidden_dim, output_dim * 3)
        
        # 2. FFN parameters
        self.ffn_generator = nn.Linear(meta_hidden_dim, output_dim * 4)
        
        # 3. Layer normalization parameters
        self.norm_generator = nn.Linear(meta_hidden_dim, output_dim * 2)
        
        # Initialize for each layer
        for _ in range(self.num_layers):
            # Generate weights for this layer
            self.weight_generators.append(nn.Linear(meta_hidden_dim, hidden_dim * output_dim))
            # Generate biases for this layer
            self.bias_generators.append(nn.Linear(meta_hidden_dim, output_dim))
    
    def forward(self, causal_graph):
        """Generate neural network parameters based on causal graph.
        
        Args:
            causal_graph: Tensor representation of causal graph
            
        Returns:
            Dictionary of generated parameters
        """
        # Encode the causal graph
        encoded_graph = self.graph_encoder(causal_graph)
        
        # Initialize parameter dictionary
        params = {
            'hidden_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': 12  # Fixed for simplicity
        }
        
        # Generate parameters for each layer
        for i, (w_gen, b_gen) in enumerate(zip(self.weight_generators, self.bias_generators)):
            # Generate weights and biases
            weight = w_gen(encoded_graph).view(-1, self.hidden_dim, self.output_dim)
            bias = b_gen(encoded_graph)
            
            # Store in parameter dictionary
            params[f'layer_{i}_weight'] = weight
            params[f'layer_{i}_bias'] = bias
            
            # Generate attention parameters
            qkv = self.qkv_generator(encoded_graph)
            params[f'layer_{i}_qkv'] = qkv
            
            # Generate FFN parameters
            ffn = self.ffn_generator(encoded_graph)
            params[f'layer_{i}_ffn'] = ffn
            
            # Generate normalization parameters
            norm = self.norm_generator(encoded_graph)
            params[f'layer_{i}_norm'] = norm
        
        return params
    
    def generate_architecture(self, causal_graph):
        """Generate a task-specific neural network architecture.
        
        Args:
            causal_graph: Tensor representation of causal graph
            
        Returns:
            Neural network model with task-specific architecture
        """
        # Generate parameters
        params = self.forward(causal_graph)
        
        # Create network with these parameters
        return DynamicCausalTransformer(
            vocab_size=50000,  # Default value, should be set properly in practice
            params=params
        )


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning for causal tasks.
    
    Args:
        model (nn.Module): Base model to meta-train
        inner_lr (float, optional): Learning rate for task adaptation. Defaults to 0.01.
        meta_lr (float, optional): Learning rate for meta-update. Defaults to 0.001.
        num_inner_steps (int, optional): Number of adaptation steps. Defaults to 5.
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
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    
    def adapt_to_task(self, X_support, y_support, loss_fn):
        """Adapt the model to a new task using support data.
        
        Args:
            X_support: Support set inputs
            y_support: Support set targets
            loss_fn: Loss function
            
        Returns:
            Adapted model parameters
        """
        # Clone model parameters
        theta = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Adaptation steps
        for _ in range(self.num_inner_steps):
            # Compute loss on support set
            outputs = self.model(X_support)
            loss = loss_fn(outputs, y_support)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            
            # Update parameters
            theta = {name: param - self.inner_lr * grad 
                    for (name, param), grad in zip(theta.items(), grads)}
        
        return theta
    
    def meta_train(self, tasks, loss_fn):
        """Meta-train the model across multiple tasks.
        
        Args:
            tasks: List of (X_support, y_support, X_query, y_query) tuples
            loss_fn: Loss function
            
        Returns:
            Meta-loss value
        """
        # Zero meta-gradients
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        for X_support, y_support, X_query, y_query in tasks:
            # Adapt to task
            adapted_params = self.adapt_to_task(X_support, y_support, loss_fn)
            
            # Compute loss with adapted parameters on query set
            with torch.set_grad_enabled(True):
                outputs = self.model(X_query)
                task_loss = loss_fn(outputs, y_query)
                meta_loss += task_loss
        
        # Average meta-loss
        meta_loss = meta_loss / len(tasks)
        
        # Backward and optimize
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class MultimodalCausalTransformer(CausalTransformer):
    """Transformer with cross-modal causal reasoning capabilities.
    
    Args:
        vocab_size (int): Size of vocabulary
        image_embedding_dim (int, optional): Image embedding dimension. Defaults to 512.
        hidden_dim (int, optional): Hidden dimension size. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        causal_rules (Union[CausalRuleSet, Dict, None], optional): Causal rules. Defaults to None.
        sparsity (float, optional): Level of sparse activation (0-1). Defaults to 0.9.
        ethical_constitution (Optional[nn.Module], optional): Ethical constraints. Defaults to None.
    """
    def __init__(
        self, 
        vocab_size: int, 
        image_embedding_dim: int = 512, 
        hidden_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12, 
        causal_rules: Union[CausalRuleSet, Dict, None] = None,
        sparsity: float = 0.9,
        ethical_constitution: Optional[nn.Module] = None
    ):
        super().__init__(
            vocab_size, 
            hidden_dim, 
            num_layers, 
            num_heads, 
            causal_rules,
            sparsity,
            ethical_constitution
        )
        
        # Image embedding components
        self.image_projection = nn.Linear(image_embedding_dim, hidden_dim)
        self.image_token_id = vocab_size  # Special token for image representation
        
        # Image encoder (placeholder - would use a real vision model)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, image_embedding_dim)
        )
        
        # Cross-modal causal rules
        self.cross_modal_rules = CausalRuleSet()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Process both text and image inputs with causal reasoning.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            images: Image tensors [batch_size, channels, height, width]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Either logits or (logits, attention_weights)
        """
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)
        
        # Process images if provided
        if images is not None:
            # Get image embeddings
            image_embeddings = self.image_projection(self.image_encoder(images))
            
            if input_ids is not None:
                # For multimodal input: insert image embeddings after the first token
                # This is a simplified approach - more sophisticated strategies could be used
                text_embeddings = self.token_embedding(input_ids)
                
                # Create new sequence with images
                seq_len = input_ids.size(1)
                new_seq_len = seq_len + 1  # +1 for image token
                
                # Create embeddings for combined sequence
                combined_embeddings = torch.zeros(
                    batch_size, new_seq_len, self.hidden_dim, 
                    device=input_ids.device
                )
                
                # Insert text and image embeddings
                combined_embeddings[:, 0] = text_embeddings[:, 0]  # First token
                combined_embeddings[:, 1] = image_embeddings  # Image embedding
                if seq_len > 1:
                    combined_embeddings[:, 2:] = text_embeddings[:, 1:]  # Rest of text
                
                # Update attention mask if provided
                if attention_mask is not None:
                    new_mask = torch.ones(
                        batch_size, new_seq_len, 
                        device=attention_mask.device
                    )
                    new_mask[:, 0] = attention_mask[:, 0]
                    new_mask[:, 2:] = attention_mask[:, 1:]
                    attention_mask = new_mask
                
                # Add positional embeddings
                positions = torch.arange(0, new_seq_len, device=input_ids.device).unsqueeze(0)
                x = combined_embeddings + self.position_embedding(positions)
            else:
                # Image-only input
                x = image_embeddings.unsqueeze(1)  # [batch, 1, hidden]
                
                # Add positional embedding
                positions = torch.zeros(1, 1, device=images.device).long()
                x = x + self.position_embedding(positions)
        else:
            # Text-only input (same as parent class)
            positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Process through transformer blocks (same as parent class)
        attentions = []
        for block in self.blocks:
            x, attn = block(x, attention_mask=attention_mask)
            attentions.append(attn)
        
        # Get logits
        logits = self.output(x)
        
        if return_attention:
            return logits, attentions
        return logits


class CounterfactualCausalTransformer(CausalTransformer):
    """Transformer with counterfactual reasoning capabilities.
    
    Args:
        vocab_size (int): Size of vocabulary
        hidden_dim (int, optional): Hidden dimension size. Defaults to 768.
        num_layers (int, optional): Number of transformer layers. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        causal_rules (Union[CausalRuleSet, Dict, None], optional): Causal rules. Defaults to None.
        sparsity (float, optional): Level of sparse activation (0-1). Defaults to 0.9.
    """
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12, 
        causal_rules: Union[CausalRuleSet, Dict, None] = None,
        sparsity: float = 0.9
    ):
        super().__init__(
            vocab_size, 
            hidden_dim, 
            num_layers, 
            num_heads, 
            causal_rules,
            sparsity
        )
        
        # Counterfactual intervention module
        self.intervention_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def imagine(
        self,
        prompt: Union[str, torch.Tensor],
        interventions: List[Dict[str, Union[str, float]]],
        max_length: int = 100, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> List[str]:
        """Generate counterfactual text with causal interventions.
        
        Args:
            prompt: Input prompt text or token IDs
            interventions: List of intervention specifications
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_samples: Number of samples to generate
            
        Returns:
            List of generated counterfactual texts
        """
        # Check if we need to tokenize the prompt
        if isinstance(prompt, str) and hasattr(self, 'tokenizer'):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = prompt if isinstance(prompt, torch.Tensor) else torch.tensor([prompt])
        
        # Create copies of input_ids for each sample
        if num_samples > 1:
            input_ids = input_ids.repeat(num_samples, 1)
        
        # Make a copy of the original rules
        original_rules = self.causal_rules.copy() if self.causal_rules else CausalRuleSet()
        
        # Apply interventions to causal rules
        for intervention in interventions:
            variable = intervention.get('variable')
            value = intervention.get('value', 1.0)
            
            if variable in original_rules:
                # Modify existing rule
                effect = original_rules[variable]['effect']
                self.causal_rules[variable] = {"effect": effect, "strength": value}
            elif 'effect' in intervention:
                # Add new rule
                effect = intervention.get('effect')
                strength = intervention.get('strength', 0.9)
                self.causal_rules.add_rule(variable, effect, strength)
        
        # Generate with modified rules
        output_ids = self.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Convert to text if tokenizer available
        outputs = []
        if hasattr(self, 'tokenizer'):
            for i in range(output_ids.size(0)):
                text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
                outputs.append(text)
        else:
            outputs = output_ids
        
        # Restore original rules
        self.causal_rules = original_rules
        
        return outputs


class EthicalConstitution(nn.Module):
    """Module that enforces ethical constraints during generation.
    
    Args:
        rules (Optional[List[Dict]], optional): List of ethical rules. Defaults to None.
        log_violations (bool, optional): Whether to log violations. Defaults to True.
    """
    def __init__(
        self, 
        rules: Optional[List[Dict]] = None, 
        log_violations: bool = True
    ):
        super().__init__()
        self.rules = rules or []
        self.log_violations = log_violations
        
        # Keep a record of violations
        self.violations = []
    
    def add_rule(self, rule: Dict):
        """Add a new rule to the constitution.
        
        Args:
            rule: Rule specification
        """
        self.rules.append(rule)
        
        # Sort rules by priority (higher first)
        self.rules.sort(key=lambda r: r.get('priority', 0), reverse=True)
    
    def forward(self, generated_output: str) -> Tuple[str, bool, List[Dict]]:
        """Check output against ethical rules.
        
        Args:
            generated_output: Generated text
            
        Returns:
            (output, passed, violations)
        """
        output = generated_output
        passed = True
        violations = []
        
        for rule in self.rules:
            # Check if output complies with rule
            detection_fn = rule.get('detection_fn')
            if detection_fn and callable(detection_fn):
                complies, reason = detection_fn(output)
                
                if not complies:
                    # Record violation
                    violation = {
                        "rule": rule.get('name', 'Unnamed rule'),
                        "reason": reason or "No reason provided",
                        "action": rule.get('action', 'warn')
                    }
                    violations.append(violation)
                    
                    # Handle violation based on rule's action
                    action = rule.get('action', 'warn')
                    if action == "block":
                        passed = False
                    elif action == "modify" and 'modification_fn' in rule:
                        # Apply modification function if provided
                        modification_fn = rule['modification_fn']
                        if callable(modification_fn):
                            output = modification_fn(output)
        
        # Record violations if logging is enabled
        if self.log_violations and violations:
            self.violations.extend(violations)
        
        return output, passed, violations


def load_default_ethical_rules():
    """Load default set of ethical rules.
    
    Returns:
        List of ethical rules
    """
    # Simple pattern-based checks (would be more sophisticated in practice)
    def check_harmful_content(text):
        harmful_patterns = [
            r"\b(kill|hurt|harm|injure)\b.*\bpeople\b",
            r"\b(hate|hateful)\b.*\b(speech|content)\b",
            r"\b(illegal|harmful)\b.*\b(activities|actions)\b"
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Matched harmful pattern: {pattern}"
        
        return True, None
    
    def check_privacy_violation(text):
        privacy_patterns = [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone numbers
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
        
        for pattern in privacy_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Contains private information"
        
        return True, None
    
    # Define default rules
    rules = [
        {
            "name": "no_harm",
            "description": "Do not generate content that could cause harm to humans",
            "detection_fn": check_harmful_content,
            "action": "block",
            "priority": 10
        },
        {
            "name": "privacy",
            "description": "Protect private information in generated content",
            "detection_fn": check_privacy_violation,
            "action": "modify",
            "priority": 9
        }
    ]
    
    return rules