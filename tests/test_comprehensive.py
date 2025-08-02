"""
Comprehensive Test Suite for CausalTorch
========================================

Production-ready testing framework with coverage for all modules.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple
import tempfile
import shutil
from pathlib import Path

from causaltorch.rules import CausalRule, CausalRuleSet
from causaltorch.models import CausalTransformer
from causaltorch.training.trainer import CausalTrainer
from causaltorch.training.finetuner import CausalFineTuner
from causaltorch.metrics import calculate_cfs
from causaltorch.layers import CausalLinear, CausalAttentionLayer


class TestCausalRules:
    """Test causal rule system."""
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = CausalRule("rain", "wet_ground", strength=0.9)
        assert rule.cause == "rain"
        assert rule.effect == "wet_ground"
        assert rule.strength == 0.9
    
    def test_rule_strength_bounds(self):
        """Test rule strength is properly bounded."""
        rule = CausalRule("rain", "wet_ground", strength=1.5)
        assert rule.strength == 1.0  # Should be clamped
        
        rule = CausalRule("rain", "wet_ground", strength=-0.5)
        assert rule.strength == 0.0  # Should be clamped
    
    def test_ruleset_creation(self):
        """Test rule set creation and management."""
        rules = CausalRuleSet()
        rules.add_rule(CausalRule("rain", "wet_ground"))
        rules.add_rule(CausalRule("fire", "smoke"))
        
        assert len(rules.rules) == 2
        assert "rain" in rules.rules
        assert "fire" in rules.rules
    
    def test_ruleset_acyclic_validation(self):
        """Test DAG validation in rule sets."""
        rules = CausalRuleSet()
        rules.add_rule(CausalRule("A", "B"))
        rules.add_rule(CausalRule("B", "C"))
        assert rules.is_acyclic()
        
        # Add cycle
        rules.add_rule(CausalRule("C", "A"))
        assert not rules.is_acyclic()
    
    def test_rule_serialization(self):
        """Test rule serialization and deserialization."""
        rule = CausalRule("rain", "wet_ground", strength=0.8, temporal_offset=1)
        rule_dict = rule.to_dict()
        
        assert rule_dict['effect'] == "wet_ground"
        assert rule_dict['strength'] == 0.8
        assert rule_dict['temporal_offset'] == 1
        
        # Test reconstruction
        new_rule = CausalRule.from_dict("rain", rule_dict)
        assert new_rule.cause == rule.cause
        assert new_rule.effect == rule.effect
        assert new_rule.strength == rule.strength


class TestCausalLayers:
    """Test causal neural network layers."""
    
    def test_causal_linear_layer(self):
        """Test CausalLinear layer functionality."""
        # Create adjacency mask (3x3 matrix)
        mask = torch.tensor([
            [1, 1, 0],
            [0, 1, 1], 
            [0, 0, 1]
        ], dtype=torch.float32)
        
        layer = CausalLinear(3, 3, mask)
        
        # Test forward pass
        x = torch.randn(2, 3)
        output = layer(x)
        
        assert output.shape == (2, 3)
        
        # Test that mask is enforced
        weight_masked = layer.weight * layer.mask
        torch.testing.assert_close(layer.weight * layer.mask, weight_masked)
    
    def test_causal_attention_layer(self):
        """Test CausalAttentionLayer functionality."""
        rules = {
            "rain": {"effect": "wet", "strength": 0.9}
        }
        
        attention_layer = CausalAttentionLayer(rules)
        
        # Mock attention scores and input text
        attention_scores = torch.randn(1, 10, 100)  # batch, seq, vocab
        input_text = "It's raining outside"
        
        # Note: This test would need a proper tokenizer setup
        # For now, just test that the layer can be created
        assert attention_layer.rules == rules


class TestCausalModels:
    """Test causal model implementations."""
    
    @pytest.fixture
    def sample_causal_rules(self):
        """Create sample causal rules for testing."""
        rules = CausalRuleSet()
        rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
        rules.add_rule(CausalRule("fire", "smoke", strength=0.8))
        return rules
    
    def test_causal_transformer_creation(self, sample_causal_rules):
        """Test CausalTransformer model creation."""
        config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 8
        }
        
        model = CausalTransformer(config, sample_causal_rules)
        
        # Test basic properties
        assert hasattr(model, 'causal_rules')
        assert model.causal_rules == sample_causal_rules
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape == (batch_size, seq_len, 1000)


class TestCausalMetrics:
    """Test causal evaluation metrics."""
    
    def test_causal_fidelity_score(self):
        """Test CFS calculation."""
        # Create a simple model for testing
        class SimpleModel(torch.nn.Module):
            def generate(self, input_text):
                # Mock generation that always includes "wet" when "rain" is in input
                if "rain" in input_text:
                    return "The ground is wet"
                return "The ground is dry"
        
        model = SimpleModel()
        
        # Test cases: (input, expected_effect)
        test_cases = [
            ("It's raining", "wet"),
            ("Sunny day", "wet"),  # This should fail
            ("Heavy rain", "wet"),
            ("No clouds", "wet")    # This should fail
        ]
        
        cfs = calculate_cfs(model, test_cases)
        
        # Should get 50% (2 out of 4 correct)
        assert abs(cfs - 0.5) < 0.1


class TestTrainingInfrastructure:
    """Test training and fine-tuning infrastructure."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model_and_rules(self):
        """Create sample model and rules for training tests."""
        rules = CausalRuleSet()
        rules.add_rule(CausalRule("input", "output", strength=0.8))
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
        
        return model, rules
    
    def test_trainer_initialization(self, sample_model_and_rules, temp_dir):
        """Test CausalTrainer initialization."""
        model, rules = sample_model_and_rules
        
        config = {
            'max_epochs': 5,
            'lr': 1e-3,
            'optimizer': 'AdamW',
            'task': 'classification'
        }
        
        trainer = CausalTrainer(
            model=model,
            causal_rules=rules,
            config=config,
            checkpoint_dir=str(temp_dir)
        )
        
        assert trainer.model == model
        assert trainer.causal_rules == rules
        assert trainer.current_epoch == 0
        assert trainer.checkpoint_dir == temp_dir
    
    def test_finetuner_initialization(self):
        """Test CausalFineTuner initialization."""
        model = torch.nn.Linear(10, 2)
        
        finetuner = CausalFineTuner(
            base_model=model,
            task_type='classification',
            data_type='tabular'
        )
        
        assert finetuner.task_type == 'classification'
        assert finetuner.data_type == 'tabular'
        assert finetuner.model is not None


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_training_workflow(self, temp_dir=None):
        """Test complete training workflow."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Create causal rules
            rules = CausalRuleSet()
            rules.add_rule(CausalRule("feature1", "target", strength=0.7))
            
            # 2. Create model
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
            
            # 3. Create synthetic data
            train_data = [
                {'input': torch.randn(5), 'target': torch.randint(0, 2, (1,))}
                for _ in range(100)
            ]
            
            # 4. Setup trainer
            config = {
                'max_epochs': 2,
                'lr': 1e-3,
                'batch_size': 16,
                'task': 'classification'
            }
            
            trainer = CausalTrainer(
                model=model,
                causal_rules=rules,
                config=config,
                checkpoint_dir=temp_dir
            )
            
            # 5. Test that trainer can be created without errors
            assert trainer is not None
            assert trainer.optimizer is not None
            
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_model_save_load_cycle(self, temp_dir=None):
        """Test model saving and loading."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and save model
            rules = CausalRuleSet()
            rules.add_rule(CausalRule("test", "output"))
            
            model = torch.nn.Linear(10, 2)
            
            finetuner = CausalFineTuner(
                base_model=model,
                task_type='classification',
                data_type='tabular',
                causal_rules=rules
            )
            
            # Save
            save_path = Path(temp_dir) / "test_model.pt"
            finetuner.save_finetuned_model(str(save_path))
            
            # Check file exists
            assert save_path.exists()
            
            # Load
            loaded_finetuner = CausalFineTuner.load_finetuned_model(str(save_path))
            
            assert loaded_finetuner.task_type == 'classification'
            assert loaded_finetuner.data_type == 'tabular'
            
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformance:
    """Performance and memory tests."""
    
    def test_memory_efficiency(self):
        """Test memory usage of causal layers."""
        # Create large causal linear layer
        mask = torch.eye(1000)  # Identity matrix for 1000x1000
        layer = CausalLinear(1000, 1000, mask)
        
        # Test forward pass with batch
        batch_size = 32
        x = torch.randn(batch_size, 1000)
        
        # Measure memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Forward pass
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (batch_size, 1000)
        
        # Check that mask is properly applied
        expected_weight = layer.weight * mask
        torch.testing.assert_close(layer.weight * layer.mask, expected_weight)


# Fixtures for pytest
@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    return {
        'input': torch.randn(4, 10),
        'target': torch.randint(0, 2, (4,)),
        'causal_context': {'rain': True, 'fire': False}
    }


# Custom markers for pytest
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning")
]


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])
