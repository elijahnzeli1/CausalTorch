"""
CausalTorch Configuration Management
===================================

Centralized configuration system for production deployments.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "CausalTransformer"
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 1000
    dropout: float = 0.1
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Causal-specific
    causal_lr: float = 1e-4
    causal_weight_decay: float = 1e-5
    
    # Loss weights
    loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                'task_loss': 1.0,
                'causal_loss': 0.5,
                'sparsity': 0.1,
                'smoothness': 0.1
            }


@dataclass
class DataConfig:
    """Data configuration."""
    data_type: str = "text"
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    preprocessing: Dict[str, Any] = None
    augmentation: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = {}
        if self.augmentation is None:
            self.augmentation = {}


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    experiment_name: str = "causal_experiment"
    project_name: str = "causaltorch"
    tags: list = None
    notes: str = ""
    log_interval: int = 100
    checkpoint_interval: int = 10
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "causaltorch"
    wandb_entity: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class CausalConfig:
    """Causal-specific configuration."""
    rules_file: str = ""
    causal_strength_threshold: float = 0.5
    enforce_dag: bool = True
    causal_regularization: float = 0.1
    intervention_probability: float = 0.1
    counterfactual_samples: int = 100


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = {}
        
        # Load from file if provided
        if self.config_path:
            config.update(self._load_from_file(self.config_path))
        
        # Override with environment variables
        config.update(self._load_from_env())
        
        # Apply defaults
        config = self._apply_defaults(config)
        
        return config
    
    def _load_from_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Environment variable mappings
        env_mappings = {
            'CAUSALTORCH_MODEL_TYPE': 'model.model_type',
            'CAUSALTORCH_HIDDEN_SIZE': 'model.hidden_size',
            'CAUSALTORCH_NUM_LAYERS': 'model.num_layers',
            'CAUSALTORCH_LEARNING_RATE': 'training.learning_rate',
            'CAUSALTORCH_BATCH_SIZE': 'training.batch_size',
            'CAUSALTORCH_MAX_EPOCHS': 'training.max_epochs',
            'CAUSALTORCH_USE_WANDB': 'experiment.use_wandb',
            'CAUSALTORCH_WANDB_PROJECT': 'experiment.wandb_project',
            'CAUSALTORCH_DATA_PATH': 'data.train_path'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_key(config, config_key, self._cast_value(value))
        
        return config
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default configurations."""
        defaults = {
            'model': asdict(ModelConfig()),
            'training': asdict(TrainingConfig()),
            'data': asdict(DataConfig()),
            'experiment': asdict(ExperimentConfig()),
            'causal': asdict(CausalConfig())
        }
        
        # Deep merge defaults with provided config
        return self._deep_merge(defaults, config)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_key(self, d: Dict, key: str, value: Any):
        """Set nested dictionary key using dot notation."""
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def _cast_value(self, value: str) -> Union[str, int, float, bool]:
        """Cast string value to appropriate type."""
        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self.config['model'])
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        return TrainingConfig(**self.config['training'])
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return DataConfig(**self.config['data'])
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return ExperimentConfig(**self.config['experiment'])
    
    def get_causal_config(self) -> CausalConfig:
        """Get causal configuration."""
        return CausalConfig(**self.config['causal'])
    
    def save_config(self, path: str):
        """Save current configuration to file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self.config = self._deep_merge(self.config, updates)
    
    def validate_config(self) -> bool:
        """Validate configuration consistency."""
        try:
            # Validate each section
            self.get_model_config()
            self.get_training_config()
            self.get_data_config()
            self.get_experiment_config()
            self.get_causal_config()
            
            # Custom validation logic
            model_config = self.get_model_config()
            training_config = self.get_training_config()
            
            # Check model-training compatibility
            if model_config.model_type == "CausalTransformer" and training_config.batch_size > 256:
                self.logger.warning("Large batch size may cause memory issues with CausalTransformer")
            
            # Validate paths
            data_config = self.get_data_config()
            if data_config.train_path and not Path(data_config.train_path).exists():
                raise ValueError(f"Training data path does not exist: {data_config.train_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


# Configuration templates
DEFAULT_CONFIG_TEMPLATE = """
# CausalTorch Configuration Template
model:
  model_type: "CausalTransformer"
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  vocab_size: 1000
  dropout: 0.1

training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "AdamW"
  scheduler: "cosine"
  loss_weights:
    task_loss: 1.0
    causal_loss: 0.5
    sparsity: 0.1

data:
  data_type: "text"
  train_path: ""
  val_path: ""
  preprocessing: {}

experiment:
  experiment_name: "causal_experiment"
  use_wandb: false
  log_interval: 100
  checkpoint_interval: 10

causal:
  rules_file: ""
  enforce_dag: true
  causal_regularization: 0.1
"""


def create_default_config(output_path: str):
    """Create a default configuration file."""
    with open(output_path, 'w') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)
    
    print(f"Default configuration created at {output_path}")


if __name__ == "__main__":
    # Example usage
    
    # Create default config
    create_default_config("config.yml")
    
    # Load and use config
    config_manager = ConfigManager("config.yml")
    
    # Get specific configurations
    model_config = config_manager.get_model_config()
    training_config = config_manager.get_training_config()
    
    print(f"Model type: {model_config.model_type}")
    print(f"Learning rate: {training_config.learning_rate}")
    
    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"Configuration valid: {is_valid}")
