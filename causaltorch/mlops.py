"""
CausalTorch MLOps Infrastructure
================================

Built-in MLOps capabilities providing experiment tracking, model registry,
hyperparameter optimization, and automation - no external dependencies required.
"""

import json
import os
import pickle
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import sqlite3
import torch
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


@dataclass
class Experiment:
    """Represents an ML experiment run."""
    id: str
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]]
    artifacts: Dict[str, str]
    created_at: datetime.datetime
    status: str = "running"
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class CausalMLOps:
    """
    Complete MLOps platform for CausalTorch models.
    
    Features:
    - Experiment tracking and versioning
    - Model registry and artifact management  
    - Hyperparameter optimization
    - Automated dashboards and visualization
    - LLM operations support
    - Workflow automation
    """
    
    def __init__(self, project_name: str, workspace_dir: str = "./causaltorch_workspace"):
        """
        Initialize CausalMLOps platform.
        
        Args:
            project_name: Name of the ML project
            workspace_dir: Directory for storing experiments and artifacts
        """
        self.project_name = project_name
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize database and storage
        self.db_path = self.workspace_dir / f"{project_name}.db"
        self._init_database()
        
        # Current experiment tracking
        self.current_experiment: Optional[Experiment] = None
        self.step_counter = 0
        
        # Model registry
        self.model_registry = ModelRegistry(self.workspace_dir / "models")
        
        # Hyperparameter optimizer
        self.optimizer = HyperparameterOptimizer(self.workspace_dir / "optimization")
        
        # Dashboard generator
        self.dashboard = DashboardGenerator(self.workspace_dir / "dashboards")
        
        print(f"🚀 CausalMLOps initialized for project '{project_name}'")
        print(f"📁 Workspace: {self.workspace_dir.absolute()}")
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT,
                    created_at TEXT,
                    status TEXT,
                    notes TEXT,
                    tags TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT,
                    metric_name TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    experiment_id TEXT,
                    artifact_name TEXT,
                    artifact_path TEXT,
                    artifact_type TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
    
    def start_experiment(self, name: str, config: Dict[str, Any] = None, 
                        tags: List[str] = None, notes: str = "") -> str:
        """
        Start a new experiment run.
        
        Args:
            name: Experiment name
            config: Configuration dictionary (hyperparameters, etc.)
            tags: List of tags for organization
            notes: Optional notes about the experiment
            
        Returns:
            Experiment ID
        """
        config = config or {}
        tags = tags or []
        
        # Generate unique experiment ID
        timestamp = datetime.datetime.now()
        exp_id = f"{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(config)) % 10000:04d}"
        
        # Create experiment
        self.current_experiment = Experiment(
            id=exp_id,
            name=name,
            config=config,
            metrics=defaultdict(list),
            artifacts={},
            created_at=timestamp,
            tags=tags,
            notes=notes
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (id, name, config, created_at, status, notes, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, name, json.dumps(config), timestamp.isoformat(),
                "running", notes, json.dumps(tags)
            ))
        
        # Create experiment directory
        exp_dir = self.workspace_dir / "experiments" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if HAS_YAML:
            with open(exp_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f)
        else:
            with open(exp_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        self.step_counter = 0
        print(f"🧪 Started experiment: {name} (ID: {exp_id})")
        return exp_id
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Store in memory
        self.current_experiment.metrics[name].append(value)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (experiment_id, metric_name, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.current_experiment.id, name, value, step,
                datetime.datetime.now().isoformat()
            ))
        
        print(f"📊 Logged {name}: {value:.4f} (step {step})")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_artifact(self, name: str, obj: Any, artifact_type: str = "pickle"):
        """
        Log an artifact (model, dataset, etc.).
        
        Args:
            name: Artifact name
            obj: Object to save
            artifact_type: Type of artifact ('pickle', 'torch', 'json', 'text')
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        exp_dir = self.workspace_dir / "experiments" / self.current_experiment.id
        
        if artifact_type == "pickle":
            path = exp_dir / f"{name}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        elif artifact_type == "torch":
            path = exp_dir / f"{name}.pt"
            torch.save(obj, path)
        elif artifact_type == "json":
            path = exp_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
        elif artifact_type == "text":
            path = exp_dir / f"{name}.txt"
            with open(path, 'w') as f:
                f.write(str(obj))
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        
        # Store reference
        self.current_experiment.artifacts[name] = str(path)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO artifacts (experiment_id, artifact_name, artifact_path, artifact_type)
                VALUES (?, ?, ?, ?)
            """, (self.current_experiment.id, name, str(path), artifact_type))
        
        print(f"💾 Logged artifact: {name} ({artifact_type})")
    
    def log_model_info(self, model, model_name: str, training_params: Dict[str, Any] = None):
        """
        Log comprehensive model information including architecture, parameters, and weights.
        
        Args:
            model: PyTorch model to log
            model_name: Name identifier for the model
            training_params: Additional training parameters to log
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        import torch.nn as nn
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Get layer information
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layer_type = type(module).__name__
                num_params = sum(p.numel() for p in module.parameters())
                layer_info.append({
                    'name': name,
                    'type': layer_type,
                    'parameters': num_params
                })
        
        # Collect weight statistics
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Safe calculations for edge cases
                if param.numel() == 0:
                    # For empty parameters, use default values
                    weight_stats[name] = {
                        'shape': list(param.shape),
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'norm': 0.0,
                        'sparsity': 0.0
                    }
                elif param.numel() == 1:
                    # For single-element parameters
                    param_value = float(param.data.item())
                    weight_stats[name] = {
                        'shape': list(param.shape),
                        'mean': param_value,
                        'std': 0.0,  # Standard deviation of single value is 0
                        'min': param_value,
                        'max': param_value,
                        'norm': abs(param_value),
                        'sparsity': float(param_value == 0.0)
                    }
                else:
                    # For normal parameters with multiple elements
                    weight_stats[name] = {
                        'shape': list(param.shape),
                        'mean': float(param.data.mean()),
                        'std': float(param.data.std(unbiased=False)),
                        'min': float(param.data.min()),
                        'max': float(param.data.max()),
                        'norm': float(param.data.norm()),
                        'sparsity': float((param.data == 0).sum()) / param.numel()
                    }
        
        # Create comprehensive model info
        model_info = {
            'model_name': model_name,
            'architecture': str(model),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'layer_count': len(layer_info),
            'layers': layer_info,
            'weight_statistics': weight_stats,
            'training_params': training_params or {},
            'model_class': type(model).__name__
        }
        
        # Log as artifact
        self.log_artifact(f"{model_name}_info", model_info, "json")
        
        # Log model state dict
        self.log_artifact(f"{model_name}_weights", model.state_dict(), "torch")
        
        # Log key metrics
        self.log_metric(f"{model_name}_total_params", total_params)
        self.log_metric(f"{model_name}_trainable_params", trainable_params)
        self.log_metric(f"{model_name}_size_mb", model_size_mb)
        
        print(f"Logged model info: {model_name}")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Model size: {model_size_mb:.2f} MB")
        print(f"   Layers: {len(layer_info)}")
    
    def finish_experiment(self, status: str = "completed"):
        """Finish the current experiment."""
        if not self.current_experiment:
            print("⚠️ No active experiment to finish.")
            return
        
        self.current_experiment.status = status
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments SET status = ? WHERE id = ?
            """, (status, self.current_experiment.id))
        
        # Generate experiment summary
        self._generate_experiment_summary()
        
        print(f"✅ Finished experiment: {self.current_experiment.name} ({status})")
        self.current_experiment = None
    
    def _generate_experiment_summary(self):
        """Generate experiment summary report."""
        if not self.current_experiment:
            return
        
        exp_dir = self.workspace_dir / "experiments" / self.current_experiment.id
        
        summary = {
            "experiment_id": self.current_experiment.id,
            "name": self.current_experiment.name,
            "config": self.current_experiment.config,
            "final_metrics": {
                name: values[-1] if values else None 
                for name, values in self.current_experiment.metrics.items()
            },
            "best_metrics": {
                name: max(values) if values else None
                for name, values in self.current_experiment.metrics.items()
            },
            "status": self.current_experiment.status,
            "artifacts": list(self.current_experiment.artifacts.keys()),
            "duration": (datetime.datetime.now() - self.current_experiment.created_at).total_seconds()
        }
        
        with open(exp_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def list_experiments(self, status: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """List experiments with optional filtering."""
        query = "SELECT * FROM experiments"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, List[Tuple[int, float]]]:
        """Get all metrics for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, value, step FROM metrics 
                WHERE experiment_id = ? ORDER BY step
            """, (experiment_id,))
            
            metrics = defaultdict(list)
            for name, value, step in cursor.fetchall():
                metrics[name].append((step, value))
            
            return dict(metrics)
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """Compare multiple experiments."""
        comparison = {
            "experiments": [],
            "metric_comparison": defaultdict(dict)
        }
        
        for exp_id in experiment_ids:
            # Get experiment info
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
                exp_data = dict(cursor.fetchone())
                comparison["experiments"].append(exp_data)
            
            # Get metrics
            metrics = self.get_experiment_metrics(exp_id)
            for metric_name, values in metrics.items():
                if values:
                    comparison["metric_comparison"][metric_name][exp_id] = {
                        "final": values[-1][1],
                        "best": max(v[1] for v in values),
                        "worst": min(v[1] for v in values)
                    }
        
        return comparison


class ModelRegistry:
    """Model registry for versioning and artifact management."""
    
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.index_file = registry_dir / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}, "versions": {}}
    
    def _save_index(self):
        """Save registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(self, model: torch.nn.Module, name: str, version: str,
                      metadata: Dict = None, tags: List[str] = None):
        """Register a model version."""
        metadata = metadata or {}
        tags = tags or []
        
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), model_dir / "model.pt")
        
        # Save full model
        torch.save(model, model_dir / "model_full.pt")
        
        # Save metadata
        model_info = {
            "name": name,
            "version": version,
            "registered_at": datetime.datetime.now().isoformat(),
            "metadata": metadata,
            "tags": tags,
            "model_class": type(model).__name__
        }
        
        with open(model_dir / "info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update index
        if name not in self.index["models"]:
            self.index["models"][name] = []
        
        self.index["models"][name].append(version)
        self.index["versions"][f"{name}:{version}"] = str(model_dir)
        self._save_index()
        
        print(f"📦 Registered model {name}:{version}")
    
    def load_model(self, name: str, version: str = "latest") -> torch.nn.Module:
        """Load a registered model."""
        if version == "latest":
            if name not in self.index["models"]:
                raise ValueError(f"Model {name} not found")
            version = self.index["models"][name][-1]
        
        model_key = f"{name}:{version}"
        if model_key not in self.index["versions"]:
            raise ValueError(f"Model {name}:{version} not found")
        
        model_dir = Path(self.index["versions"][model_key])
        # Use weights_only=False for our trusted models
        return torch.load(model_dir / "model_full.pt", weights_only=False)


class HyperparameterOptimizer:
    """Built-in hyperparameter optimization."""
    
    def __init__(self, opt_dir: Path):
        self.opt_dir = opt_dir
        self.opt_dir.mkdir(exist_ok=True, parents=True)
    
    def optimize(self, objective_fn: Callable, param_space: Dict,
                n_trials: int = 50, strategy: str = "random") -> Dict:
        """
        Optimize hyperparameters.
        
        Args:
            objective_fn: Function to optimize (should return metric to maximize)
            param_space: Dictionary defining parameter search space
            n_trials: Number of optimization trials
            strategy: Optimization strategy ('random', 'grid', 'bayesian')
        """
        results = []
        best_score = float('-inf')
        best_params = None
        
        print(f"🔍 Starting hyperparameter optimization ({n_trials} trials)")
        
        for trial in range(n_trials):
            # Sample parameters
            if strategy == "random":
                params = self._random_sample(param_space)
            elif strategy == "grid":
                params = self._grid_sample(param_space, trial, n_trials)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            # Evaluate
            try:
                score = objective_fn(params)
                results.append({"params": params, "score": score, "trial": trial})
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎯 New best: {score:.4f} (trial {trial})")
                
            except Exception as e:
                print(f"❌ Trial {trial} failed: {e}")
                continue
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        # Save results
        optimization_result = {
            "best_params": convert_types(best_params),
            "best_score": float(best_score),
            "all_results": convert_types(results),
            "param_space": param_space,
            "strategy": strategy
        }
        
        with open(self.opt_dir / f"optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        print(f"✅ Optimization complete! Best score: {best_score:.4f}")
        return optimization_result
    
    def _random_sample(self, param_space: Dict) -> Dict:
        """Randomly sample from parameter space."""
        params = {}
        for name, config in param_space.items():
            if config["type"] == "uniform":
                params[name] = np.random.uniform(config["min"], config["max"])
            elif config["type"] == "loguniform":
                params[name] = np.exp(np.random.uniform(np.log(config["min"]), np.log(config["max"])))
            elif config["type"] == "choice":
                params[name] = np.random.choice(config["choices"])
            elif config["type"] == "int":
                params[name] = np.random.randint(config["min"], config["max"] + 1)
        return params
    
    def _grid_sample(self, param_space: Dict, trial: int, total_trials: int) -> Dict:
        """Grid sampling (simplified implementation)."""
        # This is a simplified grid sampling - a full implementation would
        # generate all combinations and select the trial-th one
        return self._random_sample(param_space)


class DashboardGenerator:
    """Generate HTML dashboards for experiment visualization."""
    
    def __init__(self, dashboard_dir: Path):
        self.dashboard_dir = dashboard_dir
        self.dashboard_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_experiment_dashboard(self, mlops: 'CausalMLOps', 
                                    experiment_ids: List[str] = None) -> str:
        """Generate HTML dashboard for experiments."""
        if experiment_ids is None:
            experiments = mlops.list_experiments(limit=10)
            experiment_ids = [exp["id"] for exp in experiments]
        
        html = self._create_dashboard_html(mlops, experiment_ids)
        
        dashboard_file = self.dashboard_dir / f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"📊 Dashboard generated: {dashboard_file}")
        return str(dashboard_file)
    
    def _create_dashboard_html(self, mlops: 'CausalMLOps', experiment_ids: List[str]) -> str:
        """Create HTML dashboard content."""
        # This is a simplified HTML dashboard - in production you'd use
        # a proper template engine and more sophisticated visualization
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CausalTorch MLOps Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .experiment { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
                .metric { display: inline-block; margin: 5px 10px; }
                .header { background: #f0f0f0; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>🚀 CausalTorch MLOps Dashboard</h1>
            <h2>Recent Experiments</h2>
        """
        
        for exp_id in experiment_ids:
            exp_info = mlops.list_experiments()
            exp_data = next((e for e in exp_info if e["id"] == exp_id), None)
            if not exp_data:
                continue
                
            metrics = mlops.get_experiment_metrics(exp_id)
            
            html += f"""
            <div class="experiment">
                <div class="header">
                    <h3>{exp_data['name']} ({exp_id})</h3>
                    <p>Status: {exp_data['status']} | Created: {exp_data['created_at']}</p>
                </div>
                <div class="metrics">
                    <h4>Final Metrics:</h4>
            """
            
            for metric_name, values in metrics.items():
                if values:
                    final_value = values[-1][1]
                    html += f'<span class="metric"><strong>{metric_name}:</strong> {final_value:.4f}</span>'
            
            html += """
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


# Convenience functions for quick setup
def init_mlops(project_name: str, workspace_dir: str = "./causaltorch_workspace") -> CausalMLOps:
    """Initialize CausalMLOps with sensible defaults."""
    return CausalMLOps(project_name, workspace_dir)


# Integration with CausalTrainer
class MLOpsTrainer:
    """Enhanced trainer with built-in MLOps integration."""
    
    def __init__(self, model: torch.nn.Module, mlops: CausalMLOps, 
                 experiment_name: str, config: Dict = None):
        self.model = model
        self.mlops = mlops
        self.experiment_id = mlops.start_experiment(experiment_name, config or {})
    
    def train_step(self, loss: float, metrics: Dict[str, float] = None, step: int = None):
        """Log training step with MLOps tracking."""
        self.mlops.log_metric("loss", loss, step)
        if metrics:
            self.mlops.log_metrics(metrics, step)
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint with MLOps tracking."""
        self.mlops.log_artifact(name, self.model.state_dict(), "torch")
    
    def finish_training(self, status: str = "completed"):
        """Finish training and save final model."""
        self.mlops.log_artifact("final_model", self.model, "torch")
        self.mlops.finish_experiment(status)
        
        # Register model in registry
        self.mlops.model_registry.register_model(
            self.model, 
            self.experiment_id.split('_')[0],  # Use experiment name as model name
            f"v_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
