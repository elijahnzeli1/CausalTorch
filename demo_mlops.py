"""
CausalTorch MLOps Demo
=====================

Demonstrates the complete MLOps capabilities of CausalTorch.
This showcases experiment tracking, model registry, hyperparameter optimization,
and automated dashboards - all without external dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from causaltorch.mlops import CausalMLOps, MLOpsTrainer
from causaltorch.layers import CausalLinear
from causaltorch.rules import CausalRuleSet


def create_sample_model(input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
    """Create a sample causal model."""
    
    # Create causal rules
    rules = CausalRuleSet()
    rules.add_rule("input_feature_1", "hidden_representation", 0.8)
    rules.add_rule("input_feature_2", "output", 0.6)
    
    # For demo purposes, use standard linear layers
    # In production, you would properly configure causal masks
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Linear(hidden_size // 2, output_size)
    )
    
    return model, rules


def demo_experiment_tracking():
    """Demo experiment tracking capabilities."""
    print("\\n" + "="*60)
    print("🧪 DEMO: Experiment Tracking")
    print("="*60)
    
    # Initialize MLOps
    mlops = CausalMLOps("causal_demo", "./demo_workspace")
    
    # Run multiple experiments
    for lr in [0.001, 0.01, 0.1]:
        config = {
            "learning_rate": lr,
            "batch_size": 32,
            "model_type": "causal_neural_net",
            "optimizer": "adam"
        }
        
        # Start experiment
        exp_id = mlops.start_experiment(
            name=f"causal_regression_lr_{lr}",
            config=config,
            tags=["regression", "causal", "demo"]
        )
        
        # Simulate training
        model, rules = create_sample_model()
        
        for epoch in range(10):
            # Simulate training metrics
            loss = np.random.exponential(2.0) * np.exp(-epoch * lr)
            accuracy = min(0.95, 0.5 + epoch * 0.05 + np.random.normal(0, 0.02))
            causal_fidelity = min(0.99, 0.7 + epoch * 0.03 + np.random.normal(0, 0.01))
            
            # Log metrics
            mlops.log_metrics({
                "train_loss": loss,
                "accuracy": accuracy,
                "causal_fidelity": causal_fidelity,
                "learning_rate": lr
            }, step=epoch)
        
        # Log model artifacts
        mlops.log_artifact("model_state", model.state_dict(), "torch")
        mlops.log_artifact("causal_rules", rules.rules, "pickle")
        mlops.log_artifact("training_config", config, "json")
        
        # Finish experiment
        mlops.finish_experiment("completed")
        
        print(f"📋 Completed experiment with lr={lr}")
    
    # List experiments
    print("\\n📊 Recent Experiments:")
    experiments = mlops.list_experiments()
    for exp in experiments:
        print(f"  • {exp['name']} - {exp['status']} ({exp['created_at'][:19]})")
    
    return mlops


def demo_hyperparameter_optimization():
    """Demo hyperparameter optimization."""
    print("\\n" + "="*60)
    print("🔍 DEMO: Hyperparameter Optimization")
    print("="*60)
    
    mlops = CausalMLOps("hyperopt_demo", "./demo_workspace")
    
    # Define objective function
    def objective(params):
        """Objective function for hyperparameter optimization."""
        lr = params["learning_rate"]
        hidden_size = int(params["hidden_size"])
        
        # Simulate model training with these parameters
        model, _ = create_sample_model(hidden_size=hidden_size)
        
        # Simulate final validation accuracy (higher is better)
        # This would be actual model training in practice
        performance = 0.85 + np.random.normal(0, 0.05)
        performance += 0.1 * (1.0 - abs(lr - 0.01))  # Prefer lr around 0.01
        performance += 0.05 * (1.0 - abs(hidden_size - 64) / 100)  # Prefer hidden_size around 64
        
        return max(0.0, min(1.0, performance))
    
    # Define parameter space
    param_space = {
        "learning_rate": {
            "type": "loguniform",
            "min": 1e-5,
            "max": 1e-1
        },
        "hidden_size": {
            "type": "choice",
            "choices": [32, 64, 128, 256]
        },
        "dropout": {
            "type": "uniform",
            "min": 0.0,
            "max": 0.5
        }
    }
    
    # Run optimization
    result = mlops.optimizer.optimize(
        objective_fn=objective,
        param_space=param_space,
        n_trials=20,
        strategy="random"
    )
    
    print(f"🎯 Best parameters: {result['best_params']}")
    print(f"🏆 Best score: {result['best_score']:.4f}")
    
    return result


def demo_model_registry():
    """Demo model registry capabilities."""
    print("\\n" + "="*60)
    print("📦 DEMO: Model Registry")
    print("="*60)
    
    mlops = CausalMLOps("registry_demo", "./demo_workspace")
    
    # Create and register models
    for version, hidden_size in enumerate([32, 64, 128], 1):
        model, rules = create_sample_model(hidden_size=hidden_size)
        
        metadata = {
            "hidden_size": hidden_size,
            "accuracy": 0.85 + version * 0.05,
            "causal_score": 0.7 + version * 0.08,
            "training_time": 120 + version * 30
        }
        
        mlops.model_registry.register_model(
            model=model,
            name="causal_regressor",
            version=f"v1.{version}",
            metadata=metadata,
            tags=["regression", "causal", f"hidden_{hidden_size}"]
        )
    
    # Load and test a model
    print("\\n🔄 Loading model from registry...")
    loaded_model = mlops.model_registry.load_model("causal_regressor", "latest")
    
    # Test inference
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = loaded_model(test_input)
    
    print(f"✅ Model loaded successfully! Output shape: {output.shape}")
    
    return mlops


def demo_dashboard_generation():
    """Demo dashboard generation."""
    print("\\n" + "="*60)
    print("📊 DEMO: Dashboard Generation")
    print("="*60)
    
    # Use existing MLOps instance with experiments
    mlops = CausalMLOps("causal_demo", "./demo_workspace")
    
    # Generate dashboard
    dashboard_path = mlops.dashboard.generate_experiment_dashboard(mlops)
    
    print(f"📊 Dashboard generated at: {dashboard_path}")
    print("🌐 Open this file in your web browser to view the dashboard!")
    
    return dashboard_path


def demo_integrated_training():
    """Demo integrated training with MLOps."""
    print("\\n" + "="*60)
    print("🚀 DEMO: Integrated MLOps Training")
    print("="*60)
    
    # Initialize MLOps
    mlops = CausalMLOps("integrated_demo", "./demo_workspace")
    
    # Create model
    model, rules = create_sample_model()
    
    # Create integrated trainer
    config = {
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 20,
        "optimizer": "adam",
        "causal_strength": 0.8
    }
    
    trainer = MLOpsTrainer(
        model=model,
        mlops=mlops,
        experiment_name="integrated_causal_training",
        config=config
    )
    
    # Simulate training loop
    print("🏃‍♂️ Training model with integrated MLOps...")
    
    for epoch in range(10):
        # Simulate training step
        loss = 2.0 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
        metrics = {
            "accuracy": min(0.95, 0.6 + epoch * 0.04),
            "causal_fidelity": min(0.98, 0.75 + epoch * 0.023),
            "rule_violations": max(0, 50 - epoch * 5 + np.random.randint(-5, 5))
        }
        
        trainer.train_step(loss, metrics, epoch)
        
        # Save checkpoints periodically
        if epoch % 5 == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch}")
    
    # Finish training
    trainer.finish_training("completed")
    
    print("✅ Integrated training completed!")
    
    return trainer


def main():
    """Run all MLOps demos."""
    print("🚀 CausalTorch MLOps Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the complete MLOps capabilities:")
    print("• 🧪 Experiment tracking and versioning")
    print("• 📦 Model registry and artifact management")
    print("• 🔍 Hyperparameter optimization")
    print("• 📊 Automated dashboards")
    print("• 🚀 Integrated training workflows")
    print("• 🎯 LLM operations support")
    print("\\nAll features work without external dependencies!")
    print("=" * 80)
    
    try:
        # Run demos
        mlops1 = demo_experiment_tracking()
        result = demo_hyperparameter_optimization()
        mlops2 = demo_model_registry()
        dashboard_path = demo_dashboard_generation()
        trainer = demo_integrated_training()
        
        print("\\n" + "="*80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\\n📋 Summary of MLOps Features Demonstrated:")
        print("✅ Experiment tracking with metrics and artifacts")
        print("✅ Hyperparameter optimization (random search)")
        print("✅ Model registry with versioning")
        print("✅ Dashboard generation")
        print("✅ Integrated training workflow")
        print("\\n🏆 CausalTorch MLOps is ready for production!")
        print(f"📁 All data saved in: ./demo_workspace/")
        print(f"🌐 Dashboard available at: {dashboard_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
