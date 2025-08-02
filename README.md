# CausalTorch

[![PyPI Version](https://img.shields.io/pypi/v/causaltorch.svg)](https://pypi.org/project/causaltorch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/causaltorch.svg)](https://pypi.org/project/causaltorch/)
[![License](https://img.shields.io/github/license/elijahnzeli1/CausalTorch.svg)](https://github.com/elijahnzeli1/CausalTorch/blob/main/LICENSE)

CausalTorch is a PyTorch library for building generative models with explicit causal constraints. It integrates graph-based causal reasoning with deep learning to create AI systems that respect logical causal relationships.

## 🎉 What's New in CausalTorch v2.0

CausalTorch v2.0 introduces powerful new capabilities organized around seven core pillars:

1. **Causal First**: All models reason about cause-effect relationships with improved fidelity
2. **Sparsity as Law**: Dynamic activation of <10% of parameters for efficient computation
3. **Neuro-Symbolic Fusion**: Enhanced integration of neural and symbolic components
4. **Ethics by Architecture**: Hardcoded ethical rules as architectural constraints
5. **Decentralized Intelligence**: Federated learning preserving causal knowledge
6. **Creative Computation**: Novel concept generation via causal interventions
7. **Self-Evolving Meta-Learning**: Models that adapt their architecture to the task

New features include:

- **🧠 Causal HyperNetworks**: Generate task-specific neural architectures from causal graphs
- **⚡ Dynamic Sparse Activation**: Lottery Ticket Router for efficient parameter usage
- **🌐 Decentralized Causal DAO**: Federated learning with Byzantine-resistant causal consensus
- **🛡️ Ethical Constitution Engine**: Enforce ethical rules during generation
- **🔮 Counterfactual Dreamer**: Generate novel concepts by perturbing causal graphs
- **📉 Causal State-Space Models**: O(n) complexity alternative to attention mechanisms

## Key Features

- 🧠 **Neural-Symbolic Integration**: Combine neural networks with symbolic causal rules
- 📊 **Graph-Based Causality**: Define causal relationships as directed acyclic graphs
- 📝 **Native Text Generation**: Pure CausalTorch text models without external dependencies
- 🖼️ **Computer Vision Support**: Generate and classify images with causal constraints
- 🎬 **Video Generation**: Create temporally consistent videos with causal effects
- 🤖 **Reinforcement Learning**: RL agents with episodic memory and causal prioritization
- 🔬 **MLOps Platform**: Complete experiment tracking and model management
- 📈 **Causal Metrics**: Evaluate models with specialized causal fidelity metrics
- ⚡ **Production Ready**: Robust, stable, and production-ready architecture

## Installation

```bash
# Basic installation
pip install causaltorch

# With computer vision support
pip install causaltorch[vision]

# With reinforcement learning support
pip install causaltorch[rl]

# With MLOps platform support
pip install causaltorch[mlops]

# With text generation support
pip install causaltorch[text]

# With image generation support
pip install causaltorch[image]

# With federated learning support
pip install causaltorch[federated]

# With all features
pip install causaltorch[all]

# With development tools
pip install causaltorch[dev]
```

## Quick Start

### Native Text Generation (No External Dependencies)

```python
import torch
from causaltorch.models import cnsg
from causaltorch.rules import CausalRuleSet, CausalRule

# Create causal rules
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
rules.add_rule(CausalRule("fire", "smoke", strength=0.8))

# Initialize native CausalTorch text model (no GPT-2 dependency)
model = cnsg(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    causal_rules=rules.to_dict()
)

# Generate text with enforced causal relationships
input_ids = torch.randint(0, 1000, (1, 10))
output = model.generate(
    input_ids=input_ids,
    max_length=50,
    temperature=0.8,
    causal_constraints={'forbidden_words': [999]}
)
print(f"Generated sequence: {output[0].tolist()}")
# Native causal reasoning without external model dependencies
```

### Computer Vision with Causal Reasoning

```python
import torch
from causaltorch.models import CausalVisionTransformer, CausalCNN
from causaltorch.rules import CausalRuleSet, CausalRule

# Create vision model with causal constraints
causal_rules = CausalRuleSet()
causal_rules.add_rule(CausalRule("weather_sunny", "shadows_present", strength=0.8))
causal_rules.add_rule(CausalRule("rain_intensity", "ground_wetness", strength=0.9))

# Initialize Causal Vision Transformer
vision_model = CausalVisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    d_model=768,
    n_heads=12,
    n_layers=6,
    causal_rules=causal_rules.to_dict()
)

# Process images with causal reasoning
image = torch.randn(1, 3, 224, 224)
logits, causal_features = vision_model(image)

# Generate images with causal constraints
image_generator = CausalCNN(
    latent_dim=128,
    image_size=64,
    causal_rules=causal_rules.to_dict()
)
generated_image = image_generator.generate(
    num_samples=1,
    causal_interventions={"weather_sunny": 0.9}
)
```

### Reinforcement Learning with Episodic Memory

```python
import torch
from causaltorch.core_architecture import FromScratchModelBuilder

# Create RL agent with episodic memory and causal reasoning
rl_config = {
    'causal_config': {
        'hidden_dim': 128,
        'causal_rules': [
            {'cause': 'action', 'effect': 'reward', 'strength': 0.9},
            {'cause': 'state', 'effect': 'action_value', 'strength': 0.8}
        ]
    }
}

builder = FromScratchModelBuilder(rl_config)
agent = builder.build_model(
    'reinforcement_learning',
    state_dim=8,
    action_dim=4,
    agent_type='dqn',
    memory_capacity=10000,
    batch_size=32
)

# Agent automatically remembers actions and outcomes with causal prioritization
state = torch.randn(1, 8)
action = agent.select_action(state, explore=True)
reward = 10.0
next_state = torch.randn(1, 8)
done = False

# Store experience (causal strength computed automatically)
agent.store_experience(state, action, reward, next_state, done)

# Learning uses causal prioritization for experience replay
loss_info = agent.learn()
print(f"Training loss: {loss_info.get('total_loss', 0.0):.4f}")

# Get causal analysis of learning
causal_analysis = agent.get_causal_analysis()
print(f"High-causal episodes: {causal_analysis['memory_size']}")
```

### MLOps Platform Integration

```python
import torch
from causaltorch.mlops import CausalMLOps
from causaltorch.models import cnsg

# Initialize MLOps platform
mlops = CausalMLOps(
    project_name="causal_text_generation",
    experiment_name="native_cnsg_experiment"
)

# Create and track model
model = cnsg(vocab_size=5000, d_model=256, n_heads=8, n_layers=4)

# Log model architecture and parameters
mlops.log_model_info(model, "native_cnsg_v1")

# Track training metrics
for epoch in range(10):
    # Simulate training
    loss = torch.randn(1).item()
    accuracy = torch.rand(1).item()
    
    mlops.log_metrics({
        'loss': loss,
        'accuracy': accuracy,
        'causal_adherence': 0.85
    }, step=epoch)

# Save model to registry
model_version = mlops.model_registry.save_model(
    model=model,
    name="native_cnsg",
    version="1.0.0",
    metadata={"architecture": "native_causal_transformer"}
)

# Generate dashboard
dashboard_path = mlops.generate_dashboard()
print(f"Dashboard saved to: {dashboard_path}")
```

### v2.0: Meta-Learning with Causal HyperNetworks

```python
import torch
from causaltorch import CausalHyperNetwork, CausalRuleSet, CausalRule

# Create a set of causal graphs for different tasks
graph1 = CausalRuleSet()
graph1.add_rule(CausalRule("X", "Y", strength=0.8))

graph2 = CausalRuleSet()
graph2.add_rule(CausalRule("X", "Z", strength=0.6))
graph2.add_rule(CausalRule("Z", "Y", strength=0.7))

# Convert graphs to adjacency matrices
adj1 = torch.zeros(10, 10)
adj1[0, 1] = 0.8  # X → Y

adj2 = torch.zeros(10, 10)
adj2[0, 2] = 0.6  # X → Z
adj2[2, 1] = 0.7  # Z → Y

# Initialize CausalHyperNetwork
hyper_net = CausalHyperNetwork(
    input_dim=100,
    output_dim=1,
    hidden_dim=64,
    meta_hidden_dim=128
)

# Generate task-specific architectures
model1 = hyper_net.generate_architecture(adj1.unsqueeze(0))
model2 = hyper_net.generate_architecture(adj2.unsqueeze(0))

# Use the generated models for specific tasks
y1 = model1(torch.randn(5, 10))  # For task 1
y2 = model2(torch.randn(5, 10))  # For task 2
```

### v2.0: Creative Generation with Counterfactual Dreamer

```python
import torch
from causaltorch import CausalRuleSet, CausalRule
from causaltorch import CounterfactualDreamer, CausalIntervention

# Create a causal ruleset
rules = CausalRuleSet()
rules.add_rule(CausalRule("weather", "ground_condition", strength=0.9))
rules.add_rule(CausalRule("ground_condition", "plant_growth", strength=0.7))

# Initialize a generative model (e.g., VAE)
vae = torch.nn.Sequential(...)  # Your generative model here

# Create the Counterfactual Dreamer
dreamer = CounterfactualDreamer(
    base_generator=vae,
    rules=rules,
    latent_dim=10
)

# Generate baseline without interventions
baseline = dreamer.imagine(interventions=None, num_samples=5)

# Define a counterfactual intervention
intervention = CausalIntervention(
    variable="weather",
    value=0.9,  # Sunny weather
    strength=1.0,
    description="What if it were extremely sunny?"
)

# Generate counterfactual samples
counterfactual = dreamer.imagine(
    interventions=[intervention],
    num_samples=5
)

# Explain the intervention
print(dreamer.explain_interventions())
```

### Image Generation with Causal Constraints

```python
import torch
from causaltorch import CNSGNet
from causaltorch.rules import CausalRuleSet, CausalRule

# Define causal rules
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "ground_wet", strength=0.9))

# Create model
model = CNSGNet(latent_dim=3, causal_rules=rules.to_dict())

# Generate images with increasing rain intensity
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
rain_levels = [0.1, 0.5, 0.9]

for i, rain in enumerate(rain_levels):
    # Generate image
    image = model.generate(rain_intensity=rain)
    # Display
    axs[i].imshow(image[0, 0].detach().numpy(), cmap='gray')
    axs[i].set_title(f"Rain: {rain:.1f}")
plt.show()
```

### v2.0: Ethical Constitution for Safe Generation

```python
import torch
from causaltorch import EthicalConstitution, EthicalRule, EthicalTextFilter

# Create ethical rules
rules = [
    EthicalRule(
        name="no_harm",
        description="Do not generate content that could cause harm to humans",
        detection_fn=EthicalTextFilter.check_harmful_content,
        action="block",
        priority=10
    ),
    EthicalRule(
        name="privacy",
        description="Protect private information in generated content",
        detection_fn=EthicalTextFilter.check_privacy_violation,
        action="modify",
        priority=8
    )
]

# Create ethical constitution
constitution = EthicalConstitution(rules=rules)

# Check if output complies with ethical rules
generated_text = "Here's how to make a harmful device..."
safe_text, passed, violations = constitution(generated_text)

if not passed:
    print("Ethical violations detected:")
    for violation in violations:
        print(f"- {violation['rule']}: {violation['reason']}")
```

### Visualization of Causal Graph

```python
from causaltorch.rules import CausalRuleSet, CausalRule

# Create a causal graph
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
rules.add_rule(CausalRule("wet_ground", "slippery", strength=0.7))
rules.add_rule(CausalRule("fire", "smoke", strength=0.8))
rules.add_rule(CausalRule("smoke", "reduced_visibility", strength=0.6))

# Visualize the causal relationships
rules.visualize()
```

## How It Works

CausalTorch works by:

1. **Defining causal relationships** using a graph-based structure
2. **Integrating these relationships** into neural network architectures 
3. **Modifying the generation process** to enforce causal constraints
4. **Evaluating adherence** to causal rules using specialized metrics

The library provides multiple approaches to causal integration:

- **Native Architecture**: Built-from-scratch models with causal reasoning at every layer
- **Attention Modification**: For text models, biasing attention toward causal effects
- **Latent Space Conditioning**: For image models, enforcing relationships in latent variables
- **Temporal Constraints**: For video models, ensuring causality across frames
- **Episodic Memory**: For RL agents, prioritizing causally significant experiences
- **Dynamic Architecture Generation**: For meta-learning, creating architecture from causal graphs
- **Ethical Constitution**: For safe generation, enforcing ethical rules during generation
- **Counterfactual Reasoning**: For creative generation, exploring "what if" scenarios
- **MLOps Integration**: Complete experiment tracking and model lifecycle management

## Evaluation Metrics

```python
from causaltorch import CNSGNet, CausalVisionTransformer, calculate_image_cfs, CreativeMetrics
from causaltorch.rules import load_default_rules
from causaltorch.mlops import CausalMLOps

# Vision model evaluation
vision_model = CausalVisionTransformer(image_size=224, num_classes=1000)
image = torch.randn(1, 3, 224, 224)
logits, causal_features = vision_model(image)

# Calculate Causal Fidelity Score for images
rules = {"rain": {"threshold": 0.5}}
image_model = CNSGNet(latent_dim=3, causal_rules=load_default_rules().to_dict())
cfs_score = calculate_image_cfs(image_model, rules, num_samples=10)
print(f"Image Causal Fidelity Score: {cfs_score:.2f}")

# Calculate novelty score
output = image_model.generate(rain_intensity=0.8)
reference_outputs = [image_model.generate(rain_intensity=0.2) for _ in range(5)]
novelty = CreativeMetrics.novelty_score(output, reference_outputs)
print(f"Novelty Score: {novelty:.2f}")

# RL agent evaluation
from causaltorch.core_architecture import FromScratchModelBuilder
builder = FromScratchModelBuilder({'causal_config': {}})
agent = builder.build_model('reinforcement_learning', state_dim=8, action_dim=4, agent_type='dqn')

# Evaluate causal learning
causal_analysis = agent.get_causal_analysis()
print(f"RL Causal Analysis: {causal_analysis}")

# MLOps metrics tracking
mlops = CausalMLOps(project_name="evaluation", experiment_name="metrics_test")
mlops.log_metrics({
    'causal_fidelity': cfs_score,
    'novelty': novelty,
    'rl_memory_size': len(agent.episodic_memory)
})
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Citation

If you use CausalTorch in your research, please cite:

```bibtex
@software{nzeli2025causaltorch,
  author = {Nzeli, Elijah},
  title = {CausalTorch: Neural-Symbolic Generative Networks with Causal Constraints},
  year = {2025},
  url = {https://github.com/elijahnzeli1/CausalTorch},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.