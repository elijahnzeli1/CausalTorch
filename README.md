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
- 📝 **Text Generation**: Enforce causal rules in text with modified attention mechanisms
- 🖼️ **Image Generation**: Generate images that respect causal relationships (e.g., "rain → wet ground")
- 🎬 **Video Generation**: Create temporally consistent videos with causal effects
- 📈 **Causal Metrics**: Evaluate models with specialized causal fidelity metrics

## Installation

```bash
# Basic installation
pip install causaltorch

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

### Text Generation with Causal Rules

```python
import torch
from causaltorch import cnsg
from causaltorch.rules import CausalRuleSet, CausalRule
from transformers import GPT2Tokenizer

# Create causal rules
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
rules.add_rule(CausalRule("fire", "smoke", strength=0.8))

# Initialize model and tokenizer
model = cnsg(pretrained_model_name="gpt2", causal_rules=rules.to_dict())
model.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text with enforced causal relationships
input_ids = model.tokenizer.encode("The fire spread quickly and", return_tensors="pt")
output = model.generate(input_ids, max_length=50)
print(model.tokenizer.decode(output[0], skip_special_tokens=True))
# Expected to include mention of smoke due to causal rule
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

- **Attention Modification**: For text models, biasing attention toward causal effects
- **Latent Space Conditioning**: For image models, enforcing relationships in latent variables
- **Temporal Constraints**: For video models, ensuring causality across frames
- **Dynamic Architecture Generation**: For meta-learning, creating architecture from causal graphs
- **Ethical Constitution**: For safe generation, enforcing ethical rules during generation
- **Counterfactual Reasoning**: For creative generation, exploring "what if" scenarios

## Evaluation Metrics

```python
from causaltorch import CNSGNet, calculate_image_cfs, CreativeMetrics
from causaltorch.rules import load_default_rules

# Load model
model = CNSGNet(latent_dim=3, causal_rules=load_default_rules().to_dict())

# Calculate Causal Fidelity Score
rules = {"rain": {"threshold": 0.5}}
cfs_score = calculate_image_cfs(model, rules, num_samples=10)
print(f"Causal Fidelity Score: {cfs_score:.2f}")

# Calculate novelty score
output = model.generate(rain_intensity=0.8)
reference_outputs = [model.generate(rain_intensity=0.2) for _ in range(5)]
novelty = CreativeMetrics.novelty_score(output, reference_outputs)
print(f"Novelty Score: {novelty:.2f}")
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