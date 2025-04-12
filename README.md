# CausalTorch

[![PyPI Version](https://img.shields.io/pypi/v/causaltorch.svg)](https://pypi.org/project/causaltorch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/causaltorch.svg)](https://pypi.org/project/causaltorch/)
[![License](https://img.shields.io/github/license/elijahnzeli1/CausalTorch.svg)](https://github.com/elijahnzeli1/CausalTorch/blob/main/LICENSE)

CausalTorch is a PyTorch library for building generative models with explicit causal constraints. It integrates graph-based causal reasoning with deep learning to create AI systems that respect logical causal relationships.

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

# With development tools
pip install causaltorch[dev]
```

## Quick Start

### Text Generation with Causal Rules

```python
import torch
from causaltorch import CNSG_GPT2
from causaltorch.rules import CausalRuleSet, CausalRule
from transformers import GPT2Tokenizer

# Create causal rules
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
rules.add_rule(CausalRule("fire", "smoke", strength=0.8))

# Initialize model and tokenizer
model = CNSG_GPT2(pretrained_model_name="gpt2", causal_rules=rules.to_dict())
model.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text with enforced causal relationships
input_ids = model.tokenizer.encode("The fire spread quickly and", return_tensors="pt")
output = model.generate(input_ids, max_length=50)
print(model.tokenizer.decode(output[0], skip_special_tokens=True))
# Expected to include mention of smoke due to causal rule
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
### or

```python
from causaltorch import CausalRuleSet, plot_causal_graph
from causaltorch.visualization import plot_cfs_comparison

# Create and visualize rules
rules = CausalRuleSet()
rules.add_rule(CausalRule("rain", "wet_ground", 0.9))
rules.visualize()  # Uses plot_causal_graph internally

# Visualize metrics
models = ["CNSG-Small", "CNSG-Base", "CNSG-Large"]
scores = [0.82, 0.89, 0.94]
plot_cfs_comparison(models, scores)
```

## Evaluation Metrics

```python
from causaltorch import CNSGNet, calculate_image_cfs
from causaltorch.rules import load_default_rules

# Load model
model = CNSGNet(latent_dim=3, causal_rules=load_default_rules().to_dict())

# Calculate Causal Fidelity Score
rules = {"rain": {"threshold": 0.5}}
score = calculate_image_cfs(model, rules, num_samples=10)
print(f"Causal Fidelity Score: {score:.2f}")
```

## How It Works

CausalTorch works by:

1. **Defining causal relationships** using a graph-based structure
2. **Integrating these relationships** into neural network architectures 
3. **Modifying the generation process** to enforce causal constraints
4. **Evaluating adherence** to causal rules using specialized metrics

The library provides three main approaches to causal integration:

- **Attention Modification**: For text models, biasing attention toward causal effects
- **Latent Space Conditioning**: For image models, enforcing relationships in latent variables
- **Temporal Constraints**: For video models, ensuring causality across frames

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
@software{nzeli2023causaltorch,
  author = {Nzeli, Elija},
  title = {CausalTorch: Neural-Symbolic Generative Networks with Causal Constraints},
  year = {2023},
  url = {https://github.com/elijahnzeli1/CausalTorch},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.