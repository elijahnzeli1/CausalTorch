# CausalTorch Multimodal

This package provides capabilities for causal multimodal learning and reasoning. It integrates text, image, and other modalities with causal structure awareness, enabling counterfactual reasoning across modalities and causal interventions.

## Overview

The CausalTorch Multimodal package enables:

1. **Cross-modal causal reasoning**: Define and enforce causal relationships spanning across modalities
2. **Multimodal generation**: Generate content in multiple modalities while respecting causal constraints
3. **Counterfactual generation**: Create "what if" scenarios by performing causal interventions
4. **Causal graph analysis**: Analyze and visualize causal relationships between variables

## Components

### Models

- `MultimodalCausalModel`: Main model for multimodal causal processing
- `CausalTextEncoder`: Text encoder with causal awareness
- `CausalImageEncoder`: Image encoder with causal awareness
- `CausalModalFusion`: Fusion module for integrating information across modalities
- `MultimodalCausalGenerator`: Generator for producing outputs in multiple modalities
- `CausalAttentionLayer`: Attention mechanism that respects causal constraints
- `CounterfactualDreamer`: Module for generating counterfactual examples

### Data

- `MultimodalCausalDataset`: Dataset class for handling multimodal data with causal annotations

### Utilities

- **Causal Graph Utilities**:
  - `MultimodalCausalGraph`: Class for defining and manipulating causal graphs
  - `calculate_causal_consistency`: Measure causal consistency in model outputs
  - `apply_intervention`: Apply causal interventions to data
  - `generate_synthetic_causal_graph`: Create synthetic graphs for testing

- **Metrics**:
  - `calculate_multimodal_causal_consistency`: Measure causal consistency across modalities
  - `evaluate_counterfactual_quality`: Evaluate the quality of counterfactual examples
  - `evaluate_model_causal_fidelity`: Measure how well model outputs follow causal rules

- **Visualization**:
  - `visualize_causal_graph`: Visualize a causal graph with modality markers
  - `visualize_attention_weights`: Visualize attention patterns
  - `visualize_counterfactual_comparison`: Compare original and counterfactual examples
  - `display_multimodal_outputs`: Display text and image outputs side by side

- **Training**:
  - `MultimodalTrainer`: Trainer class for multimodal causal models
  - `prepare_dataloaders`: Prepare DataLoaders with appropriate splits
  - `load_checkpoint`: Load model checkpoints
  - `GradualWarmupScheduler`: Learning rate scheduler with gradual warmup

## Example Usage

```python
import torch
from transformers import AutoTokenizer
from causaltorch.multimodal import (
    MultimodalCausalModel,
    CausalTextEncoder,
    CausalImageEncoder,
    CausalModalFusion,
    MultimodalCausalGenerator,
    CounterfactualDreamer,
    MultimodalCausalGraph,
    visualize_causal_graph,
    visualize_counterfactual_comparison
)

# Create a causal graph
causal_graph = MultimodalCausalGraph([
    {"cause": "text_weather", "effect": "image_weather", "strength": 0.9},
    {"cause": "text_subject", "effect": "image_subject", "strength": 0.8}
])

# Create model components
text_encoder = CausalTextEncoder(model_name="distilbert-base-uncased", hidden_dim=512)
image_encoder = CausalImageEncoder(model_name="resnet18", hidden_dim=512)
fusion = CausalModalFusion(hidden_dim=512, fusion_method="attention")
generator = MultimodalCausalGenerator(hidden_dim=512, target_modalities=["text", "image"])
counterfactual = CounterfactualDreamer(hidden_dim=512)

# Create the full model
model = MultimodalCausalModel(
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    fusion_module=fusion,
    generator=generator,
    counterfactual_module=counterfactual
)

# Process inputs
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "A sunny beach with palm trees"
text_tokens = tokenizer(text, return_tensors="pt")
image = torch.randn(1, 3, 224, 224)  # Dummy image

# Generate outputs
outputs = model.generate(
    text_tokens=text_tokens,
    image=image,
    causal_graph=causal_graph.to_dict()
)

# Generate counterfactual
counterfactual_outputs = model.imagine_counterfactual(
    text_tokens=text_tokens,
    image=image,
    intervention={"variable": "text_weather", "value": "rainy"},
    causal_graph=causal_graph.to_dict()
)

# Visualize comparison
original = {
    'text': outputs.get('text', text),
    'image': outputs.get('image_output', image.squeeze(0))
}
counterfactual = {
    'text': counterfactual_outputs.get('text', text),
    'image': counterfactual_outputs.get('image_output', image.squeeze(0))
}
visualize_counterfactual_comparison(
    original, 
    counterfactual, 
    {"variable": "text_weather", "value": "rainy"}
)
```

## Documentation

For more details and advanced usage, see the example notebook at `examples/multimodal_causal_example.ipynb` and the example script at `examples/multimodal_causal.py`.

## Requirements

- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- transformers >= 4.5.0
- matplotlib
- networkx
- numpy
- PIL
- tqdm 