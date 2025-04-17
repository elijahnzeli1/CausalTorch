#!/usr/bin/env python
"""
Multimodal Causal Example
========================

This example demonstrates the multimodal causal capabilities of CausalTorch.
It shows how to:
1. Create a multimodal causal model
2. Define causal relationships across modalities
3. Generate counterfactual examples by performing causal interventions
4. Visualize the causal relationships and counterfactual results
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torchvision import transforms
from transformers import AutoTokenizer

# Import CausalTorch multimodal components
from causaltorch.multimodal import (
    MultimodalCausalModel,
    CausalTextEncoder,
    CausalImageEncoder,
    CausalModalFusion,
    MultimodalCausalGenerator,
    CounterfactualDreamer,
    MultimodalCausalGraph,
    visualize_causal_graph,
    visualize_counterfactual_comparison,
    display_multimodal_outputs,
    apply_intervention
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Causal Example")
    parser.add_argument("--text", type=str, default="A sunny beach with palm trees",
                        help="Text input for the model")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file (optional)")
    parser.add_argument("--intervention", type=str, default="weather=rainy",
                        help="Causal intervention in format 'variable=value'")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def create_dummy_image(size=(3, 224, 224)):
    """Create a dummy image for testing."""
    return torch.randn(size)


def load_image(image_path, image_size=224):
    """Load and preprocess an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def create_causal_graph():
    """Create a simple causal graph for demonstration."""
    # Define causal relationships
    relations = [
        {"cause": "text_subject", "effect": "image_subject", "strength": 0.8},
        {"cause": "text_weather", "effect": "image_weather", "strength": 0.9},
        {"cause": "text_mood", "effect": "image_colors", "strength": 0.7},
        {"cause": "image_subject", "effect": "text_description", "strength": 0.6},
        {"cause": "image_weather", "effect": "text_weather_description", "strength": 0.8}
    ]
    
    # Create a causal graph from these relations
    graph = MultimodalCausalGraph(relations)
    
    return graph


def create_model(device="cuda"):
    """Create a multimodal causal model for demonstration."""
    hidden_dim = 512
    
    # Create text encoder
    text_encoder = CausalTextEncoder(
        model_name="distilbert-base-uncased",
        hidden_dim=hidden_dim
    )
    
    # Create image encoder
    image_encoder = CausalImageEncoder(
        model_name="resnet18",
        hidden_dim=hidden_dim,
        pretrained=True
    )
    
    # Create fusion module
    fusion = CausalModalFusion(
        hidden_dim=hidden_dim,
        fusion_method="attention"
    )
    
    # Create counterfactual dreamer
    counterfactual_module = CounterfactualDreamer(
        hidden_dim=hidden_dim
    )
    
    # Create generator
    generator = MultimodalCausalGenerator(
        hidden_dim=hidden_dim,
        target_modalities=["text", "image"]
    )
    
    # Create the full model
    model = MultimodalCausalModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_module=fusion,
        generator=generator,
        counterfactual_module=counterfactual_module
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model


def process_intervention(intervention_str):
    """Process intervention string into variable and value."""
    if "=" not in intervention_str:
        raise ValueError("Intervention must be in format 'variable=value'")
    
    variable, value = intervention_str.split("=", 1)
    
    # Try to convert value to float if it looks numeric
    try:
        value = float(value)
    except ValueError:
        pass  # Keep as string if not numeric
    
    return {"variable": variable, "value": value}


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model(device=args.device)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create causal graph
    print("Creating causal graph...")
    causal_graph = create_causal_graph()
    
    # Visualize the causal graph
    print("Visualizing causal graph...")
    fig = visualize_causal_graph(causal_graph.to_dict())
    fig.savefig(os.path.join(args.output_dir, "causal_graph.png"), dpi=300)
    plt.close(fig)
    
    # Preprocess inputs
    print("Processing inputs...")
    text_tokens = tokenizer(args.text, return_tensors="pt").to(args.device)
    
    if args.image is not None:
        try:
            image = load_image(args.image).to(args.device)
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Using a dummy image instead.")
            image = create_dummy_image().unsqueeze(0).to(args.device)
    else:
        print("No image provided, using a dummy image.")
        image = create_dummy_image().unsqueeze(0).to(args.device)
    
    # Process inputs
    print("Generating outputs...")
    with torch.no_grad():
        # Forward pass through model
        outputs = model.generate(
            text_tokens=text_tokens,
            image=image,
            causal_graph=causal_graph.to_dict()
        )
    
    # Process the intervention
    intervention = process_intervention(args.intervention)
    print(f"Applying intervention: {intervention['variable']} = {intervention['value']}")
    
    # Generate counterfactual
    with torch.no_grad():
        counterfactual_outputs = model.imagine_counterfactual(
            text_tokens=text_tokens,
            image=image,
            intervention=intervention,
            causal_graph=causal_graph.to_dict()
        )
    
    # Visualize original vs counterfactual
    print("Visualizing comparison...")
    
    # Convert text tokens back to text
    if 'text' in outputs:
        # If text is already decoded
        original_text = outputs['text']
    else:
        # If we have token IDs, decode them
        original_text = tokenizer.decode(outputs.get('text_output', text_tokens['input_ids'][0]), 
                                       skip_special_tokens=True)
    
    if 'text' in counterfactual_outputs:
        counterfactual_text = counterfactual_outputs['text']
    else:
        counterfactual_text = tokenizer.decode(
            counterfactual_outputs.get('text_output', text_tokens['input_ids'][0]),
            skip_special_tokens=True
        )
    
    # Prepare for visualization
    original = {
        'text': original_text,
        'image': outputs.get('image_output', outputs.get('image', image.squeeze(0)))
    }
    
    counterfactual = {
        'text': counterfactual_text,
        'image': counterfactual_outputs.get('image_output', counterfactual_outputs.get('image', image.squeeze(0)))
    }
    
    # Visualize the comparison
    fig = visualize_counterfactual_comparison(original, counterfactual, intervention)
    fig.savefig(os.path.join(args.output_dir, "counterfactual_comparison.png"), dpi=300)
    plt.close(fig)
    
    # Display individual outputs
    fig = display_multimodal_outputs(original_text, original['image'])
    fig.savefig(os.path.join(args.output_dir, "original_output.png"), dpi=300)
    plt.close(fig)
    
    fig = display_multimodal_outputs(counterfactual_text, counterfactual['image'])
    fig.savefig(os.path.join(args.output_dir, "counterfactual_output.png"), dpi=300)
    plt.close(fig)
    
    print(f"All visualizations saved to {args.output_dir}")
    print(f"Original text: {original_text}")
    print(f"Counterfactual text: {counterfactual_text}")


if __name__ == "__main__":
    main() 