"""
Counterfactual Dreamer Example
============================

This example demonstrates the use of CounterfactualDreamer for generating
novel concepts by perturbing causal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import causaltorch
from causaltorch import CausalRuleSet, CausalRule
from causaltorch import CausalIntervention, CounterfactualDreamer, CreativeMetrics

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a simple VAE as the base generator
class SimpleVAE(nn.Module):
    """Simple Variational Autoencoder for MNIST-like images."""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input images to latent mean and log variance."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent vectors to images."""
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Create a causal graph for MNIST-like images
def create_image_causal_graph():
    """Create a causal graph for MNIST-like images.
    
    Variables:
    - digit_type: The type of digit (0-9)
    - stroke_width: Width of the strokes
    - rotation: Rotation angle
    - background: Background intensity
    - contrast: Contrast level
    """
    rules = CausalRuleSet()
    
    # Causal relationships
    rules.add_rule(CausalRule(
        cause="digit_type",
        effect="stroke_width",
        strength=0.7,  # Different digits have different natural stroke widths
    ))
    
    rules.add_rule(CausalRule(
        cause="rotation",
        effect="background",
        strength=0.3,  # Rotation can affect background due to artifacts
    ))
    
    rules.add_rule(CausalRule(
        cause="stroke_width",
        effect="contrast",
        strength=0.5,  # Thicker strokes may appear with different contrast
    ))
    
    return rules

# Function to visualize generated images
def visualize_images(images, title, save_path=None):
    """Visualize a batch of images."""
    # Convert to grid
    grid = make_grid(images, nrow=5, normalize=True)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

# Main script
def main():
    print("Counterfactual Dreamer Example")
    print("==============================")
    
    # Step 1: Create a VAE model (pretrained or train one)
    # In a real example, you would train this on MNIST or similar data
    print("Initializing VAE model...")
    vae = SimpleVAE(latent_dim=10)
    
    # For demonstration, we'll simulate a pretrained model
    # In reality, you would load weights from a trained model
    print("Simulating pretrained model...")
    
    # Step 2: Create causal graph
    print("Creating causal graph...")
    causal_graph = create_image_causal_graph()
    print(f"Causal graph has {len(causal_graph.rules)} rules")
    
    # Step 3: Initialize CounterfactualDreamer
    print("Initializing CounterfactualDreamer...")
    dreamer = CounterfactualDreamer(
        base_generator=vae,
        rules=causal_graph,
        latent_dim=10
    )
    
    # Step 4: Generate baseline images without intervention
    print("Generating baseline images...")
    baseline_images = dreamer.imagine(
        interventions=None,  # No interventions
        num_samples=10,
        random_seed=42
    )
    
    visualize_images(baseline_images, "Baseline Images (No Intervention)", "baseline_images.png")
    
    # Step 5: Generate images with various interventions
    print("\nGenerating counterfactual images with interventions...")
    
    # Intervention 1: Increase stroke width
    print("\nIntervention 1: Increase stroke width")
    intervention1 = CausalIntervention(
        variable="stroke_width",
        value=0.8,  # High value for thick strokes
        strength=1.0,
        description="Increase stroke width"
    )
    
    thick_stroke_images = dreamer.imagine(
        interventions=[intervention1],
        num_samples=10,
        random_seed=42
    )
    
    visualize_images(thick_stroke_images, "Thick Stroke Images", "thick_stroke_images.png")
    
    # Intervention 2: Rotate digits
    print("\nIntervention 2: Rotate digits")
    intervention2 = CausalIntervention(
        variable="rotation",
        value=0.7,  # High value for significant rotation
        strength=1.0,
        description="Rotate digits"
    )
    
    rotated_images = dreamer.imagine(
        interventions=[intervention2],
        num_samples=10,
        random_seed=42
    )
    
    visualize_images(rotated_images, "Rotated Images", "rotated_images.png")
    
    # Intervention 3: Combined interventions
    print("\nIntervention 3: Combined (High contrast and rotation)")
    intervention3a = CausalIntervention(
        variable="contrast",
        value=0.9,  # High contrast
        strength=1.0,
        description="Increase contrast"
    )
    
    intervention3b = CausalIntervention(
        variable="rotation",
        value=0.5,  # Medium rotation
        strength=0.8,
        description="Medium rotation"
    )
    
    combined_images = dreamer.imagine(
        interventions=[intervention3a, intervention3b],
        num_samples=10,
        random_seed=42
    )
    
    visualize_images(combined_images, "Combined Intervention Images", "combined_images.png")
    
    # Step 6: Measure novelty and diversity
    print("\nMeasuring novelty and diversity...")
    
    # Measure novelty of interventions compared to baseline
    novelty1 = CreativeMetrics.novelty_score(
        output=thick_stroke_images[0],  # First image from intervention 1
        reference_outputs=[baseline_images[i] for i in range(10)]
    )
    
    novelty2 = CreativeMetrics.novelty_score(
        output=rotated_images[0],  # First image from intervention 2
        reference_outputs=[baseline_images[i] for i in range(10)]
    )
    
    novelty3 = CreativeMetrics.novelty_score(
        output=combined_images[0],  # First image from intervention 3
        reference_outputs=[baseline_images[i] for i in range(10)]
    )
    
    print(f"Novelty scores compared to baseline:")
    print(f"Intervention 1 (Thick Strokes): {novelty1:.4f}")
    print(f"Intervention 2 (Rotation): {novelty2:.4f}")
    print(f"Intervention 3 (Combined): {novelty3:.4f}")
    
    # Measure diversity within each intervention
    diversity1 = CreativeMetrics.diversity_score(
        outputs=[thick_stroke_images[i] for i in range(10)]
    )
    
    diversity2 = CreativeMetrics.diversity_score(
        outputs=[rotated_images[i] for i in range(10)]
    )
    
    diversity3 = CreativeMetrics.diversity_score(
        outputs=[combined_images[i] for i in range(10)]
    )
    
    print(f"\nDiversity scores within each intervention:")
    print(f"Intervention 1 (Thick Strokes): {diversity1:.4f}")
    print(f"Intervention 2 (Rotation): {diversity2:.4f}")
    print(f"Intervention 3 (Combined): {diversity3:.4f}")
    
    print("\nExperiment complete. Review the generated images to see how causal interventions affect generation.")

if __name__ == "__main__":
    main() 