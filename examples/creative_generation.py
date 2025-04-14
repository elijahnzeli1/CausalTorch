# examples/creative_generation.py
"""
Example of creative generation with CounterfactualDreamer.

This script demonstrates how to use causal interventions to generate
novel concepts and creative variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from causaltorch import CounterfactualDreamer, CausalIntervention, CreativeMetrics, NoveltySearch
from causaltorch.rules import CausalRuleSet, CausalRule


# Simple VAE for demonstration
class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        return h.view(h.size(0), 1, 28, 28)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def visualize_images(images, title):
    """Helper to visualize a batch of images."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    grid = make_grid(images, nrow=4).permute(1, 2, 0)
    plt.imshow(grid.numpy())
    plt.axis('off')
    plt.show()


def main():
    print("Creative Generation with CounterfactualDreamer")
    print("-" * 50)
    
    # Create causal ruleset
    rules = CausalRuleSet()
    rules.add_rule(CausalRule("brightness", "texture", strength=0.8))
    rules.add_rule(CausalRule("texture", "shape", strength=0.6))
    rules.add_rule(CausalRule("shape", "style", strength=0.7))
    
    print("Causal Rules:")
    print(rules)
    
    # Initialize VAE as base generator
    latent_dim = 10
    vae = SimpleVAE(latent_dim=latent_dim)
    
    # Create CounterfactualDreamer
    dreamer = CounterfactualDreamer(
        base_generator=vae,
        rules=rules,
        latent_dim=latent_dim
    )
    
    # Generate baseline samples
    print("\nGenerating baseline samples...")
    baseline_samples = dreamer.imagine(num_samples=8)
    
    # Define interventions
    interventions = [
        CausalIntervention(
            variable="brightness",
            value=0.9,
            strength=1.0,
            description="High brightness"
        ),
        CausalIntervention(
            variable="texture",
            value=0.8,
            strength=1.0,
            description="Smooth texture"
        ),
        CausalIntervention(
            variable="shape",
            value=0.7,
            strength=1.0,
            description="Rounded shapes"
        )
    ]
    
    # Generate for each intervention
    print("\nGenerating counterfactual samples...")
    for intervention in interventions:
        print(f"\nIntervention: {intervention.description}")
        
        # Generate samples with this intervention
        counterfactual_samples = dreamer.imagine(
            interventions=[intervention],
            num_samples=8
        )
        
        # Calculate novelty compared to baseline
        novelty = CreativeMetrics.novelty_score(
            counterfactual_samples[0],
            baseline_samples
        )
        
        # Calculate diversity within generated samples
        diversity = CreativeMetrics.diversity_score(counterfactual_samples)
        
        print(f"Novelty score: {novelty:.4f}")
        print(f"Diversity score: {diversity:.4f}")
    
    # Generate with multiple interventions
    print("\nGenerating with multiple stacked interventions...")
    multi_intervention_samples = dreamer.imagine(
        interventions=interventions,
        num_samples=8
    )
    
    # Run novelty search
    print("\nRunning novelty search for creative concepts...")
    
    def extract_features(image):
        """Extract simple features for novelty search."""
        # Average brightness and texture features
        flat = image.view(-1)
        brightness = flat.mean().item()
        texture = torch.std(flat).item()
        
        # Simple shape feature - ratio of bright to dark pixels
        shape = (flat > 0.5).float().mean().item()
        
        return np.array([brightness, texture, shape])
    
    # Create novelty search
    novelty_search = NoveltySearch(
        base_model=dreamer,
        behavior_fn=extract_features,
        population_size=20,
        num_generations=10
    )
    
    # Run search
    best_params, novelty_scores = novelty_search.run_search()
    
    # Generate with best parameters
    novel_sample = dreamer.imagine(
        interventions=None,  # No explicit interventions
        num_samples=1,
        z=best_params.unsqueeze(0)  # Use discovered latent code
    )
    
    print("\nNovelty search complete!")
    print(f"Best novelty score: {novelty_scores[-1]:.4f}")
    print("\nNovelty progression over generations:")
    print(novelty_scores)
    
    print("\nThe CounterfactualDreamer demonstrates how causal interventions")
    print("can be used to generate creative variations and novel concepts.")


if __name__ == "__main__":
    main()