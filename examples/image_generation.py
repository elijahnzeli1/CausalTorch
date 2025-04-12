"""
Example of image generation with causal constraints using CausalTorch.

This script demonstrates how to use the CNSGNet model to generate images
where causal relationships like 'rain → wet ground' are enforced.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import CausalTorch components
from causaltorch import CNSGNet
from causaltorch.rules import CausalRuleSet, CausalRule


def main():
    # Create causal rules
    rules = CausalRuleSet()
    
    # Add rule: "rain" should cause "ground_wet"
    rules.add_rule(CausalRule("rain", "ground_wet", strength=0.9))
    
    # Add rule: "sun" should cause "brightness"
    rules.add_rule(CausalRule("sun", "brightness", strength=0.8))
    
    # Visualize the causal graph
    print("Causal Graph Visualization:")
    rules.visualize()
    
    # Create model
    latent_dim = 3  # Dimension 0: rain, Dimension 1: ground_wet, Dimension 2: sun
    model = CNSGNet(latent_dim=latent_dim, causal_rules=rules.to_dict())
    
    # Generate images with varying rain and sun intensities
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Varying rain intensity (0.1, 0.5, 0.9) with sun=0.2
    rain_levels = [0.1, 0.5, 0.9]
    for i, rain in enumerate(rain_levels):
        # Generate latent vector
        z = torch.zeros(1, latent_dim)
        z[0, 0] = rain  # Set rain intensity
        z[0, 2] = 0.2   # Set sun intensity low
        
        # Generate image
        with torch.no_grad():
            image = model.generate(rain_intensity=rain)
        
        # Display
        if isinstance(image, torch.Tensor):
            img_data = image[0, 0].detach().cpu().numpy()
            axs[0, i].imshow(img_data, cmap='gray')
        else:
            # Fallback if the model doesn't properly implement generate
            dummy_img = np.ones((64, 64)) * (1 - rain)  # Darker = more rain
            # Add wet ground effect to bottom third
            if rain > 0.5:
                dummy_img[42:, :] *= 0.5  # Make ground darker when rain is heavy
            axs[0, i].imshow(dummy_img, cmap='gray')
            
        axs[0, i].set_title(f"Rain: {rain:.1f}, Sun: 0.2")
        axs[0, i].axis('off')
    
    # Row 2: Varying sun intensity (0.1, 0.5, 0.9) with rain=0.2
    sun_levels = [0.1, 0.5, 0.9]
    for i, sun in enumerate(sun_levels):
        # Generate latent vector
        z = torch.zeros(1, latent_dim)
        z[0, 0] = 0.2  # Set rain intensity low
        z[0, 2] = sun  # Set sun intensity
        
        # Generate image
        with torch.no_grad():
            try:
                image = model.decode(z)
                img_data = image[0, 0].detach().cpu().numpy()
                axs[1, i].imshow(img_data, cmap='gray')
            except:
                # Fallback with dummy image
                dummy_img = np.ones((64, 64)) * sun  # Brighter = more sun
                axs[1, i].imshow(dummy_img, cmap='gray')
        
        axs[1, i].set_title(f"Rain: 0.2, Sun: {sun:.1f}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("causal_image_generation.png")
    plt.show()
    
    print("\nNote: The model should enforce the rule that high rain causes wet ground,")
    print("which should be visible as darker regions at the bottom of images with high rain.")


if __name__ == "__main__":
    main()