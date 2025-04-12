"""
Image generation models with causal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import CausalSymbolicLayer


class CNSGImageGenerator(nn.Module):
    """Causal Neuro-Symbolic Generative Network for image generation.
    
    This model implements a VAE/GAN with causal structure in the latent space.
    
    Args:
        latent_dim (int): Dimension of the latent space
        causal_rules (dict, optional): Dictionary of causal rules
        img_size (int, optional): Size of generated images
    """
    def __init__(self, latent_dim=3, causal_rules=None, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and variance for VAE
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Causal layer to enforce relationships in latent space
        self.causal_layer = CausalSymbolicLayer(causal_rules)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            tuple: (mu, log_var) parameters of latent distribution
        """
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Generated image
        """
        # Apply causal constraints to latent vector
        z = self.causal_layer(z)
        
        # Decode to image
        h = self.decoder(z)
        return h.view(h.size(0), 1, self.img_size, self.img_size)
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            tuple: (reconstructed_x, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
    
    def generate(self, rain_intensity=None, num_samples=1):
        """Generate images with causal constraints.
        
        Args:
            rain_intensity (float, optional): Rain intensity value (0-1)
            num_samples (int, optional): Number of images to generate
            
        Returns:
            torch.Tensor: Generated images
        """
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_samples, self.latent_dim)
            
            # If rain_intensity is specified, set the rain dimension
            if rain_intensity is not None:
                z[:, 0] = rain_intensity
            
            # Generate images with causal constraints
            images = self.decode(z)
            return images