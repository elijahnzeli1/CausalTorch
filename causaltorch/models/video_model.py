"""
Video generation models with causal-temporal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TemporalCausalConv


class CNSG_VideoGenerator(nn.Module):
    """Causal Neuro-Symbolic Video Generator.
    
    This model generates temporally consistent video with causal constraints
    between frames.
    
    Args:
        frame_size (tuple): Height and width of video frames
        latent_dim (int): Dimension of the latent space
        causal_rules (dict, optional): Dictionary of temporal causal rules
    """
    def __init__(self, frame_size=(64, 64), latent_dim=16, causal_rules=None):
        super().__init__()
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        height, width = frame_size
        
        # Frame generator network
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 3, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Latent dynamics network (predicts next latent state)
        self.latent_encoder = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        
        # Temporal causal layer
        self.temporal_causal = TemporalCausalConv(3, 3, kernel_size=3, causal_rules=causal_rules or {})
    
    def forward(self, initial_frame, initial_latent, seq_length=24, metadata=None):
        """Generate a video sequence with causal temporal constraints.
        
        Args:
            initial_frame (torch.Tensor): Starting frame [batch, 3, H, W]
            initial_latent (torch.Tensor): Initial latent state [batch, latent_dim]
            seq_length (int): Number of frames to generate
            metadata (dict, optional): Frame metadata for causal rules
            
        Returns:
            torch.Tensor: Generated video sequence [batch, seq_length, 3, H, W]
        """
        batch_size = initial_frame.size(0)
        device = initial_frame.device
        frames = [initial_frame]
        latent = initial_latent
        
        # Generate frames sequentially
        for t in range(seq_length - 1):
            # Get previous frame
            prev_frame = frames[-1]
            
            # Update latent state
            latent_input = latent.unsqueeze(1)  # Add sequence dimension
            latent_output, _ = self.latent_encoder(latent_input)
            latent = latent_output.squeeze(1)  # Remove sequence dimension
            
            # Generate next frame
            gen_input = torch.cat([prev_frame, latent.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 1)], dim=1)
            next_frame = self.generator(gen_input)
            
            # Apply temporal causal effects
            if metadata is not None:
                # Get metadata for current frame
                frame_metadata = {k: v[t] if isinstance(v, list) else v for k, v in metadata.items()}
                next_frame = self.temporal_causal(next_frame.unsqueeze(2), frame_metadata).squeeze(2)
            
            frames.append(next_frame)
        
        # Stack frames to make video
        video = torch.stack(frames, dim=1)  # [batch, seq_length, 3, H, W]
        return video
    
    def generate_battle_scene(self, num_frames=24):
        """Generate a battle scene with horses, arrows, and causal effects.
        
        Args:
            num_frames (int): Number of frames to generate
            
        Returns:
            torch.Tensor: Generated battle video
        """
        # Create initial inputs
        batch_size = 1
        initial_frame = torch.randn(batch_size, 3, self.frame_size[0], self.frame_size[1])
        initial_latent = torch.zeros(batch_size, self.latent_dim)
        
        # Set up metadata with causal events
        metadata = {
            # Hoof contacts ground at specific frames
            "hoof_contact": [1.0 if i % 6 == 0 else 0.0 for i in range(num_frames)],
            
            # Arrow hits at frame 10
            "arrow_hit": [1.0 if i == 10 else 0.0 for i in range(num_frames)]
        }
        
        # Generate video
        return self.forward(initial_frame, initial_latent, seq_length=num_frames, metadata=metadata)