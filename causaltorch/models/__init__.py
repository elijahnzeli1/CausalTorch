"""CausalTorch models package."""

from .text_model import CNSG_GPT2
from .image_model import CNSGImageGenerator 
from .video_model import CNSG_VideoGenerator

# Create alias for consistency
CNSGTextGenerator = CNSG_GPT2
CNSGNet = CNSGImageGenerator  # Legacy alias

__all__ = [
    "CNSG_GPT2", 
    "CNSGTextGenerator",
    "CNSGImageGenerator",
    "CNSGNet", 
    "CNSG_VideoGenerator"
]