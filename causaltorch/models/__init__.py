"""CausalTorch models package."""

from .text_model import (
    CausalTransformer, 
    CausalLanguageModel,
    SelfEvolvingTextGenerator,
    FewShotCausalTransformer,
    MultimodalCausalTransformer,
    CounterfactualCausalTransformer
)

from .image_model import CNSGImageGenerator 
from .video_model import CNSG_VideoGenerator

# Legacy alias for backward compatibility
CNSG_GPT2 = CausalTransformer

# Create aliases for consistency
CNSGTextGenerator = CausalTransformer
CNSGNet = CNSGImageGenerator  # Legacy alias

__all__ = [
    "CausalTransformer",
    "CausalLanguageModel",
    "SelfEvolvingTextGenerator", 
    "FewShotCausalTransformer",
    "MultimodalCausalTransformer",
    "CounterfactualCausalTransformer",
    "CNSG_GPT2",  # Legacy alias
    "CNSGTextGenerator",
    "CNSGImageGenerator",
    "CNSGNet", 
    "CNSG_VideoGenerator"
]