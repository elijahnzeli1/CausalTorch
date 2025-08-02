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

# Import the new native cnsg implementation by importing the parent models module
try:
    # Import from the parent directory's models.py file
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(parent_dir, 'models.py')
    
    if os.path.exists(models_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location("parent_models", models_file)
        parent_models = importlib.util.module_from_spec(spec)
        
        # Set up required imports for the parent models
        parent_models.torch = __import__('torch')
        parent_models.nn = parent_models.torch.nn
        parent_models.F = parent_models.torch.nn.functional
        parent_models.math = __import__('math')
        
        # Import typing
        typing_module = __import__('typing')
        parent_models.Optional = typing_module.Optional
        parent_models.Dict = typing_module.Dict
        parent_models.Any = typing_module.Any
        parent_models.Tuple = typing_module.Tuple
        
        # Import layers
        layers_module = __import__('causaltorch.layers', fromlist=['CausalAttentionLayer', 'CausalSymbolicLayer', 'TemporalCausalConv'])
        parent_models.CausalAttentionLayer = layers_module.CausalAttentionLayer
        parent_models.CausalSymbolicLayer = layers_module.CausalSymbolicLayer
        parent_models.TemporalCausalConv = layers_module.TemporalCausalConv
        
        # Execute the parent models
        spec.loader.exec_module(parent_models)
        
        # Use the new native models
        cnsg = parent_models.cnsg
        CNSGNet = parent_models.CNSGNet
        CNSG_VideoGenerator = parent_models.CNSG_VideoGenerator
        
        # Import new vision models
        CausalVisionTransformer = parent_models.CausalVisionTransformer
        CausalObjectDetector = parent_models.CausalObjectDetector
        CausalSegmentationModel = parent_models.CausalSegmentationModel
        CausalVisionPatchEmbedding = parent_models.CausalVisionPatchEmbedding
        CausalVisionTransformerBlock = parent_models.CausalVisionTransformerBlock
        
    else:
        # Fallback to legacy
        cnsg = CausalTransformer
        CNSGNet = CNSGImageGenerator
        CausalVisionTransformer = None
        CausalObjectDetector = None
        CausalSegmentationModel = None
        CausalVisionPatchEmbedding = None
        CausalVisionTransformerBlock = None
        
except Exception as e:
    # Fallback to legacy implementation if anything fails
    cnsg = CausalTransformer
    CNSGNet = CNSGImageGenerator
    CausalVisionTransformer = None
    CausalObjectDetector = None
    CausalSegmentationModel = None
    CausalVisionPatchEmbedding = None
    CausalVisionTransformerBlock = None

# Create aliases for consistency
CNSGTextGenerator = CausalTransformer

__all__ = [
    "CausalTransformer",
    "CausalLanguageModel", 
    "SelfEvolvingTextGenerator",
    "FewShotCausalTransformer",
    "MultimodalCausalTransformer",
    "CounterfactualCausalTransformer",
    "cnsg",  # New native implementation when available, fallback to legacy
    "CNSGTextGenerator",
    "CNSGImageGenerator",
    "CNSGNet",  # New native implementation
    "CNSG_VideoGenerator",  # New native implementation
    # Vision models
    "CausalVisionTransformer",
    "CausalObjectDetector", 
    "CausalSegmentationModel",
    "CausalVisionPatchEmbedding",
    "CausalVisionTransformerBlock"
]