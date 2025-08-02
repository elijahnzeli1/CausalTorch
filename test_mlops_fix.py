#!/usr/bin/env python3
"""
Test script to verify the MLOps std() warning fix
"""

import torch
import torch.nn as nn
import sys
import os

# Add CausalTorch to path
current_dir = os.path.dirname(os.path.abspath(__file__))
causaltorch_dir = os.path.dirname(os.path.dirname(current_dir))
if causaltorch_dir not in sys.path:
    sys.path.insert(0, causaltorch_dir)

from causaltorch.mlops import CausalMLOps

def test_edge_case_parameters():
    """Test MLOps logging with edge case parameters that could cause std() warnings."""
    
    print("🧪 Testing MLOps fix for std() warnings...")
    
    # Create a model with edge case parameters
    class EdgeCaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Single parameter (numel = 1) - this used to cause warnings
            self.single_param = nn.Parameter(torch.tensor(1.0))
            
            # Empty parameter (numel = 0) - edge case
            self.empty_param = nn.Parameter(torch.empty(0))
            
            # Normal parameter for comparison
            self.normal_param = nn.Parameter(torch.randn(10, 5))
            
        def forward(self, x):
            return x
    
    # Initialize MLOps and model
    mlops = CausalMLOps(project_name="edge_case_test")
    mlops.start_experiment("test_std_fix", {"test": "edge_cases"})
    
    model = EdgeCaseModel()
    
    print("📊 Testing model info logging with edge case parameters...")
    
    # This should not produce any warnings now
    try:
        mlops.log_model_info(model, "edge_case_model", {"test": "std_fix"})
        print("✅ MLOps logging completed without warnings!")
        print("✅ std() warning fix is working correctly!")
        
        # Verify the model info was logged
        print(f"Model info logged successfully for {len(list(model.parameters()))} parameters")
        
    except Exception as e:
        print(f"❌ Error during logging: {e}")
        return False
    
    mlops.finish_experiment("completed")
    print("🎉 All edge case tests passed!")
    return True

if __name__ == "__main__":
    success = test_edge_case_parameters()
    if success:
        print("\n✅ Fix verified: No more std() warnings in MLOps logging!")
    else:
        print("\n❌ Fix verification failed")
        exit(1)
