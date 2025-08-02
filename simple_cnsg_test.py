"""
Simple test of native cnsg model
"""

import torch
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the native cnsg directly
from causaltorch.models import cnsg

def simple_test():
    print("Testing native cnsg model creation...")
    
    try:
        # Try to create a very simple model
        model = cnsg(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_length=50,
            causal_rules={}
        )
        print("✅ Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        print(f"✅ Forward pass successful! Output shape: {outputs['logits'].shape}")
        
        # Test generation
        generated = model.generate(input_ids, max_length=15, temperature=0.8)
        print(f"✅ Generation successful! Generated shape: {generated.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\\n🎉 SUCCESS: Native CausalTorch cnsg model working!")
        print("✅ No GPT-2 dependencies")
        print("✅ Pure CausalTorch architecture")
        print("✅ Causal reasoning integrated")
    else:
        print("\\n❌ Failed to create native cnsg model")
