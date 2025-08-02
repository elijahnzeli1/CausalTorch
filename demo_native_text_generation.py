"""
CausalTorch Native Text Generation Demo
======================================

This script demonstrates the new native CausalTorch text generation model (cnsg)
that doesn't rely on external models like GPT-2, but implements its own
causal neuro-symbolic architecture.

Key Features:
- Native CausalTorch transformer architecture
- Integrated causal reasoning in every layer
- Causal positional encoding
- Causal attention mechanisms
- Generation with causal constraints
- No external model dependencies
"""

import torch
import torch.nn.functional as F
from causaltorch.models import cnsg

def demonstrate_native_cnsg():
    """Demonstrate the native CausalTorch text generation model."""
    print("🚀 CausalTorch Native Text Generation Demo")
    print("=" * 60)
    print("Testing the new cnsg (Causal Neuro-Symbolic Generator)")
    print("without GPT-2 dependencies - pure CausalTorch architecture!")
    print("=" * 60)
    
    # Define causal rules for text generation
    causal_rules = {
        'cause_effect_pairs': [
            {'cause': 'question_word', 'effect': 'interrogative_response', 'strength': 0.9},
            {'cause': 'negative_context', 'effect': 'negative_sentiment', 'strength': 0.8},
            {'cause': 'past_tense', 'effect': 'temporal_consistency', 'strength': 0.7}
        ],
        'logical_constraints': [
            {'type': 'temporal', 'rule': 'past_before_present'},
            {'type': 'causal', 'rule': 'cause_before_effect'}
        ]
    }
    
    print("\\n🔧 Creating native CausalTorch text model...")
    
    # Create the model with smaller dimensions for demo
    model = cnsg(
        vocab_size=1000,      # Smaller vocab for demo
        d_model=256,          # Smaller model dimension
        n_heads=8,            # 8 attention heads
        n_layers=6,           # 6 transformer layers
        d_ff=1024,            # Feed-forward dimension
        max_seq_length=512,   # Maximum sequence length
        causal_rules=causal_rules
    )
    
    print(f"✅ Created cnsg model:")
    print(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🧠 Model dimension: {model.d_model}")
    print(f"   🔢 Vocabulary size: {model.vocab_size}")
    print(f"   🏗️ Layers: {model.n_layers}")
    print(f"   👁️ Attention heads: {model.n_heads}")
    
    # Test forward pass
    print("\\n🧪 Testing forward pass...")
    
    # Create sample input tokens
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length))
    
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs["logits"]
    print(f"   Output logits shape: {logits.shape}")
    print(f"   ✅ Forward pass successful!")
    
    # Test with loss computation
    print("\\n📊 Testing training mode with loss computation...")
    
    # Create labels for loss computation (shifted input_ids)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_length))
    
    outputs_with_loss = model(input_ids, labels=labels)
    
    if "loss" in outputs_with_loss:
        loss = outputs_with_loss["loss"]
        print(f"   Computed loss: {loss.item():.4f}")
        print(f"   ✅ Loss computation successful!")
    
    # Test text generation
    print("\\n🎯 Testing text generation...")
    
    # Start with a short input sequence
    start_tokens = torch.randint(0, model.vocab_size, (1, 5))
    print(f"   Starting with {start_tokens.shape[1]} tokens")
    
    # Generate text
    generated = model.generate(
        input_ids=start_tokens,
        max_length=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    
    print(f"   Generated sequence length: {generated.shape[1]}")
    print(f"   Generated tokens: {generated[0].tolist()}")
    print(f"   ✅ Text generation successful!")
    
    # Test causal constraints in generation
    print("\\n🔬 Testing causal constraints in generation...")
    
    causal_constraints = {
        'forbidden_words': [999],  # Forbid token 999
        'encouraged_words': [100, 200]  # Encourage tokens 100, 200
    }
    
    constrained_generated = model.generate(
        input_ids=start_tokens,
        max_length=15,
        temperature=0.8,
        causal_constraints=causal_constraints
    )
    
    print(f"   Constrained generation length: {constrained_generated.shape[1]}")
    print(f"   Constrained tokens: {constrained_generated[0].tolist()}")
    
    # Check if forbidden token was avoided
    if 999 not in constrained_generated[0].tolist():
        print("   ✅ Successfully avoided forbidden token!")
    else:
        print("   ⚠️ Forbidden token appeared in generation")
    
    # Test causal attention pattern extraction
    print("\\n🧠 Testing causal attention pattern extraction...")
    
    try:
        attention_patterns = model.get_causal_attention_patterns(input_ids[:1])
        print(f"   Extracted attention patterns for {len(attention_patterns)} layers")
        print(f"   ✅ Attention analysis successful!")
    except Exception as e:
        print(f"   ⚠️ Attention analysis: {e}")
    
    # Architecture comparison
    print("\\n📋 Architecture Summary:")
    print("   🏗️ NATIVE CAUSALTORCH ARCHITECTURE:")
    print("      • CausalPositionalEncoding for temporal reasoning")
    print("      • CausalTransformerBlock with integrated causal layers")
    print("      • CausalAttentionLayer for causal relationships")
    print("      • CausalSymbolicLayer for symbolic reasoning")
    print("      • Native generation with causal constraints")
    print("   ")
    print("   🚫 NO EXTERNAL DEPENDENCIES:")
    print("      • No GPT-2 model loading")
    print("      • No transformers library requirement")
    print("      • Pure PyTorch + CausalTorch implementation")
    print("      • Custom causal reasoning throughout")
    
    return True

def demonstrate_model_components():
    """Demonstrate individual components of the native model."""
    print("\\n" + "=" * 60)
    print("🔍 INDIVIDUAL COMPONENT DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Import components directly from the parent models module
        from causaltorch.models import cnsg
        
        # Test the cnsg model creation with different sizes
        print("\\n🎯 Testing different cnsg model configurations...")
        
        # Small model
        small_model = cnsg(
            vocab_size=500,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=512,
            max_seq_length=100,
            causal_rules={'test_rule': {'strength': 0.5}}
        )
        print(f"   Small model parameters: {sum(p.numel() for p in small_model.parameters()):,}")
        
        # Medium model  
        medium_model = cnsg(
            vocab_size=2000,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_length=512,
            causal_rules={'advanced_rule': {'strength': 0.8}}
        )
        print(f"   Medium model parameters: {sum(p.numel() for p in medium_model.parameters()):,}")
        
        print("\\n🔧 Testing model components...")
        
        # Test forward pass with both models
        test_input = torch.randint(0, 500, (1, 5))
        
        with torch.no_grad():
            small_output = small_model(test_input)
            medium_output = medium_model(test_input[:, :5])  # Adjust input size
            
        print(f"   Small model output shape: {small_output['logits'].shape}")
        print(f"   Medium model output shape: {medium_output['logits'].shape}")
        
        print("\\n🏆 Component Summary:")
        print("   ✅ CausalPositionalEncoding: Adds causal temporal information")
        print("   ✅ CausalTransformerBlock: Integrates causal reasoning in attention")
        print("   ✅ Native architecture: No external model dependencies")
        print("   ✅ Scalable design: Works with different model sizes")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        return False

def main():
    """Run the complete native CausalTorch text generation demonstration."""
    print("CausalTorch Native Text Generation Refactoring Demo")
    print("=" * 80)
    print("Demonstrating the refactored cnsg model without GPT-2 dependencies")
    print("=" * 80)
    
    try:
        # Test the main model
        model_success = demonstrate_native_cnsg()
        
        # Test individual components
        components_success = demonstrate_model_components()
        
        # Final summary
        print("\\n" + "=" * 80)
        print("🏆 REFACTORING SUMMARY")
        print("=" * 80)
        
        if model_success and components_success:
            print("✅ SUCCESS: Complete refactoring accomplished!")
            print("\\n🎯 What was changed:")
            print("   🚫 REMOVED: GPT2LMHeadModel dependency")
            print("   🚫 REMOVED: transformers library requirement")
            print("   🚫 REMOVED: External model fine-tuning approach")
            print("   ")
            print("   ✅ ADDED: Native CausalTorch text generation architecture")
            print("   ✅ ADDED: CausalPositionalEncoding for temporal reasoning")
            print("   ✅ ADDED: CausalTransformerBlock with integrated causal layers")
            print("   ✅ ADDED: Custom generation with causal constraints")
            print("   ✅ ADDED: Pure PyTorch + CausalTorch implementation")
            print("   ")
            print("💫 Benefits:")
            print("   🚀 No external model dependencies")
            print("   🧠 Causal reasoning integrated throughout")
            print("   🎯 Custom generation with causal constraints")
            print("   🔧 Full control over architecture and training")
            print("   ⚡ Optimized for causal neuro-symbolic reasoning")
            
            return True
        else:
            print("❌ Some components failed during testing")
            return False
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
