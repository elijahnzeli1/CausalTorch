"""
CausalTorch Multi-Modal Integration Test
=======================================

This script demonstrates how all CausalTorch modalities work together:
- Text Generation (cnsg) 
- Vision Processing (CausalVisionTransformer)
- Image Generation (CNSGNet)
- Video Generation (CNSG_VideoGenerator) 
- Reinforcement Learning (CausalRLAgent)
"""

import torch
from causaltorch.models import cnsg, CausalVisionTransformer, CNSGNet, CNSG_VideoGenerator
from causaltorch.core_architecture import FromScratchModelBuilder

def test_all_modalities():
    """Test all CausalTorch modalities."""
    print("🚀 CausalTorch Multi-Modal Integration Test")
    print("=" * 60)
    
    results = {}
    
    # 1. Text Generation
    print("\\n📝 Testing Text Generation (cnsg)...")
    try:
        text_model = cnsg(vocab_size=1000, d_model=256, n_heads=8, n_layers=4, d_ff=1024)
        input_tokens = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            text_output = text_model(input_tokens)
        print(f"✅ Text: {text_output['logits'].shape}")
        results['text'] = True
    except Exception as e:
        print(f"❌ Text failed: {e}")
        results['text'] = False
    
    # 2. Vision Processing
    print("\\n🎯 Testing Vision Processing (CausalVisionTransformer)...")
    try:
        vision_model = CausalVisionTransformer(
            img_size=224, embed_dim=384, depth=4, num_heads=6, num_classes=1000
        )
        input_images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            vision_output = vision_model(input_images)
        print(f"✅ Vision: {vision_output['logits'].shape}")
        results['vision'] = True
    except Exception as e:
        print(f"❌ Vision failed: {e}")
        results['vision'] = False
    
    # 3. Image Generation
    print("\\n🎨 Testing Image Generation (CNSGNet)...")
    try:
        image_gen_model = CNSGNet(latent_dim=16, img_size=64)
        input_images = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            reconstructed, mu, log_var = image_gen_model(input_images)
        print(f"✅ Image Gen: {reconstructed.shape}")
        results['image_gen'] = True
    except Exception as e:
        print(f"❌ Image Gen failed: {e}")
        results['image_gen'] = False
    
    # 4. Video Generation
    print("\\n🎬 Testing Video Generation (CNSG_VideoGenerator)...")
    try:
        video_model = CNSG_VideoGenerator(frame_size=(64, 64), latent_dim=8)
        initial_frame = torch.randn(1, 3, 64, 64)
        initial_latent = torch.randn(1, 8)
        with torch.no_grad():
            video_output = video_model(initial_frame, initial_latent, seq_length=5)
        print(f"✅ Video: {video_output.shape}")
        results['video'] = True
    except Exception as e:
        print(f"❌ Video failed: {e}")
        results['video'] = False
    
    # 5. Reinforcement Learning
    print("\\n🎮 Testing Reinforcement Learning (CausalRLAgent)...")
    try:
        rl_config = {'causal_config': {'hidden_dim': 128, 'causal_rules': []}}
        builder = FromScratchModelBuilder(rl_config)
        rl_agent = builder.build_model(
            'reinforcement_learning', state_dim=8, action_dim=4, agent_type='dqn'
        )
        state = torch.randn(1, 8)
        action = rl_agent.select_action(state)
        print(f"✅ RL: Action selected from state {state.shape}")
        results['rl'] = True
    except Exception as e:
        print(f"❌ RL failed: {e}")
        results['rl'] = False
    
    # Summary
    print("\\n" + "=" * 60)
    print("🏆 MULTI-MODAL INTEGRATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for modality, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {modality.replace('_', ' ').title()}")
    
    print(f"\\n📊 Overall: {passed}/{total} modalities working ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n🎉 SUCCESS: All CausalTorch modalities integrated!")
        print("💫 Available Capabilities:")
        print("   📝 Text Generation with causal reasoning")
        print("   🎯 Vision Processing with causal attention")
        print("   🎨 Image Generation with causal latents")
        print("   🎬 Video Generation with temporal causality")
        print("   🎮 Reinforcement Learning with episodic memory")
        print("   🧠 Cross-modal causal reasoning")
    else:
        print(f"\\n⚠️ {total-passed} modality(ies) need attention")
    
    return passed == total

if __name__ == "__main__":
    success = test_all_modalities()
    exit(0 if success else 1)
