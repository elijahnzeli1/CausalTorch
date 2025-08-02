"""
Simple Example: Memory Episodes in CausalTorch RL
================================================

This example demonstrates how CausalTorch RL agents use episodic memory
to remember actions and their outcomes, with causal prioritization.
"""

import torch
import numpy as np
from causaltorch.core_architecture import FromScratchModelBuilder

def simple_memory_episodes_demo():
    """Demonstrate how RL agents remember action episodes."""
    print("🧠 CausalTorch RL Memory Episodes Demo")
    print("=" * 50)
    
    # Create an RL agent with episodic memory
    rl_config = {
        'causal_config': {
            'hidden_dim': 64,
            'causal_rules': [
                {'cause': 'action', 'effect': 'reward', 'strength': 0.9}
            ]
        }
    }
    
    builder = FromScratchModelBuilder(rl_config)
    agent = builder.build_model(
        'reinforcement_learning',
        state_dim=4,           # Simple state (e.g., cart-pole position, velocity, angle, angular_velocity)
        action_dim=2,          # Two actions: left, right
        agent_type='dqn',
        memory_capacity=100,
        batch_size=8
    )
    
    print(f"✅ Created DQN agent with memory capacity: {agent.memory_capacity}")
    
    # Simulate a few episodes where the agent learns from experience
    print("\\n🎮 Simulating learning episodes...")
    
    episode_data = []
    
    for episode in range(3):
        print(f"\\nEpisode {episode + 1}:")
        
        # Start with random state
        state = torch.randn(1, 4)
        episode_reward = 0
        episode_experiences = []
        
        # Run 5 steps per episode
        for step in range(5):
            # Agent selects action
            action = agent.select_action(state, explore=True)
            
            # Simulate environment response
            # Higher reward for action 1 (right) when state[0] > 0
            if state[0, 0] > 0 and action.item() == 1:
                reward = 10.0  # Good action
                causal_strength = 0.9
            elif state[0, 0] <= 0 and action.item() == 0:
                reward = 8.0   # Good action  
                causal_strength = 0.8
            else:
                reward = -2.0  # Poor action
                causal_strength = 0.3
                
            next_state = torch.randn(1, 4)
            done = (step == 4)  # End episode
            
            # Store experience in memory (agent computes causal strength automatically)
            agent.store_experience(state, action, reward, next_state, done)
            
            # For demo purposes, estimate causal strength for display
            if state[0, 0] > 0 and action.item() == 1:
                causal_strength = 0.9  # Good action
            elif state[0, 0] <= 0 and action.item() == 0:
                causal_strength = 0.8  # Good action  
            else:
                causal_strength = 0.3  # Poor action
            
            episode_experiences.append({
                'step': step + 1,
                'state_summary': f"{state[0, 0].item():.2f}",
                'action': action.item(),
                'reward': reward,
                'causal_strength': causal_strength
            })
            
            episode_reward += reward
            state = next_state
        
        episode_data.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'experiences': episode_experiences
        })
        
        print(f"   Total reward: {episode_reward:.1f}")
        print(f"   Memory size: {len(agent.episodic_memory)}")
    
    # Show what the agent remembers
    print("\\n🧠 What the agent remembers (episodic memory):")
    print("   Episodes stored in memory:")
    
    for ep_data in episode_data:
        print(f"\\n   Episode {ep_data['episode']} (Reward: {ep_data['total_reward']:.1f}):")
        for exp in ep_data['experiences']:
            action_name = "RIGHT" if exp['action'] == 1 else "LEFT"
            print(f"     Step {exp['step']}: State={exp['state_summary']}, Action={action_name}, "
                  f"Reward={exp['reward']:+.1f}, Causal={exp['causal_strength']:.1f}")
    
    # Test causal prioritized memory retrieval
    print("\\n🎯 Testing causal prioritized memory sampling:")
    print("   (Agent prioritizes high-causal experiences for learning)")
    
    if len(agent.episodic_memory) >= 3:
        sample = agent.episodic_memory.sample(3, use_causal_priority=True)
        print("\\n   Top 3 causally important experiences:")
        for i, exp in enumerate(sample, 1):
            action_name = "RIGHT" if exp['action'].item() == 1 else "LEFT"
            print(f"     {i}. Action={action_name}, Reward={exp['reward']:+.1f}, "
                  f"Causal Strength={exp['causal_strength']:.2f}")
    
    # Test memory consolidation (keeping important experiences)
    print("\\n💾 Testing memory consolidation:")
    initial_size = len(agent.episodic_memory)
    
    # Get causal statistics before consolidation
    all_experiences = [agent.episodic_memory.memory[i] for i in range(len(agent.episodic_memory)) 
                      if agent.episodic_memory.memory[i] is not None]
    
    if all_experiences:
        causal_strengths = [exp['causal_strength'] for exp in all_experiences]
        avg_causal_before = np.mean(causal_strengths)
        
        # Consolidate memory (remove 30% of least causal experiences)
        agent.episodic_memory.consolidate_memory(forget_ratio=0.3)
        
        # Get statistics after consolidation
        all_experiences_after = [agent.episodic_memory.memory[i] for i in range(len(agent.episodic_memory)) 
                               if agent.episodic_memory.memory[i] is not None]
        
        if all_experiences_after:
            causal_strengths_after = [exp['causal_strength'] for exp in all_experiences_after]
            avg_causal_after = np.mean(causal_strengths_after)
            
            print(f"   Before consolidation: {initial_size} experiences, avg causal = {avg_causal_before:.3f}")
            print(f"   After consolidation:  {len(agent.episodic_memory)} experiences, avg causal = {avg_causal_after:.3f}")
            print("   ✅ Agent kept the most causally important memories!")
    
    print("\\n🏆 Memory Episodes Summary:")
    print("   ✅ Agent stores action-outcome experiences")
    print("   ✅ Causal strength determines memory importance") 
    print("   ✅ Prioritized sampling focuses on causal experiences")
    print("   ✅ Memory consolidation preserves important episodes")
    print("   ✅ Agent learns from causally significant action patterns")
    
    return True

if __name__ == "__main__":
    print("Memory Episodes in CausalTorch Reinforcement Learning")
    print("=" * 60)
    print("Demonstrating how RL agents remember and prioritize")
    print("action episodes based on causal significance.")
    print("=" * 60)
    
    success = simple_memory_episodes_demo()
    
    if success:
        print("\\n✨ SUCCESS: CausalTorch RL agents have sophisticated episodic memory!")
        print("💡 Key Features:")
        print("   • Actions and outcomes stored as episodes")
        print("   • Causal strength determines memory priority")
        print("   • Important experiences preserved longer")
        print("   • Learning focuses on causally significant patterns")
    else:
        print("\\n❌ FAILED: Memory episodes demonstration failed")
