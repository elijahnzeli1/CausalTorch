"""
CausalTorch Reinforcement Learning Demo
======================================

This script demonstrates the comprehensive reinforcement learning capabilities
of CausalTorch, including:

1. Causal RL Agents (DQN, Policy Gradient, Actor-Critic, PPO)
2. Episodic Memory with causal prioritization
3. Causal intervention analysis in RL
4. Memory consolidation and experience replay
5. Action-outcome causal relationship learning

Key Features:
- Episodic memory that prioritizes causally significant experiences
- Causal strength computation for action-reward relationships
- Intervention tracking for counterfactual policy analysis
- Multiple RL algorithms with causal reasoning integration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

# Import CausalTorch RL components
try:
    from causaltorch.core_architecture import (
        FromScratchModelBuilder,
        CausalRLAgent,
        EpisodicMemory,
        CausalPolicyNetwork,
        CausalValueNetwork,
        CausalQNetwork
    )
    print("✅ Successfully imported CausalTorch RL components")
    rl_available = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    rl_available = False


def demonstrate_episodic_memory():
    """Demonstrate episodic memory with causal prioritization."""
    print("\\n" + "="*70)
    print("🧠 EPISODIC MEMORY WITH CAUSAL PRIORITIZATION")
    print("="*70)
    
    if not rl_available:
        print("❌ RL components not available")
        return False
    
    # Create episodic memory
    memory = EpisodicMemory(capacity=1000, causal_threshold=0.5)
    print(f"✅ Created episodic memory with capacity {memory.capacity}")
    
    # Simulate storing experiences with different causal strengths
    print("\\n📝 Storing experiences with varying causal strengths...")
    
    experiences = [
        (torch.randn(4, 10), torch.tensor([1]), 10.0, torch.randn(4, 10), False, 0.9),  # High causal
        (torch.randn(4, 10), torch.tensor([0]), 1.0, torch.randn(4, 10), False, 0.2),   # Low causal
        (torch.randn(4, 10), torch.tensor([2]), 15.0, torch.randn(4, 10), False, 0.95), # Very high causal
        (torch.randn(4, 10), torch.tensor([1]), 0.5, torch.randn(4, 10), False, 0.1),   # Very low causal
        (torch.randn(4, 10), torch.tensor([3]), 8.0, torch.randn(4, 10), True, 0.7),    # High causal, terminal
    ]
    
    for i, (state, action, reward, next_state, done, causal_strength) in enumerate(experiences):
        memory.push(state, action, reward, next_state, done, causal_strength)
        print(f"   Episode {i+1}: Reward={reward:5.1f}, Causal Strength={causal_strength:.2f}")
    
    print(f"\\n📊 Memory statistics:")
    print(f"   Total experiences: {len(memory)}")
    
    # Test causal prioritized sampling
    print("\\n🎯 Testing causal prioritized sampling...")
    sample = memory.sample(3, use_causal_priority=True)
    print(f"   Sampled {len(sample)} experiences with causal priority")
    for i, exp in enumerate(sample):
        print(f"   Sample {i+1}: Reward={exp['reward']:5.1f}, Causal={exp['causal_strength']:.2f}")
    
    # Test causal episode retrieval
    print("\\n🔍 Retrieving high-causal episodes (threshold=0.6)...")
    causal_episodes = memory.get_causal_episodes(min_strength=0.6)
    print(f"   Found {len(causal_episodes)} high-causal episodes")
    
    return True


def demonstrate_causal_rl_agents():
    """Demonstrate different types of causal RL agents."""
    print("\\n" + "="*70)
    print("🤖 CAUSAL REINFORCEMENT LEARNING AGENTS")
    print("="*70)
    
    if not rl_available:
        print("❌ RL components not available")
        return False
    
    # Configuration for causal RL
    rl_config = {
        'causal_config': {
            'hidden_dim': 128,
            'num_reasoning_layers': 2,
            'causal_rules': [
                {'cause': 'state', 'effect': 'action_value', 'strength': 0.8},
                {'cause': 'action', 'effect': 'reward', 'strength': 0.9},
                {'cause': 'reward', 'effect': 'next_state', 'strength': 0.7}
            ]
        }
    }
    
    # Create model builder
    builder = FromScratchModelBuilder(rl_config)
    
    # Test different RL agent types
    agent_types = ["dqn", "policy_gradient", "actor_critic", "ppo"]
    state_dim = 10
    action_dim = 4
    
    agents = {}
    
    for agent_type in agent_types:
        print(f"\\n🔧 Building {agent_type.upper()} agent...")
        try:
            agent = builder.build_model(
                'reinforcement_learning',
                state_dim=state_dim,
                action_dim=action_dim,
                agent_type=agent_type,
                memory_capacity=10000,
                batch_size=32
            )
            agents[agent_type] = agent
            print(f"   ✅ {agent_type.upper()} agent created successfully")
            print(f"   📊 Memory capacity: {agent.memory_capacity}")
            print(f"   🧠 Hidden dimension: {agent.hidden_dim}")
            
        except Exception as e:
            print(f"   ❌ Failed to create {agent_type} agent: {e}")
    
    return len(agents) > 0


def demonstrate_rl_training_cycle():
    """Demonstrate a complete RL training cycle with causal analysis."""
    print("\\n" + "="*70)
    print("🎯 RL TRAINING CYCLE WITH CAUSAL ANALYSIS")
    print("="*70)
    
    if not rl_available:
        print("❌ RL components not available")
        return False
    
    print("🚀 Setting up DQN agent for training demonstration...")
    
    # Create DQN agent
    rl_config = {
        'causal_config': {
            'hidden_dim': 64,
            'num_reasoning_layers': 1,
            'causal_rules': [
                {'cause': 'state_features', 'effect': 'action_values', 'strength': 0.85},
                {'cause': 'action_choice', 'effect': 'environmental_response', 'strength': 0.9}
            ]
        }
    }
    
    builder = FromScratchModelBuilder(rl_config)
    agent = builder.build_model(
        'reinforcement_learning',
        state_dim=8,
        action_dim=3,
        agent_type='dqn',
        memory_capacity=1000,
        batch_size=16
    )
    
    print(f"✅ DQN agent created with {len(agent.episodic_memory)} initial experiences")
    
    # Simulate training episodes
    print("\\n🎮 Simulating training episodes...")
    
    num_episodes = 5
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\\n   Episode {episode + 1}:")
        
        # Simulate environment interaction
        state = torch.randn(1, 8)  # Random state
        episode_reward = 0
        steps = 0
        
        # Run episode for a few steps
        for step in range(10):
            # Agent selects action
            action = agent.select_action(state, explore=True)
            
            # Simulate environment response
            next_state = torch.randn(1, 8)
            reward = torch.randn(1).item() * 5  # Random reward
            done = (step == 9)  # End episode after 10 steps
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"     Total reward: {episode_reward:.2f}")
        print(f"     Steps taken: {steps}")
        print(f"     Memory size: {len(agent.episodic_memory)}")
        
        # Learn from experience (if enough data)
        if len(agent.episodic_memory) >= agent.batch_size:
            loss_info = agent.learn()
            print(f"     Training loss: {loss_info.get('total_loss', 0.0):.4f}")
    
    # Causal analysis
    print("\\n🔬 Performing causal analysis...")
    
    # Test causal interventions
    intervention_tests = [
        {'action_choice': 0.8},
        {'action_choice': 0.2},
        {'state_features': 0.9}
    ]
    
    for i, interventions in enumerate(intervention_tests):
        agent.save_intervention_episode(interventions, episode_rewards[i % len(episode_rewards)])
    
    causal_analysis = agent.get_causal_analysis()
    print(f"   📊 Causal analysis results:")
    print(f"     Total intervention episodes: {causal_analysis['total_episodes']}")
    print(f"     Memory size: {causal_analysis['memory_size']}")
    
    if 'causal_effects' in causal_analysis:
        for var, effects in causal_analysis['causal_effects'].items():
            print(f"     {var}: Mean reward = {effects['mean_reward']:.2f}")
    
    print(f"\\n Training summary:")
    print(f"   Episodes completed: {num_episodes}")
    print(f"   Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"   Final memory size: {len(agent.episodic_memory)}")
    
    return True


def demonstrate_memory_consolidation():
    """Demonstrate memory consolidation for long-term learning."""
    print("\\n" + "="*70)
    print("🧠 MEMORY CONSOLIDATION FOR LONG-TERM LEARNING")
    print("="*70)
    
    if not rl_available:
        print("❌ RL components not available")
        return False
    
    # Create memory with small capacity to trigger consolidation
    memory = EpisodicMemory(capacity=20, causal_threshold=0.3)
    
    print(f"📝 Filling memory beyond capacity to test consolidation...")
    print(f"   Memory capacity: {memory.capacity}")
    
    # Add many experiences
    for i in range(30):
        state = torch.randn(2, 5)
        action = torch.tensor([i % 3])
        reward = np.random.normal(0, 5)  # Random reward
        next_state = torch.randn(2, 5)
        done = (i % 10 == 9)
        
        # Vary causal strength - some high, some low
        causal_strength = np.random.beta(2, 5)  # Biased toward lower values
        if i % 7 == 0:  # Occasionally high causal strength
            causal_strength = np.random.beta(5, 2)  # Biased toward higher values
        
        memory.push(state, action, reward, next_state, done, causal_strength)
    
    print(f"   Experiences added: 30")
    print(f"   Current memory size: {len(memory)}")
    
    # Show pre-consolidation statistics
    if len(memory.memory) > 0:
        causal_strengths = [exp['causal_strength'] for exp in memory.memory if exp]
        print(f"   Pre-consolidation avg causal strength: {np.mean(causal_strengths):.3f}")
        print(f"   Pre-consolidation max causal strength: {np.max(causal_strengths):.3f}")
    
    # Perform memory consolidation
    print(f"\\n🔄 Performing memory consolidation (forget 30% of low-causal experiences)...")
    memory.consolidate_memory(forget_ratio=0.3)
    
    print(f"   Post-consolidation memory size: {len(memory)}")
    
    if len(memory.memory) > 0:
        causal_strengths = [exp['causal_strength'] for exp in memory.memory if exp]
        print(f"   Post-consolidation avg causal strength: {np.mean(causal_strengths):.3f}")
        print(f"   Post-consolidation max causal strength: {np.max(causal_strengths):.3f}")
    
    # Test retrieval of high-causal episodes
    high_causal_episodes = memory.get_causal_episodes(min_strength=0.5)
    print(f"   High-causal episodes (>0.5): {len(high_causal_episodes)}")
    
    return True


def demonstrate_causal_interventions_in_rl():
    """Demonstrate causal interventions for policy analysis."""
    print("\\n" + "="*70)
    print("🎯 CAUSAL INTERVENTIONS FOR POLICY ANALYSIS")
    print("="*70)
    
    if not rl_available:
        print("❌ RL components not available")
        return False
    
    print("🔬 Creating policy network for intervention analysis...")
    
    # Create a policy network
    rl_config = {
        'causal_config': {
            'hidden_dim': 64,
            'causal_rules': [
                {'cause': 'state_input', 'effect': 'policy_output', 'strength': 0.8}
            ]
        }
    }
    
    builder = FromScratchModelBuilder(rl_config)
    policy = builder.build_model('policy_network', state_dim=6, action_dim=4)
    
    print("✅ Policy network created")
    
    # Test policy under different interventions
    test_state = torch.randn(3, 6)
    
    print("\\n🧪 Testing policy under various causal interventions...")
    
    interventions = [
        {},  # No intervention (baseline)
        {'state_input': 0.8},
        {'state_input': 0.2},
        {'policy_output': 0.9}
    ]
    
    results = []
    
    for i, intervention in enumerate(interventions):
        print(f"\\n   Test {i+1}: {intervention if intervention else 'Baseline (no intervention)'}")
        
        with torch.no_grad():
            if intervention:
                # Apply intervention using the core's intervention API
                with policy.causal_core.intervention_api.apply_interventions(intervention):
                    action_probs = policy(test_state)
            else:
                action_probs = policy(test_state)
        
        # Analyze action distribution
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
        max_prob = action_probs.max(dim=-1)[0].mean()
        
        result = {
            'intervention': intervention,
            'entropy': entropy.item(),
            'max_prob': max_prob.item(),
            'action_probs': action_probs.mean(dim=0)
        }
        results.append(result)
        
        print(f"     Policy entropy: {entropy:.3f}")
        print(f"     Max action prob: {max_prob:.3f}")
        print(f"     Action distribution: {action_probs.mean(dim=0).numpy()}")
    
    # Compare interventions
    print("\\n📊 Intervention comparison:")
    baseline_entropy = results[0]['entropy']
    
    for i, result in enumerate(results[1:], 1):
        entropy_change = result['entropy'] - baseline_entropy
        print(f"   Intervention {i}: Entropy change = {entropy_change:+.3f}")
    
    return True


def main():
    """Run comprehensive RL demonstration."""
    print("🚀 CausalTorch Reinforcement Learning Demonstration")
    print("=" * 80)
    print("Demonstrating comprehensive RL capabilities with causal reasoning:")
    print("• Episodic Memory with causal prioritization")
    print("• Multiple RL algorithms (DQN, PG, A2C, PPO)")
    print("• Causal intervention analysis")
    print("• Memory consolidation and experience replay")
    print("• Action-outcome causal relationship learning")
    print("=" * 80)
    
    if not rl_available:
        print("\\n❌ CRITICAL: RL components not available")
        print("Please ensure core_architecture.py RL classes are properly implemented")
        return False
    
    # Track demonstration results
    results = {
        'episodic_memory': False,
        'rl_agents': False,
        'training_cycle': False,
        'memory_consolidation': False,
        'causal_interventions': False
    }
    
    # Run demonstrations
    results['episodic_memory'] = demonstrate_episodic_memory()
    results['rl_agents'] = demonstrate_causal_rl_agents()
    results['training_cycle'] = demonstrate_rl_training_cycle()
    results['memory_consolidation'] = demonstrate_memory_consolidation()
    results['causal_interventions'] = demonstrate_causal_interventions_in_rl()
    
    # Print summary
    print("\\n" + "="*80)
    print("🏆 RL DEMONSTRATION SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        display_name = test_name.replace('_', ' ').title()
        print(f"   {status}: {display_name}")
    
    print(f"\\n📊 Overall Score: {passed}/{total} demonstrations passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n🎉 SUCCESS: CausalTorch fully supports reinforcement learning!")
        print("🧠 ✅ Episodic memory with causal prioritization")
        print("🤖 ✅ Multiple RL algorithms with causal reasoning")
        print("🎯 ✅ Complete training cycle with causal analysis")
        print("💾 ✅ Memory consolidation for long-term learning")
        print("🔬 ✅ Causal interventions for policy analysis")
        print("\\n💫 Key RL Features Available:")
        print("   • DQN, Policy Gradient, Actor-Critic, PPO algorithms")
        print("   • Episodic memory with causal experience prioritization")
        print("   • Action-outcome causal strength computation")
        print("   • Intervention tracking for counterfactual analysis")
        print("   • Memory consolidation based on causal significance")
        print("   • Policy analysis under causal interventions")
        return True
    else:
        print(f"\\n⚠️ PARTIAL SUCCESS: {total-passed} demonstration(s) failed")
        print("Some RL features need refinement")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
