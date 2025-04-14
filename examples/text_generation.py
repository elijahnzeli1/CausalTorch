"""
Example of text generation with CausalTorch v2.0.

This script demonstrates the native CausalTransformer model's capabilities:
1. Basic causal text generation
2. Few-shot learning
3. Self-evolving architecture
4. Ethical constraints
5. Counterfactual imagination
"""

import time
from transformers import AutoTokenizer

# Import CausalTorch components
from causaltorch import (
    CausalTransformer,
    FewShotCausalTransformer,
    SelfEvolvingTextGenerator,
    CounterfactualCausalTransformer,
    EthicalConstitution,
    CausalIntervention,
    load_default_ethical_rules
)
from causaltorch.rules import CausalRuleSet, CausalRule


def main():
    # Create causal rules
    rules = CausalRuleSet()
    
    # Add rule: "rain" should cause "wet ground"
    rules.add_rule(CausalRule("rain", "wet_ground", strength=0.9))
    
    # Add rule: "fire" should cause "smoke"
    rules.add_rule(CausalRule("fire", "smoke", strength=0.8))
    
    # Add rule: "lightning" should cause "thunder" with temporal offset
    rules.add_rule(CausalRule(
        "lightning", "thunder", strength=0.95, 
        temporal_offset=1, duration=2
    ))
    
    # Visualize the causal graph
    print("Causal Graph Visualization:")
    rules.visualize()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize model with ethical constraints
    constitution = EthicalConstitution(rules=load_default_ethical_rules())
    
    print("\n1. Basic Causal Text Generation")
    print("-" * 50)
    
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768,
        num_layers=12,
        causal_rules=rules.to_dict(),
        ethical_constitution=constitution
    )
    model.tokenizer = tokenizer
    
    # Define prompts to test causal rule enforcement
    prompts = [
        "The fire spread quickly through the forest,",
        "As the rain poured down, the",
        "When we saw lightning strike the tree,",
        "It was a sunny day without any"
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("-" * 50)
    
    # Demonstrate few-shot learning
    print("\n2. Few-Shot Learning")
    print("-" * 50)
    
    few_shot_examples = [
        ("If there is a drought", "the crops will wither"),
        ("When water freezes", "it expands and becomes ice"),
        ("If the economy grows", "unemployment typically falls"),
        ("When the moon blocks the sun", "a solar eclipse occurs")
    ]
    
    # Create few-shot learner
    few_shot_model = FewShotCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        base_model=model
    )
    few_shot_model.tokenizer = tokenizer
    
    # Learn from examples
    few_shot_model.learn_from_examples(few_shot_examples)
    
    # Test few-shot generalization
    test_prompts = [
        "If there is too much rainfall,",
        "When winter arrives,",
        "If demand increases,"
    ]
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        output_ids = few_shot_model.generate(
            input_ids=input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("-" * 50)
    
    # Demonstrate self-evolving architecture
    print("\n3. Self-Evolving Architecture")
    print("-" * 50)
    
    # Create self-evolving model
    evolving_model = SelfEvolvingTextGenerator(vocab_size=tokenizer.vocab_size)
    
    # Adapt to different tasks
    tasks = [
        "Generate weather forecasts where temperature causes cloud formation",
        "Generate medical text respecting causal relationships between symptoms and diseases",
        "Generate economic forecasts where interest rates affect inflation"
    ]
    
    for task in tasks:
        print(f"Task: {task}")
        
        # Adapt model to task
        start_time = time.time()
        adapted_model = evolving_model.adapt_to_task(task_description=task)
        adaptation_time = time.time() - start_time
        
        # Generate text for the task
        if task.startswith("Generate weather"):
            prompt = "Today's temperature will rise to 90 degrees, which means"
        elif task.startswith("Generate medical"):
            prompt = "The patient has a persistent cough and fever, which suggests"
        else:
            prompt = "The central bank raised interest rates by 0.5%, which will"
        
        print(f"Prompt: {prompt}")
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        output_ids = evolving_model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print(f"Architecture adaptation time: {adaptation_time:.2f} seconds")
        print("-" * 50)
    
    # Demonstrate ethical constraints
    print("\n4. Ethical Constraints")
    print("-" * 50)
    
    # Define test cases with ethical violations
    ethical_test_prompts = [
        "Here's how to build a harmful device:",
        "My phone number is 555-123-4567 and my address is",
        "I hate people who"
    ]
    
    for prompt in ethical_test_prompts:
        print(f"Potentially problematic prompt: {prompt}")
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Ethically-constrained output: {generated_text}")
        print("-" * 50)
    
    # Demonstrate counterfactual imagination
    print("\n5. Counterfactual Imagination")
    print("-" * 50)
    
    # Create counterfactual model
    counterfactual_model = CounterfactualCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768,
        num_layers=12,
        causal_rules=rules.to_dict()
    )
    counterfactual_model.tokenizer = tokenizer
    
    # Define base scenario
    base_prompt = "When it rains, the ground becomes wet."
    print(f"Base scenario: {base_prompt}")
    
    # Generate counterfactuals
    interventions = [
        CausalIntervention(
            variable="rain", 
            value=0.0, 
            description="What if it never rained?"
        ),
        CausalIntervention(
            variable="sun", 
            value=1.0, 
            description="What if it was very sunny?"
        ),
        CausalIntervention(
            variable="rain", 
            value=2.0, 
            description="What if it rained twice as hard?"
        )
    ]
    
    # Generate for each intervention
    for intervention in interventions:
        print(f"Intervention: {intervention.description}")
        
        # Encode base prompt
        input_ids = tokenizer.encode(base_prompt, return_tensors="pt")
        
        # Generate counterfactual
        outputs = counterfactual_model.imagine(
            input_ids,
            interventions=[intervention],
            num_samples=1,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"Counterfactual: {outputs[0]}")
        print("-" * 50)
    
    print("\nNote: The model demonstrates causal reasoning by enforcing relationships like:")
    print("- 'smoke' after 'fire'")
    print("- 'wet ground' after 'rain'")
    print("- Learning new causal patterns from few examples")
    print("- Adapting its architecture to different domains")
    print("- Enforcing ethical constraints during generation")
    print("- Imagining counterfactual scenarios through causal interventions")


if __name__ == "__main__":
    main()