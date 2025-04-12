"""
Example of text generation with causal constraints using CausalTorch.

This script demonstrates how to use the CNSG_GPT2 model to generate text
that follows specified causal rules.
"""

import torch
from transformers import GPT2Tokenizer

# Import CausalTorch components
from causaltorch import CNSG_GPT2
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
    
    # Initialize model and tokenizer
    model = CNSG_GPT2(pretrained_model_name="gpt2", causal_rules=rules.to_dict())
    model.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Define prompts to test causal rule enforcement
    prompts = [
        "The fire spread quickly through the forest,",
        "As the rain poured down, the",
        "When we saw lightning strike the tree,",
        "It was a sunny day without any"
    ]
    
    # Generate text for each prompt
    print("\nGenerating text with causal constraints:")
    print("-" * 50)
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        
        # Encode the prompt
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        # Decode the generated text
        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("-" * 50)
    
    print("\nNote: The model should bias generation to include effects like 'smoke' after 'fire', "
          "or 'wet ground' after 'rain' due to the causal rules we defined.")


if __name__ == "__main__":
    main()