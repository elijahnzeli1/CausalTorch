"""
Demo of CausalTorch rules engine usage.

This example demonstrates how to define causal rules and apply them to tensors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from causaltorch import CausalRule, CausalRuleSet

# Create a simple rule set
ruleset = CausalRuleSet()

# Add some causal rules with different strengths
ruleset.add_rule("temperature", "ice_melting", 0.8)
ruleset.add_rule("sunlight", "temperature", 0.6)
ruleset.add_rule("cloud_cover", "sunlight", -0.7)  # Negative effect
ruleset.add_rule("humidity", "cloud_cover", 0.5)

# Print the rule set
print(f"Created rule set: {ruleset}")
for rule in ruleset.rules:
    print(f"  {rule}")

# Create a sample tensor (representing temperature values)
temperature = torch.ones(5, 5) * 20  # Start with 20°C everywhere

# Define context for applying rules
context = {"time_of_day": "noon"}

# Apply the causal effects to modify the tensor
modified = ruleset.apply_rules(temperature, context)

print("\nOriginal temperature tensor:")
print(temperature)
print("\nModified temperature tensor after applying rules:")
print(modified)

# Convert to adjacency matrix for visualization
adj_matrix, variables = ruleset.to_adjacency_matrix()
print("\nAdjacency matrix:")
print(adj_matrix)
print("Variables:", variables)

# Visualize the causal graph
plt.figure(figsize=(8, 6))
n = len(variables)
pos = {}
for i, var in enumerate(variables):
    angle = 2 * np.pi * i / n
    pos[i] = [np.cos(angle), np.sin(angle)]

# Draw the nodes
for i, var in enumerate(variables):
    plt.plot(pos[i][0], pos[i][1], 'o', markersize=20, 
             color='skyblue', alpha=0.8)
    plt.text(pos[i][0], pos[i][1], var, ha='center', va='center')

# Draw the edges
for i in range(n):
    for j in range(n):
        if adj_matrix[i, j] != 0:
            strength = adj_matrix[i, j]
            color = 'green' if strength > 0 else 'red'
            alpha = abs(strength)
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            plt.arrow(pos[i][0], pos[i][1], dx*0.8, dy*0.8, 
                     head_width=0.1, color=color, alpha=alpha)
            # Add strength label
            plt.text(pos[i][0] + dx*0.4, pos[i][1] + dy*0.4, 
                     f"{strength:.1f}", color=color)

plt.title("Causal Graph Visualization")
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('causal_graph.png')
plt.show()

print("\nCausal graph visualization saved as 'causal_graph.png'")

# Demonstrate finding causes and effects
print("\nCauses of temperature:", ruleset.get_causes("temperature"))
print("Effects of sunlight:", ruleset.get_effects("sunlight"))