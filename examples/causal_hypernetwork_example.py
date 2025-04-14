"""
CausalHyperNetwork Example
=========================

This example demonstrates the use of CausalHyperNetwork for generating 
task-specific neural architectures based on causal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import causaltorch
from causaltorch import CausalRuleSet, CausalRule
from causaltorch import CausalHyperNetwork

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a simple causal graph
def create_causal_graph(num_nodes=5):
    """Create a simple causal graph with random connections."""
    rules = CausalRuleSet()
    
    # Create nodes
    nodes = [f"X{i}" for i in range(num_nodes)]
    
    # Create edges (ensuring DAG structure)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Create edge with 50% probability
            if np.random.random() < 0.5:
                strength = np.random.uniform(0.5, 1.0)
                rule = CausalRule(
                    cause=nodes[i],
                    effect=nodes[j],
                    strength=strength
                )
                rules.add_rule(rule)
    
    return rules

# Convert causal graph to adjacency matrix
def graph_to_adjacency(graph, num_nodes=5):
    """Convert causal graph to adjacency matrix."""
    nodes = [f"X{i}" for i in range(num_nodes)]
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    for i, cause in enumerate(nodes):
        for rule in graph.get_rules_for_cause(cause):
            # Find index of effect
            effect = rule.effect
            j = nodes.index(effect)
            adj_matrix[i, j] = rule.strength
    
    return adj_matrix

# Generate synthetic data based on causal graph
def generate_data(graph, num_samples=1000, num_nodes=5):
    """Generate synthetic data based on causal graph."""
    nodes = [f"X{i}" for i in range(num_nodes)]
    adj_matrix = graph_to_adjacency(graph, num_nodes)
    
    # Initialize data
    data = torch.zeros(num_samples, num_nodes)
    
    # Generate data following causal structure
    for i in range(num_samples):
        # For each node, compute value based on parents
        for j in range(num_nodes):
            # Exogenous noise
            noise = torch.normal(0, 0.1, size=(1,))
            
            # Sum of parent influences
            parent_sum = 0
            for k in range(num_nodes):
                if adj_matrix[k, j] > 0:
                    parent_sum += adj_matrix[k, j] * data[i, k]
            
            # Node value = parent influence + noise
            data[i, j] = parent_sum + noise
    
    # Split into features and target
    X = data[:, :-1]
    y = data[:, -1].unsqueeze(1)
    
    return X, y

# Function to train model for specific graph
def train_model(model, X, y, epochs=100, lr=0.01):
    """Train model for a specific causal graph."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

# Main script
def main():
    print("CausalHyperNetwork Example")
    print("==========================")
    
    # Parameters
    num_nodes = 5
    num_samples = 1000
    input_dim = 16  # Power of 2 near num_nodes^2
    
    # Create multiple causal graphs
    print("Creating causal graphs...")
    graphs = [create_causal_graph(num_nodes) for _ in range(3)]
    
    # Initialize CausalHyperNetwork
    print("Initializing CausalHyperNetwork...")
    hyper_net = CausalHyperNetwork(
        input_dim=input_dim,
        output_dim=1,  # Predicting one target variable
        hidden_dim=64,
        meta_hidden_dim=128,
        num_layers=3,
        activation="relu"
    )
    
    # Training loop for multiple graphs
    all_losses = []
    print("Training on multiple causal graphs...")
    
    for i, graph in enumerate(graphs):
        print(f"\nGraph {i+1}:")
        
        # Convert graph to adjacency matrix
        adj_matrix = graph_to_adjacency(graph, num_nodes)
        print("Adjacency Matrix:")
        print(adj_matrix)
        
        # Generate data for this graph
        X, y = generate_data(graph, num_samples, num_nodes)
        
        # Pad adjacency matrix to input_dim
        padded_adj = torch.zeros(input_dim, input_dim)
        padded_adj[:num_nodes, :num_nodes] = adj_matrix
        
        # Generate model for this graph
        batch_adj = padded_adj.unsqueeze(0)  # Add batch dimension
        task_model = hyper_net.generate_architecture(batch_adj)
        
        # Train model
        losses = train_model(task_model, X, y, epochs=100)
        all_losses.append(losses)
        
        # Test model
        test_X = X[:10]  # Use first 10 samples for testing
        test_y = y[:10]
        
        with torch.no_grad():
            predictions = task_model(test_X)
        
        print("Test Results:")
        for j in range(5):  # Show first 5 predictions
            print(f"True: {test_y[j].item():.4f}, Predicted: {predictions[j].item():.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label=f"Graph {i+1}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss for Different Causal Graphs")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("causal_hypernetwork_training.png")
    plt.show()

if __name__ == "__main__":
    main() 