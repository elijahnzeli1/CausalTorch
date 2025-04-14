# examples/federated_learning.py
"""
Example of federated learning with CausalTorch.

This script demonstrates how multiple clients can collaboratively learn
and share causal knowledge without exchanging raw data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from causaltorch import CausalTransformer, CausalDAO, FederatedClient
from causaltorch.rules import CausalRuleSet, CausalRule


def create_synthetic_data(client_id, num_samples=1000):
    """Create synthetic data with client-specific causal patterns."""
    np.random.seed(int(client_id.split('_')[1]))
    
    # Variables
    X = np.random.normal(0, 1, (num_samples, 10))
    
    # Different causal patterns for different clients
    if client_id == "client_1":
        # X[0] causes X[1]
        X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.normal(0, 1, num_samples)
        Y = X[:, 0] + X[:, 1] + 0.1 * np.random.normal(0, 1, num_samples)
        local_rules = CausalRuleSet()
        local_rules.add_rule("X0", "X1", strength=0.8)
    elif client_id == "client_2":
        # X[2] causes X[3]
        X[:, 3] = 0.7 * X[:, 2] + 0.3 * np.random.normal(0, 1, num_samples)
        Y = X[:, 2] + X[:, 3] + 0.1 * np.random.normal(0, 1, num_samples)
        local_rules = CausalRuleSet()
        local_rules.add_rule("X2", "X3", strength=0.7)
    elif client_id == "client_3":
        # X[4] causes X[5] and X[5] causes X[6]
        X[:, 5] = 0.9 * X[:, 4] + 0.1 * np.random.normal(0, 1, num_samples)
        X[:, 6] = 0.6 * X[:, 5] + 0.4 * np.random.normal(0, 1, num_samples)
        Y = X[:, 4] + X[:, 5] + X[:, 6] + 0.1 * np.random.normal(0, 1, num_samples)
        local_rules = CausalRuleSet()
        local_rules.add_rule("X4", "X5", strength=0.9)
        local_rules.add_rule("X5", "X6", strength=0.6)
    else:
        # No specific causal patterns
        Y = np.sum(X, axis=1) + 0.1 * np.random.normal(0, 1, num_samples)
        local_rules = CausalRuleSet()
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader, local_rules, X.shape[1]


def main():
    print("Federated Learning with CausalTorch")
    print("-" * 50)
    
    # Initialize CausalDAO on server
    dao = CausalDAO(
        initial_graph=None,
        consensus_threshold=0.6,
        model_aggregation='fedavg'
    )
    
    # Create clients
    client_ids = ["client_1", "client_2", "client_3"]
    clients = []
    
    for client_id in client_ids:
        # Create synthetic data for this client
        dataloader, local_rules, input_dim = create_synthetic_data(client_id)
        
        # Create simple model for regression
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Create client
        client = FederatedClient(
            client_id=client_id,
            model=model,
            data_size=len(dataloader.dataset),
            local_causal_graph=local_rules
        )
        
        # Register with DAO
        dao.register_client(client_id, data_size=len(dataloader.dataset))
        
        clients.append((client, dataloader))
        
        print(f"Registered {client_id} with {len(dataloader.dataset)} samples")
        print(f"Local causal rules: {local_rules}")
    
    print("\nStarting federated training")
    print("-" * 50)
    
    # Train for multiple rounds
    rounds = 5
    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}/{rounds}")
        
        # Each client trains locally
        for client, dataloader in clients:
            print(f"Training {client.client_id}...")
            
            # Create optimizer
            optimizer = torch.optim.Adam(client.model.parameters(), lr=0.01)
            
            # Train
            metrics = client.train(
                dataloader=dataloader,
                optimizer=optimizer,
                loss_fn=F.mse_loss,
                num_epochs=3
            )
            
            print(f"  Loss: {metrics['loss'][-1]:.4f}")
            
            # Get model update
            model_update = client.get_model_update()
            
            # Discover local causal graph (in this case, we already have it)
            local_graph = client.discover_causal_graph(dataloader)
            
            # Submit updates to DAO
            dao.update_local_model(client.client_id, model_update, client.data_size)
            dao.update_local_graph(client.client_id, local_graph)
        
        # Get global model and graph
        global_model = dao.get_global_model()
        global_graph = dao.get_global_graph()
        
        print("\nGlobal causal graph after round:")
        print(global_graph)
        
        # Update clients with global knowledge
        for client, _ in clients:
            client.update_model(global_model)
            client.update_local_graph(global_graph)
    
    print("\nFinal global causal graph:")
    print("-" * 50)
    print(global_graph)
    print("\nNote how clients have collaboratively discovered the full causal structure")
    print("without sharing their raw data.")


if __name__ == "__main__":
    main()