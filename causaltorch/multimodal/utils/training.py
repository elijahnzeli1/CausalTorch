"""
Training Utilities for Multimodal Causal Models
==============================================

This module provides training utilities for multimodal causal models,
including training loop helpers, evaluation metrics tracking, and
optimization utilities.
"""

import os
import time
import json
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Union, Optional, Any

from causaltorch.multimodal.utils.metrics import (
    calculate_multimodal_causal_consistency,
    evaluate_model_causal_fidelity,
    evaluate_counterfactual_quality
)


class MultimodalTrainer:
    """Trainer class for multimodal causal models.
    
    This class handles training, evaluation, and saving of multimodal causal models.
    It tracks metrics, implements early stopping, and provides visualization of training progress.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        device=None,
        output_dir="./outputs",
        patience=5,
        grad_clip=1.0,
        log_interval=10,
        wandb_logging=False,
        causal_consistency_weight=0.5,
        counterfactual_supervision=False,
        modalities=["text", "image"]
    ):
        """Initialize the trainer.
        
        Args:
            model: The multimodal causal model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            optimizer: Optimizer for training (if None, will create AdamW)
            lr_scheduler: Learning rate scheduler (optional)
            device: Device to train on (if None, will use CUDA if available)
            output_dir: Directory to save outputs
            patience: Patience for early stopping
            grad_clip: Gradient clipping value
            log_interval: Interval for logging training progress
            wandb_logging: Whether to use Weights & Biases for logging
            causal_consistency_weight: Weight for causal consistency loss
            counterfactual_supervision: Whether to use counterfactual supervision
            modalities: List of modalities to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=2e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.output_dir = output_dir
        self.patience = patience
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.wandb_logging = wandb_logging
        self.causal_consistency_weight = causal_consistency_weight
        self.counterfactual_supervision = counterfactual_supervision
        self.modalities = modalities
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize training metrics
        self.train_metrics = {
            'loss': [],
            'text_loss': [],
            'image_loss': [],
            'causal_consistency': []
        }
        
        self.val_metrics = {
            'loss': [],
            'text_loss': [],
            'image_loss': [],
            'causal_consistency': []
        }
        
        # Initialize W&B if enabled
        if self.wandb_logging:
            try:
                import wandb # type: ignore
                self.wandb = wandb
            except ImportError:
                print("wandb not installed. Running without W&B logging.")
                self.wandb_logging = False
    
    def train(self, epochs=10, eval_every=1, save_every=1, eval_causal_fidelity=False):
        """Train the model for a specified number of epochs.
        
        Args:
            epochs: Number of epochs to train for
            eval_every: Evaluate on validation set every N epochs
            save_every: Save model checkpoint every N epochs
            eval_causal_fidelity: Whether to evaluate causal fidelity (computationally expensive)
            
        Returns:
            dict: Training metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Training phase
            train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            if self.val_dataloader is not None and epoch % eval_every == 0:
                val_metrics = self._validate_epoch(epoch, eval_causal_fidelity)
                
                # Update metrics history
                for key, value in val_metrics.items():
                    if key in self.val_metrics:
                        self.val_metrics[key].append(value)
                
                # Early stopping logic
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save best model
                    self._save_checkpoint(
                        os.path.join(self.output_dir, "best_model.pt"),
                        epoch, val_metrics
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        break
            
            # Save checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(
                    os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt"),
                    epoch, train_metrics
                )
            
            # Step learning rate scheduler if provided
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # This scheduler requires validation loss
                    if self.val_dataloader is not None:
                        self.lr_scheduler.step(val_metrics['loss'])
                else:
                    self.lr_scheduler.step()
        
        # Save final model
        self._save_checkpoint(
            os.path.join(self.output_dir, "final_model.pt"),
            epochs, train_metrics
        )
        
        # Plot training curves
        self._plot_training_curves()
        
        # Calculate total training time
        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds")
        
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }
    
    def _train_epoch(self, epoch):
        """Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Training metrics for this epoch
        """
        self.model.train()
        
        epoch_loss = 0
        epoch_text_loss = 0
        epoch_image_loss = 0
        epoch_causal_consistency = 0
        
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}/{self.patience}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Extract losses
            loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
            text_loss = outputs.get('text_loss', torch.tensor(0.0, device=self.device))
            image_loss = outputs.get('image_loss', torch.tensor(0.0, device=self.device))
            causal_consistency = outputs.get('causal_consistency', torch.tensor(0.0, device=self.device))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update epoch metrics
            epoch_loss += loss.item()
            epoch_text_loss += text_loss.item()
            epoch_image_loss += image_loss.item()
            epoch_causal_consistency += causal_consistency.item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    text_loss=f"{text_loss.item():.4f}",
                    image_loss=f"{image_loss.item():.4f}",
                    causal_consistency=f"{causal_consistency.item():.4f}"
                )
                
                # Log to W&B if enabled
                if self.wandb_logging:
                    self.wandb.log({
                        'batch_loss': loss.item(),
                        'batch_text_loss': text_loss.item(),
                        'batch_image_loss': image_loss.item(),
                        'batch_causal_consistency': causal_consistency.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'batch': batch_idx
                    })
        
        # Calculate epoch averages
        epoch_metrics = {
            'loss': epoch_loss / num_batches,
            'text_loss': epoch_text_loss / num_batches,
            'image_loss': epoch_image_loss / num_batches,
            'causal_consistency': epoch_causal_consistency / num_batches
        }
        
        # Update metrics history
        for key, value in epoch_metrics.items():
            self.train_metrics[key].append(value)
        
        # Print epoch summary
        print(f"Epoch {epoch} - "
              f"Loss: {epoch_metrics['loss']:.4f}, "
              f"Text Loss: {epoch_metrics['text_loss']:.4f}, "
              f"Image Loss: {epoch_metrics['image_loss']:.4f}, "
              f"Causal Consistency: {epoch_metrics['causal_consistency']:.4f}")
        
        # Log to W&B if enabled
        if self.wandb_logging:
            self.wandb.log({
                'train_loss': epoch_metrics['loss'],
                'train_text_loss': epoch_metrics['text_loss'],
                'train_image_loss': epoch_metrics['image_loss'],
                'train_causal_consistency': epoch_metrics['causal_consistency'],
                'epoch': epoch
            })
        
        return epoch_metrics
    
    def _validate_epoch(self, epoch, eval_causal_fidelity=False):
        """Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            eval_causal_fidelity: Whether to evaluate causal fidelity
            
        Returns:
            dict: Validation metrics for this epoch
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        val_loss = 0
        val_text_loss = 0
        val_image_loss = 0
        val_causal_consistency = 0
        
        num_batches = len(self.val_dataloader)
        causal_fidelity_scores = []
        counterfactual_quality_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc=f"Validating Epoch {epoch}", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Extract losses
                loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
                text_loss = outputs.get('text_loss', torch.tensor(0.0, device=self.device))
                image_loss = outputs.get('image_loss', torch.tensor(0.0, device=self.device))
                causal_consistency = outputs.get('causal_consistency', torch.tensor(0.0, device=self.device))
                
                # Update epoch metrics
                val_loss += loss.item()
                val_text_loss += text_loss.item()
                val_image_loss += image_loss.item()
                val_causal_consistency += causal_consistency.item()
                
                # Evaluate causal fidelity if requested (expensive)
                if eval_causal_fidelity and 'causal_graph' in batch:
                    # Get model outputs
                    model_outputs = {
                        'text': outputs.get('text_output', None),
                        'image': outputs.get('image_output', None)
                    }
                    
                    # Get ground truth
                    ground_truth = {
                        'text': batch.get('text_target', batch.get('input_ids', None)),
                        'image': batch.get('image_target', batch.get('image', None))
                    }
                    
                    # Calculate causal fidelity
                    fidelity_scores = evaluate_model_causal_fidelity(
                        model_outputs,
                        ground_truth,
                        batch['causal_graph'],
                        self.modalities
                    )
                    
                    causal_fidelity_scores.append(fidelity_scores)
                    
                    # Generate counterfactual outputs if model supports it
                    if hasattr(self.model, 'imagine_counterfactual') and self.counterfactual_supervision:
                        # Get first causal variable for intervention
                        causal_graph = batch['causal_graph'][0] if isinstance(batch['causal_graph'], list) else batch['causal_graph']
                        
                        # Extract a variable for intervention (simplified)
                        intervention_var = None
                        intervention_val = 1.0
                        
                        if isinstance(causal_graph, dict):
                            if causal_graph and 'causal_relations' in causal_graph:
                                relations = causal_graph['causal_relations']
                                if relations and isinstance(relations, list) and relations[0]:
                                    intervention_var = relations[0].get('cause', None)
                            elif causal_graph:
                                # Just get the first key
                                intervention_var = next(iter(causal_graph.keys()), None)
                        
                        if intervention_var:
                            # Create intervention
                            intervention = {'variable': intervention_var, 'value': intervention_val}
                            
                            # Generate counterfactual
                            counterfactual_outputs = self.model.imagine_counterfactual(
                                batch, intervention
                            )
                            
                            # Evaluate counterfactual quality
                            quality = evaluate_counterfactual_quality(
                                model_outputs,
                                counterfactual_outputs,
                                intervention,
                                batch['causal_graph']
                            )
                            
                            counterfactual_quality_scores.append(quality)
        
        # Calculate epoch averages
        val_metrics = {
            'loss': val_loss / num_batches,
            'text_loss': val_text_loss / num_batches,
            'image_loss': val_image_loss / num_batches,
            'causal_consistency': val_causal_consistency / num_batches
        }
        
        # Add causal fidelity metrics if evaluated
        if causal_fidelity_scores:
            # Average fidelity scores across batches
            avg_fidelity = {}
            for score_dict in causal_fidelity_scores:
                for key, value in score_dict.items():
                    if key not in avg_fidelity:
                        avg_fidelity[key] = []
                    avg_fidelity[key].append(value)
            
            for key, values in avg_fidelity.items():
                val_metrics[f'fidelity_{key}'] = sum(values) / len(values)
        
        # Add counterfactual quality metrics if evaluated
        if counterfactual_quality_scores:
            # Average quality scores across batches
            avg_quality = {}
            for score_dict in counterfactual_quality_scores:
                for key, value in score_dict.items():
                    if key not in avg_quality:
                        avg_quality[key] = []
                    avg_quality[key].append(value)
            
            for key, values in avg_quality.items():
                val_metrics[f'counterfactual_{key}'] = sum(values) / len(values)
        
        # Print validation summary
        print(f"Validation Epoch {epoch} - "
              f"Loss: {val_metrics['loss']:.4f}, "
              f"Text Loss: {val_metrics['text_loss']:.4f}, "
              f"Image Loss: {val_metrics['image_loss']:.4f}, "
              f"Causal Consistency: {val_metrics['causal_consistency']:.4f}")
        
        # Log to W&B if enabled
        if self.wandb_logging:
            log_dict = {
                'val_loss': val_metrics['loss'],
                'val_text_loss': val_metrics['text_loss'],
                'val_image_loss': val_metrics['image_loss'],
                'val_causal_consistency': val_metrics['causal_consistency'],
                'epoch': epoch
            }
            
            # Add causal fidelity metrics if present
            for key, value in val_metrics.items():
                if key.startswith('fidelity_') or key.startswith('counterfactual_'):
                    log_dict[key] = value
            
            self.wandb.log(log_dict)
        
        return val_metrics
    
    def _save_checkpoint(self, path, epoch, metrics):
        """Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Add lr_scheduler state if available
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        metrics_to_plot = [
            ('loss', 'Loss'),
            ('text_loss', 'Text Loss'),
            ('image_loss', 'Image Loss'),
            ('causal_consistency', 'Causal Consistency')
        ]
        
        num_plots = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        for i, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[i] if num_plots > 1 else axes
            
            # Plot training curve
            if metric_key in self.train_metrics and self.train_metrics[metric_key]:
                ax.plot(
                    range(1, len(self.train_metrics[metric_key]) + 1),
                    self.train_metrics[metric_key],
                    label='Train'
                )
            
            # Plot validation curve
            if self.val_dataloader is not None and metric_key in self.val_metrics and self.val_metrics[metric_key]:
                ax.plot(
                    range(1, len(self.val_metrics[metric_key]) + 1),
                    self.val_metrics[metric_key],
                    label='Validation'
                )
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs. Epoch')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    def get_best_model_path(self):
        """Get the path to the best model checkpoint.
        
        Returns:
            str: Path to the best model checkpoint
        """
        return os.path.join(self.output_dir, "best_model.pt")


def prepare_dataloaders(
    dataset,
    batch_size=16,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    num_workers=4,
    collate_fn=None
):
    """Split dataset and create DataLoaders.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for DataLoaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoaders
        collate_fn: Custom collate function
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if val_size > 0 else None
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if test_size > 0 else None
    
    return train_dataloader, val_dataloader, test_dataloader


def load_checkpoint(path, model, optimizer=None, lr_scheduler=None, device=None):
    """Load a model checkpoint.
    
    Args:
        path: Path to the checkpoint
        model: Model to load the state dict into
        optimizer: Optimizer to load the state dict into (optional)
        lr_scheduler: Learning rate scheduler to load the state dict into (optional)
        device: Device to load the model onto (optional)
        
    Returns:
        tuple: (model, optimizer, lr_scheduler, epoch, metrics)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load lr_scheduler state if provided
    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, lr_scheduler, epoch, metrics


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    
    Adapted from: https://github.com/ildoonet/pytorch-gradual-warmup-lr
    """
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            multiplier: Target learning rate = base lr * multiplier if multiplier > 1.0
            total_epoch: Target learning rate is reached at total_epoch
            after_scheduler: After target_epoch, use this scheduler(eg. ReduceLROnPlateau)
        """
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def log_hyperparameters(output_dir, hyperparams):
    """Log hyperparameters to a JSON file.
    
    Args:
        output_dir: Directory to save the hyperparameters file
        hyperparams: Dictionary of hyperparameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert non-serializable values to strings
    serializable_params = {}
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
            serializable_params[key] = value
        else:
            serializable_params[key] = str(value)
    
    # Save to JSON file
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(serializable_params, f, indent=4) 