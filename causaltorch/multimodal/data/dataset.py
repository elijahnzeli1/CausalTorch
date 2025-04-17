"""
Multimodal Causal Dataset
========================

This module provides a dataset class for multimodal data with causal annotations.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import warnings

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers package not found. Using dummy tokenizer.")
    TRANSFORMERS_AVAILABLE = False
    # Create a simple dummy tokenizer for testing
    class DummyTokenizer:
        def __init__(self, vocab_size=30000):
            self.vocab_size = vocab_size
            print("Using dummy tokenizer with vocab size:", vocab_size)
            
        def __call__(self, text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"):
            if not isinstance(text, list):
                text = [text]
            
            # Create random input IDs and attention masks
            batch_size = len(text)
            input_ids = torch.randint(1, self.vocab_size, (batch_size, max_length))
            attention_mask = torch.ones_like(input_ids)
            
            # Convert to specified tensor type
            if return_tensors == "pt":
                pass  # Already PyTorch tensors
            elif return_tensors == "np":
                input_ids = input_ids.numpy()
                attention_mask = attention_mask.numpy()
            
            return {"input_ids": input_ids, "attention_mask": attention_mask}


class MultimodalCausalDataset(Dataset):
    """Dataset for multimodal data with causal annotations.
    
    This dataset handles text-image pairs with causal relationships
    between elements in the data.
    
    Args:
        text_data (list): List of text inputs or path to text file
        image_data (list): List of image paths or path to image directory
        causal_annotations (dict): Dictionary mapping sample IDs to causal annotations
        text_tokenizer (str): Name of pre-trained tokenizer to use
        image_size (int): Size to resize images to
        max_text_length (int): Maximum text length to use
        transform (callable, optional): Transform to apply to images
    """
    def __init__(
        self,
        text_data,
        image_data,
        causal_annotations=None,
        text_tokenizer="bert-base-uncased",
        image_size=224,
        max_text_length=128,
        transform=None
    ):
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Load text data
        if isinstance(text_data, str) and os.path.exists(text_data):
            # Load from file
            try:
                with open(text_data, 'r', encoding='utf-8') as f:
                    if text_data.endswith('.json'):
                        self.text_data = json.load(f)
                    else:
                        self.text_data = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.text_data)} text samples from {text_data}")
            except Exception as e:
                print(f"Error loading text data from {text_data}: {e}")
                self.text_data = []
        else:
            # Use provided list
            if isinstance(text_data, list):
                self.text_data = text_data
                print(f"Using provided list of {len(self.text_data)} text samples")
            else:
                print(f"Warning: text_data is not a list or a valid file path: {text_data}")
                self.text_data = []
        
        # Load image data
        if isinstance(image_data, str) and os.path.isdir(image_data):
            # Load images from directory
            try:
                image_files = [f for f in os.listdir(image_data) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_files.sort()  # Ensure consistent ordering
                self.image_data = [os.path.join(image_data, f) for f in image_files]
                print(f"Loaded {len(self.image_data)} image paths from {image_data}")
            except Exception as e:
                print(f"Error loading images from directory {image_data}: {e}")
                self.image_data = []
        else:
            # Use provided list
            if isinstance(image_data, list):
                self.image_data = image_data
                print(f"Using provided list of {len(self.image_data)} image samples")
            else:
                print(f"Warning: image_data is not a list or a valid directory path: {image_data}")
                self.image_data = []
        
        # Ensure same number of text and image samples
        if len(self.text_data) == 0 or len(self.image_data) == 0:
            print("Warning: Either text_data or image_data is empty. Dataset will be empty.")
            self.text_data = []
            self.image_data = []
        else:
            min_len = min(len(self.text_data), len(self.image_data))
            if min_len < len(self.text_data):
                print(f"Warning: More text samples ({len(self.text_data)}) than image samples ({len(self.image_data)}). Truncating text data.")
            if min_len < len(self.image_data):
                print(f"Warning: More image samples ({len(self.image_data)}) than text samples ({len(self.text_data)}). Truncating image data.")
            self.text_data = self.text_data[:min_len]
            self.image_data = self.image_data[:min_len]
        
        # Load causal annotations
        if causal_annotations is None:
            self.causal_annotations = {}
            print("No causal annotations provided. Using empty dictionary.")
        elif isinstance(causal_annotations, dict):
            self.causal_annotations = causal_annotations
            print(f"Using provided causal annotations with {len(self.causal_annotations)} entries")
        else:
            print(f"Warning: causal_annotations must be a dictionary. Got {type(causal_annotations)}. Using empty dictionary instead.")
            self.causal_annotations = {}
        
        # Initialize tokenizer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
                print(f"Loaded tokenizer {text_tokenizer} with vocab size {self.tokenizer.vocab_size}")
            except Exception as e:
                print(f"Error loading tokenizer {text_tokenizer}: {e}. Using dummy tokenizer.")
                self.tokenizer = DummyTokenizer()
        else:
            self.tokenizer = DummyTokenizer()
        
        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            # Instead of lambda, use a method reference which is picklable
            self.transform = self._basic_transform
    
    def _basic_transform(self, img):
        """Basic image transformation.
        
        Args:
            img (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Resize
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert to float and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Convert to channels-first format (C, H, W)
        if img_array.ndim == 2:  # Grayscale
            img_array = np.expand_dims(img_array, axis=0)
        else:  # RGB
            img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array)
        
        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        
        return img_tensor
    
    def __len__(self):
        """Get dataset length."""
        return len(self.text_data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            dict: Sample with text, image, and causal annotations
        """
        text = self.text_data[idx]
        img_path = self.image_data[idx]
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension (added by tokenizer with return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Load image
        if isinstance(img_path, str):
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    warnings.warn(f"Error loading image {img_path}: {e}. Creating dummy image.")
                    img = Image.new('RGB', (224, 224), color=(200, 200, 200))
            else:
                warnings.warn(f"Image file not found at {img_path}. Creating dummy image.")
                img = Image.new('RGB', (224, 224), color=(200, 200, 200))
        else:
            # Handle case where img_path is already an image or tensor
            img = img_path
            
        if self.transform:
            img = self.transform(img)
        
        # Get causal graph - handle different key types and missing keys
        causal_graph = {"nodes": [], "edges": []}
        
        # Check for integer key
        if idx in self.causal_annotations:
            causal_graph = self.causal_annotations[idx]
        # Check for string key
        elif str(idx) in self.causal_annotations:
            causal_graph = self.causal_annotations[str(idx)]
        # Use a modulo operation to get a valid key if we have any annotations
        elif len(self.causal_annotations) > 0:
            # Get a list of keys and use modulo to get a valid index
            keys = list(self.causal_annotations.keys())
            fallback_key = keys[idx % len(keys)]
            causal_graph = self.causal_annotations[fallback_key]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img,
            "causal_graph": causal_graph,
            "text": text  # Include original text for reference
        }
    
    def get_causal_graph(self):
        """Get the combined causal graph from all annotations.
        
        Returns:
            dict: Combined causal graph
        """
        # Initialize combined graph
        combined_graph = {"nodes": set(), "edges": []}
        
        # Gather all nodes and edges
        for graph in self.causal_annotations.values():
            if "nodes" in graph:
                for node in graph["nodes"]:
                    combined_graph["nodes"].add(node)
            
            if "edges" in graph:
                for edge in graph["edges"]:
                    combined_graph["edges"].append(edge)
            
            # Also check for causal_relations format
            if "causal_relations" in graph:
                for relation in graph["causal_relations"]:
                    if isinstance(relation, dict) and "cause" in relation and "effect" in relation:
                        combined_graph["nodes"].add(relation["cause"])
                        combined_graph["nodes"].add(relation["effect"])
                        combined_graph["edges"].append({
                            "source": relation["cause"],
                            "target": relation["effect"],
                            "strength": relation.get("strength", 0.8)
                        })
        
        # Convert nodes set to list
        combined_graph["nodes"] = list(combined_graph["nodes"])
        
        return combined_graph
    
    def to_causal_rule_set(self):
        """Convert causal annotations to a CausalRuleSet.
        
        Returns:
            CausalRuleSet: Set of causal rules
        """
        from causaltorch.rules import CausalRuleSet, CausalRule
        
        # Create ruleset
        ruleset = CausalRuleSet()
        
        # Get combined graph
        graph = self.get_causal_graph()
        
        # Add edges as rules
        for edge in graph["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge:
                ruleset.add_rule(CausalRule(
                    edge["source"],
                    edge["target"],
                    strength=edge.get("strength", 0.8)
                ))
        
        # Also check for causal_relations directly
        for annotation in self.causal_annotations.values():
            if "causal_relations" in annotation:
                for relation in annotation["causal_relations"]:
                    if isinstance(relation, dict) and "cause" in relation and "effect" in relation:
                        ruleset.add_rule(CausalRule(
                            relation["cause"],
                            relation["effect"],
                            strength=relation.get("strength", 0.8)
                        ))
        
        return ruleset 