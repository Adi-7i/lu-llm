"""
Neural Network Layers for Mini Language Model
Implements individual layers: Embedding, Hidden, and Output
"""

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that converts token IDs to dense vector representations
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: Number of unique tokens in vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        super(EmbeddingLayer, self).__init__()
        
        # Create embedding matrix: vocab_size x embedding_dim
        # Each row represents the embedding for one token
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        print(f"Embedding Layer initialized: {vocab_size} tokens -> {embedding_dim} dimensions")
    
    def forward(self, token_ids):
        """
        Convert token IDs to embeddings
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_length)
            
        Returns:
            Embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(token_ids)


class HiddenLayer(nn.Module):
    """
    Hidden layer with linear transformation and activation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Input dimension (from previous layer)
            hidden_dim: Hidden layer dimension
        """
        super(HiddenLayer, self).__init__()
        
        # Linear transformation: input_dim -> hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        
        # ReLU activation for non-linearity
        self.activation = nn.ReLU()
        
        print(f"Hidden Layer initialized: {input_dim} -> {hidden_dim}")
    
    def forward(self, x):
        """
        Apply linear transformation and activation
        
        Args:
            x: Input tensor
            
        Returns:
            Activated output tensor
        """
        # Linear transformation
        x = self.linear(x)
        
        # Apply activation function
        x = self.activation(x)
        
        return x


class OutputLayer(nn.Module):
    """
    Output layer that projects to vocabulary size for next-token prediction
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        """
        Args:
            hidden_dim: Hidden layer dimension
            vocab_size: Number of unique tokens (output dimension)
        """
        super(OutputLayer, self).__init__()
        
        # Linear projection: hidden_dim -> vocab_size
        # Output will be logits (raw scores) for each token in vocabulary
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        print(f"Output Layer initialized: {hidden_dim} -> {vocab_size} (vocab size)")
    
    def forward(self, x):
        """
        Project hidden representation to vocabulary logits
        
        Args:
            x: Hidden representation tensor
            
        Returns:
            Logits of shape (..., vocab_size)
        """
        return self.linear(x)
