"""
Mini Language Model Architecture
Combines layers into a sequence-to-sequence model with LSTM
"""

import torch
import torch.nn as nn
from model.layers import EmbeddingLayer, HiddenLayer, OutputLayer
from utils.config import Config


class MiniLanguageModel(nn.Module):
    """
    An improved language model with LSTM for sequence processing
    
    Architecture:
        Input Token IDs -> Embedding -> LSTM -> Hidden Layer -> Output -> Logits
    """
    
    def __init__(self, vocab_size: int):
        """
        Initialize the Mini Language Model
        
        Args:
            vocab_size: Size of the vocabulary
        """
        super(MiniLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = Config.EMBEDDING_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        
        # Layer 1: Embedding layer
        # Converts token IDs to dense vectors
        self.embedding_layer = EmbeddingLayer(vocab_size, self.embedding_dim)
        
        # Layer 2: LSTM for sequence processing
        # Helps model understand sequential patterns in Q&A
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Layer 3: Hidden layer
        # Processes LSTM output through linear transformation + activation
        self.hidden_layer = HiddenLayer(self.embedding_dim, self.hidden_dim)
        
        # Layer 4: Output layer
        # Projects to vocabulary size for next-token prediction
        self.output_layer = OutputLayer(self.hidden_dim, vocab_size)
        
        print(f"\nMini Language Model Architecture (Enhanced):")
        print(f"  Vocabulary Size: {vocab_size}")
        print(f"  Embedding Dim: {self.embedding_dim}")
        print(f"  LSTM Hidden: {self.embedding_dim}")
        print(f"  Hidden Dim: {self.hidden_dim}")
        print(f"  Total Parameters: {self.count_parameters():,}")
    
    def forward(self, token_ids):
        """
        Forward pass through the model
        
        Args:
            token_ids: Input tensor of token IDs, shape (batch_size, seq_length)
            
        Returns:
            logits: Output logits for next token prediction
                   shape (batch_size, seq_length, vocab_size)
        """
        # Step 1: Convert token IDs to embeddings
        # Shape: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embeddings = self.embedding_layer(token_ids)
        
        # Step 2: Process through LSTM
        # LSTM captures sequential patterns and dependencies
        # Shape: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)
        
        # Step 3: Process through hidden layer
        # Apply to each timestep independently
        # Shape: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_dim)
        hidden = self.hidden_layer(lstm_out)
        
        # Step 4: Project to vocabulary size
        # Shape: (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length, vocab_size)
        logits = self.output_layer(hidden)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath: str):
        """
        Save model weights to disk
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model weights from disk
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")

