"""
Loss Function for Mini Language Model
Implements cross-entropy loss for next-token prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelLoss:
    """
    Cross-entropy loss for language model training
    """
    
    def __init__(self, pad_token_id: int):
        """
        Args:
            pad_token_id: ID of the padding token (to ignore in loss)
        """
        self.pad_token_id = pad_token_id
        
        # CrossEntropyLoss combines LogSoftmax and NLLLoss
        # ignore_index ensures padding tokens don't contribute to loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss between predictions and targets
        
        Args:
            logits: Model output logits, shape (batch_size, seq_length, vocab_size)
            targets: Target token IDs, shape (batch_size, seq_length)
            
        Returns:
            loss: Scalar loss value
        """
        # Reshape logits and targets for cross-entropy
        # CrossEntropyLoss expects:
        #   - logits: (N, C) where N = batch_size * seq_length, C = vocab_size
        #   - targets: (N,) where N = batch_size * seq_length
        
        batch_size, seq_length, vocab_size = logits.shape
        
        # Reshape logits: (batch_size, seq_length, vocab_size) -> (batch_size * seq_length, vocab_size)
        logits_flat = logits.view(-1, vocab_size)
        
        # Reshape targets: (batch_size, seq_length) -> (batch_size * seq_length,)
        targets_flat = targets.view(-1)
        
        # Compute cross-entropy loss
        loss = self.criterion(logits_flat, targets_flat)
        
        return loss


def get_predictions(logits):
    """
    Convert logits to predicted token IDs
    
    Args:
        logits: Model output logits, shape (..., vocab_size)
        
    Returns:
        predictions: Predicted token IDs
    """
    # Take argmax along vocabulary dimension
    predictions = torch.argmax(logits, dim=-1)
    return predictions


def apply_temperature(logits, temperature=1.0):
    """
    Apply temperature scaling to logits for sampling
    Higher temperature = more random, lower = more deterministic
    
    Args:
        logits: Model output logits
        temperature: Temperature parameter
        
    Returns:
        scaled_logits: Temperature-scaled logits
    """
    return logits / temperature


def sample_from_logits(logits, temperature=1.0):
    """
    Sample token from logits using temperature
    
    Args:
        logits: Model output logits, shape (..., vocab_size)
        temperature: Sampling temperature
        
    Returns:
        sampled_token: Sampled token ID
    """
    # Apply temperature scaling
    scaled_logits = apply_temperature(logits, temperature)
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample from distribution
    sampled_token = torch.multinomial(probs, num_samples=1)
    
    return sampled_token
