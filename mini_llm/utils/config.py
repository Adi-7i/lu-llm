"""
Configuration file for Mini Language Model
Contains all hyperparameters and paths
"""

import os

class Config:
    """Configuration class containing all model and training parameters"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATASET_PATH = os.path.join(DATA_DIR, 'dataset.txt')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_model.pth')
    VOCAB_SAVE_PATH = os.path.join(BASE_DIR, 'vocabulary.pkl')
    
    # Model Architecture Parameters
    EMBEDDING_DIM = 128          # Size of token embeddings
    HIDDEN_DIM = 256             # Size of hidden layer
    
    # Training Parameters
    LEARNING_RATE = 0.001        # Learning rate for optimizer
    EPOCHS = 500                 # Number of training epochs (increased for better learning)
    BATCH_SIZE = 8               # Number of samples per batch
    
    # Sequence Parameters
    MAX_SEQ_LENGTH = 50          # Maximum sequence length for training
    
    # Generation Parameters
    MAX_GENERATION_LENGTH = 100  # Maximum tokens to generate during inference
    TEMPERATURE = 1.0            # Sampling temperature (higher = more random)
    
    # Special Tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
