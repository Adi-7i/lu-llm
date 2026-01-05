"""
Training Loop for Mini Language Model
Implements data preparation and training from scratch
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os

from tokenizer.tokenizer import SimpleTokenizer
from model.model import MiniLanguageModel
from training.loss import LanguageModelLoss
from utils.config import Config


class QADataset(Dataset):
    """
    Custom Dataset for Question-Answer pairs
    """
    
    def __init__(self, qa_pairs: List[Tuple[str, str]], tokenizer: SimpleTokenizer):
        """
        Args:
            qa_pairs: List of (question, answer) tuples
            tokenizer: Tokenizer instance
        """
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = Config.MAX_SEQ_LENGTH
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        """
        Get a single training example
        
        Returns:
            input_ids: Token IDs for input (question)
            target_ids: Token IDs for target (answer)
        """
        question, answer = self.qa_pairs[idx]
        
        # Encode question and answer
        input_ids = self.tokenizer.encode(question, add_special_tokens=True)
        target_ids = self.tokenizer.encode(answer, add_special_tokens=True)
        
        # Pad or truncate to max length
        input_ids = self._pad_sequence(input_ids)
        target_ids = self._pad_sequence(target_ids)
        
        return torch.tensor(input_ids), torch.tensor(target_ids)
    
    def _pad_sequence(self, token_ids: List[int]) -> List[int]:
        """Pad or truncate sequence to max_length"""
        pad_id = self.tokenizer.word2idx[Config.PAD_TOKEN]
        
        if len(token_ids) >= self.max_length:
            # Truncate
            return token_ids[:self.max_length]
        else:
            # Pad
            padding_length = self.max_length - len(token_ids)
            return token_ids + [pad_id] * padding_length


def load_dataset(filepath: str) -> List[Tuple[str, str]]:
    """
    Load Q&A pairs from dataset file
    
    Args:
        filepath: Path to dataset.txt
        
    Returns:
        List of (question, answer) tuples
    """
    qa_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse dataset (expects alternating question/answer lines with blank line separator)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line and i + 1 < len(lines):
            question = line
            answer = lines[i + 1].strip()
            
            if answer:  # Only add if answer is not empty
                qa_pairs.append((question, answer))
            
            i += 2
        else:
            i += 1
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs from dataset")
    return qa_pairs


def train_model():
    """
    Main training function
    """
    print("=" * 60)
    print("MINI LANGUAGE MODEL - TRAINING")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    qa_pairs = load_dataset(Config.DATASET_PATH)
    
    # Also extract all text for vocabulary building
    all_texts = []
    for q, a in qa_pairs:
        all_texts.append(q)
        all_texts.append(a)
    
    # Step 2: Build tokenizer
    print("\n[2/6] Building tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocabulary(all_texts)
    tokenizer.save(Config.VOCAB_SAVE_PATH)
    
    # Step 3: Create dataset and dataloader
    print("\n[3/6] Preparing data loader...")
    dataset = QADataset(qa_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    print(f"Created DataLoader with batch size {Config.BATCH_SIZE}")
    
    # Step 4: Initialize model
    print("\n[4/6] Initializing model...")
    model = MiniLanguageModel(vocab_size=tokenizer.vocab_size)
    
    # Step 5: Setup loss and optimizer
    print("\n[5/6] Setting up training components...")
    pad_token_id = tokenizer.word2idx[Config.PAD_TOKEN]
    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    print(f"Optimizer: Adam with learning rate {Config.LEARNING_RATE}")
    
    # Step 6: Training loop
    print("\n[6/6] Starting training...")
    print("=" * 60)
    
    model.train()  # Set model to training mode
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: get model predictions
            # input_ids shape: (batch_size, seq_length)
            # We use input for prediction, shift target by 1 for next-token prediction
            logits = model(input_ids)
            
            # Compute loss
            # For simplicity, we predict the entire target sequence
            loss = loss_fn.compute_loss(logits, target_ids)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            model.save_model(checkpoint_path)
    
    # Final save
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final Loss: {avg_loss:.4f}")
    print("\nSaving final model...")
    model.save_model(Config.MODEL_SAVE_PATH)
    print("=" * 60)
    
    return model, tokenizer


if __name__ == "__main__":
    train_model()
