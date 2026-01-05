"""
Custom Tokenizer for Mini Language Model
Implements word-level tokenization from scratch
"""

import pickle
from typing import List, Dict
from utils.config import Config


class SimpleTokenizer:
    """
    A simple word-level tokenizer that builds vocabulary from text
    and converts between text and token IDs
    """
    
    def __init__(self):
        """Initialize tokenizer with special tokens"""
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.START_TOKEN,
            Config.END_TOKEN
        ]
        
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
        
        self.vocab_size = len(self.word2idx)
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from a list of text strings
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        print("Building vocabulary...")
        
        for text in texts:
            # Split text into words (simple whitespace tokenization)
            words = text.lower().split()
            
            for word in words:
                # Add word to vocabulary if not already present
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built! Size: {self.vocab_size} tokens")
        print(f"Sample words: {list(self.word2idx.keys())[:10]}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to list of token IDs
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add START and END tokens
            
        Returns:
            List of token IDs
        """
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Convert words to token IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word2idx[Config.START_TOKEN])
        
        for word in words:
            # Use UNK token for unknown words
            token_id = self.word2idx.get(word, self.word2idx[Config.UNK_TOKEN])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.word2idx[Config.END_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert list of token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        words = []
        special_token_ids = {
            self.word2idx[Config.PAD_TOKEN],
            self.word2idx[Config.START_TOKEN],
            self.word2idx[Config.END_TOKEN]
        }
        
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            # Stop at END token
            if token_id == self.word2idx[Config.END_TOKEN]:
                break
            
            # Convert ID to word
            word = self.idx2word.get(token_id, Config.UNK_TOKEN)
            words.append(word)
        
        return ' '.join(words)
    
    def save(self, filepath: str):
        """Save tokenizer vocabulary to disk"""
        print(f"Saving tokenizer to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
        print("Tokenizer saved!")
    
    def load(self, filepath: str):
        """Load tokenizer vocabulary from disk"""
        print(f"Loading tokenizer from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
        print(f"Tokenizer loaded! Vocabulary size: {self.vocab_size}")
