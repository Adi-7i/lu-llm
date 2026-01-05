"""
Mini Language Model - Main Entry Point
Command-line interface for training and chatting
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mini_llm'))

from training.train import train_model
from inference.chat import chat_loop
from tokenizer.tokenizer import SimpleTokenizer
from utils.config import Config


def print_banner():
    """Print welcome banner"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "     MINI LANGUAGE MODEL - Built from Scratch".center(58) + "║")
    print("║" + "     Educational LLM using PyTorch".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_help():
    """Print usage instructions"""
    print("Usage: python main.py [command]")
    print()
    print("Commands:")
    print("  train       - Train the language model on dataset")
    print("  chat        - Start interactive chat with trained model")
    print("  test-token  - Test tokenizer functionality")
    print("  help        - Show this help message")
    print()
    print("Example:")
    print("  python main.py train")
    print("  python main.py chat")
    print()


def test_tokenizer():
    """Test tokenizer functionality"""
    print("=" * 60)
    print("TESTING TOKENIZER")
    print("=" * 60)
    
    # Create sample texts
    texts = [
        "Hello, how are you?",
        "What is machine learning?",
        "I am doing well, thank you!"
    ]
    
    # Build tokenizer
    print("\nBuilding vocabulary from sample texts...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocabulary(texts)
    
    # Test encoding and decoding
    print("\n" + "-" * 60)
    print("Testing Encoding and Decoding:")
    print("-" * 60)
    
    for text in texts:
        print(f"\nOriginal: {text}")
        
        # Encode
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        print(f"Token IDs: {token_ids}")
        
        # Decode
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Decoded: {decoded_text}")
        
        # Verify
        if decoded_text.lower() == text.lower():
            print("✓ Encoding/Decoding successful!")
        else:
            print("✗ Encoding/Decoding mismatch!")
    
    print("\n" + "=" * 60)
    print("Tokenizer test completed!")
    print("=" * 60)


def main():
    """Main entry point"""
    print_banner()
    
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("ERROR: No command provided!")
        print()
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    # Route to appropriate function
    if command == "train":
        train_model()
    
    elif command == "chat":
        chat_loop()
    
    elif command == "test-token" or command == "test-tokenizer":
        test_tokenizer()
    
    elif command == "help" or command == "--help" or command == "-h":
        print_help()
    
    else:
        print(f"ERROR: Unknown command '{command}'")
        print()
        print_help()


if __name__ == "__main__":
    main()
