"""
Interactive Chat Interface for Mini Language Model
Implements hybrid retrieval + generation for better responses
"""

import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.model import MiniLanguageModel
from utils.config import Config
import os


def load_trained_model():
    """
    Load trained model and tokenizer from disk
    
    Returns:
        model: Trained MiniLanguageModel
        tokenizer: Trained tokenizer
    """
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.load(Config.VOCAB_SAVE_PATH)
    
    print("Loading model...")
    model = MiniLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_model(Config.MODEL_SAVE_PATH)
    model.eval()  # Set to evaluation mode
    
    print("Model and tokenizer loaded successfully!\n")
    return model, tokenizer


def load_qa_dataset():
    """Load Q&A pairs from dataset for retrieval"""
    qa_pairs = []
    
    with open(Config.DATASET_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line and i + 1 < len(lines):
            question = line.lower()
            answer = lines[i + 1].strip()
            
            if answer:
                qa_pairs.append((question, answer))
            
            i += 2
        else:
            i += 1
    
    return qa_pairs


def find_best_match(user_input: str, qa_pairs: list):
    """
    Find best matching Q&A pair from dataset
    
    Args:
        user_input: User's question
        qa_pairs: List of (question, answer) tuples
        
    Returns:
        Best matching answer or None
    """
    import re
    
    # Normalize user input: lowercase and remove punctuation
    user_input_normalized = re.sub(r'[^\w\s]', '', user_input.lower()).strip()
    
    # Try exact match first (normalized)
    for question, answer in qa_pairs:
        question_normalized = re.sub(r'[^\w\s]', '', question.lower()).strip()
        if user_input_normalized == question_normalized:
            return answer
    
    # Try partial match - check if user input contains key words from questions
    user_words = set(user_input_normalized.split())
    best_match = None
    best_score = 0
    
    for question, answer in qa_pairs:
        question_normalized = re.sub(r'[^\w\s]', '', question.lower()).strip()
        question_words = set(question_normalized.split())
        
        # Calculate word overlap
        overlap = len(user_words & question_words)
        
        # Also consider if question words are subset of user input
        if len(question_words) > 0:
            coverage = overlap / len(question_words)
        else:
            coverage = 0
        
        # Combined score: overlap count + coverage bonus
        score = overlap + (coverage * 0.5)
        
        if score > best_score:
            best_score = score
            best_match = answer
    
    # Return match if we have good overlap (at least 2 words or 50% coverage)
    if best_score >= 1.5:
        return best_match
    
    return None



def generate_with_model(model, tokenizer, prompt: str, max_length: int = 30):
    """
    Generate response using the trained model
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Max tokens to generate
        
    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Pad to model's expected length
    pad_id = tokenizer.word2idx[Config.PAD_TOKEN]
    while len(input_ids) < Config.MAX_SEQ_LENGTH:
        input_ids.append(pad_id)
    
    input_ids = input_ids[:Config.MAX_SEQ_LENGTH]
    input_tensor = torch.tensor([input_ids])
    
    end_token_id = tokenizer.word2idx[Config.END_TOKEN]
    start_token_id = tokenizer.word2idx[Config.START_TOKEN]
    
    generated_ids = []
    
    with torch.no_grad():
        # Get model output for input
        logits = model(input_tensor)
        
        # Use the last position's logits
        last_logits = logits[0, -1, :]
        
        # Generate tokens
        for _ in range(max_length):
            # Get next token
            next_token_id = torch.argmax(last_logits).item()
            
            # Stop conditions
            if next_token_id == end_token_id:
                break
            
            if next_token_id not in [start_token_id, pad_id]:
                generated_ids.append(next_token_id)
            
            # Update logits for next prediction
            if len(generated_ids) > 0:
                new_input = torch.tensor([[next_token_id]])
                logits = model(new_input)
                last_logits = logits[0, -1, :]
            
            # Stop if repetitive
            if len(generated_ids) >= 3:
                if generated_ids[-1] == generated_ids[-2] == generated_ids[-3]:
                    break
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_response(model, tokenizer, user_input: str, qa_pairs: list):
    """
    Generate response using hybrid approach: retrieval + generation
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        user_input: User's input
        qa_pairs: Dataset Q&A pairs
        
    Returns:
        Response string
    """
    # First, try to find a matching answer from dataset
    matched_answer = find_best_match(user_input, qa_pairs)
    
    if matched_answer:
        return matched_answer
    
    # If no match, try model generation
    generated = generate_with_model(model, tokenizer, user_input)
    
    if generated and len(generated) > 5:
        return generated
    
    # Fallback message
    return "I'm not sure about that. Try asking: 'What is your name?', 'What is Python?', or 'What is AI?'"


def chat_loop():
    """
    Interactive chat loop with hybrid retrieval-generation
    """
    print("=" * 60)
    print("MINI LANGUAGE MODEL - CHAT INTERFACE")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_trained_model()
    except FileNotFoundError:
        print("ERROR: Model not found!")
        print("Please train the model first by running: python main.py train")
        return
    
    # Load Q&A dataset for retrieval
    print("Loading Q&A dataset for retrieval...")
    qa_pairs = load_qa_dataset()
    print(f"Loaded {len(qa_pairs)} Q&A pairs\n")
    
    print("Chat started! Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60)
    print()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("MiniLM: Goodbye! Have a great day!")
            break
        
        if not user_input:
            continue
        
        # Generate response using hybrid approach
        try:
            response = generate_response(model, tokenizer, user_input, qa_pairs)
            print(f"MiniLM: {response}")
        except Exception as e:
            print(f"MiniLM: Sorry, I encountered an error. Please try again.")
        
        print()  # Blank line for readability


if __name__ == "__main__":
    chat_loop()

