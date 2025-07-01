#!/usr/bin/env python3
"""
Debug tokenization issues in KrunchWrapper compression
"""
import sys
import tiktoken
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

def test_tokenization_context():
    """Test how tiktoken handles multiple Greek letters in context."""
    print("🔍 Debugging Tokenization Context Issues")
    print("=" * 50)
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Test individual symbols
    symbols = ['α', 'β', 'γ', 'δ', 'ε', 'η', 'θ', 'ι', 'κ', 'λ']
    
    print("📊 Individual Symbol Tokenization:")
    for symbol in symbols:
        tokens = tokenizer.encode(symbol)
        print(f"  '{symbol}' → {len(tokens)} tokens: {tokens}")
    
    # Test symbols in combination
    combined_text = " ".join(symbols)
    combined_tokens = tokenizer.encode(combined_text)
    expected_tokens = sum(len(tokenizer.encode(symbol)) for symbol in symbols) + len(symbols) - 1  # +spaces
    
    print(f"\n🔗 Combined Symbol Tokenization:")
    print(f"Text: '{combined_text}'")
    print(f"Actual tokens: {len(combined_tokens)} {combined_tokens}")
    print(f"Expected tokens: ~{expected_tokens}")
    
    # Test realistic compression scenario
    original_text = """
    log_message = "Starting processing"
    completions = get_completions()
    compression_ratio = calculate_ratio()
    """
    
    # Simulate compression with our dictionary
    compressed_text = original_text.replace("log_message", "α").replace("completions", "β").replace("compression_ratio", "γ")
    
    print(f"\n🧪 Realistic Compression Test:")
    print(f"Original text: {len(original_text)} chars")
    print(f"Compressed text: {len(compressed_text)} chars")
    
    original_tokens = tokenizer.encode(original_text)
    compressed_tokens = tokenizer.encode(compressed_text)
    
    print(f"Original tokens: {len(original_tokens)}")
    print(f"Compressed tokens: {len(compressed_tokens)}")
    print(f"Token difference: {len(original_tokens) - len(compressed_tokens)}")
    print(f"Token ratio: {(len(original_tokens) - len(compressed_tokens)) / len(original_tokens) * 100:.2f}%")
    
    # Test dense symbol usage (like in KrunchWrapper output)
    dense_symbols = "α β γ δ ε η θ ι κ λ μ ν ο π ρ σ τ υ φ χ ψ ω"
    dense_tokens = tokenizer.encode(dense_symbols)
    print(f"\n⚡ Dense Symbol Test:")
    print(f"Dense symbols: '{dense_symbols}'")
    print(f"Tokens: {len(dense_tokens)} {dense_tokens}")
    
    # Check for multi-token symbols
    print(f"\n🚨 Multi-token Symbols Detected:")
    for symbol in symbols:
        tokens = tokenizer.encode(symbol)
        if len(tokens) > 1:
            print(f"  ⚠️  '{symbol}' = {len(tokens)} tokens: {tokens}")

if __name__ == "__main__":
    test_tokenization_context() 