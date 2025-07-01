#!/usr/bin/env python3
"""
Test the minimal dictionary format implementation.
"""

import sys
import os
import tempfile
import pathlib

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.system_prompt import (
    encode_dictionary_minimal, 
    decode_dictionary_minimal,
    build_system_prompt,
    _format_dictionary_minimal,
    _format_dictionary_verbose
)


def test_basic_encoding_decoding():
    """Test basic encoding and decoding of minimal format."""
    print("=" * 60)
    print("Testing Basic Encoding/Decoding")
    print("=" * 60)
    
    # Test data
    test_pairs = [
        ("α", "import numpy as np"),
        ("β", "def __init__(self,"),
        ("γ", "variable")
    ]
    
    # Encode
    encoded = encode_dictionary_minimal(test_pairs)
    print(f"Encoded: {encoded}")
    
    # Decode
    decoded = decode_dictionary_minimal(encoded)
    print(f"Decoded: {decoded}")
    
    # Verify roundtrip
    expected = {k: v for k, v in test_pairs}
    assert decoded == expected, f"Expected {expected}, got {decoded}"
    
    print("✅ Basic encoding/decoding works!")
    print()


def test_delimiter_conflicts():
    """Test handling of delimiter conflicts in phrases."""
    print("=" * 60)
    print("Testing Delimiter Conflict Handling")
    print("=" * 60)
    
    # Test data with conflicting delimiters
    test_pairs = [
        ("α", "function(param1; param2)"),  # Contains semicolon
        ("β", "obj: {key: value}"),        # Contains colon
        ("γ", "normal_text")               # No conflicts
    ]
    
    # Test with config for alternative delimiters
    config = {
        "alternative_delimiters": {
            "pair_separator": "‖",
            "key_value_separator": "⟦"
        }
    }
    
    # Encode
    encoded = encode_dictionary_minimal(test_pairs, config)
    print(f"Encoded with alternative delimiters: {encoded}")
    
    # Decode
    decoded = decode_dictionary_minimal(encoded)
    print(f"Decoded: {decoded}")
    
    # Verify roundtrip
    expected = {k: v for k, v in test_pairs}
    assert decoded == expected, f"Expected {expected}, got {decoded}"
    
    print("✅ Delimiter conflict handling works!")
    print()


def test_format_comparison():
    """Test comparison between verbose and minimal formats."""
    print("=" * 60)
    print("Testing Format Comparison")
    print("=" * 60)
    
    # Test dictionary
    test_dict = {
        "α": "import numpy as np",
        "β": "def __init__(self,",
        "γ": "variable",
        "δ": "function",
        "ε": "return",
        "ζ": "class MyClass:"
    }
    
    config = {
        "style": "minimal",
        "minimal_header": "You are given a symbol dictionary. Expand symbols when reading and writing.",
        "alternative_delimiters": {
            "pair_separator": "‖",
            "key_value_separator": "⟦"
        }
    }
    
    # Generate both formats
    verbose_format = _format_dictionary_verbose(test_dict, "python")
    minimal_format = _format_dictionary_minimal(test_dict, "python", config)
    
    print("Verbose format:")
    print(verbose_format)
    print(f"Length: {len(verbose_format)} characters")
    print()
    
    print("Minimal format:")
    print(minimal_format)
    print(f"Length: {len(minimal_format)} characters")
    print()
    
    # Calculate savings
    char_savings = len(verbose_format) - len(minimal_format)
    token_savings_estimate = char_savings // 4
    percentage_savings = (char_savings / len(verbose_format)) * 100
    
    print(f"Character savings: {char_savings} ({percentage_savings:.1f}%)")
    print(f"Estimated token savings: {token_savings_estimate} tokens")
    
    assert char_savings > 0, "Minimal format should be shorter than verbose format"
    print("✅ Minimal format provides savings!")
    print()


def test_build_system_prompt_integration():
    """Test integration with build_system_prompt function."""
    print("=" * 60)
    print("Testing build_system_prompt Integration")
    print("=" * 60)
    
    # Create a temporary config file to control the format
    config_content = """{
  "dictionary_format": {
    "style": "minimal",
    "minimal_format_threshold": 3,
    "minimal_header": "You are given a symbol dictionary. Expand symbols when reading and writing.",
    "alternative_delimiters": {
      "pair_separator": "‖",
      "key_value_separator": "⟦"
    },
    "enable_debug_view": true
  }
}"""
    
    # Test with minimal format (>= 3 entries)
    test_dict_large = {
        "α": "import numpy as np",
        "β": "def __init__(self,",
        "γ": "variable",
        "δ": "function"
    }
    
    content, metadata = build_system_prompt(test_dict_large, "python", "chatml")
    print("System prompt with minimal format (4 entries):")
    print(content)
    print(f"Length: {len(content)} characters")
    print()
    
    # Test with verbose format (< 3 entries)
    test_dict_small = {
        "α": "import numpy as np",
        "β": "def __init__(self,"
    }
    
    content_small, metadata_small = build_system_prompt(test_dict_small, "python", "chatml")
    print("System prompt with verbose format (2 entries):")
    print(content_small)
    print(f"Length: {len(content_small)} characters")
    print()
    
    # Verify minimal format is used for larger dictionaries
    assert "#DICT" in content, "Large dictionary should use minimal format"
    assert "#DICT" not in content_small, "Small dictionary should use verbose format"
    
    print("✅ build_system_prompt integration works!")
    print()


def test_empty_dictionary():
    """Test handling of empty dictionaries."""
    print("=" * 60)
    print("Testing Empty Dictionary Handling")
    print("=" * 60)
    
    # Test empty encoding
    encoded = encode_dictionary_minimal([])
    print(f"Empty encoding: '{encoded}'")
    assert encoded == "", "Empty dictionary should encode to empty string"
    
    # Test empty decoding
    decoded = decode_dictionary_minimal("")
    print(f"Empty decoding: {decoded}")
    assert decoded == {}, "Empty string should decode to empty dictionary"
    
    # Test build_system_prompt with empty dictionary
    content, metadata = build_system_prompt({}, "python", "chatml")
    print(f"Empty dictionary system prompt: '{content}'")
    assert "You will read python code. Reason about it as-is." in content
    
    print("✅ Empty dictionary handling works!")
    print()


def test_special_characters():
    """Test handling of special characters in phrases."""
    print("=" * 60)
    print("Testing Special Character Handling")
    print("=" * 60)
    
    # Test data with various special characters
    test_pairs = [
        ("α", "print(\"Hello, World!\")"),  # Quotes
        ("β", "regex = r'\\d+\\.\\d+'"),    # Backslashes and quotes  
        ("γ", "# This is a comment"),       # Hash character
        ("δ", "unicode: 中文字符"),          # Unicode characters
    ]
    
    # Encode and decode
    encoded = encode_dictionary_minimal(test_pairs)
    print(f"Encoded: {encoded}")
    
    decoded = decode_dictionary_minimal(encoded)
    print(f"Decoded: {decoded}")
    
    # Verify roundtrip
    expected = {k: v for k, v in test_pairs}
    assert decoded == expected, f"Expected {expected}, got {decoded}"
    
    print("✅ Special character handling works!")
    print()


def main():
    """Run all tests."""
    print("🧪 Testing Minimal Dictionary Format Implementation")
    print("=" * 80)
    
    try:
        test_basic_encoding_decoding()
        test_delimiter_conflicts()
        test_format_comparison()
        test_build_system_prompt_integration()
        test_empty_dictionary()
        test_special_characters()
        
        print("=" * 80)
        print("🎉 All tests passed! Minimal dictionary format is working correctly.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 