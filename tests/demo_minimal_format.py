#!/usr/bin/env python3
"""
Demonstration of the minimal dictionary format with real compression scenarios.
"""

import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.system_prompt import build_system_prompt, encode_dictionary_minimal, decode_dictionary_minimal
from core.conversation_compress import _estimate_system_prompt_overhead

def demo_format_comparison():
    """Demonstrate the difference between verbose and minimal formats."""
    print("🎭 DEMONSTRATION: Minimal Dictionary Format")
    print("=" * 80)
    
    # Simulate a real compression dictionary
    compression_dict = {
        "α": "import numpy as np",
        "β": "def __init__(self,",
        "γ": "return",
        "δ": "self.",
        "ε": "function",
        "ζ": "class",
        "η": "if __name__ == '__main__':",
        "θ": "print(",
        "ι": "for i in range(",
        "κ": "try:",
        "λ": "except Exception as e:",
        "μ": "with open(",
        "ν": ".format(",
        "ξ": "def test_",
        "ο": "assert",
        "π": "import unittest",
        "ρ": "from typing import",
        "σ": "List[",
        "τ": "Dict[",
        "υ": "Optional[",
    }
    
    # Build system prompts with both formats by temporarily modifying the config
    
    # First, test verbose format
    print("\n📝 VERBOSE FORMAT (Traditional):")
    print("-" * 50)
    
    # Create a test verbose format manually
    pairs = ", ".join(f"{k}={v}" for k, v in list(compression_dict.items())[:10])
    verbose_content = (
        f"You will read python code in a compressed DSL. "
        f"Apply these symbol substitutions when understanding and responding: {pairs}. "
        f"This reduces token usage."
    )
    print(verbose_content)
    print(f"\nCharacters: {len(verbose_content)}")
    print(f"Estimated tokens: {len(verbose_content) // 4}")
    
    # Now test minimal format
    print("\n🎯 MINIMAL FORMAT (New):")
    print("-" * 50)
    
    config = {
        "minimal_header": "You are given a symbol dictionary. Expand symbols when reading and writing.",
        "alternative_delimiters": {
            "pair_separator": "‖",
            "key_value_separator": "⟦"
        }
    }
    
    # Create minimal format
    pairs_list = [(k, v) for k, v in list(compression_dict.items())[:10]]
    encoded_dict = encode_dictionary_minimal(pairs_list, config)
    minimal_content = f"{config['minimal_header']}\n{encoded_dict}"
    
    print(minimal_content)
    print(f"\nCharacters: {len(minimal_content)}")
    print(f"Estimated tokens: {len(minimal_content) // 4}")
    
    # Calculate savings
    char_savings = len(verbose_content) - len(minimal_content)
    token_savings = char_savings // 4
    percentage_savings = (char_savings / len(verbose_content)) * 100
    
    print(f"\n💰 SAVINGS ANALYSIS:")
    print("-" * 50)
    print(f"Character savings: {char_savings} ({percentage_savings:.1f}%)")
    print(f"Token savings: {token_savings} tokens")
    print(f"Cost savings (GPT-4): ~${token_savings * 0.00002:.6f} per request")
    print(f"Cost savings (1M requests): ~${token_savings * 0.00002 * 1000000:.2f}")
    
    # Test overhead calculation
    print(f"\n⚖️  OVERHEAD CALCULATION:")
    print("-" * 50)
    
    # Test the actual overhead calculation function
    overhead_tokens = _estimate_system_prompt_overhead(dict(list(compression_dict.items())[:10]))
    print(f"Accurate overhead estimation: {overhead_tokens} tokens")
    print(f"This accounts for the actual format being used (minimal vs verbose)")
    
    return char_savings, token_savings


def demo_delimiter_handling():
    """Demonstrate automatic delimiter conflict resolution."""
    print(f"\n🔧 DELIMITER CONFLICT RESOLUTION:")
    print("-" * 50)
    
    # Dictionary with conflicting characters
    conflict_dict = {
        "α": "function(param1; param2)",
        "β": "obj: {key: value}",
        "γ": "normal_text"
    }
    
    pairs_list = [(k, v) for k, v in conflict_dict.items()]
    encoded = encode_dictionary_minimal(pairs_list)
    
    print("Dictionary with conflicts:")
    for k, v in conflict_dict.items():
        print(f"  {k}: {v}")
    
    print(f"\nAutomatically encoded:")
    print(encoded)
    
    # Decode to verify
    decoded = decode_dictionary_minimal(encoded)
    print(f"\nDecoded back:")
    for k, v in decoded.items():
        print(f"  {k}: {v}")
    
    # Verify roundtrip
    assert decoded == conflict_dict, "Roundtrip failed!"
    print("\n✅ Roundtrip successful - no data loss!")


def demo_real_world_scenario():
    """Demonstrate with a realistic compression scenario."""
    print(f"\n🌍 REAL-WORLD SCENARIO:")
    print("-" * 50)
    
    # Simulate actual system prompt building
    real_compression = {
        "α": "import",
        "β": "def ",
        "γ": "class ",
        "δ": "return ",
        "ε": "self.",
        "ζ": "print(",
        "η": "__init__",
        "θ": "Exception",
        "ι": "try:",
        "κ": "except:",
        "λ": "for i in",
        "μ": "if __name__",
    }
    
    # Build actual system prompt (will use current config)
    content, metadata = build_system_prompt(real_compression, "python", "chatml")
    
    print("Generated system prompt:")
    print(content)
    print(f"\nMetadata: {metadata}")
    
    # Calculate overhead with the actual function
    overhead = _estimate_system_prompt_overhead(real_compression)
    print(f"\nCalculated overhead: {overhead} tokens")
    print(f"This overhead is automatically accounted for in compression decisions")


def main():
    """Run the demonstration."""
    char_savings, token_savings = demo_format_comparison()
    demo_delimiter_handling()
    demo_real_world_scenario()
    
    print("\n" + "=" * 80)
    print("🎉 SUMMARY:")
    print(f"✅ Implemented minimal dictionary format")
    print(f"💰 Achieves {char_savings} character savings ({token_savings} tokens)")
    print(f"🔧 Handles delimiter conflicts automatically")
    print(f"⚖️  Accurate overhead calculation")
    print(f"🔄 Fully backward compatible")
    print(f"📊 Configurable via system-prompts.jsonc")
    print("=" * 80)


if __name__ == "__main__":
    main() 