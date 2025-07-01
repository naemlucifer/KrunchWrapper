#!/usr/bin/env python3
"""
Test script for case-insensitive model tokenizer detection and custom mappings.
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from core.model_tokenizer_validator import ModelTokenizerValidator

def test_case_insensitive_detection():
    """Test that model detection is case-insensitive."""
    print("🔍 Testing Case-Insensitive Model Detection")
    print("=" * 50)
    
    validator = ModelTokenizerValidator()
    
    # Test various case combinations
    test_cases = [
        # OpenAI
        ("gpt-4", "gpt-4"),
        ("GPT-4", "gpt-4"),
        ("Gpt-4", "gpt-4"),
        ("GPT4", "gpt-4"),
        ("openai/GPT-4", "gpt-4"),
        
        # Claude
        ("claude", "claude"),
        ("CLAUDE", "claude"),
        ("Claude-3-5-Sonnet", "claude"),
        ("ANTHROPIC/CLAUDE-3-HAIKU", "claude"),
        
        # LLaMA
        ("llama", "llama"),
        ("LLAMA", "llama"),
        ("Llama-2-7B", "llama"),
        ("META-LLAMA/LLAMA-3-8B-INSTRUCT", "llama3"),
        ("llama3", "llama3"),
        ("LLAMA-3", "llama3"),
        
        # Qwen
        ("qwen", "qwen"),
        ("QWEN", "qwen"),
        ("Qwen2.5-Coder", "qwen2"),
        ("QWEN2", "qwen2"),
        ("qwen3", "qwen3"),
        
        # Others
        ("mistral", "mistral"),
        ("MISTRAL", "mistral"),
        ("deepseek", "deepseek"),
        ("DEEPSEEK", "deepseek"),
        
        # Should not match
        ("unknown-model", None),
        ("random", None),
        ("", None),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for model_name, expected_family in test_cases:
        detected_family = validator.detect_model_family(model_name)
        status = "✅" if detected_family == expected_family else "❌"
        
        print(f"{status} {model_name:<35} → {detected_family or 'None':<15} (expected: {expected_family or 'None'})")
        
        if detected_family == expected_family:
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} tests passed")
    return success_count == total_count

def test_custom_model_mappings():
    """Test that custom model mappings work correctly."""
    print("\n🔧 Testing Custom Model Mappings")
    print("=" * 50)
    
    # This test demonstrates how custom mappings would work
    # The actual custom mappings need to be configured in config.jsonc
    
    print("To test custom model mappings:")
    print("1. Add custom mappings to config/config.jsonc:")
    print('   "custom_model_mappings": {')
    print('     "generic_model": ["a", "model", "llm", "ai"],')
    print('     "my_custom_gpt": ["custom-gpt", "company-model"]')
    print('   }')
    print("\n2. Restart the server to reload configuration")
    print("\n3. Test with these model names:")
    
    test_names = ["a", "A", "model", "MODEL", "custom-gpt", "CUSTOM-GPT"]
    
    validator = ModelTokenizerValidator()
    
    print("\nCurrent detection results:")
    for name in test_names:
        family = validator.detect_model_family(name)
        print(f"  {name:<15} → {family or 'None'}")
    
    print("\n💡 To fix the 'Unknown model family for: a' warning,")
    print("   add this to your config/config.jsonc:")
    print('   "custom_model_mappings": {')
    print('     "gpt-4": ["a", "model", "llm"]')
    print('   }')
    
    return True

def test_pattern_priority():
    """Test that pattern matching priority works correctly."""
    print("\n🎯 Testing Pattern Priority")
    print("=" * 50)
    
    validator = ModelTokenizerValidator()
    
    # Test that more specific patterns take priority
    priority_tests = [
        # Should detect llama3, not llama
        ("llama3", "llama3"),
        ("llama-3", "llama3"),
        ("meta-llama/Llama-3-8B-Instruct", "llama3"),
        
        # Should detect qwen2/qwen3, not qwen
        ("qwen2", "qwen2"),
        ("qwen-2", "qwen2"),
        ("qwen3", "qwen3"),
        ("qwen-3", "qwen3"),
        
        # Should still detect base versions
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("llama2", "llama"),
    ]
    
    success_count = 0
    for model_name, expected_family in priority_tests:
        detected_family = validator.detect_model_family(model_name)
        status = "✅" if detected_family == expected_family else "❌"
        
        print(f"{status} {model_name:<35} → {detected_family or 'None':<15} (expected: {expected_family})")
        
        if detected_family == expected_family:
            success_count += 1
    
    print(f"\nPriority Results: {success_count}/{len(priority_tests)} tests passed")
    return success_count == len(priority_tests)

def main():
    """Run all tests."""
    print("🚀 Model Tokenizer Case-Insensitive & Custom Mapping Test Suite")
    print("=" * 70)
    
    try:
        # Run all tests
        test1_passed = test_case_insensitive_detection()
        test2_passed = test_custom_model_mappings()
        test3_passed = test_pattern_priority()
        
        print("\n" + "=" * 70)
        if test1_passed and test3_passed:
            print("✅ All core tests passed! Case-insensitive detection works correctly.")
        else:
            print("❌ Some tests failed. Check the implementation.")
            
        print("\n📝 Configuration Examples:")
        print("For the 'Unknown model family for: a' issue, add to config.jsonc:")
        print('  "model_tokenizer": {')
        print('    "custom_model_mappings": {')
        print('      "gpt-4": ["a", "model", "llm"],')
        print('      "claude": ["assistant", "ai"],')
        print('      "my_local_model": ["local", "custom-model"]')
        print('    }')
        print('  }')
        
        return 0 if (test1_passed and test3_passed) else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 