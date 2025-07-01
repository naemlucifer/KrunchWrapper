#!/usr/bin/env python3
"""
Test script for model-specific tokenizer validation.
Demonstrates how the model tokenizer validator works with different model families.
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from core.model_tokenizer_validator import ModelTokenizerValidator, get_model_tokenizer_validator, validate_with_model
from core.model_context import ModelContext, set_global_model_context, extract_model_from_provider_format, normalize_model_name

def test_model_family_detection():
    """Test model family detection for various model names."""
    print("🔍 Testing Model Family Detection")
    print("=" * 50)
    
    validator = ModelTokenizerValidator()
    
    test_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "openai/gpt-4",
        "anthropic/claude-3-5-sonnet-20241022",
        "claude-3-haiku",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "01-ai/Yi-34B-Chat",
        "google/gemini-pro",
        "THUDM/chatglm3-6b",
        "microsoft/phi-2",
        "tiiuae/falcon-7b-instruct",
        "unknown-provider/random-model"
    ]
    
    for model in test_models:
        normalized = normalize_model_name(model)
        family = validator.detect_model_family(normalized)
        clean_name = extract_model_from_provider_format(model)
        
        status = "✅" if family else "❓"
        print(f"{status} {model:<40} → {clean_name:<30} → {family or 'Unknown'}")

def test_tokenizer_availability():
    """Test which tokenizer libraries are available."""
    print("\n🔧 Testing Tokenizer Library Availability")
    print("=" * 50)
    
    validator = ModelTokenizerValidator()
    
    for library, available in validator.available_tokenizers.items():
        status = "✅" if available else "❌"
        print(f"{status} {library:<15} {'Available' if available else 'Not available'}")

def test_basic_validation():
    """Test basic tokenizer validation with different models."""
    print("\n🧪 Testing Basic Tokenizer Validation")
    print("=" * 50)
    
    # Test text
    original_text = "def process_user_input(user_message: str) -> str:"
    compressed_text = "def α(β: str) -> str:"
    
    test_models = ["gpt-4", "claude-3-5-sonnet", "llama-3-8b-instruct", "qwen2.5-coder-32b"]
    
    for model in test_models:
        print(f"\n📝 Testing with model: {model}")
        print("-" * 30)
        
        # Test with model context
        with ModelContext(model):
            result = validate_with_model(original_text, compressed_text, model)
            validator = get_model_tokenizer_validator()
            detailed_result = validator.validate_token_efficiency(original_text, compressed_text, model)
            
            print(f"   Token ratio: {result:.3f}")
            print(f"   Method: {detailed_result.get('method', 'unknown')}")
            print(f"   Model family: {detailed_result.get('model_family', 'unknown')}")
            
            if detailed_result.get("method") == "model_specific":
                print(f"   Original tokens: {detailed_result.get('original_tokens', 0)}")
                print(f"   Compressed tokens: {detailed_result.get('compressed_tokens', 0)}")
                print(f"   Tokenizer type: {detailed_result.get('tokenizer_type', 'unknown')}")
            else:
                print(f"   Fallback used: {detailed_result.get('method', 'unknown')}")

def test_model_context():
    """Test model context management."""
    print("\n🎯 Testing Model Context Management")
    print("=" * 50)
    
    # Test context manager
    print("Testing context manager:")
    with ModelContext("anthropic/claude-3-5-sonnet-20241022"):
        from core.model_context import get_current_model, get_effective_model
        print(f"   Current model: {get_current_model()}")
        print(f"   Effective model: {get_effective_model()}")
    
    # Test global context
    print("\nTesting global context:")
    set_global_model_context("openai/gpt-4")
    from core.model_context import get_effective_model
    print(f"   Global model: {get_effective_model()}")

def test_provider_format_extraction():
    """Test extraction of model names from provider/model format."""
    print("\n🔧 Testing Provider Format Extraction")
    print("=" * 50)
    
    test_cases = [
        ("anthropic/claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
        ("openai/gpt-4", "gpt-4"),
        ("meta-llama/Llama-3-8B-Instruct", "Llama-3-8B-Instruct"),
        ("qwen/qwen2.5-coder-32b-instruct", "qwen2.5-coder-32b-instruct"),
        ("just-a-model-name", "just-a-model-name"),
        ("", ""),
    ]
    
    for original, expected in test_cases:
        result = extract_model_from_provider_format(original)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{original}' → '{result}' (expected: '{expected}')")

def test_integration_with_compression():
    """Test integration with the compression system."""
    print("\n🔗 Testing Integration with Compression System")
    print("=" * 50)
    
    # Test the enhanced validate_tokenization_efficiency function
    from core.compress import validate_tokenization_efficiency
    
    original_text = """
    def analyze_code_complexity(source_code: str, language: str) -> dict:
        complexity_metrics = {}
        token_count = len(source_code.split())
        line_count = len(source_code.split('\\n'))
        complexity_metrics['tokens'] = token_count
        complexity_metrics['lines'] = line_count
        return complexity_metrics
    """
    
    compressed_text = original_text.replace("analyze_code_complexity", "α").replace("complexity_metrics", "β").replace("source_code", "γ")
    used_dict = {"α": "analyze_code_complexity", "β": "complexity_metrics", "γ": "source_code"}
    
    test_models = ["gpt-4", "claude-3-sonnet", "llama-3-8b"]
    
    for model in test_models:
        print(f"\n📝 Testing compression validation with {model}:")
        print("-" * 40)
        
        # Set model context
        set_global_model_context(model)
        
        # Test validation
        ratio = validate_tokenization_efficiency(original_text, compressed_text, used_dict, model)
        print(f"   Token compression ratio: {ratio:.3f}")
        print(f"   Result: {'✅ Beneficial' if ratio > 0 else '❌ Not beneficial'}")

def main():
    """Run all tests."""
    print("🚀 Model-Specific Tokenizer Validator Test Suite")
    print("=" * 60)
    
    try:
        test_model_family_detection()
        test_tokenizer_availability()
        test_basic_validation()
        test_model_context()
        test_provider_format_extraction()
        test_integration_with_compression()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n💡 To use model-specific validation in your application:")
        print("   1. Set model context: set_global_model_context('your-model')")
        print("   2. Use enhanced validation: validate_tokenization_efficiency(...)")
        print("   3. Configure in config.jsonc: 'model_tokenizer.enabled: true'")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 