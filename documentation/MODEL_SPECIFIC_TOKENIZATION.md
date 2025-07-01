# Model-Specific Tokenizer Validation

This document explains the model-specific tokenizer validation system implemented in KrunchWrapper, which provides accurate token efficiency validation based on the specific model being used.

## Overview

Different language models use different tokenizers, which means that compression efficiency can vary significantly between models. The model-specific tokenizer validation system automatically detects the model family and uses the appropriate tokenizer for accurate token counting.

## Key Features

- **Automatic Model Detection**: Detects model family from model names (including provider/model format)
- **Multiple Tokenizer Support**: Supports tiktoken, transformers, and SentencePiece tokenizers
- **Fallback Mechanisms**: Graceful fallback to generic validation when model-specific tokenizers are unavailable
- **Context Management**: Tracks current model context throughout the compression pipeline
- **Caching**: Caches loaded tokenizers for better performance

## Supported Model Families

### OpenAI Models
- **GPT-4**: Uses `cl100k_base` encoding via tiktoken
- **GPT-3.5**: Uses `cl100k_base` encoding via tiktoken
- **GPT-3**: Uses `p50k_base` encoding via tiktoken

### Anthropic Models
- **Claude**: Uses custom SentencePiece tokenizer

### Meta LLaMA Models
- **LLaMA 1/2**: Uses SentencePiece tokenizer (32k vocab)
- **LLaMA 3**: Uses tiktoken-style tokenizer via transformers (128k vocab)
- **CodeLlama**: Uses SentencePiece tokenizer

### Mistral Models
- **Mistral/Mixtral**: Uses SentencePiece tokenizer (32k vocab)

### Qwen Models
- **Qwen 1/3**: Uses tiktoken-style tokenizer via transformers (152k vocab)
- **Qwen 2**: Uses tiktoken-style tokenizer via transformers (152k vocab)

### Google Models
- **Gemini**: Uses custom SentencePiece tokenizer
- **PaLM**: Uses SentencePiece tokenizer

### Other Models
- **Yi**: Uses SentencePiece tokenizer (64k vocab)
- **DeepSeek**: Uses tiktoken-style tokenizer via transformers
- **ChatGLM**: Uses custom SentencePiece tokenizer
- **Phi**: Uses transformers tokenizer
- **Falcon**: Uses transformers tokenizer
- **StarCoder**: Uses transformers tokenizer

## Configuration

Add the following to your `config/config.jsonc`:

```jsonc
{
    "model_tokenizer": {
        /* Enable model-specific tokenizer validation */
        "enabled": true,
        
        /* Default model family when detection fails */
        "default_model_family": "gpt-4",
        
        /* Fallback method when model-specific tokenizer unavailable */
        "fallback_method": "tiktoken",
        
        /* Cache tokenizers for better performance */
        "cache_tokenizers": true
    }
}
```

### Configuration Options

- `enabled`: Enable/disable model-specific validation (default: true)
- `default_model_family`: Fallback model family for unknown models
- `fallback_method`: Validation method when model-specific fails
  - `"tiktoken"`: Use generic tiktoken validation
  - `"character_estimation"`: Use character-based estimation
  - `"word_count"`: Use word count estimation
- `cache_tokenizers`: Cache loaded tokenizers for performance

## Installation Requirements

For maximum compatibility, install all tokenizer libraries:

```bash
# For OpenAI models
pip install tiktoken

# For modern models (Qwen, LLaMA 3, etc.)
pip install transformers

# For LLaMA and other SentencePiece models
pip install sentencepiece

# Install all for maximum compatibility
pip install tiktoken transformers sentencepiece
```

## Usage

### Automatic Usage (Recommended)

The system automatically detects and uses model-specific tokenizers when the model is specified in API requests:

```python
# The system automatically extracts model info from requests
# and uses appropriate tokenizers for validation
```

### Manual Usage

```python
from core.model_tokenizer_validator import get_model_tokenizer_validator, validate_with_model
from core.model_context import ModelContext, set_global_model_context

# Method 1: Using context manager
with ModelContext("anthropic/claude-3-5-sonnet-20241022"):
    result = validate_with_model(original_text, compressed_text, "claude-3-5-sonnet")

# Method 2: Using global context
set_global_model_context("openai/gpt-4")
result = validate_with_model(original_text, compressed_text, "gpt-4")

# Method 3: Direct validation
validator = get_model_tokenizer_validator()
result = validator.validate_token_efficiency(original_text, compressed_text, "gpt-4")
```

### Enhanced Compression Validation

The existing validation functions now support model-specific validation:

```python
from core.compress import validate_tokenization_efficiency

# Enhanced function with model parameter
ratio = validate_tokenization_efficiency(
    original_text="def process_user_input(message: str) -> str:",
    compressed_text="def α(β: str) -> str:",
    used_dict={"α": "process_user_input", "β": "message"},
    model_name="gpt-4"  # Optional - will use context if not provided
)
```

## Model Name Format Support

The system supports various model name formats:

### Provider/Model Format (Cline)
```
anthropic/claude-3-5-sonnet-20241022  → claude-3-5-sonnet-20241022
openai/gpt-4                          → gpt-4
meta-llama/Llama-3-8B-Instruct        → Llama-3-8B-Instruct
qwen/qwen2.5-coder-32b-instruct       → qwen2.5-coder-32b-instruct
```

### Direct Model Names
```
gpt-4
claude-3-haiku
llama-3-8b-instruct
qwen2.5-coder-7b
```

### Model Name Normalization

The system normalizes model names for consistent detection:

- Converts to lowercase
- Replaces underscores and spaces with hyphens
- Removes common suffixes like `-instruct`, `-chat`, `-preview`

## Architecture

### Core Components

1. **ModelTokenizerValidator**: Main validation class
2. **ModelContext**: Context management for tracking current models
3. **Model Detection**: Pattern-based model family detection
4. **Tokenizer Loading**: Dynamic tokenizer loading based on model family
5. **Fallback System**: Graceful degradation when specific tokenizers unavailable

### Validation Flow

1. **Model Detection**: Extract and normalize model name
2. **Family Mapping**: Map model to tokenizer family
3. **Tokenizer Loading**: Load appropriate tokenizer (with caching)
4. **Token Counting**: Use model-specific tokenization
5. **Fallback**: Use generic validation if model-specific fails

### Integration Points

The system integrates with:

- **compress.py**: Enhanced `validate_tokenization_efficiency` function
- **dynamic_dictionary.py**: Model-specific validation in `_validate_tokenization_efficiency`
- **server.py**: Automatic model context setting from API requests

## Performance Considerations

### Tokenizer Caching

Tokenizers are cached after first load to improve performance:

```python
# First call - loads tokenizer
result1 = validator.validate_token_efficiency(text1, compressed1, "gpt-4")

# Subsequent calls - uses cached tokenizer
result2 = validator.validate_token_efficiency(text2, compressed2, "gpt-4")
```

### Library Availability Checking

The system checks tokenizer library availability at startup:

```python
Available Libraries:
✅ tiktoken      Available
✅ transformers  Available  
❌ sentencepiece Not available
```

### Graceful Fallbacks

When model-specific validation fails, the system falls back gracefully:

1. Try model-specific tokenizer
2. Fall back to generic tiktoken validation
3. Fall back to character-based estimation
4. Return conservative estimate

## Debugging and Monitoring

### Debug Logging

Enable debug logging to see tokenizer selection:

```python
# In logs, you'll see:
# 🔧 Set model context: claude-3-5-sonnet (from anthropic/claude-3-5-sonnet-20241022)
# Model-specific token validation (claude) - Original: 45 tokens, Compressed: 32 tokens
```

### Validation Results

The validation returns detailed information:

```python
{
    "valid": True,
    "token_savings": 13,
    "original_tokens": 45,
    "compressed_tokens": 32,
    "token_ratio": 0.289,
    "method": "model_specific",
    "model_family": "claude",
    "tokenizer_type": "SentencePieceProcessor"
}
```

## Testing

Run the test suite to verify functionality:

```bash
python tests/test_model_tokenizer_validator.py
```

The test suite covers:

- Model family detection
- Tokenizer availability checking
- Basic validation with different models
- Model context management
- Provider format extraction
- Integration with compression system

## Troubleshooting

### Common Issues

1. **Missing Tokenizer Libraries**
   ```
   Warning: sentencepiece not available
   Solution: pip install sentencepiece
   ```

2. **Unknown Model Family**
   ```
   Warning: Unknown model family for: custom-model-name
   Solution: Add patterns to MODEL_PATTERNS or use fallback
   ```

3. **Tokenizer Loading Failures**
   ```
   Error: Failed to load tokenizer for qwen2.5-coder
   Solution: Ensure transformers library is installed and model name is correct
   ```

### Fallback Behavior

When model-specific validation fails, the system logs the fallback method used:

```
Model-specific validation failed for custom-model: No tokenizer found
Fallback tiktoken validation - Original: 45 tokens, Compressed: 32 tokens
```

## Future Enhancements

Potential improvements:

1. **Dynamic Model Loading**: Automatically download and cache model tokenizers
2. **Custom Tokenizer Registration**: Allow users to register custom tokenizers
3. **Performance Optimization**: Further optimize tokenizer loading and caching
4. **Extended Model Support**: Add support for more model families
5. **Tokenizer Benchmarking**: Compare tokenization efficiency across models

## Summary

The model-specific tokenizer validation system provides:

- **Accuracy**: Uses actual model tokenizers for precise validation
- **Flexibility**: Supports multiple model families and tokenizer types
- **Reliability**: Graceful fallbacks ensure system always works
- **Performance**: Caching and efficient loading for production use
- **Integration**: Seamless integration with existing compression pipeline

This ensures that compression validation is as accurate as possible for the specific model being used, leading to better compression decisions and more efficient token usage. 