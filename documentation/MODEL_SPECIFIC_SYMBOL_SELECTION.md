# Model-Specific Symbol Selection

## Overview

The KrunchWrapper system now features intelligent model-specific symbol selection that automatically chooses the most efficient compression symbols based on the target model's tokenizer architecture. This ensures optimal token compression for each specific model being used.

## Key Features

### 1. Unified Model-Specific Dictionary

- **Location**: `dictionaries/model_specific_symbols.json`
- **Structure**: Each symbol contains token counts for multiple model architectures
- **Coverage**: 3,750 symbols tested across 14+ model families
- **Data**: Includes best/worst models, average tokens, and source priority information

### 2. Automatic Model Detection

The system automatically detects model families including:
- **OpenAI**: GPT-4, GPT-3.5, GPT-3 (davinci, curie, babbage, ada)
- **Anthropic**: Claude models
- **Meta**: LLaMA, LLaMA-2, LLaMA-3, CodeLLaMA
- **Mistral**: Mistral, Mixtral
- **Qwen**: Qwen 1.x, 2.x, 3.x
- **Google**: Gemini, PaLM
- **Others**: Yi, DeepSeek, Phi, Falcon, etc.

### 3. Dynamic Symbol Prioritization

```python
# Example: Symbols optimized for different models
GPT-4:     ['¡', '¢', '£', '¤', '¥'] → 1 token each
GPT-3:     ['¢', '£', '§', '¨', '©'] → 1 token each
Qwen:      ['¡', '¢', '£', '¤', '¥'] → 1 token each
```

## Performance Results

### Model Efficiency Comparison
```
Model Family    | Avg Tokens | Single-Token Symbols | Total Symbols
GPT-4          | 2.73       | 65                   | 3,750
GPT-3.5        | 2.73       | 65                   | 3,750
Qwen           | 2.73       | 65                   | 3,750
GPT-3          | 2.78       | 72                   | 3,750
```

### Model-to-Model Comparison (GPT-4 vs GPT-3)
- **GPT-4 more efficient**: 226 symbols
- **GPT-3 more efficient**: 39 symbols
- **Equal performance**: 3,485 symbols
- **Average difference**: -0.05 tokens (GPT-4 slightly better)

## Configuration

### Enable/Disable Model-Specific Selection

In `config/config.jsonc`:
```jsonc
{
  "dynamic_dictionary": {
    "use_model_specific_symbols": true  // Enable model-specific selection
  }
}
```

### Fallback Behavior

1. **Primary**: Uses model-specific token counts from unified dictionary
2. **Secondary**: Falls back to average token counts if model not found
3. **Tertiary**: Uses hardcoded priority symbols if dictionary unavailable

## Usage Examples

### Programmatic Usage

```python
from core.model_specific_symbol_selector import ModelSpecificSymbolSelector

# Initialize selector
selector = ModelSpecificSymbolSelector()

# Get optimal symbols for a specific model
symbols = selector.get_symbols_for_model("gpt-4", max_symbols=10)
# Returns: [('¡', 1), ('¢', 1), ('£', 1), ...]

# Compare model efficiency
comparison = selector.compare_model_efficiency("gpt-4", "gpt-3")

# Get model statistics
stats = selector.get_model_efficiency_stats("gpt-4")
```

### Automatic Integration

The system automatically integrates with:
- **Dynamic Dictionary Analyzer**: Uses model-specific symbols for compression
- **Symbol Generator**: Prioritizes most efficient symbols for current model
- **Compression Pipeline**: Optimizes token usage based on model context

## Benefits

### 1. Optimized Token Usage
- Automatically selects symbols with lowest token count for target model
- Reduces token consumption by choosing model-specific efficient symbols
- Improves compression ratios through intelligent symbol selection

### 2. Model-Aware Compression
- Different symbol sets for different model architectures
- Accounts for tokenizer differences between model families
- Prevents using symbols that are inefficient for specific models

### 3. Future-Proof Design
- Easily extensible to new model families
- Unified dictionary structure supports any tokenizer
- Automatic fallback ensures compatibility

## Regenerating the Dictionary

To update the model-specific dictionary with new symbols or models:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install required tokenizer libraries
pip install tiktoken transformers sentencepiece

# Generate updated dictionary
python utils/generate_model_specific_dictionary.py
```

## Technical Implementation

### Dictionary Structure
```json
{
  "symbol": {
    "source": "priority_1",
    "token_counts": {
      "gpt-4": 1,
      "gpt-3": 2,
      "qwen": 1
    },
    "best_models": ["gpt-4", "qwen"],
    "worst_models": ["gpt-3"],
    "average_tokens": 1.33
  }
}
```

### Selection Algorithm
1. Detect model family from model name
2. Load unified dictionary with model-specific token counts
3. Sort symbols by token count for target model (ascending)
4. Return top N most efficient symbols
5. Fallback to average counts if model not found

## Testing

Run the test suite to verify functionality:
```bash
python utils/test_model_specific_selection.py
```

This will demonstrate:
- Symbol selection for different models
- Model efficiency comparisons
- Dynamic selection based on model context

## Conclusion

Model-specific symbol selection represents a significant advancement in compression efficiency, automatically tailoring symbol choice to the target model's tokenizer characteristics. This ensures optimal token usage across different AI model architectures while maintaining backward compatibility and extensibility. 