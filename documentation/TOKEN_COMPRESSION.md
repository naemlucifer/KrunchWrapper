# Token-Level Compression

This document explains the token-level compression system implemented in KrunchWrapper.

## Overview

The token-level compression system evaluates each potential substitution independently based on its estimated token savings. This approach ensures that only substitutions that actually save tokens are applied, unlike the standard compression which applies all substitutions from the dictionary.

## Key Features

- **Token-by-Token Evaluation**: Each substitution is evaluated independently
- **Multithreaded Processing**: Parallel analysis of substitutions for better performance
- **Configurable Parameters**: Adjustable thresholds for token savings, occurrences, etc.
- **Aggressive Mode**: Special mode for large files that prioritizes character savings
- **Smart Token Estimation**: Heuristics to estimate token counts for different patterns

## How It Works

1. **Load Dictionary**: Load the language-specific substitution dictionary
2. **Parallel Analysis**: Split substitutions into batches and analyze in parallel
3. **Evaluate Each Substitution**: For each potential substitution:
   - Count occurrences in the text
   - Estimate token savings
   - Apply only if it meets the configured thresholds
4. **Apply Substitutions**: Apply beneficial substitutions in order of highest savings

## Token Estimation

The system uses heuristics to estimate token counts:

- Common programming keywords are usually 1 token
- CamelCase and snake_case often split at boundaries
- Special characters often cause token splits
- Unicode characters (used for substitutions) are often 1 token each

## Configuration

Token-level compression is highly configurable through `config/config.jsonc`:

```json
{
    "compression": {
        "threads": 4,
        "min_token_savings": 0,
        "min_occurrences": 3,
        "min_compression_ratio": 0.05,
        "use_token_compression": true,
        "aggressive_mode": false,
        "large_file_threshold": 5000
    }
}
```

### Options

- `threads`: Number of threads for parallel processing
- `min_token_savings`: Minimum token savings required for a substitution
- `min_occurrences`: Minimum times a token must appear to be considered
- `min_compression_ratio`: Minimum overall compression ratio required
- `use_token_compression`: Whether to use token-level compression
- `aggressive_mode`: Whether to use aggressive mode for large files
- `large_file_threshold`: Size threshold for automatic aggressive mode

## Aggressive Mode

Aggressive mode is designed for large files:

- Uses a lower occurrence threshold (2 instead of configured value)
- Prioritizes character savings over token savings
- Automatically activates for files larger than the threshold
- May achieve better compression ratios for large files

## Testing

You can test and compare compression methods using the available test utilities:

```bash
# Test enhanced compression with current dictionaries
python tests/test_enhanced_compression.py

# Debug compression behavior and see what's happening
python utils/debug_compression.py

# Test model-specific tokenizer validation
python tests/test_model_tokenizer_validator.py

# Test dynamic compression analysis
python utils/test_dynamic_dictionary.py --sample repetitive --all-tests

# Test token boundary calculation
python utils/test_optimized_compression.py
```

## Performance

Token-level compression generally produces less compression than standard compression but is more token-efficient. The standard compression may achieve better character reduction but could potentially use more tokens due to inefficient substitutions.

For very large files, aggressive mode can improve compression ratios while still being more token-efficient than standard compression. 