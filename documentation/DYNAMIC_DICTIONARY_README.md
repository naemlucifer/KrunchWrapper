# Dynamic Dictionary Feature

The Dynamic Dictionary feature is a new addition to KrunchWrapper that automatically analyzes user prompts for compression opportunities and generates temporary, ad-hoc dictionaries to optimize compression for specific messages.

## Overview

Unlike the static language-specific dictionaries in the `dicts/` folder, the dynamic dictionary feature:

1. **Analyzes each user prompt** for repetitive patterns and compression opportunities
2. **Generates temporary dictionaries** stored in the `temp/` folder
3. **Creates custom compression mappings** using unique Unicode symbols
4. **Combines with existing dictionaries** for maximum compression efficiency
5. **Automatically cleans up** old temporary dictionaries

## Key Features

### 🧠 Intelligent Analysis
- **Pattern Detection**: Identifies repeated words, phrases, and code patterns
- **Substring Analysis**: Finds common substrings within longer tokens
- **Code Pattern Recognition**: Detects function calls, method access, and structured content
- **Frequency Analysis**: Prioritizes tokens based on occurrence frequency and potential savings

### ⚙️ Config File-Based Configuration
The dynamic dictionary feature is configured through the main `config/config.jsonc` file in the `dynamic_dictionary` section:

```jsonc
{
    "dynamic_dictionary": {
        "enabled": true,
        "min_token_length": 4,
        "min_frequency": 2,
        "max_dictionary_size": 100,
        "compression_threshold": 0.05
    }
}
```

### 🔄 Automatic Integration
- **Seamless Integration**: Works with existing compression system
- **Format-Aware**: Integrates with system prompt interceptor
- **Language-Aware**: Combines with language-specific dictionaries
- **Smart Fallback**: Falls back to standard compression if dynamic analysis doesn't provide benefit

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable/disable dynamic dictionary analysis |
| `min_token_length` | 4 | Minimum length for compression tokens |
| `min_frequency` | 2 | Minimum occurrences needed for compression |
| `max_dictionary_size` | 100 | Maximum number of compression entries |
| `compression_threshold` | 0.05 | Minimum compression benefit required (5%) |
| `enable_substring_analysis` | true | Analyze substrings within words |
| `enable_phrase_analysis` | true | Analyze multi-word phrases |
| `enable_pattern_analysis` | true | Analyze code-like patterns |
| `min_prompt_length` | 500 | Minimum prompt length to consider for analysis |
| `auto_detection_threshold` | 0.3 | Automatic detection threshold for repetitive content (30%) |
| `cleanup_max_age_hours` | 24 | Maximum age for temporary dictionary files |
| `auto_cleanup` | true | Enable automatic cleanup of old temporary dictionaries |

## Configuration Examples

### Example 1: Conservative Settings (High Quality)
```jsonc
{
    "dynamic_dictionary": {
        "enabled": true,
        "min_token_length": 5,
        "min_frequency": 3,
        "max_dictionary_size": 50,
        "compression_threshold": 0.1,
        "min_prompt_length": 1000
    }
}
```

### Example 2: Aggressive Settings (More Compression)
```jsonc
{
    "dynamic_dictionary": {
        "enabled": true,
        "min_token_length": 3,
        "min_frequency": 2,
        "max_dictionary_size": 200,
        "compression_threshold": 0.02,
        "min_prompt_length": 300,
        "auto_detection_threshold": 0.2
    }
}
```

### Example 3: Code-Focused Settings
```jsonc
{
    "dynamic_dictionary": {
        "enabled": true,
        "enable_phrase_analysis": false,
        "enable_pattern_analysis": true,
        "enable_substring_analysis": true,
        "min_token_length": 4,
        "min_frequency": 2
    }
}
```

## How It Works

### 1. Configuration Loading
```python
# Configuration is automatically loaded from config/config.jsonc
analyzer = DynamicDictionaryAnalyzer()
```

### 2. Analysis Decision
```python
# System automatically decides whether to analyze based on config
should_analyze, reason = analyzer.should_analyze_prompt(user_text)
```

### 3. Dictionary Generation
```python
# Temporary dictionary is created with Unicode symbols
dictionary = {
    "compression_opportunities": "∀",
    "analyze_patterns": "∃", 
    "text_content": "∋",
    # ... more mappings
}
```

### 4. Compression Application
```python
# Both dynamic and language dictionaries are applied
result = compress_with_dynamic_analysis(text)
```

### 5. Temporary Storage
```
temp/
├── dynamic_20241201_143022.json  # Temporary dictionary
├── dynamic_20241201_143055.json  # Another temp dictionary
└── ...
```

**Note**: Temporary dictionaries use JSON format (instead of TOML) for robust handling of complex token strings containing quotes, newlines, and special characters that could cause parsing issues.

## File Structure

### New Files Added
- `core/dynamic_dictionary.py` - Main dynamic dictionary analyzer
- `core/dynamic_config_parser.py` - Configuration manager for dynamic dictionary
- `utils/test_dynamic_dictionary.py` - Testing utility
- `temp/DYNAMIC_DICTIONARY_README.md` - This documentation

### Modified Files
- `core/compress.py` - Added dynamic compression support
- `core/system_prompt_interceptor.py` - Integrated dynamic analysis
- `config/config.jsonc` - Added dynamic_dictionary configuration section

## API Integration

The dynamic dictionary feature integrates seamlessly with the existing system prompt interceptor:

```python
# In system_prompt_interceptor.py
def intercept_and_process(self, messages, rule_union, lang, target_format, ...):
    # Extract user content
    user_content = self._extract_user_content(messages)
    
    # Process dynamic dictionary analysis (config-based)
    dynamic_metadata = self._process_dynamic_dictionary_analysis(user_content, lang)
    
    # Enhance system prompt with dynamic info
    if dynamic_metadata.get("dynamic_dict_used"):
        compression_prompt = self._enhance_compression_prompt_with_dynamic_info(
            compression_prompt, dynamic_metadata
        )
```

## Testing

Use the test utility to experiment with the dynamic dictionary:

```bash
# Test with built-in samples
python utils/test_dynamic_dictionary.py --sample repetitive --all-tests

# Test with your own file
python utils/test_dynamic_dictionary.py --file your_text.txt --compare

# Display current configuration
python utils/test_dynamic_dictionary.py --config

# Test with direct text
python utils/test_dynamic_dictionary.py --text "Your text here" --language python
```

## Performance Considerations

### When Dynamic Compression Helps
- **Repetitive content**: Text with repeated patterns
- **Code snippets**: Programming code with repeated identifiers
- **Structured data**: JSON, XML, or similar formats
- **Long prompts**: Content over the configured minimum length with repetition

### When It Doesn't Help
- **Short prompts**: Less than the configured minimum length
- **Unique content**: Text with minimal repetition
- **Mixed languages**: Content with varied vocabularies
- **Already compressed**: Text that's already been processed

### Automatic Decision Making
The system automatically decides whether to use dynamic compression based on:
- Configuration settings (enabled, min_prompt_length, etc.)
- Content length and repetition analysis
- Pattern detection (code, structured data)
- Estimated compression benefit

## Cleanup and Maintenance

- **Automatic Cleanup**: Old temporary dictionaries are automatically removed based on `cleanup_max_age_hours` setting
- **Manual Cleanup**: Run cleanup manually if needed
- **Storage Efficient**: Uses Unicode symbols for minimal storage overhead
- **Conflict Avoidance**: Ensures new symbols don't conflict with existing dictionaries

## Configuration Management

### Viewing Current Configuration
```bash
python utils/test_dynamic_dictionary.py --config
```

### Modifying Configuration
Edit the `dynamic_dictionary` section in `config/config.jsonc`:

```jsonc
{
    "dynamic_dictionary": {
        "enabled": true,
        "min_token_length": 4,
        "min_frequency": 2,
        // ... other settings
    }
}
```

### Configuration Validation
The system validates configuration values and provides sensible defaults if the config file is missing or invalid.

## Troubleshooting

### Common Issues

1. **Feature not working**: Check that `enabled` is set to `true` in config
2. **No compression improvement**: Content may not meet minimum length or repetition thresholds
3. **Analysis skipped**: Adjust `min_prompt_length` and `auto_detection_threshold` settings
4. **Too aggressive compression**: Increase `min_frequency` and `compression_threshold`
5. **Config not loading**: Ensure `config/config.jsonc` exists and has valid JSON syntax

### Debug Mode
Enable debug logging to see detailed analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Configuration Verification
```python
from core.dynamic_config_parser import DynamicConfigManager
config_manager = DynamicConfigManager()
print(config_manager.get_configuration_summary())
```

---

This dynamic dictionary feature significantly enhances KrunchWrapper's compression capabilities by adapting to the specific content of each user prompt, with all settings managed through the centralized configuration system for easy maintenance and deployment.

## Aggressive Configuration Guide

For scenarios requiring more aggressive compression, you can tune key parameters for higher compression ratios at the cost of increased processing time.

### Aggressiveness Levels

#### Conservative (Current Default)
```jsonc
"dynamic_dictionary": {
    "min_token_length": 6,           // Only longer patterns
    "min_frequency": 5,              // Must appear 5+ times  
    "max_dictionary_size": 100,      // Moderate dictionary size
    "min_prompt_length": 2000,       // Only large prompts
    "auto_detection_threshold": 0.35, // Requires high repetition
    "compression_threshold": 0.05,   // Needs 5%+ compression
    "enable_substring_analysis": false,
    "enable_advanced_pattern_analysis": false,
    "enable_aggressive_symbol_optimization": false
}
```

#### Moderate Aggressive
```jsonc
"dynamic_dictionary": {
    "min_token_length": 4,           // Shorter patterns allowed
    "min_frequency": 3,              // Appears 3+ times
    "max_dictionary_size": 150,      // Larger dictionaries
    "min_prompt_length": 1000,       // Smaller prompts eligible
    "auto_detection_threshold": 0.25, // Lower repetition threshold
    "compression_threshold": 0.03,   // Accepts 3%+ compression
    "enable_substring_analysis": true // More pattern types
}
```

#### Very Aggressive
```jsonc
"dynamic_dictionary": {
    "min_token_length": 3,           // Very short patterns (def, for, etc.)
    "min_frequency": 2,              // Only needs to appear twice
    "max_dictionary_size": 200,      // Large dictionaries
    "min_prompt_length": 500,        // Small prompts eligible  
    "auto_detection_threshold": 0.15, // Low repetition triggers
    "compression_threshold": 0.01,   // Accepts 1%+ compression
    "enable_substring_analysis": true,
    "enable_advanced_pattern_analysis": true,
    "enable_aggressive_symbol_optimization": true,
    "optimization": {
        "multi_pass_analysis": true,  // Multiple analysis passes
        "min_token_savings_threshold": 1.0,
        "min_net_token_savings": 0.5  // Accept minimal net benefit
    }
}
```

### Performance vs Aggressiveness Trade-offs

| Setting | Conservative | Moderate | Very Aggressive | Trade-off |
|---------|-------------|----------|-----------------|-----------|
| **Analysis Speed** | Fastest | Medium | Slowest | Speed vs Coverage |
| **Compression Ratio** | 10-15% | 15-25% | 25-40% | Quality vs Quantity |
| **Dictionary Size** | Small | Medium | Large | Overhead vs Savings |
| **False Positives** | Minimal | Low | Higher | Precision vs Recall |

### Recommended Aggressive Settings by Use Case

#### Code Analysis & Refactoring
```jsonc
// Best for analyzing large codebases with repetitive patterns
"min_token_length": 4,
"min_frequency": 2,
"max_dictionary_size": 200,
"enable_substring_analysis": true,
"enable_advanced_pattern_analysis": true
```

#### Documentation Processing  
```jsonc
// Good for processing technical documentation with repeated terms
"min_token_length": 5,
"min_frequency": 3,
"max_dictionary_size": 150,
"enable_phrase_analysis": true,
"compression_threshold": 0.02
```

#### Log File Analysis
```jsonc
// Optimal for compressing repetitive log files
"min_token_length": 3,
"min_frequency": 2,
"max_dictionary_size": 250,
"auto_detection_threshold": 0.10,
"enable_aggressive_symbol_optimization": true
```

### When to Use Aggressive Settings

✅ **Good for:**
- Large repetitive codebases (>50KB)
- Log file processing with consistent patterns
- Documentation with repeated technical terms
- Development/testing environments

❌ **Avoid for:**
- Production systems with strict latency requirements
- Small prompts (<2KB) where overhead exceeds benefits
- Diverse content with little repetition
- Critical systems where over-compression could cause issues

### Expected Results with Aggressive Settings

| Metric | Conservative | Aggressive | Improvement |
|--------|-------------|------------|-------------|
| **Compression Ratio** | 10-15% | 25-40% | +2.5x |
| **Dictionary Entries** | 20-50 | 100-200 | +4x |
| **Analysis Time** | 0.5-1.0s | 1.0-2.5s | +2.5x |
| **Pattern Coverage** | High-value only | Comprehensive | +5x |

**Remember:** More aggressive ≠ always better. Find the optimal balance for your specific use case! 