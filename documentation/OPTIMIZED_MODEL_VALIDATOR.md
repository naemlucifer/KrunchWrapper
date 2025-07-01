# Optimized Model Validator

The `OptimizedModelTokenizerValidator` provides enhanced performance through aggressive caching, batch operations, and performance monitoring for model-specific tokenization validation.

## Overview

The optimized validator extends the base `ModelTokenizerValidator` with the following enhancements:

- **Result Caching**: Identical validations are cached to avoid redundant tokenization
- **LRU Cache**: Model family detection uses LRU caching
- **Batch Operations**: Process multiple validations efficiently
- **Performance Tracking**: Monitor validation times and cache effectiveness
- **Thread Safety**: All operations are thread-safe with proper locking
- **Memory Management**: Automatic cache size limits and cleanup

## Quick Start

### Basic Usage

```python
from core.optimized_model_validator import get_optimized_validator

# Get the singleton optimized validator (default configuration)
validator = get_optimized_validator()

# Validate token efficiency (with caching)
result = validator.validate_token_efficiency_cached(
    original_text="def calculate_sum(a, b): return a + b",
    compressed_text="def calc_sum(a,b):return a+b",
    model_name="gpt-4"
)

print(f"Token savings: {result['token_savings']}")
print(f"Cache hit: {result.get('cache_hit', False)}")
```

### Configuration Options

The optimized validator supports several configuration options:

```python
# Default configuration (recommended)
validator = get_optimized_validator()

# Custom configuration
validator = get_optimized_validator(
    model_specific_cache=True,        # Enable model-specific caching (default: True)
    max_cache_size=2000,             # Increase cache size (default: 1000)
    max_validation_time_samples=200  # More performance samples (default: 100)
)

# Model-agnostic caching (faster but less accurate)
validator = get_optimized_validator(
    model_specific_cache=False,  # All models share same cache
    force_new=True              # Force new instance with this config
)
```

### Batch Processing

```python
# Process multiple validations efficiently
validations = [
    ("original_text_1", "compressed_text_1", "gpt-4"),
    ("original_text_2", "compressed_text_2", "claude-3-5-sonnet"),
    ("original_text_3", "compressed_text_3", "qwen2.5-coder")
]

results = validator.validate_batch(validations)
for result in results:
    print(f"Model: {result.get('model_family')}, Savings: {result['token_savings']}")
```

## Performance Benefits

### Caching Effectiveness

The validator caches validation results based on a hash of:
- Original text
- Compressed text  
- Normalized model name

This means identical validations return instantly from cache:

```python
# First call - cache miss, full validation
result1 = validator.validate_token_efficiency_cached(text1, text2, "gpt-4")
print(f"Time: {result1['validation_time']:.4f}s")  # e.g., 0.0045s

# Second identical call - cache hit, instant return
result2 = validator.validate_token_efficiency_cached(text1, text2, "gpt-4") 
print(f"Time: {result2.get('validation_time', 0):.4f}s")  # 0.0000s
print(f"Cache hit: {result2.get('cache_hit', True)}")  # True
```

### Performance Statistics

Monitor cache performance:

```python
stats = validator.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Average validation time: {stats['avg_validation_time']:.4f}s")
print(f"Cached validations: {stats['cached_validations']}")
```

## Integration Examples

### Replace Standard Validator

Replace the standard validator in your compression pipeline:

```python
# Instead of:
from core.model_tokenizer_validator import validate_with_model
efficiency = validate_with_model(original, compressed, model)

# Use optimized version:
from core.optimized_model_validator import validate_with_optimized_model
efficiency = validate_with_optimized_model(original, compressed, model)
```

### Context Integration

Use with the fast model context:

```python
from core.optimized_model_validator import get_fast_context, get_optimized_validator

# Set up fast context
fast_context = get_fast_context()
fast_context.set_context_batch({
    'global_model': 'gpt-4',
    'backup_model': 'claude-3-5-sonnet'
})

# Use effective model detection
validator = get_optimized_validator()
current_model = fast_context.get_effective_model_fast()
if current_model:
    result = validator.validate_token_efficiency_cached(
        original, compressed, current_model
    )
```

### Performance Monitoring

Use the performance monitor for detailed timing:

```python
from core.optimized_model_validator import PerformanceMonitor

with PerformanceMonitor("batch_validation") as monitor:
    results = validator.validate_batch(large_validation_set)
    
# Timing automatically logged
```

## Advanced Features

### Cache Management

Control cache behavior:

```python
# Clear specific caches
validator.clear_cache("validation")  # Clear validation results only
validator.clear_cache("tokenizer")   # Clear tokenizer cache only
validator.clear_cache("all")         # Clear everything

# Get detailed cache statistics
stats = validator.get_cache_stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

### Memory Optimization

The validator automatically manages memory:

- **Cache Size Limit**: Validation cache limited to 1000 entries
- **FIFO Replacement**: Oldest entries removed when limit reached
- **Weak References**: Temporary objects cleaned up automatically
- **Stale Cleanup**: Remove unused contexts after 1 hour

```python
# Manual cleanup of stale contexts
fast_context = get_fast_context()
removed = fast_context.cleanup_stale_contexts(max_age_seconds=1800)  # 30 minutes
print(f"Removed {removed} stale contexts")
```

### Comprehensive Statistics

Get system-wide performance data:

```python
from core.optimized_model_validator import get_comprehensive_stats

stats = get_comprehensive_stats()
print("Validator Stats:")
for key, value in stats['validator_stats'].items():
    print(f"  {key}: {value}")
    
print("Context Stats:")
for key, value in stats['context_stats'].items():
    print(f"  {key}: {value}")
```

## Caching Modes

### Model-Specific Caching (Default - Recommended)

**Enabled by default** - Each model gets its own cache entries:

```python
validator = get_optimized_validator(model_specific_cache=True)  # Default

# These create separate cache entries (correct behavior)
result1 = validator.validate_token_efficiency_cached(text, compressed, "gpt-4")
result2 = validator.validate_token_efficiency_cached(text, compressed, "claude-3-5-sonnet")
```

**Pros:**
- ✅ **Accurate**: Different models use different tokenizers
- ✅ **Correct**: GPT-4 tokens ≠ Claude tokens for same text
- ✅ **Reliable**: Compression decisions based on actual model behavior

**Cons:**
- ⚠️ Lower cache hit rate when switching between models
- ⚠️ Higher memory usage with many different models

### Model-Agnostic Caching (Advanced Use)

**Use with caution** - All models share the same cache:

```python
validator = get_optimized_validator(model_specific_cache=False, force_new=True)

# These share the same cache entry (faster but potentially incorrect)
result1 = validator.validate_token_efficiency_cached(text, compressed, "gpt-4")       # Cache miss
result2 = validator.validate_token_efficiency_cached(text, compressed, "claude")     # Cache hit!
```

**Pros:**
- ✅ Higher cache hit rates in multi-model environments
- ✅ Lower memory usage
- ✅ Faster when switching between models frequently

**Cons:**
- ❌ **Less accurate**: Uses tokenization results from whatever model was cached first
- ❌ **Potentially incorrect**: May make wrong compression decisions
- ❌ **Misleading**: Results don't reflect actual model behavior

### When to Use Each Mode

| Scenario | Recommended Mode | Reason |
|----------|------------------|---------|
| **Production systems** | Model-specific (default) | Accuracy and correctness are critical |
| **Single model family** | Model-specific (default) | No downside, full benefits |
| **Multi-model environments** | Model-specific (default) | Slight performance cost for much better accuracy |
| **Memory-constrained systems** | Model-agnostic | Reduce memory usage when accuracy is less critical |
| **Approximate validation** | Model-agnostic | When speed matters more than precision |

## Configuration Options

### Cache Size Tuning

Modify cache sizes for your use case:

```python
# Access validator internals for custom configuration
validator = get_optimized_validator()

# Change validation cache limit (default: 1000)
validator._validation_cache_limit = 2000

# Change performance tracking samples (default: 100)
validator._max_validation_time_samples = 200
```

### Threading Configuration

The validator is thread-safe by default, but you can optimize for single-threaded use:

```python
# For single-threaded applications, you can disable some locking overhead
# by using the validator directly without the singleton pattern
from core.optimized_model_validator import OptimizedModelTokenizerValidator

validator = OptimizedModelTokenizerValidator()
# Use validator in single-threaded context
```

## Best Practices

### 1. Use Singleton Instances

Always use the singleton accessors for best performance:

```python
# Good - uses singleton
validator = get_optimized_validator()

# Avoid - creates new instance each time
validator = OptimizedModelTokenizerValidator()
```

### 2. Batch Similar Operations

Group validations for the same model when possible:

```python
# Efficient - batch processing
gpt4_validations = [(orig, comp, "gpt-4") for orig, comp in gpt4_pairs]
results = validator.validate_batch(gpt4_validations)
```

### 3. Monitor Cache Performance

Regular monitoring helps optimize usage patterns:

```python
# Log cache stats periodically
import logging
stats = validator.get_cache_stats()
logging.info(f"Cache hit rate: {stats['hit_rate']:.2%}")

# Clear caches if hit rate is low
if stats['hit_rate'] < 0.3:
    validator.clear_cache("validation")
```

### 4. Handle Model Name Variations

The validator normalizes model names, but be consistent:

```python
# These are treated as the same model due to normalization
result1 = validator.validate_token_efficiency_cached(text1, text2, "gpt-4")
result2 = validator.validate_token_efficiency_cached(text1, text2, "openai/gpt-4")  # Cache hit
```

## Error Handling

The optimized validator maintains the same error handling as the base validator:

```python
try:
    result = validator.validate_token_efficiency_cached(
        original_text, compressed_text, "unknown-model"
    )
    if not result['valid']:
        print("Validation failed - compression may not be beneficial")
except Exception as e:
    print(f"Validation error: {e}")
    # Fallback to character-based estimation
```

## Performance Expectations

Typical performance improvements with the optimized validator:

- **Cache Hits**: 95%+ faster than standard validator
- **Batch Processing**: 20-40% faster than individual calls
- **Memory Usage**: Stable with automatic cleanup
- **Thread Safety**: No performance penalty in multi-threaded use

### Benchmarking

Run the provided benchmark script to measure performance on your system:

```bash
cd utils
python test_optimized_validator.py
```

Example output:
```
=== Results ===
Standard validator average: 0.0234s
Optimized validator (cached) average: 0.0012s
Batch validation average: 0.0089s

Performance improvements:
Cached validation: 94.9% faster
Batch validation: 62.0% faster
```

## Migration Guide

### From Standard Validator

1. **Import Changes**:
   ```python
   # Old
   from core.model_tokenizer_validator import get_model_tokenizer_validator
   
   # New  
   from core.optimized_model_validator import get_optimized_validator
   ```

2. **Method Changes**:
   ```python
   # Old
   validator = get_model_tokenizer_validator()
   result = validator.validate_token_efficiency(original, compressed, model)
   
   # New
   validator = get_optimized_validator()
   result = validator.validate_token_efficiency_cached(original, compressed, model)
   ```

3. **Additional Features**:
   ```python
   # New capabilities
   stats = validator.get_cache_stats()
   results = validator.validate_batch(validations)
   validator.clear_cache("all")
   ```

### Gradual Adoption

You can use both validators simultaneously during migration:

```python
from core.model_tokenizer_validator import get_model_tokenizer_validator
from core.optimized_model_validator import get_optimized_validator

# Use optimized for frequently repeated validations
optimized = get_optimized_validator()

# Keep standard for one-off validations if preferred
standard = get_model_tokenizer_validator()
```

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**:
   - Check if model names are consistent
   - Verify text normalization isn't affecting cache keys
   - Monitor with `get_cache_stats()`

2. **Memory Usage**:
   - Cache limits are automatic but can be adjusted
   - Use `clear_cache()` periodically if needed
   - Monitor with `get_comprehensive_stats()`

3. **Thread Safety**:
   - All operations are thread-safe by default
   - Performance scales well with concurrent access

### Debug Mode

Enable debug logging to monitor cache behavior:

```python
import logging
logging.getLogger('core.optimized_model_validator').setLevel(logging.DEBUG)

# Now cache hits/misses will be logged
result = validator.validate_token_efficiency_cached(text1, text2, "gpt-4")
```

## API Reference

### OptimizedModelTokenizerValidator

| Method | Description |
|--------|-------------|
| `__init__(base_validator, model_specific_cache, max_cache_size, max_validation_time_samples)` | Initialize with configuration options |
| `validate_token_efficiency_cached()` | Cached version of token validation |
| `validate_batch()` | Process multiple validations efficiently |
| `get_cache_stats()` | Get cache performance statistics |
| `clear_cache()` | Clear specific or all caches |
| `detect_model_family_cached()` | Cached model family detection |

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_validator` | `ModelTokenizerValidator` | `None` | Base validator instance |
| `model_specific_cache` | `bool` | `True` | Enable model-specific caching |
| `max_cache_size` | `int` | `1000` | Maximum validation cache entries |
| `max_validation_time_samples` | `int` | `100` | Maximum performance samples to track |

### FastModelContext

| Method | Description |
|--------|-------------|
| `set_context_batch()` | Set multiple contexts at once |
| `get_effective_model_fast()` | Fast model retrieval |
| `cleanup_stale_contexts()` | Remove old context entries |
| `get_context_stats()` | Get context performance statistics |

### Utility Functions

| Function | Description |
|----------|-------------|
| `get_optimized_validator(model_specific_cache, max_cache_size, max_validation_time_samples, force_new)` | Get singleton validator instance with optional configuration |
| `get_fast_context()` | Get singleton context instance |
| `validate_with_optimized_model()` | Drop-in replacement wrapper |
| `clear_all_caches()` | Clear all optimization caches |
| `get_comprehensive_stats()` | Get system-wide statistics |

#### get_optimized_validator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_specific_cache` | `bool` | `None` | Override model-specific caching setting |
| `max_cache_size` | `int` | `None` | Override maximum cache size |
| `max_validation_time_samples` | `int` | `None` | Override performance sample limit |
| `force_new` | `bool` | `False` | Force creation of new instance | 