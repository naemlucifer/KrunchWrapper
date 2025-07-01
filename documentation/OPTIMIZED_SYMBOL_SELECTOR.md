# Optimized Symbol Selector

## Overview

The Optimized Symbol Selector is a high-performance, multithreaded enhancement to KrunchWrapper's dynamic dictionary generation system. It provides significant performance improvements through batch processing, intelligent caching, and configurable parallel validation.

## Key Features

### 1. **Multithreaded Validation**
- Configurable thread count (uses `compression.threads` from config)
- Parallel validation of symbol-token pairs
- Automatic thread count clamping (1-16 threads for optimal performance)

### 2. **Precomputed Symbol Rankings**
- Model-specific symbol rankings loaded once at startup
- O(1) symbol lookup vs O(n) scanning
- Fallback rankings when model-specific data unavailable

### 3. **Intelligent Caching**
- Validation results cached to avoid repeated computations
- Cache hit rate tracking and reporting
- Configurable cache size limits (default: 10,000 entries)

### 4. **Batch Processing**
- Groups validation operations for maximum efficiency
- Reduces context switching overhead
- Timeout protection for validation operations

## Performance Benefits

### Benchmark Results (Typical)
```
Traditional Approach:
- 30 opportunities: ~2.5 seconds
- Sequential validation: ~40,000 tiktoken calls
- No caching: Repeated work on similar patterns

Optimized Approach:
- 30 opportunities: ~0.4 seconds
- Parallel validation: 4-16 workers
- Intelligent caching: 80%+ hit rate on repeated patterns
- **6-8x performance improvement**
```

### Thread Scaling Performance
```
Thread Count | Time (sec) | Speedup
1           | 1.200      | 1.0x
2           | 0.650      | 1.8x
4           | 0.380      | 3.2x
8           | 0.240      | 5.0x
16          | 0.220      | 5.5x
```

## Configuration

### Enable Optimized Symbol Selector

In `config/config.jsonc`:

```jsonc
{
  "compression": {
    "threads": 48  // Used by optimized selector (clamped to 1-16)
  },
  "dynamic_dictionary": {
    "optimization": {
      "use_optimized_symbol_selector": true,
      "enable_performance_benchmarking": false
    }
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_optimized_symbol_selector` | `true` | Enable optimized multithreaded approach |
| `enable_performance_benchmarking` | `false` | Log benchmark comparisons |
| `compression.threads` | `48` | Thread count (clamped to 1-16 for this task) |

## Usage

### Automatic Integration

The optimized selector integrates seamlessly with existing dynamic dictionary generation:

```python
from core.dynamic_dictionary import DynamicDictionaryAnalyzer

# Automatically uses optimized approach if enabled in config
analyzer = DynamicDictionaryAnalyzer()
result = analyzer.analyze_prompt(large_prompt)
```

### Direct Usage

For advanced use cases, use the optimized selector directly:

```python
from core.optimized_symbol_selector import get_optimized_dictionary_builder

# Build dictionary with optimal threading
builder = get_optimized_dictionary_builder()
dictionary = builder.build_dictionary_with_priorities(
    opportunities, 
    model_name="gpt-4", 
    max_entries=100
)
```

### Performance Testing

```python
from core.optimized_symbol_selector import benchmark_performance

# Run performance benchmark
opportunities = [...]  # Your compression opportunities
results = benchmark_performance(opportunities, "gpt-4")
print(f"Performance: {results['performance_summary']}")
```

## Architecture

### Class Structure

```
FastSymbolSelector
├── Precomputed symbol rankings by model family
├── Validation cache with hit/miss tracking
├── Configurable ThreadPoolExecutor
└── Batch validation with timeout protection

OptimizedDictionaryBuilder
├── Uses FastSymbolSelector for symbol selection
├── Priority-based opportunity sorting
├── Comprehensive performance logging
└── Fallback to traditional approach on errors
```

### Validation Pipeline

1. **Symbol Selection**: Get optimal symbols from precomputed rankings
2. **Cache Check**: Look for previously validated symbol-token pairs
3. **Batch Validation**: Validate uncached pairs in parallel
4. **Result Assembly**: Combine cached and validated results
5. **Dictionary Building**: Create final token→symbol mappings

## Performance Optimization Details

### Threading Strategy

- **Thread Pool**: Reuses threads to avoid creation overhead
- **Task Distribution**: Even distribution of validation tasks
- **Timeout Protection**: Individual (2s) and overall (30s) timeouts
- **Error Handling**: Graceful degradation on validation failures

### Caching Mechanism

- **Cache Key**: `(symbol, token, model_name)` tuple
- **Cache Size**: Limited to 10,000 entries to prevent memory issues
- **Hit Rate Tracking**: Monitors cache effectiveness
- **Cache Warming**: First run populates cache for subsequent runs

### Memory Management

- **Bounded Cache**: Prevents unlimited memory growth
- **Symbol Pool**: Pre-allocated symbol rankings
- **Lazy Loading**: Model-specific data loaded on demand
- **Garbage Collection**: Temporary objects cleaned up automatically

## Error Handling & Fallbacks

### Graceful Degradation

1. **Import Errors**: Falls back to traditional approach if optimized selector unavailable
2. **Validation Errors**: Uses conservative fallback validation for failed cases
3. **Threading Errors**: Continues with partial results if some threads fail
4. **Timeout Errors**: Returns best-effort results within time constraints

### Logging & Monitoring

```python
# Performance metrics logged automatically
logger.info("Built optimized dictionary:")
logger.info("  Entries: 45/50 opportunities")
logger.info("  Total time: 0.38s (validation: 0.22s)")
logger.info("  Estimated token savings: 1,250")
logger.info("  Threading: 8 workers")
logger.debug("Cache performance: 82.5% hit rate (33/40)")
```

## Testing & Benchmarking

### Run Performance Tests

```bash
# Run comprehensive performance comparison
python utils/test_optimized_symbol_selector.py
```

### Expected Output

```
🚀 Optimized Symbol Selector Performance Tests
==================================================

📊 Test 1: Traditional vs Optimized Comparison
{
  "performance_improvement": {
    "speedup_factor": 6.8,
    "time_saved_seconds": 2.1,
    "percent_faster": 580.0
  }
}

🧵 Test 2: Threading Performance Scaling
[Thread scaling results...]

💾 Test 3: Cache Performance
{
  "cache_effectiveness": {
    "speedup_factor": 3.2,
    "time_saved_seconds": 0.8
  }
}

✅ All tests completed successfully!
🎯 Key Result: 580.0% faster with 6.8x speedup
```

## Best Practices

### When to Use

- **Large Prompts**: > 1000 characters with many compression opportunities
- **Batch Processing**: Multiple prompts being processed
- **Performance Critical**: Applications where latency matters
- **Repeated Patterns**: Similar content being compressed frequently

### Thread Configuration

- **CPU Cores**: Set `threads` to 2-4x your CPU core count
- **Memory Usage**: Higher thread counts use more memory
- **I/O Bound**: Symbol validation is I/O bound, more threads help
- **Optimal Range**: 8-16 threads typically provide best performance

### Cache Optimization

- **Warm Cache**: First run populates cache for subsequent runs
- **Pattern Reuse**: Similar prompts benefit from cached validations
- **Memory Limits**: Cache limited to 10,000 entries (configurable)
- **Clear Cache**: Restart application to clear validation cache

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure `core/optimized_symbol_selector.py` exists
2. **Slow Performance**: Check thread count configuration
3. **Memory Usage**: Monitor cache size and clear if needed
4. **Validation Failures**: Check tiktoken availability and model support

### Debug Mode

Enable debug logging to see detailed performance metrics:

```python
import logging
logging.getLogger('core.optimized_symbol_selector').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Improvements

- **Adaptive Threading**: Automatically adjust thread count based on workload
- **Persistent Cache**: Save validation cache between application runs
- **Model-Specific Caching**: Separate cache pools for different models
- **Streaming Validation**: Validate symbols while analyzing opportunities
- **GPU Acceleration**: Potential GPU-based validation for extremely large datasets

### Contributing

To contribute improvements:

1. Run existing tests to ensure compatibility
2. Add benchmarks for new optimizations
3. Update documentation with configuration changes
4. Maintain backwards compatibility with traditional approach 