# Persistent Token Cache

## Overview

The KrunchWrapper system now includes a persistent token caching mechanism that saves cached tokens to disk and loads them lazily when needed after server restarts. This improves performance by avoiding re-tokenization of the same text patterns across server sessions.

## How It Works

### 🗄️ Cache Storage

**Location**: `temp/` directory (created automatically)
**Format**: JSON files with MD5-hashed filenames for safe filesystem storage
**Structure**: Each cache file contains:
```json
{
  "key": "original_cache_key",
  "value": "cached_result", 
  "timestamp": 1234567890.123,
  "prefix": "cache_type"
}
```

### 🔄 Cache Lifecycle

1. **Startup**: Server automatically cleans expired cache files and initializes cache system
2. **Runtime**: 
   - Checks RAM cache first (fastest)
   - Falls back to disk cache if not in RAM (lazy loading)
   - Saves new results to both RAM and disk
3. **Memory Management**: LRU eviction keeps RAM cache bounded while keeping disk cache intact

### ⚡ Performance Benefits

- **Token Count Caching**: Avoids re-tokenizing identical text patterns
- **Symbol Efficiency Caching**: Stores pre-computed symbol tokenization tests
- **Cross-Session Persistence**: Cache survives server restarts
- **Lazy Loading**: Only loads disk cache entries when actually needed

## Cache Types

### 1. Token Count Cache
- **Prefix**: `token_count`
- **Purpose**: Cache tokenizer results for text strings
- **Key Format**: `tokenizer:{model_name}:{text_content}`
- **Location**: `core/dynamic_dictionary.py:get_token_count()`

### 2. Symbol Efficiency Cache  
- **Prefix**: `symbol_efficiency`
- **Purpose**: Cache symbol tokenization efficiency in different contexts
- **Key Format**: `symbol_test:{symbol}:{contexts}`
- **Location**: `core/model_optimized_symbol_pool.py:test_symbol_in_contexts()`

## Configuration

### Cache Settings
```python
# Default settings (automatically configured)
temp_dir = "temp"                 # Cache directory
max_ram_entries = 1000            # Max entries in RAM
cache_ttl_hours = 24              # Cache expiration time
```

### Environment Integration
- Cache initializes automatically on server startup
- No manual configuration required
- Gracefully degrades if cache directory unavailable

## Cache Statistics

Access cache performance via the persistent cache instance:

```python
from core.persistent_token_cache import get_persistent_cache

cache = get_persistent_cache()
stats = cache.get_stats()
print(f"RAM entries: {stats['ram_entries']}")
print(f"Disk files: {stats['disk_files']}")  
print(f"Hit rate: {stats['hit_rate_percent']}%")
```

Example output:
```
🗄️ Persistent token cache initialized
  📁 Cache directory: temp
  📊 Cache files: 247, RAM entries: 156
  🔄 Existing cache files found - lazy loading enabled
```

## Cache Management

### Automatic Cleanup
- **Startup**: Expired files automatically removed
- **TTL**: 24-hour default expiration
- **LRU**: RAM cache automatically evicts least-used entries

### Manual Cleanup
```python
cache = get_persistent_cache()

# Remove expired files only
cache.clear_expired()

# Clear everything (RAM + disk)
cache.clear_all()
```

## Integration Points

The persistent cache is automatically integrated into:

1. **Dynamic Dictionary Analysis** (`core/dynamic_dictionary.py`)
   - Token counting for pattern analysis
   - Symbol assignment value calculations

2. **Model-Optimized Symbol Pool** (`core/model_optimized_symbol_pool.py`)
   - Symbol efficiency testing in multiple contexts
   - Model-specific tokenization results

3. **Server Initialization** (`api/server.py`)
   - Automatic cache setup and expired file cleanup
   - Performance logging and statistics

## Benefits

### ✅ Performance Improvements
- **Faster Analysis**: Avoid re-tokenizing repeated patterns
- **Reduced API Calls**: Cache expensive tokenization operations
- **Better Responsiveness**: Immediate results for cached patterns

### ✅ Reliability
- **Graceful Degradation**: Works without cache if needed  
- **Error Handling**: Corrupted cache files automatically cleaned up
- **Thread Safety**: All operations are thread-safe

### ✅ Efficiency
- **Memory Bounded**: RAM cache limited to prevent memory issues
- **Disk Efficient**: Only stores necessary data with compression
- **Lazy Loading**: Disk cache loaded only when needed

## Testing

Run the comprehensive test suite:
```bash
python tests/test_persistent_token_cache.py
```

Tests cover:
- Basic cache operations (get/set)
- Cache expiration and TTL handling
- RAM cache LRU eviction
- Token counting integration
- Cache cleanup functionality

## Migration Notes

**Existing Installations**: The persistent cache is backward compatible and requires no migration. It will automatically start caching new tokenization operations while maintaining existing functionality.

**Performance Impact**: Minimal - cache checks add microseconds to tokenization calls while providing significant speedup for repeated operations. 