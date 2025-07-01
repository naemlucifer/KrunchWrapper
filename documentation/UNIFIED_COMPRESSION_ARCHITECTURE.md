# Unified Compression Architecture

## Overview

This document describes the consolidated compression architecture in KrunchWrapper, where all compression paths have been unified to use a single core dynamic compression system, with different system prompt interception implementations for different modes.

## Architecture Principles

### 🎯 **Core Principle: One Compression System**
- **Single Entry Point**: All compression flows through `compress_with_dynamic_analysis()`
- **No Separate Paths**: No mode-specific compression logic
- **Consistent Behavior**: Same compression for standard and cline modes

### 🔄 **System Prompt Interception Differences Only**
- **Standard Mode**: Uses `SystemPromptInterceptor` with configurable target format
- **Cline Mode**: Uses `ClineSystemPromptInterceptor` with automatic format detection
- **Same Core Logic**: Both inherit compression from base interceptor

## Unified Compression Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Incoming Request                            │
└─────────────────────┬───────────────────────────────────────────┘
                     │
┌─────────────────────▼───────────────────────────────────────────┐
│             Core Dynamic Compression                            │
│         compress_with_dynamic_analysis()                       │
│                                                                │
│  • Dynamic dictionary analysis                                 │
│  • Comment stripping                                          │
│  • Token validation                                           │
│  • Conversation-aware (uses dynamic internally)               │
└─────────────────────┬───────────────────────────────────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
    ┌──────▼──────┐    ┌──────▼──────┐
    │ Standard    │    │ Cline       │
    │ Mode        │    │ Mode        │
    └──────┬──────┘    └──────┬──────┘
           │                   │
┌──────────▼──────────┐ ┌──────▼──────────┐
│ SystemPrompt        │ │ ClineSystemPrompt│
│ Interceptor         │ │ Interceptor      │
│                     │ │                  │
│ • Fixed target      │ │ • Auto format    │
│   format            │ │   detection      │
│ • Manual config     │ │ • Provider aware │
└─────────────────────┘ └──────────────────┘
```

## Core Components

### 1. **Dynamic Compression Engine** (`core/compress.py`)
- **`compress_with_dynamic_analysis()`**: The single compression entry point
- **Features**:
  - Dynamic dictionary analysis
  - Comment stripping integration
  - Token validation
  - Tool call detection (skips compression)
  - Markdown detection (skips compression)

### 2. **Conversation Compression** (`core/conversation_compress.py`)
- **Built on Dynamic**: Uses `compress_with_dynamic_analysis()` internally
- **Features**:
  - State management across turns
  - KV cache optimization for short responses
  - Efficiency tracking
  - Falls back to dynamic compression when needed

### 3. **System Prompt Interceptors**

#### Standard Mode (`core/system_prompt_interceptor.py`)
```python
class SystemPromptInterceptor:
    def intercept_and_process(self, messages, rule_union, lang, target_format, ...):
        # Uses rule_union from compression or calls compress_with_dynamic_analysis()
        # Processes system prompts with fixed target format
```

#### Cline Mode (`core/cline_system_prompt_interceptor.py`)
```python
class ClineSystemPromptInterceptor(SystemPromptInterceptor):
    def intercept_and_process_cline(self, messages, rule_union, lang, model_id, ...):
        # Determines target format from model_id (provider/model)
        # Calls parent intercept_and_process() with detected format
        # SAME compression logic, DIFFERENT format detection
```

## Elimination of Duplicate Paths

### ❌ **Before: Multiple Compression Paths**
- Main endpoint: `compress_with_dynamic_analysis()`
- Cline proxy: Separate `compress_with_dynamic_analysis()` calls
- Legacy API: Old `compress()` function (didn't exist but was imported)
- Various utilities: Mixed compression approaches

### ✅ **After: Single Compression Path**
- **All endpoints**: `compress_with_dynamic_analysis()`
- **All modes**: Same core compression
- **All utilities**: Updated to use dynamic compression
- **Consistent behavior**: Across all entry points

## Mode Differences

| Aspect | Standard Mode | Cline Mode |
|--------|---------------|------------|
| **Compression** | `compress_with_dynamic_analysis()` | `compress_with_dynamic_analysis()` |
| **System Prompt Format** | Fixed (config: `system_prompt.format`) | Auto-detected from `model_id` |
| **Format Detection** | Manual configuration | Provider-aware (`anthropic/claude` → `claude`) |
| **Interceptor** | `SystemPromptInterceptor` | `ClineSystemPromptInterceptor` |
| **Core Logic** | Base implementation | Inherits from base |

## Configuration

### System Prompt Modes
```jsonc
{
  "system_prompt": {
    "format": "chatml",        // Standard mode: fixed format
    "use_cline": true         // Enable cline mode auto-detection
  }
}
```

### Compression Settings
```jsonc
{
  "compression": {
    "min_characters": 250,           // Unified threshold
    "min_compression_ratio": 0.02    // Unified efficiency threshold
  },
  "dynamic_dictionary": {
    "enabled": true,                 // Core dynamic compression
    "compression_threshold": 0.01    // Dynamic analysis threshold
  }
}
```

## Benefits of Unified Architecture

### 🎯 **Consistency**
- Same compression behavior across all modes
- Predictable performance characteristics
- Unified configuration and tuning

### 🛠️ **Maintainability**
- Single compression codebase to maintain
- Easier debugging and testing
- Clear separation of concerns

### 🚀 **Performance**
- No duplicate compression logic
- Shared optimizations benefit all modes
- Consistent memory usage patterns

### 🔧 **Extensibility**
- New modes only need system prompt interceptors
- Core compression improvements benefit all modes
- Easy to add new format support

## Migration Notes

### Updated Components
1. **`api/main.py`**: Updated to use `compress_with_dynamic_analysis()`
2. **Server proxy paths**: Consolidated to use unified compression
3. **Cline interceptor**: Added clarifying comments about inheritance
4. **Import statements**: Removed references to non-existent `compress()` function

### Backward Compatibility
- All existing configurations continue to work
- API endpoints remain unchanged
- Compression quality and efficiency maintained or improved

## Future Enhancements

The unified architecture enables:
- **New Provider Support**: Add new providers by extending format detection
- **Compression Improvements**: All modes benefit from core enhancements
- **Advanced Features**: Easier to add features like compression analytics
- **Testing**: Simplified test scenarios with consistent behavior

## Validation

To verify the unified architecture:

1. **Both modes use same compression**:
   ```bash
   # Should show compress_with_dynamic_analysis usage
   grep -r "compress_with_dynamic_analysis" api/ core/
   ```

2. **No separate compression paths**:
   ```bash
   # Should find no separate cline compression logic
   grep -r "cline.*compress" --exclude-dir=.git
   ```

3. **System prompt differences only**:
   ```bash
   # Should show format detection differences only
   diff core/system_prompt_interceptor.py core/cline_system_prompt_interceptor.py
   ```