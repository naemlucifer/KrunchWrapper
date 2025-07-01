# Debug Categories System

The KrunchWrapper debug categories system provides fine-grained control over debug logging output. Instead of being overwhelmed by all debug messages when using `DEBUG` log level, you can selectively enable only the categories you're interested in.

> **📚 For a complete guide to KrunchWrapper logging, see [LOGGING_GUIDE.md](LOGGING_GUIDE.md)** which explains the difference between log levels and verbose logging, plus all configuration options.

## Overview

When `log_level` is set to `DEBUG`, the debug categories system allows you to:
- **Enable specific debug categories** while filtering out others
- **Focus on particular subsystems** during debugging
- **Reduce noise** in debug logs by showing only relevant messages
- **Use predefined scenarios** for common debugging tasks

## Configuration

Debug categories are configured in `config/server.jsonc` under the `logging.debug_categories` section:

```jsonc
{
    "logging": {
        "log_level": "DEBUG",  // Must be DEBUG to use categories
        "debug_categories": {
            "request_processing": true,     // Enable request processing debug messages
            "cline_integration": false,     // Disable Cline integration debug messages
            "compression_core": true,       // Enable compression core debug messages
            // ... other categories
        }
    }
}
```

## Available Debug Categories

### 🔍 Request Processing & Client Detection
- **`request_processing`** - Raw request analysis, parsing, and general debug messages `[ALL REQUESTS]`, `[DEBUG]`
- **`cline_integration`** - Cline client detection and special handling `[CLINE]`
- **`session_management`** - Session ID generation and tracking `[SESSION]`

### 🗜️ Compression Operations & Analysis
- **`compression_core`** - Core compression analysis, dynamic dictionary operations
- **`compression_proxy`** - Proxy-level compression decisions and metrics `[PROXY]`
- **`conversation_detection`** - Multi-turn conversation analysis `[CONVERSATION DETECTION]`
- **`kv_cache_optimization`** - KV cache optimization logic `[KV DEBUG]`, `[KV CACHE]`
- **`conversation_compression`** - Conversation-aware compression `[CONVERSATION COMPRESS]`
- **`symbol_management`** - Symbol assignment, model-specific tokenization

### 🌊 Response Processing & Decompression
- **`streaming_responses`** - Streaming response processing `[STREAMING DEBUG]`, `[PROXY STREAMING DEBUG]`
- **`response_decompression`** - Response decompression `[NON-STREAMING DEBUG]`, `[PROXY DECOMPRESSION DEBUG]`
- **`token_calculations`** - Token calculation and compression ratio reporting

### 🔧 System Operations & Infrastructure
- **`system_fixes`** - System corrections and adaptations `[FIX]`
- **`request_filtering`** - Request parameter filtering `[TIMEOUT FILTER]`
- **`cleanup_operations`** - Compression artifact cleanup `[CLEANUP]`
- **`error_handling`** - Detailed error reporting and diagnostics `[ERROR]`

### 📡 Server Communication & Forwarding
- **`server_communication`** - Target server communication, props forwarding
- **`model_context`** - Model context setting and management

### 🔬 Development & Testing
- **`payload_debugging`** - Detailed payload inspection (very verbose) `[PAYLOAD DEBUG]`
- **`test_utilities`** - Test-specific debug output (very verbose)

## How It Works

1. **Automatic Category Detection**: The system automatically categorizes debug messages based on their content patterns (e.g., `[CLINE]`, `[PROXY]`, etc.)

2. **Filtering Logic**: When a DEBUG message is logged:
   - The system extracts the category from the message content
   - Checks if that category is enabled in the configuration
   - Only displays the message if the category is enabled

3. **Non-DEBUG Messages**: INFO, WARNING, and ERROR messages are **never filtered** - only DEBUG level messages are affected

## Common Usage Scenarios

### 🎯 Debugging Cline Integration Issues
```jsonc
"debug_categories": {
    "cline_integration": true,
    "error_handling": true,
    // All others: false
}
```

### 🗜️ Debugging Compression Performance
```jsonc
"debug_categories": {
    "compression_core": true,
    "compression_proxy": true,
    "conversation_detection": true,
    "kv_cache_optimization": true,
    "token_calculations": true,
    "error_handling": true,
    // All others: false
}
```

### 🌊 Debugging Streaming Responses
```jsonc
"debug_categories": {
    "streaming_responses": true,
    "response_decompression": true,
    "cleanup_operations": true,
    "error_handling": true,
    // All others: false
}
```

## Example Configurations

See `config/debug-categories-examples.jsonc` for complete example configurations covering:
- **Cline-only debugging**
- **Compression system debugging**
- **Streaming and response processing**
- **Request processing and client detection**
- **Minimal (errors only)**
- **Development and testing (verbose)**
- **Performance analysis**

## Usage Instructions

1. **Set log level to DEBUG** in `config/server.jsonc`:
   ```jsonc
   "logging": {
       "log_level": "DEBUG"
   }
   ```

2. **Configure debug categories** in the same file:
   ```jsonc
   "logging": {
       "log_level": "DEBUG",
       "debug_categories": {
           "cline_integration": true,
           "error_handling": true,
           // Set others to false to filter them out
       }
   }
   ```

3. **Start the server** - you'll now see only the enabled debug categories

## Benefits

- **🎯 Focused Debugging**: See only the debug messages relevant to your current investigation
- **📊 Reduced Noise**: Eliminate overwhelming debug output from irrelevant subsystems
- **⚡ Faster Analysis**: Quickly identify issues without scrolling through thousands of debug messages
- **🔧 Flexible Configuration**: Easy to switch between different debugging scenarios
- **🔄 Backward Compatible**: If no categories are configured, all debug messages are shown (existing behavior)

## Testing

Run the test suite to verify the debug categories system:

```bash
cd tests
python test_debug_categories.py
```

This will test:
- Category extraction from message content
- Category filtering logic
- Configuration file structure

## Implementation Details

- **Category Extraction**: Uses pattern matching on message content to automatically determine categories
- **Performance**: Minimal overhead - filtering only occurs for DEBUG level messages
- **Extensibility**: New categories can be easily added by updating the pattern matching function
- **Fallback**: Unknown patterns default to `request_processing` category

The debug categories system provides a powerful way to manage debug output in KrunchWrapper, making it easier to debug specific issues without being overwhelmed by irrelevant log messages. 