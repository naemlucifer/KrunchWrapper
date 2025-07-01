# KrunchWrapper Configuration Reference

Detailed configuration options for KrunchWrapper server and compression settings.

## 🚀 Quick Start - Configuration Presets

**For most users**: Edit `server.jsonc` and uncomment the configuration preset you want to use:

### 🏠 **Local Server** (Default Active)
```json
"target_host": "localhost",
"target_port": 1234,              // Change to match your server
"target_use_https": false,
"api_key": ""
```
**Perfect for**: LM Studio, Ollama, Text Generation WebUI, vLLM, LocalAI  

### 🤖 **Direct Anthropic API** ✅ TESTED & WORKING
```json
// "target_host": "api.anthropic.com",
// "target_port": 443,
// "target_use_https": true,
// "api_key": "sk-ant-your-key-here"
```
**Perfect for**: Cline with direct Anthropic access  
**Status**: ✅ Fully tested and debugged

### 🧠 **Direct OpenAI API** ⚠️ EXPERIMENTAL
```json
// "target_host": "api.openai.com",          // NOT TESTED
// "target_port": 443,                       // MAY NOT WORK  
// "target_use_https": true,                 // EXPERIMENTAL
// "api_key": "sk-your-openai-key-here"
```
**Warning**: Theoretical configuration - not tested! Use Local Server setup for reliable OpenAI access.

**Other presets in `server.jsonc` are experimental/theoretical examples only.**

## Configuration Files

- **`server.jsonc`**: Server connection and API settings (⭐ **Start here for presets**)
- **`config.jsonc`**: Advanced compression and feature settings
- **`system-prompts.jsonc`**: System prompt format definitions
- **`async_logging.jsonc`**: Async logging configuration

## Configuration Priority

Settings are applied in order of precedence:
1. **Command line arguments** (highest priority)
2. **Environment variables**
3. **Configuration files** (server.jsonc, config.jsonc)
4. **Default values** (lowest priority)

## Server Configuration (`server.jsonc`)

Core server and API connection settings:

```json
{
    // Network Configuration
    "host": "0.0.0.0",                  // Server bind address
    "port": 5001,                       // Server listen port
    "target_host": "localhost",         // Target LLM API host
    "target_port": 5002,               // Target LLM API port

    // API Configuration
    "api_key": "",                      // Target API key (if required)
    "require_api_key": false,          // Require API key validation
    "min_compression_ratio": 0.05,     // Minimum compression ratio (5%)

    // Logging Configuration
    "verbose_logging": false,          // Show original/compressed content
    "file_logging": true,              // Save logs to dated files
    "log_level": "INFO",               // DEBUG, INFO, WARNING, ERROR

    // Advanced Features
    "system_prompt_format": "chatml",  // Default system prompt format
    "filter_timeout_parameters": true  // Filter unsupported timeout params
    
    // NOTE: Cline integration is now controlled by "system_prompt.use_cline" in config.jsonc
}
```

### Environment Variable Mappings

| Configuration Key | Environment Variable | Description |
|-------------------|---------------------|-------------|
| `host` | `KRUNCHWRAPPER_HOST` | Server bind address |
| `port` | `KRUNCHWRAPPER_PORT` | Server listen port |
| `target_host` | `LLM_API_HOST` | Target LLM API host |
| `target_port` | `LLM_API_PORT` | Target LLM API port |
| `target_url` | `LLM_API_URL` | Full target URL (overrides host/port) |
| `api_key` | `LLM_API_KEY` | Target API authentication key |
| `min_compression_ratio` | `MIN_COMPRESSION_RATIO` | Minimum compression threshold |
| `verbose_logging` | `KRUNCHWRAPPER_VERBOSE` | Enable verbose content logging (shows full text content) |
| `file_logging` | `KRUNCHWRAPPER_FILE_LOGGING` | Enable file-based logging |
| `log_level` | `KRUNCHWRAPPER_LOG_LEVEL` | Message severity level (DEBUG/INFO/WARNING/ERROR) |

## Advanced Configuration (`config.jsonc`)

### 🔍 Understanding KrunchWrapper Logging

KrunchWrapper has **two separate, independent** logging controls that are often confused:

#### 1. **Log Level** (`log_level`) - Message Severity Filter
Controls **which severity messages** appear in logs:
- `DEBUG`: All messages (basic operational info like "Request received", "Model: gpt-4")  
- `INFO`: Standard operations and performance metrics only
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

**Important**: Both file and console use the **same log level** - there's no separate file vs terminal control.

#### 2. **Verbose Logging** (`verbose_logging`) - Content Detail Control  
Controls **how detailed the content** is within those messages:
- `true`: Shows **full request/response text**, compression before/after content, detailed rules
- `false`: Shows only basic operational messages

**Warning**: Verbose logging can generate **massive log files** with complete text content!

#### Examples:
```bash
# Shows basic DEBUG messages only:
"log_level": "DEBUG", "verbose_logging": false
# Output: "Request received", "Model: gpt-4", "Compression ratio: 0.65"

# Shows full content with DEBUG messages:  
"log_level": "DEBUG", "verbose_logging": true
# Output: Same as above PLUS full original text, compressed text, etc.

# Shows minimal INFO messages:
"log_level": "INFO", "verbose_logging": false  
# Output: Only performance metrics, no debug details

# Shows full content but only for INFO+ messages:
"log_level": "INFO", "verbose_logging": true
# Output: Performance metrics with full content details
```

### Compression Settings

```json
{
    "compression": {
        // Basic Thresholds
        "min_characters": 250,              // Minimum content size for compression
        "min_compression_ratio": 0.05,      // Minimum efficiency threshold
        "min_token_savings": 1,             // Minimum tokens saved per substitution

        // Processing Configuration
        "threads": 4,                       // Compression analysis threads
        "aggressive_mode": false,           // Enable aggressive compression
        "large_file_threshold": 5000,       // Auto-aggressive mode threshold

        // Advanced Options
        "use_token_compression": true,      // Use token-aware compression
        "min_occurrences": 3               // Minimum pattern occurrences
    }
}
```

### Dynamic Dictionary Configuration

Advanced pattern analysis and compression generation:

```json
{
    "dynamic_dictionary": {
        // Core Settings
        "enabled": true,                    // Enable dynamic compression
        "compression_threshold": 0.01,      // Analysis efficiency threshold
        "multipass_enabled": true,          // Enable multi-pass optimization
        "max_passes": 3,                    // Maximum optimization passes

        // Analysis Parameters
        "min_frequency": 2,                 // Minimum pattern frequency
        "min_length": 3,                   // Minimum pattern length
        "max_dictionary_size": 500,        // Maximum symbols per analysis

        // Performance Tuning
        "analysis_threads": 4,              // Pattern analysis threads
        "symbol_assignment_threads": 2,     // Symbol assignment threads
        "enable_caching": true             // Enable result caching
    }
}
```

### Comment Stripping Configuration

Intelligent code comment removal with safety features:

```json
{
    "comment_stripping": {
        // Global Settings
        "enabled": true,                    // Enable comment stripping
        "preserve_license_headers": true,   // Keep copyright/license info
        "preserve_shebang": true,          // Keep shebang lines
        "preserve_docstrings": true,       // Keep function/class docs
        "min_line_length_after_strip": 3,  // Minimum line length to keep

        // Language-Specific Settings
        "languages": {
            "python": true,                // Python (.py) files
            "javascript": true,            // JavaScript (.js, .ts) files
            "c_cpp": true,                // C/C++ (.c, .cpp, .h) files
            "html": true,                 // HTML (.html, .htm) files
            "css": true,                  // CSS (.css) files
            "sql": true,                  // SQL (.sql) files
            "shell": true                 // Shell (.sh, .bash) files
        }
    }
}
```

### Model Tokenizer Configuration

Model-specific tokenizer validation for accurate token counting. KrunchWrapper automatically detects your model family and uses the appropriate tokenizer library for precise validation.

```json
{
    "model_tokenizer": {
        /* Enable model-specific tokenizer validation
           When enabled, KrunchWrapper detects model families and uses appropriate tokenizers
           When disabled, falls back to generic tiktoken validation */
        "enabled": true,
        
        /* Default model family to use when model detection fails
           Available model families:
           
           OpenAI Models:
           - "gpt-4"     -> Detects: gpt-4, gpt4, openai/gpt-4
           - "gpt-3.5"   -> Detects: gpt-3.5, gpt-35, turbo, gpt-3.5-turbo
           - "gpt-3"     -> Detects: davinci, curie, babbage, ada
           
           Anthropic Models:
           - "claude"    -> Detects: claude, anthropic, anthropic/claude-3-5-sonnet
           
           Meta LLaMA Models:
           - "llama"     -> Detects: llama, llama2, llama-2, meta-llama/Llama-2-7b-chat-hf
           - "llama3"    -> Detects: llama3, llama-3, meta-llama/Llama-3-8B-Instruct
           - "codellama" -> Detects: codellama, code-llama
           
           Mistral Models:
           - "mistral"   -> Detects: mistral, mixtral, mistralai/Mistral-7B-Instruct
           
           Google Models:
           - "gemini"    -> Detects: gemini, bard, google/gemini-pro
           - "palm"      -> Detects: palm, palm2
           
           Qwen Models:
           - "qwen"      -> Detects: qwen, Qwen/Qwen2.5-Coder-32B-Instruct
           - "qwen2"     -> Detects: qwen2, qwen-2
           - "qwen3"     -> Detects: qwen3, qwen-3
           
           Other Models:
           - "yi"        -> Detects: yi-, 01-ai, 01-ai/Yi-34B-Chat
           - "deepseek"  -> Detects: deepseek, deepseek-ai/deepseek-coder
           - "phi"       -> Detects: phi-, microsoft/phi
           - "falcon"    -> Detects: falcon, tiiuae/falcon-7b-instruct
           - "starcoder" -> Detects: starcoder, starcode
           - "vicuna"    -> Detects: vicuna
           - "alpaca"    -> Detects: alpaca
           - "chatglm"   -> Detects: chatglm, glm-, THUDM/chatglm3-6b
        */
        "default_model_family": "gpt-4",
        
        /* Fallback validation method when model-specific tokenizer is unavailable
           Options:
           - "tiktoken"             -> Use generic GPT-4 tokenizer (cl100k_base)
           - "character_estimation" -> Estimate tokens from character count (chars/4)
           - "word_count"           -> Estimate tokens from word count */
        "fallback_method": "tiktoken",
        
        /* Cache tokenizers for better performance (recommended: true)
           When true, tokenizers are loaded once and cached for subsequent use
           When false, tokenizers are loaded fresh for each validation (slower) */
        "cache_tokenizers": true,

        // Performance Settings
        "validation_timeout": 5.0,         // Validation timeout (seconds)
        "max_cache_size": 1000,           // Maximum cached validations
        "enable_batch_validation": true,   // Enable batch processing
        
        /* Advanced model family mappings (optional)
           You can override or extend model detection patterns here
           Format: "model_family": ["pattern1", "pattern2", ...]
           
           Example custom mappings:
           "custom_model_mappings": {
               "my_custom_model": ["custom-model", "my-org/custom"],
               "local_llama": ["local-llama", "localhost/llama"]
           }
        */
        
        /* Tokenizer library requirements:
           For full compatibility, install these libraries:
           
           pip install tiktoken      # For OpenAI models (GPT-4, GPT-3.5, etc.)
           pip install transformers  # For modern models (Qwen, LLaMA 3, DeepSeek, etc.)
           pip install sentencepiece # For LLaMA 1/2, Mistral, Claude, etc.
           
           The system will gracefully fall back if libraries are missing.
        */
    }
}
```

### System Prompt Configuration

Advanced system prompt processing and format handling:

```json
{
    "system_prompt": {
        // Format Settings
        "format": "chatml",                // Default format (see formats below)
        "use_cline": false,               // Enable Cline-specific handling
        
        // Processing Options
        "merge_strategy": "priority",      // priority, append, replace
        "max_system_prompt_length": 4000, // Maximum combined prompt length
        "enable_format_detection": true,   // Auto-detect existing formats
        
        // Compression Integration
        "compression_instructions_priority": "high", // high, medium, low
        "include_decoder_when_ratio_below": 0.02    // Include decoder threshold
    }
}
```

#### Supported System Prompt Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `chatml` | Standard ChatML format | Most providers |
| `claude` | Anthropic Claude with system parameter | Claude API |
| `chatgpt` | OpenAI ChatGPT format | OpenAI API |
| `gemini` | Google Gemini with system_instruction | Gemini API |
| `qwen` | Qwen format (OpenAI-compatible) | Qwen models |
| `deepseek` | DeepSeek format | DeepSeek models |
| `gemma` | Gemma format with turn templates | Gemma models |

### Async Logging Configuration

High-performance async logging system:

```json
{
    "async_logging": {
        // Core Settings
        "enabled": true,                   // Enable async logging
        "log_level": "INFO",              // DEBUG, INFO, WARNING, ERROR
        "max_queue_size": 50000,          // Message queue size

        // Performance Settings
        "batch_size": 100,                // Messages per batch
        "flush_interval": 1.0,            // Flush interval (seconds)
        "worker_threads": 1,              // Background worker threads

        // File Logging
        "file_logging": true,             // Enable file output
        "log_file_prefix": "krunchwrapper", // Log file prefix
        "log_directory": "logs",          // Log directory
        "max_file_size": 100000000,       // Max file size (bytes)
        "backup_count": 5                 // Number of backup files
    }
}
```

### Conversation State Configuration

Multi-turn conversation optimization:

```json
{
    "conversation_state": {
        // State Management
        "enabled": true,                   // Enable conversation tracking
        "max_conversations": 1000,        // Maximum tracked conversations
        "conversation_timeout": 3600,      // Timeout (seconds)
        
        // Compression Persistence
        "reuse_dictionaries": true,       // Reuse compression across turns
        "dictionary_expansion": true,     // Expand dictionaries over time
        "min_reuse_threshold": 0.3,       // Minimum reuse efficiency
        
        // Cache Settings
        "enable_kv_cache": true,          // Enable key-value caching
        "cache_size_limit": 10000,        // Maximum cache entries
        "auto_cleanup": true              // Automatic cleanup
    }
}
```

## Configuration Examples

### Development Environment

```json
{
    "host": "127.0.0.1",
    "port": 5001,
    "target_host": "localhost", 
    "target_port": 11434,
    "verbose_logging": true,
    "log_level": "DEBUG",
    "compression": {
        "min_characters": 100,
        "threads": 2,
        "aggressive_mode": false
    },
    "async_logging": {
        "log_level": "DEBUG",
        "max_queue_size": 20000
    }
}
```

### Production Environment

```json
{
    "host": "0.0.0.0",
    "port": 5001,
    "target_host": "internal-llm-api",
    "target_port": 80,
    "verbose_logging": false,
    "log_level": "INFO",
    "file_logging": true,
    "compression": {
        "min_characters": 250,
        "threads": 8,
        "aggressive_mode": true,
        "large_file_threshold": 5000
    },
    "async_logging": {
        "log_level": "INFO",
        "max_queue_size": 100000,
        "batch_size": 500
    }
}
```

### High-Performance Setup

```json
{
    "compression": {
        "threads": 16,
        "aggressive_mode": true,
        "use_token_compression": true
    },
    "dynamic_dictionary": {
        "analysis_threads": 8,
        "symbol_assignment_threads": 4,
        "max_dictionary_size": 1000
    },
    "async_logging": {
        "max_queue_size": 200000,
        "batch_size": 1000,
        "worker_threads": 2
    },
    "model_tokenizer": {
        "cache_tokenizers": true,
        "max_cache_size": 5000,
        "enable_batch_validation": true
    }
}
```

## Command Line Overrides

Key command line arguments that override configuration files:

```bash
# Server Settings
python server/run_server.py \
    --port 8080 \
    --host 127.0.0.1 \
    --target-url http://localhost:11434/v1 \
    --min-compression-ratio 0.1

# With Environment Variables
KRUNCHWRAPPER_VERBOSE=true \
KRUNCHWRAPPER_LOG_LEVEL=DEBUG \
LLM_API_URL=http://api.example.com/v1 \
python server/run_server.py
```

## Validation and Troubleshooting

### Configuration Validation

KrunchWrapper validates configuration on startup and reports errors:

```
🔧 Loading configuration from config/server.jsonc
✅ Server configuration validated successfully
🔧 Loading advanced settings from config/config.jsonc  
⚠️  Warning: Unknown configuration key 'invalid_option' in compression section
✅ Advanced configuration loaded with 1 warning
```

### Common Configuration Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Invalid JSON | Startup failure | Validate JSON syntax with `python -m json.tool config.jsonc` |
| Port conflicts | Address already in use | Change port or stop conflicting service |
| Target unreachable | 502 Bad Gateway | Verify target_host and target_port |
| Low compression | No compression applied | Lower min_compression_ratio |
| High memory usage | OOM errors | Reduce max_queue_size and max_cache_size |
| Unknown model family | `WARNING - Unknown model family for: a` | See Model Tokenizer Troubleshooting below |

### How to Configure Custom Model Mappings

**🎯 Quick Setup for "Unknown model family" Warning**:

1. **Open `config/config.jsonc`** and find the `model_tokenizer` section
2. **Add your custom mappings**:
   ```json
   {
       "model_tokenizer": {
           "custom_model_mappings": {
               "gpt-4": ["exactly-a", "my-local-model"],
               "claude": ["custom-claude", "internal-ai"]
           }
       }
   }
   ```
3. **Save and restart** your KrunchWrapper server
4. **Check logs** for confirmation:
   ```
   INFO - Loading 2 custom model mappings
   INFO - Model tokenizer loaded with 15 families (including 2 custom)
   ```

**🔧 Advanced Configuration Options**:

- **Case-insensitive matching**: All patterns automatically converted to lowercase
- **Substring matching**: Pattern `"gpt"` matches `"my-gpt-model"`, `"custom-gpt-4"`
- **Family extension**: Add patterns to existing families (like `gpt-4`) or create new ones
- **Pattern specificity**: Use specific patterns like `"exactly-a"` instead of broad ones like `"a"`

**💡 Configuration Tips**:
- Test with `python tests/test_case_insensitive_tokenizer.py`
- Enable debug logging to see pattern matching: `"log_level": "DEBUG"`
- Restart server after any configuration changes

### Model Tokenizer Troubleshooting

**Problem**: `WARNING - Unknown model family for: a`

This warning indicates that your API is sending a generic model name (like `"a"`, `"model"`, or `"llm"`) that doesn't match any supported model patterns. While compression still works, token validation falls back to less accurate character-based estimation.

**Root Causes & Solutions**:

1. **Generic model name in API requests**
   ```bash
   # Problem: Your client/provider sends generic names
   curl -X POST http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "a", "messages": [...]}'  # ❌ Generic name
   
   # Solution: Use specific model names
   curl -X POST http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4", "messages": [...]}'  # ✅ Specific name
   ```

2. **Provider configuration issue**
   ```bash
   # Check your LLM provider settings
   # Many providers allow setting default model names:
   
   # For Ollama:
   OLLAMA_MODEL_NAME=llama-3-8b-instruct
   
   # For LocalAI:
   export MODEL_NAME=gpt-3.5-turbo
   
   # For OpenAI-compatible APIs:
   export OPENAI_MODEL=gpt-4
   ```

3. **Cline/IDE configuration**
   ```json
   // In your Cline/IDE settings, set a proper model name:
   {
     "model": "gpt-4",              // ✅ Instead of "a"
     "provider": "openai",
     "baseUrl": "http://localhost:5001/v1"
   }
   ```

4. **Custom model patterns**
   ```json
   // Add custom detection patterns to config/config.jsonc:
   {
     "model_tokenizer": {
       "custom_model_mappings": {
         "gpt-4": ["exactly-a", "my-local-gpt"],
         "claude": ["my-claude", "internal-assistant"],
         "generic_model": ["company-llm", "custom-model-v1"]
       }
     }
   }
   ```
   
   ⚠️ **Important**: Patterns are matched as case-insensitive substrings.
   - Use specific patterns like `"exactly-a"` instead of `"a"`
   - Avoid overly broad patterns that might match unintended models
   - Pattern `"a"` would incorrectly match `"llama"`, `"claude"`, etc.

**Verification**:
- ✅ **Good**: Logs show `Loaded tiktoken tokenizer for gpt-4 model: gpt-4`
- ❌ **Problem**: Logs show `Unknown model family for: a`
- ✅ **Fallback working**: Compression still occurs with `method: character_estimation`

### Performance Tuning

For optimal performance based on your system:

**CPU-Bound Workloads:**
- Increase `compression.threads` to match CPU cores
- Enable `aggressive_mode` for better compression
- Increase `dynamic_dictionary.analysis_threads`

**Memory-Constrained Systems:**
- Reduce `async_logging.max_queue_size`
- Lower `model_tokenizer.max_cache_size`
- Disable `conversation_state.enabled` if not needed

**High-Throughput Systems:**
- Increase `async_logging.batch_size`
- Enable `model_tokenizer.enable_batch_validation`
- Tune `dynamic_dictionary.max_dictionary_size`

## Additional Resources

- **Main Configuration Guide**: [README.md#configuration](../README.md#configuration)
- **API Reference**: [api/README.md](../api/README.md)
- **System Architecture**: [charts/README.md](../charts/README.md)
- **Performance Documentation**: [documentation/](../documentation/)

## Async Logging Configuration Examples

The `async_logging.jsonc` file controls the performance characteristics of KrunchWrapper's logging system. Here are optimized configurations for different use cases:

### 🐛 **Debugging/Development Scenario**
```json
{
    "async_logging": {
        "enabled": true,
        "log_level": "DEBUG",
        "max_queue_size": 0,
        "batch_size": 1,
        "worker_timeout": 0.001,
        "performance_monitoring": {
            "enabled": true,
            "track_system_prompts": true,
            "track_compression": true,
            "max_tracked_operations": 50
        }
    }
}
```
**Benefits**: Immediate log processing, full debug info, detailed operation tracking.

### 🚀 **High Performance/Production Scenario**
```json
{
    "async_logging": {
        "enabled": true,
        "log_level": "INFO",
        "max_queue_size": 0,
        "batch_size": 100,
        "worker_timeout": 0.1,
        "performance_monitoring": {
            "enabled": true,
            "track_system_prompts": false,
            "track_compression": true,
            "max_tracked_operations": 100
        }
    }
}
```
**Benefits**: Optimized throughput, reduced CPU overhead, focused monitoring.

### 📊 **High Volume/Traffic Spikes Scenario**
```json
{
    "async_logging": {
        "enabled": true,
        "log_level": "WARNING",
        "max_queue_size": 0,
        "batch_size": 500,
        "worker_timeout": 0.5,
        "performance_monitoring": {
            "enabled": false
        }
    }
}
```
**Benefits**: Maximum efficiency, minimal resource usage, handles traffic bursts.

### 💾 **Memory Constrained Scenario**
```json
{
    "async_logging": {
        "enabled": true,
        "log_level": "INFO",
        "max_queue_size": 10000,
        "batch_size": 25,
        "worker_timeout": 0.05,
        "performance_monitoring": {
            "enabled": true,
            "track_system_prompts": false,
            "track_compression": false,
            "max_tracked_operations": 25
        }
    }
}
```
**Benefits**: Bounded memory usage, frequent queue cleanup, minimal tracking overhead.

## Environment Variable Overrides

You can override any async logging setting with environment variables:

```bash
# Enable/disable async logging
export KRUNCHWRAPPER_GLOBAL_ASYNC_LOGGING=true

# Adjust performance parameters
export ASYNC_LOG_BATCH_SIZE=200
export ASYNC_LOG_WORKER_TIMEOUT=0.2
export KRUNCHWRAPPER_LOG_LEVEL=DEBUG

# Start server with custom settings
python api/server.py
```

## Monitoring Performance

Check async logging performance at runtime:

```bash
# Get current stats
curl http://localhost:5002/v1/compression/stats

# Look for the async_logging section:
{
  "async_logging": {
    "enabled": true,
    "stats": {
      "messages_logged": 1250,
      "messages_dropped": 0,
      "queue_size": 45,
      "batches_processed": 25,
      "avg_batch_size": 48.2,
      "SYSTEM_PROMPT_PROCESSING": {
        "count": 120,
        "frequency_per_sec": 2.4
      }
    }
  }
}
```

## Performance Impact

**Batching Benefits:**
- `batch_size=1`: Individual processing (debugging)
- `batch_size=50`: ~40% better performance 
- `batch_size=100`: ~60% better performance
- `batch_size=500`: ~80% better performance (high volume)

**Queue Size:**
- `max_queue_size=0`: Unlimited (recommended for most cases)
- `max_queue_size=N`: Bounded memory usage, may drop logs under load

**Worker Timeout:**
- Lower values: More responsive, higher CPU usage
- Higher values: More efficient batching, higher latency 