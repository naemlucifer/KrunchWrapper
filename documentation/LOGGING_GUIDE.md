# KrunchWrapper Logging Configuration Guide

A comprehensive guide to understanding and configuring logging in KrunchWrapper, including the key differences between **Log Levels** and **Verbose Logging**.

## 🎯 **Understanding the Two Dimensions of Logging**

KrunchWrapper uses **two separate controls** for logging that work together:

### **1. Log Level - WHICH Messages Appear**
### **2. Verbose Logging - HOW DETAILED Messages Are**

This dual-control system gives you precise control over what you see and how much detail is included.

---

## 📊 **Log Level (`log_level`) - Message Filtering**

**Log Level** controls **which severity levels** of messages appear in your logs:

```jsonc
{
    "logging": {
        "log_level": "DEBUG"    // Shows: DEBUG + INFO + WARNING + ERROR
        "log_level": "INFO"     // Shows: INFO + WARNING + ERROR  
        "log_level": "WARNING"  // Shows: WARNING + ERROR
        "log_level": "ERROR"    // Shows: ERROR only
    }
}
```

### **Examples by Log Level:**

| Level | Example Messages |
|-------|-----------------|
| **DEBUG** | `"🔧 Set model context: gpt-4 (from gpt-4-turbo)"` |
| **INFO** | `"📊 Performance Metrics: 45.2 t/s"` |
| **WARNING** | `"⚠️ Target server returned 429"` |
| **ERROR** | `"❌ Failed to connect to target server"` |

### **What Each Level Shows:**

- **`"DEBUG"`**: All diagnostic messages + operational info + warnings + errors
- **`"INFO"`**: Operational information + warnings + errors (no debugging details)
- **`"WARNING"`**: Only problems and errors (clean production output)
- **`"ERROR"`**: Only serious errors (minimal output)

---

## 🔍 **Verbose Logging (`verbose_logging`) - Content Detail**

**Verbose Logging** controls **how much content detail** is shown within those messages:

### **`"verbose_logging": false`** (Concise Output)
```
INFO - 🔄 Processing chat completion request  
INFO - 🗜️ Compression applied: 45.2% ratio
INFO - ✅ Request completed successfully
```

### **`"verbose_logging": true`** (Detailed Content)
```
INFO - 🔄 Processing chat completion request
INFO - 🔍 Verbose Logging (streaming):
INFO - ================================================================================
INFO - 📝 ORIGINAL MESSAGES:
INFO -    [user] what day is today
INFO -    [assistant] I don't have access to real-time information, so I can't tell you...
INFO - 
INFO - 🗜️ COMPRESSED MESSAGES:  
INFO -    [user] what day is today
INFO -    [assistant] I don't have access to real-time information...
INFO -
INFO - 🤖 LLM RESPONSE:
INFO -    I don't have access to real-time information, so I can't tell you...
INFO - ================================================================================
INFO - ✅ Request completed successfully
```

### **What Verbose Logging Shows:**

When **enabled**, verbose logging adds:
- ✅ **Full request/response content** (original messages, compressed messages, LLM responses)
- ✅ **Before/after compression text** (see exactly what changed)
- ✅ **Detailed compression rules** (which symbols were used)
- ✅ **Token calculations** (exact compression ratios and savings)
- ✅ **Performance breakdowns** (timing details for each operation)

---

## 📈 **How Log Level + Verbose Logging Work Together**

| Log Level | Verbose Logging | Result |
|-----------|----------------|---------|
| `"INFO"` | `false` | **Clean operational info** - Perfect for production monitoring |
| `"INFO"` | `true` | **Operational info + full content details** - Good for content analysis |
| `"DEBUG"` | `false` | **All debug messages but concise** - Good for debugging logic |
| `"DEBUG"` | `true` | **Everything + full content** - Maximum detail for development |
| `"WARNING"` | `true` | **Only warnings/errors** (verbose has no effect at this level) |

---

## 🎛️ **Complete Logging Configuration Reference**

### **Basic Settings**

```jsonc
{
    "logging": {
        // WHICH messages appear
        "log_level": "INFO",              // DEBUG | INFO | WARNING | ERROR
        
        // HOW DETAILED messages are  
        "verbose_logging": false,         // true = show full content, false = concise
        
        // WHERE messages go
        "file_logging": true,             // true = save to files, false = console only
        "file_log_level": "DEBUG",        // Can be different from console level
        
        // CONSOLE CONTROL
        "hide_verbose_from_console": true, // true = verbose only in files, false = everywhere
        "simplify_terminal_output": true   // true = clean terminal format
    }
}
```

### **Advanced Settings**

```jsonc
{
    "logging": {
        // SPECIALIZED LOGGING
        "show_passthrough_requests": true,    // Show requests that skip compression
        "cline_stream_content_logging": false, // "terminal" | "file" | "both" | false
        
        // DEBUG FILTERING (requires log_level: "DEBUG")
        "debug_categories": {
            "cline_integration": true,        // Show Cline debug messages
            "compression_core": false,        // Hide compression debug messages
            "streaming_responses": true       // Show streaming debug messages
            // ... 18 more categories available
        },
        
        // ASYNC PERFORMANCE  
        "async_logging_enabled": true,        // High-performance async logging
        "async_batch_size": 50,              // Messages processed per batch
        "async_worker_timeout": 0.1          // Batch collection timeout
    }
}
```

---

## 🎭 **Common Usage Scenarios**

### **🔬 Development & Debugging**
```jsonc
{
    "logging": {
        "log_level": "DEBUG",           // See everything
        "verbose_logging": true,        // Full content details
        "hide_verbose_from_console": false, // Show in terminal
        "file_logging": true            // Also save to files
    }
}
```
**Perfect for:** Debugging compression issues, understanding request/response flow

### **🏭 Production Monitoring**
```jsonc
{
    "logging": {
        "log_level": "INFO",            // Skip debug noise
        "verbose_logging": false,       // Clean, manageable output
        "hide_verbose_from_console": true, // Clean terminal
        "file_logging": true            // Detailed logs in files
    }
}
```
**Perfect for:** Clean terminal output with detailed file logs for later analysis

### **📊 Performance Analysis**
```jsonc
{
    "logging": {
        "log_level": "INFO",            // Operational info
        "verbose_logging": true,        // See compression details
        "hide_verbose_from_console": true, // Keep terminal clean
        "show_passthrough_requests": true  // Analyze what's not compressed
    }
}
```
**Perfect for:** Understanding compression performance and effectiveness

### **🚨 Troubleshooting Issues**
```jsonc
{
    "logging": {
        "log_level": "DEBUG",           // All diagnostic info
        "verbose_logging": false,       // Keep output manageable
        "debug_categories": {
            "error_handling": true,     // Show error details
            "cline_integration": true,  // If Cline-related issue
            // Set others to false to focus
        }
    }
}
```
**Perfect for:** Focused debugging without overwhelming content

### **🧹 Minimal Output** 
```jsonc
{
    "logging": {
        "log_level": "WARNING",         // Only problems
        "verbose_logging": false,       // Concise
        "file_logging": false          // Console only
    }
}
```
**Perfect for:** Quiet operation, only see when something's wrong

---

## 🔍 **Debug Categories System**

When `log_level` is set to `"DEBUG"`, you can enable/disable specific categories:

### **Available Categories (20 total):**

#### **🔍 Request Processing & Client Detection**
- `request_processing` - Raw request analysis `[ALL REQUESTS]`, `[DEBUG]`
- `cline_integration` - Cline client detection `[CLINE]`
- `session_management` - Session ID generation `[SESSION]`

#### **🗜️ Compression Operations**
- `compression_core` - Core compression analysis
- `compression_proxy` - Proxy-level decisions `[PROXY]`
- `conversation_detection` - Multi-turn analysis `[CONVERSATION DETECTION]`
- `kv_cache_optimization` - KV cache logic `[KV DEBUG]`, `[KV CACHE]`
- `conversation_compression` - Conversation-aware compression `[CONVERSATION COMPRESS]`
- `symbol_management` - Symbol assignment and tokenization

#### **🌊 Response Processing**
- `streaming_responses` - Streaming decompression `[STREAMING DEBUG]`
- `response_decompression` - Response decompression `[NON-STREAMING DEBUG]`
- `token_calculations` - Token calculation reporting

#### **🔧 System Operations**
- `system_fixes` - System corrections `[FIX]`
- `request_filtering` - Parameter filtering `[TIMEOUT FILTER]`
- `cleanup_operations` - Artifact cleanup `[CLEANUP]`
- `error_handling` - Error reporting `[ERROR]`

#### **📡 Server Communication**
- `server_communication` - Target server communication
- `model_context` - Model context management

#### **🔬 Development & Testing**
- `payload_debugging` - Detailed payload inspection `[PAYLOAD DEBUG]` (very verbose)
- `test_utilities` - Test-specific output (very verbose)

### **Debug Category Examples:**

```jsonc
// Only show Cline-related issues
"debug_categories": {
    "cline_integration": true,
    "error_handling": true,
    // All others default to false
}

// Focus on compression performance
"debug_categories": {
    "compression_core": true,
    "compression_proxy": true, 
    "token_calculations": true,
    "error_handling": true
}
```

---

## 📁 **File vs Console Control**

### **Separate Log Levels**
```jsonc
{
    "logging": {
        "log_level": "INFO",        // Console: clean operational info
        "file_log_level": "DEBUG", // Files: detailed diagnostic info
        "file_logging": true
    }
}
```

### **Verbose Content Control**
```jsonc
{
    "logging": {
        "verbose_logging": true,           // Enable detailed content
        "hide_verbose_from_console": true  // But only show in files
    }
}
```

**Result:** Clean terminal with detailed file logs for later analysis.

---

## 🚀 **Quick Configuration Examples**

### **Copy-Paste Configurations:**

#### **Development (See Everything)**
```jsonc
{
    "logging": {
        "log_level": "DEBUG",
        "verbose_logging": true,
        "hide_verbose_from_console": false,
        "file_logging": true
    }
}
```

#### **Production (Clean + Detailed Files)**
```jsonc
{
    "logging": {
        "log_level": "INFO", 
        "verbose_logging": true,
        "hide_verbose_from_console": true,
        "file_logging": true,
        "file_log_level": "DEBUG"
    }
}
```

#### **Troubleshooting (Focused Debug)**
```jsonc
{
    "logging": {
        "log_level": "DEBUG",
        "verbose_logging": false,
        "debug_categories": {
            "error_handling": true,
            "cline_integration": true
            // Others will be false by default
        }
    }
}
```

#### **Quiet (Problems Only)**
```jsonc
{
    "logging": {
        "log_level": "WARNING",
        "verbose_logging": false
    }
}
```

---

## 🔧 **Environment Variable Overrides**

You can override any setting with environment variables:

```bash
# Basic controls
export KRUNCHWRAPPER_LOG_LEVEL=DEBUG
export KRUNCHWRAPPER_VERBOSE=true
export KRUNCHWRAPPER_FILE_LOGGING=true

# Console control  
export KRUNCHWRAPPER_HIDE_VERBOSE_CONSOLE=true
export KRUNCHWRAPPER_SHOW_PASSTHROUGH=false

# Debug category control
export KRUNCHWRAPPER_SUPPRESS_DEBUG_MODULES="core.dynamic_dictionary,core.compress"
```

---

## 📊 **Performance Considerations**

### **High-Performance Settings (Production)**
```jsonc
{
    "logging": {
        "log_level": "INFO",                // Skip debug overhead
        "verbose_logging": false,           // Minimal content processing
        "async_logging_enabled": true,      // Non-blocking logging
        "async_batch_size": 100,           // Higher throughput
        "hide_verbose_from_console": true   // Reduce I/O
    }
}
```

### **Development Settings (Full Detail)**
```jsonc
{
    "logging": {
        "log_level": "DEBUG",              // All diagnostic info
        "verbose_logging": true,           // Full content analysis
        "async_batch_size": 1,             // Immediate processing
        "file_logging": true               // Capture everything
    }
}
```

---

## 📚 **Related Documentation**

- **[Async Logging Technical Details](ASYNC_LOGGING.md)** - Performance optimization and technical implementation
- **[Debug Categories](DEBUG_CATEGORIES.md)** - Detailed guide to the 20 debug categories
- **Configuration Examples** - See `config/debug-categories-examples.jsonc`

---

## 💡 **Key Takeaways**

1. **Log Level** = **"How much"** (quantity of messages)
2. **Verbose Logging** = **"How detailed"** (quality/depth of content) 
3. **DEBUG level** shows diagnostic messages like `"🔧 Set model context"`
4. **Verbose logging** shows the actual request/response content
5. You can have **DEBUG level** with **verbose off** for diagnostics without content dumps
6. You can have **INFO level** with **verbose on** for operational monitoring with content details
7. Use **debug categories** to filter specific types of DEBUG messages
8. Use **console vs file controls** to keep terminal clean while capturing details

The verbose logging system gives you complete control over both **what you see** and **how much detail** is included, allowing you to tailor the logging output to your exact needs. 