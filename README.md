# 🚀 KrunchWrapper

**Intelligent Compression Proxy for Large Language Model APIs**

![KrunchWrapper Overview](images/Screenshot-1)

KrunchWrapper is a sophisticated, high-performance compression proxy that acts as a middleman between your applications and LLM APIs. It intelligently compresses prompts using dynamic analysis to reduce token count, forwards requests to target LLMs, and decompresses responses - all while maintaining full OpenAI API compatibility.

## ✨ Key Features

### 🧠 **Intelligent Dynamic Compression**
- **Content-Agnostic Analysis**: Analyzes each prompt on-the-fly to find the most valuable compression patterns
- **Model-Aware Validation**: Uses correct tokenizers (tiktoken, transformers, SentencePiece) to ensure real token savings
- **Multi-Pass Optimization**: Advanced compression with up to 3 optimization passes for maximum efficiency
- **Conversation State Management**: Maintains compression context across conversation turns for improved efficiency

### 🔌 **Seamless API Compatibility**
- **OpenAI-Compatible**: Drop-in replacement for OpenAI API - just change the `base_url`
- **Multi-Provider Support**: Works with any OpenAI-compatible API (LocalAI, Ollama, etc.)
- **Native Anthropic Support**: Direct Claude API integration with native format support
- **Intelligent Interface Detection**: Auto-detects Cline, WebUI, SillyTavern, and Anthropic requests
- **Streaming Support**: Full support for both streaming and non-streaming responses
- **Multiple Endpoints**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`

### ⚡ **High-Performance Architecture**
- **Async Logging**: 1000x performance improvement with non-blocking logging system
- **Persistent Token Cache**: Intelligent caching with automatic cleanup and memory management
- **Optimized Model Validation**: 95%+ faster cached validations with thread-safe operations
- **Adaptive Threading**: Multi-threaded compression analysis with intelligent thread scaling

### 🎯 **Smart Content Handling**
- **Comment Stripping**: Optional removal of code comments with language-specific safety rules
- **Tool Call Protection**: Automatically preserves JSON tool calls and structured data
- **Markdown Preservation**: Maintains formatting for tables, lists, and links
- **System Prompt Intelligence**: Advanced system prompt interception and merging

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thad0ctor/KrunchWrapper.git
cd KrunchWrapper

# Run the installation script
./install.sh          # Linux/Mac
# or
.\install.ps1         # Windows
```

### Running the Server

```bash
# Start the server (automatically starts on port 5002)
./start.sh          # Linux/Mac
# or
.\start.ps1         # Windows

# This will start both the KrunchWrapper server and the WebUI
# Server: http://localhost:5002
# WebUI: http://localhost:5173
```

### Basic Usage

```python
import openai

# Point to your KrunchWrapper server
client = openai.OpenAI(
    base_url="http://localhost:5002/v1",
    api_key="dummy-key"  # Not used but required by the client
)

# Use exactly like a regular OpenAI client
response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
    ]
)

print(response.choices[0].message.content)
```

### Cline + Anthropic Integration

**⚡ Quick Setup for Cline Users:**

1. **Configure Cline Settings** - Create/edit `.vscode/settings.json` **in your project root**:
   ```json
   {
       "cline.anthropicBaseUrl": "http://localhost:5002",
       "cline.anthropicApiKey": "sk-ant-your-actual-anthropic-api-key-here"
   }
   ```
   
   📁 **File Location Example:**
   ```
   your-project/
   ├── .vscode/
   │   └── settings.json    ← Create this file here
   ├── src/
   └── README.md
   ```

2. **Start KrunchWrap** - Run the server (default port 5002):
   ```bash
   ./start.sh          # Linux/Mac
   .\start.ps1         # Windows
   ```

3. **Use Cline Normally** - KrunchWrap automatically:
   - 🔍 **Detects Cline requests** via auto-detection
   - 🗜️ **Compresses prompts** before sending to Anthropic
   - ✨ **Decompresses responses** back to Cline
   - 💰 **Saves 15-40% tokens** on every request

**🎯 Key Points:**
- **Port**: Use `5002` (KrunchWrap server port) 
- **No `/v1/messages`**: Don't add endpoint paths to base URL
- **Real API Key**: Replace with your actual `sk-ant-...` Anthropic key
- **Auto-Detection**: No manual configuration needed - works automatically!

**🔍 Troubleshooting:**
- **Not seeing requests in terminal?** Set `"log_level": "DEBUG"` in `config/server.jsonc` 
- **Still no activity?** Check your API key starts with `sk-ant-` and restart VS Code
- **404 errors?** Restart KrunchWrap server after adding Anthropic integration

---

### Direct Anthropic API Integration

For non-Cline usage, KrunchWrap provides native Anthropic API support:

```python
import anthropic

# Point to KrunchWrapper for automatic compression
client = anthropic.Anthropic(
    api_key="your-anthropic-api-key",
    base_url="http://localhost:5002"  # KrunchWrap proxy URL
)

# Native Anthropic API format with automatic compression
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="You are a helpful coding assistant.",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    max_tokens=1024
)
```

**Features:**
- 🎯 **Auto-Detection**: Automatically detects Anthropic API requests
- 📋 **Native Format**: Supports Anthropic's system parameter structure
- 🗜️ **Full Compression**: 15-40% token savings with Claude models
- ⚡ **Streaming Support**: Real-time response streaming
- 🎛️ **Multiple Interfaces**: Works with SDK, HTTP requests, and other frontends

See [`documentation/ANTHROPIC_INTEGRATION.md`](documentation/ANTHROPIC_INTEGRATION.md) for complete usage guide.

## ⚙️ Configuration

### 🎯 Quick Configuration Presets

KrunchWrapper includes pre-configured setups for common scenarios. Simply edit `config/server.jsonc` and uncomment the configuration you want to use:

#### 🏠 **Local Server Setup** (Default)
Perfect for LM Studio, Ollama, Text Generation WebUI, vLLM, LocalAI, etc.

**Flow**: `Client → KrunchWrap (compression) → Local Server → External APIs`

**Configuration**: Already active in `config/server.jsonc`
```json
{
    "target_host": "localhost",
    "target_port": 1234,              // Change to match your server
    "target_use_https": false,
    "api_key": ""
}
```

**Common Local Server Ports**:
- LM Studio: `1234`
- Ollama: `11434`  
- Text Generation WebUI: `5000` or `7860`
- vLLM: `8000`
- LocalAI: `8080`

**Client Setup Options**:
- **🌐 Embedded WebUI**: `http://localhost:5173` (starts automatically - **recommended for beginners!**)
- 🎭 SillyTavern: API URL = `http://localhost:5002/v1`
- 🔧 Cline: Use OpenAI provider with `http://localhost:5002/v1`

#### 🤖 **Direct Anthropic API Setup** ✅ TESTED & WORKING
Perfect for Cline with direct Anthropic API access.

**Flow**: `Cline → KrunchWrap (compression) → api.anthropic.com`  
**Status**: ✅ Fully tested and debugged - compression fix implemented

**Configuration**: In `config/server.jsonc`, comment out localhost config and uncomment:
```json
{
    // "target_host": "api.anthropic.com",
    // "target_port": 443,
    // "target_use_https": true,
    // "api_key": "sk-ant-your-actual-anthropic-api-key-here"
}
```

**Cline Setup** (`.vscode/settings.json`):
```json
{
    "cline.anthropicBaseUrl": "http://localhost:5002",
    "cline.anthropicApiKey": "sk-ant-your-actual-anthropic-api-key-here"
}
```

#### 🧠 **Direct OpenAI API Setup** ⚠️ EXPERIMENTAL
⚠️ **Warning**: This configuration is theoretical and has not been tested!

The direct Anthropic integration required significant debugging and fixes. Direct OpenAI integration may have similar issues that need to be resolved.

**Use at your own risk** - may not work properly without additional development.  
**For reliable OpenAI access**, use the Local Server setup with your local proxy.

**Theoretical Configuration**: In `config/server.jsonc`:
```json
{
    // "target_host": "api.openai.com",          // NOT TESTED
    // "target_port": 443,                       // MAY NOT WORK
    // "target_use_https": true,                 // EXPERIMENTAL
    // "api_key": "sk-your-actual-openai-api-key-here"
}
```

#### ⚡ **Other API Providers** ⚠️ NOT IMPLEMENTED

⚠️ **Warning**: These configurations are theoretical examples only!

Additional configurations in `config/server.jsonc` are **not implemented or tested**:
- **Google Gemini**: Would need custom endpoint handlers and testing
- **DeepSeek**: Would need testing and possible custom handling  
- **Custom Remote Servers**: Only works if server uses OpenAI-compatible format

**To actually implement these**: See `documentation/EXTENDING_KRUNCHWRAP.md` for development guide

### 🌐 **Embedded WebUI** - The Easiest Way to Get Started!

KrunchWrap includes a built-in browser-based chat interface that automatically gets compression benefits:

**🚀 Quick Start:**
1. Run `./start.sh` (Linux/Mac) or `.\start.ps1` (Windows)  
2. Open `http://localhost:5173` in your browser
3. Start chatting with automatic 15-40% token compression!

**Features:**
- 📱 Responsive design (works on desktop and mobile)
- 🗜️ Automatic compression on all messages
- ⚙️ Built-in settings panel 
- 🎨 Modern React-based interface
- 🔧 No external client configuration needed

**Flow:** `Browser → WebUI (5173) → KrunchWrap (5002) → Your Local Server`

### 🔄 How to Switch Configurations

1. **Open** `config/server.jsonc`
2. **Comment out** current active configuration (add `//` before each line)
3. **Uncomment** your desired configuration (remove `//` from each line)  
4. **Update** any specific values (ports, API keys, etc.)
5. **Restart** KrunchWrap: `./start.sh` or `.\start.ps1`

### Advanced Configuration

KrunchWrapper can be configured via command line arguments, environment variables, or JSON configuration files.

### Server Configuration (`config/server.jsonc`)

```json
{
    "host": "0.0.0.0",
    "port": 5002,
    "target_host": "localhost",
    "target_port": 1234,
    "min_compression_ratio": 0.05,
    "api_key": "your-llm-api-key",
    "verbose_logging": false,
    "file_logging": true,
    "log_level": "INFO"
}
```

### Advanced Compression Settings (`config/config.jsonc`)

```json
{
    "compression": {
        "min_characters": 250,
        "threads": 4,
        "min_token_savings": 1,
        "min_compression_ratio": 0.05,
        "aggressive_mode": false,
        "large_file_threshold": 5000,
        "cline_preserve_system_prompt": true,
        "selective_tool_call_compression": true
    },
    "dynamic_dictionary": {
        "enabled": true,
        "compression_threshold": 0.01,
        "multipass_enabled": true,
        "max_passes": 3
    },
    "comment_stripping": {
        "enabled": true,
        "preserve_license_headers": true,
        "preserve_shebang": true,
        "preserve_docstrings": true
    },
    "conversation_compression": {
        "kv_cache_threshold": 20
    },
    "streaming": {
        "preserve_sse_format": true,
        "validate_json_chunks": true,
        "cline_compatibility_mode": true
    },
    "model_tokenizer": {
        "custom_model_mappings": {
            "qwen3": ["qwen3", "qwen-3", "your-custom-qwen3-variant"]
        }
    },
    "logging": {
        "verbose": true,
        "console_level": "DEBUG"
    }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KRUNCHWRAPPER_PORT` | `5002` | Server port |
| `KRUNCHWRAPPER_HOST` | `0.0.0.0` | Server host |
| `LLM_API_URL` | `http://localhost:1234/v1` | Target LLM API URL |
| `MIN_COMPRESSION_RATIO` | `0.05` | Minimum compression ratio |
| `KRUNCHWRAPPER_VERBOSE` | `false` | Enable verbose logging |
| `KRUNCHWRAPPER_FILE_LOGGING` | `false` | Enable file logging |

## 📊 Performance & Monitoring

### Real-Time Metrics

KrunchWrapper provides comprehensive performance metrics for every request:

![Performance Metrics](images/Screenshot-2)

```
📊 Performance Metrics (chat/completions):
   t/s (avg): 42.5              // Average tokens per second
   pp t/s: 150.2                // Prompt processing tokens per second  
   gen t/s: 35.8                // Generation tokens per second
   compression %: 15.3%         // Content size reduction
   compression tokens used: 89  // Tokens saved through compression
   total context used: 1,847    // Total tokens consumed
   input tokens: 1,245          // Input tokens
   output tokens: 602           // Generated tokens
   total time: 1.85s (prep: 0.08s, llm: 1.77s)  // Timing breakdown
```

### Verbose Logging

Enable detailed content logging to see exactly what's being compressed:

```
🔍 Verbose Logging (chat/completions):
================================================================================
📝 ORIGINAL MESSAGES:
   [user] Here's some Python code that needs optimization...

🗜️  COMPRESSED MESSAGES:  
   [user] Here's α code β optimization...

🤖 LLM RESPONSE:
   Great question! Here are several ways to optimize your Python code...
================================================================================
```

## 🎯 Compression Behavior

### Automatic Compression

KrunchWrapper automatically:
1. **Analyzes Content**: Identifies repeated patterns, tokens, and structures
2. **Generates Symbols**: Assigns optimal Unicode symbols from priority-based pools  
3. **Validates Efficiency**: Uses model-specific tokenizers to ensure real token savings
4. **Adds Decoder**: Includes minimal decompression instructions only when beneficial
5. **Decompresses Responses**: Restores original tokens in responses seamlessly

### Compression Modes

- **Normal Mode** (250-999 characters): Token-optimized compression prioritizing actual token savings
- **Aggressive Mode** (1000+ characters): Character-optimized compression for maximum reduction
- **Multipass Mode**: Up to 3 optimization passes for complex content

### Typical Results

- **20-30%** compression for typical source code files
- **40-50%** compression for files with repeated patterns  
- **10-15%** compression for unique/generated content
- **30-60%** additional savings with comment stripping enabled

## 🔧 Advanced Features

### Model-Specific Tokenizer Validation

KrunchWrapper automatically detects your model and uses the appropriate tokenizer for accurate token counting:

#### Supported Model Families

| **Model Family** | **Detection Patterns** | **Tokenizer Library** | **Examples** |
|------------------|------------------------|----------------------|--------------|
| **OpenAI** | `gpt-4`, `gpt-3.5`, `turbo` | tiktoken | `gpt-4`, `gpt-3.5-turbo`, `openai/gpt-4` |
| **Anthropic** | `claude`, `anthropic` | SentencePiece | `claude-3-5-sonnet`, `anthropic/claude-3-haiku` |
| **LLaMA** | `llama`, `llama2`, `llama-3` | SentencePiece/tiktoken | `meta-llama/Llama-3-8B-Instruct`, `llama-2-7b` |
| **Mistral** | `mistral`, `mixtral` | SentencePiece | `mistralai/Mistral-7B-Instruct`, `mixtral-8x7b` |
| **Qwen** | `qwen`, `qwen2`, `qwen3` | tiktoken | `Qwen/Qwen2.5-Coder-32B-Instruct`, `qwen-7b` |
| **Google** | `gemini`, `bard`, `palm` | SentencePiece | `google/gemini-pro`, `palm2` |
| **Others** | `yi-`, `deepseek`, `phi-` | Various | `01-ai/Yi-34B-Chat`, `deepseek-coder`, `microsoft/phi-2` |

#### How to Use Custom Model Mappings

**📋 Step-by-Step Guide**:

1. **Edit your configuration file** (`config/config.jsonc`):
   ```json
   {
       "model_tokenizer": {
           "custom_model_mappings": {
               "gpt-4": ["my-custom-gpt", "exactly-a"],
               "claude": ["my-claude", "internal-assistant"],
               "llama": ["local-llama", "company-llm"]
           }
       }
   }
   ```

2. **Restart KrunchWrapper** to reload the configuration:
   ```bash
   ./start.sh          # Linux/Mac
   # or
   .\start.ps1         # Windows
   ```

3. **Verify in logs** that custom mappings are loaded:
   ```
   INFO - Loading 3 custom model mappings
   INFO - Extended gpt-4 patterns with: ['my-custom-gpt', 'exactly-a']
   ```

4. **Test your model detection**:
   ```bash
   # Your API calls with custom model names will now work:
   curl -X POST http://localhost:5002/v1/chat/completions \
     -d '{"model": "my-custom-gpt", "messages": [...]}'
   ```

#### Troubleshooting Model Detection

**❌ Common Issue**: `WARNING - Unknown model family for: a`

This warning appears when your model name doesn't match any supported patterns. The system falls back to character-based estimation, which still works but is less accurate.

**✅ Solutions**:
1. **Check your API configuration** - Ensure you're sending a real model name like `gpt-4` instead of generic names like `"a"`
2. **Verify provider settings** - Many providers allow setting the model name in environment variables or config files
3. **Add custom patterns** - You can extend model detection in `config/config.jsonc`:

```json
{
    "model_tokenizer": {
        "custom_model_mappings": {
            "gpt-4": ["my-custom-gpt", "company-model"],
            "claude": ["my-claude", "internal-assistant"],
            "generic_model": ["exactly-a", "model-v1"]
        }
    }
}
```

⚠️ **Pattern Matching Notes**:
- Patterns are matched as **case-insensitive substrings**
- Use **specific patterns** to avoid false matches (e.g., `"exactly-a"` instead of `"a"`)
- Pattern `"a"` would incorrectly match `"llama"`, `"claude"`, etc.
- All patterns are automatically converted to lowercase

**Expected Model Names**:
- ✅ `gpt-4`, `claude-3-5-sonnet`, `llama-3-8b-instruct`
- ❌ `a`, `model`, `llm`, `ai`

### Comment Stripping

Language-aware comment removal with safety features:
- **Multi-language support**: Python, JavaScript, C/C++, HTML, CSS, SQL, Shell
- **Smart preservation**: License headers, shebangs, docstrings
- **Significant savings**: 30-60% token reduction on heavily commented code

### System Prompt Intelligence

Advanced system prompt processing:
- **Multi-source interception**: Handles various prompt formats and sources
- **Intelligent merging**: Priority-based combination of user and compression instructions
- **Format conversion**: Seamless transformation between ChatML, Claude, Gemini formats
- **Cline integration**: Specialized handling for Cline development tool requests

## 📚 Documentation Reference

### Quick Links
- **Anthropic Integration**: [`documentation/ANTHROPIC_INTEGRATION.md`](documentation/ANTHROPIC_INTEGRATION.md) - Native Claude API support guide
- **Logging Guide**: [`documentation/LOGGING_GUIDE.md`](documentation/LOGGING_GUIDE.md) - Complete logging configuration guide
- **Debug Categories**: [`documentation/DEBUG_CATEGORIES.md`](documentation/DEBUG_CATEGORIES.md) - Fine-grained debug logging control
- **Model Tokenizer Setup**: [How to Use Custom Model Mappings](#how-to-use-custom-model-mappings)
- **Troubleshooting**: [Model Detection Issues](#troubleshooting-model-detection)
- **Configuration**: [`config/README.md`](config/README.md) - Detailed configuration guide
- **API Reference**: [`api/README.md`](api/README.md) - Complete API documentation
- **Architecture**: [`charts/README.md`](charts/README.md) - System flow diagrams

### Common Tasks
- **Fix "Unknown model family" warning**: Add custom patterns in `config/config.jsonc`
- **Test configuration**: Run `python tests/test_case_insensitive_tokenizer.py`
- **Monitor performance**: Enable verbose logging to see compression stats
- **Troubleshoot compression**: Check logs for compression ratios and token savings

## 📈 Architecture

### Project Structure

- **`api/`**: FastAPI server and request handling
- **`core/`**: Compression engine and intelligence modules  
- **`config/`**: Configuration files and schemas
- **`dictionaries/`**: Priority-based symbol pools for compression
- **`documentation/`**: Detailed feature documentation
- **`charts/`**: System flow diagrams and architecture charts
- **`tests/`**: Comprehensive test suite
- **`utils/`**: Analysis and debugging utilities

### Core Modules

- **Dynamic Analysis**: `dynamic_dictionary.py` - On-the-fly pattern analysis
- **Compression Engine**: `compress.py` - Main compression orchestration
- **System Prompts**: `system_prompt_interceptor.py` - Intelligent prompt handling
- **Model Validation**: `model_tokenizer_validator.py` - Accurate token counting
- **Performance**: `async_logger.py`, `persistent_token_cache.py` - High-performance utilities

## 🚀 Performance Optimizations

### Async Logging System
- **Enabled by default** for 1000x performance improvement
- **Environment detection**: Smart defaults for development vs production
- **100,000+ messages/second** throughput capability

### Optimized Model Validator  
- **Result caching**: 95%+ faster for repeated validations
- **Batch operations**: Efficient processing of multiple validations
- **Thread safety**: Proper locking with no performance penalty

### Persistent Token Cache
- **Intelligent caching**: Automatic cleanup and memory management
- **Disk persistence**: Survives server restarts
- **Statistics monitoring**: Built-in performance tracking

## 📄 License

[MIT License](LICENSE)

---

**KrunchWrapper**: Making LLM APIs more efficient, one token at a time. 🗜️✨ 

## 🚀 Troubleshooting

### "ModuleNotFoundError: No module named 'uvicorn'" when running start script

If you get this error when running `.\start.ps1` or `./start.sh`, it means the virtual environment isn't properly activated. This has been fixed in recent versions, but if you encounter it:

**Solution:**
1. **Run the install script first**: `.\install.ps1` (Windows) or `./install.sh` (Linux/Mac)
2. **Verify installation**: The install script now includes dependency verification
3. **Try starting again**: `.\start.ps1` or `./start.sh`

**Manual verification** (if needed):
```bash
# Activate virtual environment manually
.venv\Scripts\Activate.ps1    # Windows PowerShell
# or
source .venv/bin/activate     # Linux/Mac

# Test dependencies
python -c "import uvicorn, fastapi; print('✅ Dependencies OK')"

# Start server using the provided scripts
./start.sh          # Linux/Mac
# or
.\start.ps1         # Windows
```

### Start script exits immediately

This is **normal behavior**! The start script:
1. ✅ Shows startup message
2. ✅ Creates a **separate window** where the server runs
3. ✅ Returns control to your original terminal

Look for a **new PowerShell/Terminal window** where the actual services are running.

### WebUI not starting

If only the server (port 5002) starts but not the WebUI (port 5173):
1. **Check Node.js**: `node --version` and `npm --version`
2. **Install WebUI deps**: `cd webui && npm install`
3. **Start manually**: `cd webui && npm run dev`

## Common Issues and Solutions

#### API Request Failures with Short Messages

**Symptoms:**
- API requests fail when sending very short messages (1-5 characters)
- Logs show "KV cache threshold: 0 chars" 
- Compression disabled due to "poor efficiency trend"

**Cause:** KV cache optimization is disabled when threshold is set to 0

**Solution:**
```json
// In config/config.jsonc
"conversation_compression": {
    "kv_cache_threshold": 20  // Enable KV cache for messages < 20 chars
}
```

#### Tokenizer Validation Failures

**Symptoms:**
- Logs show "tiktoken not available", "transformers not available"
- All tokenizer validation falls back to character estimation
- Poor compression efficiency calculations

**Solution:**
```bash
# Install required tokenizer libraries
source venv/bin/activate
pip install tiktoken transformers sentencepiece
```

#### Model Family "Unknown" Warnings

**Symptoms:**
- Logs show "Unknown model family for: [model_name]"
- Model-specific tokenization falls back to generic methods

**Solution:**
Add custom model mappings in `config/config.jsonc`:
```json
"model_tokenizer": {
    "custom_model_mappings": {
        "qwen3": ["qwen3", "qwen-3", "your-custom-qwen3-variant"]
    }
}
```

#### Cline/Cursor Integration Issues

**Symptoms:**
- Responses not showing up in Cline/Cursor
- "Unexpected API Response" errors
- SSE streaming failures

**Solution:**
Ensure proper configuration in `config/config.jsonc`:
```json
"compression": {
    "cline_preserve_system_prompt": true,
    "selective_tool_call_compression": true
},
"streaming": {
    "preserve_sse_format": true,
    "validate_json_chunks": true,
    "cline_compatibility_mode": true
}
```

### Performance Optimization

#### For Maximum Compression
- Set `"kv_cache_threshold": 0` to disable KV cache
- Enable `"multi_pass_adaptive": true` for advanced compression
- Increase `"max_dictionary_size": 300` for larger dictionaries

#### For Best Responsiveness  
- Set `"kv_cache_threshold": 30` for aggressive KV cache usage
- Enable `"smart_decompression": true` for faster streaming
- Use `"min_compression_ratio": 0.05` to skip marginal compression

### Debug Logging

Enable verbose logging to troubleshoot issues:
```json
"logging": {
    "verbose": true,
    "console_level": "DEBUG"
}
```

Common debug patterns to look for:
- `🚀 [KV CACHE]` - KV cache optimization triggers
- `🗜️ Dynamic compression` - Compression analysis
- `❌ Error in` - System errors requiring attention
- `⚠️ WARNING` - Non-critical issues that may affect performance 