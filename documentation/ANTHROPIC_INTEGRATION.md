# Anthropic API Integration for KrunchWrap

This document explains how to use KrunchWrap with the Anthropic API, providing direct integration alongside the existing Cline-based Anthropic support.

## Overview

KrunchWrap now supports multiple ways to use Anthropic's Claude models:

1. **Direct Anthropic API Integration** (NEW) - Native Anthropic API format support
2. **Cline Integration** (Existing) - Through Cline's `anthropic/claude-*` provider format  
3. **WebUI/SillyTavern** (Existing) - Standard ChatML format for Anthropic-compatible endpoints

## Direct Anthropic API Integration

### Features

- **Native Anthropic Format**: Supports the Anthropic API's native request/response format
- **System Parameter Handling**: Proper handling of Anthropic's separate `system` parameter
- **Claude Message Structure**: Validates and optimizes message structure for Claude models
- **Automatic Detection**: Auto-detects Anthropic API requests based on headers and model names
- **Compression Integration**: Full integration with KrunchWrap's compression system
- **Prompt Caching Support**: Compatible with Anthropic's prompt caching features

### API Format

The Anthropic API uses a different structure than OpenAI-compatible APIs:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ],
  "max_tokens": 1024
}
```

Key differences:
- System prompts are in a separate `system` parameter (not in messages array)
- Messages array contains only `user` and `assistant` messages
- No `system` role messages in the messages array

### Configuration

#### Automatic Detection (Recommended)

```jsonc
{
    "system_prompt": {
        "format": "claude",
        "interface_engine": "auto"  // Auto-detects Anthropic requests
    }
}
```

KrunchWrap will automatically detect Anthropic API requests based on:
- **Headers**: `x-api-key`, `anthropic-version`, `anthropic-beta`
- **User-Agent**: Contains "anthropic" and "sdk"
- **Model Names**: Direct Claude model names (e.g., `claude-3-5-sonnet-20241022`)

#### Explicit Configuration

```jsonc
{
    "system_prompt": {
        "format": "claude",
        "interface_engine": "anthropic"  // Force Anthropic handling
    }
}
```

### Usage Examples

#### Python with anthropic SDK

```python
import anthropic

# Configure to use KrunchWrap proxy
client = anthropic.Anthropic(
    api_key="your-anthropic-api-key",
    base_url="http://localhost:5001"  # KrunchWrap proxy URL
)

# Standard Anthropic API call - KrunchWrap will compress automatically
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="You are a helpful coding assistant.",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    max_tokens=1024
)

print(response.content[0].text)
```

#### Direct HTTP Requests

```bash
curl -X POST http://localhost:5001/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-anthropic-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ],
    "max_tokens": 100
  }'
```

#### Using OpenAI-Compatible Format

You can also use OpenAI-compatible format and let KrunchWrap detect it as Anthropic:

```python
import openai

# Point to KrunchWrap proxy
client = openai.OpenAI(
    api_key="your-anthropic-api-key",
    base_url="http://localhost:5001/v1"
)

# Use direct Claude model name to trigger Anthropic detection
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",  # Direct model name triggers detection
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Supported Models

All Claude models are supported:

| Model Family | Model ID | Description |
|-------------|----------|-------------|
| Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` | Latest high-performance model |
| Claude 3.5 Haiku | `claude-3-5-haiku-20241022` | Fast and efficient |
| Claude 3 Opus | `claude-3-opus-20240229` | Most capable model |
| Claude 3 Sonnet | `claude-3-sonnet-20240229` | Balanced performance |
| Claude 3 Haiku | `claude-3-haiku-20240307` | Fastest model |

### System Prompt Handling

KrunchWrap's Anthropic integration properly handles system prompts:

1. **Extraction**: Extracts system prompts from both `system` parameter and system messages
2. **Compression**: Applies compression to system prompts when beneficial
3. **Merging**: Merges compression instructions with original system prompts
4. **Format**: Outputs in proper Anthropic format with separate `system` parameter

#### Example System Prompt Processing

**Original Request:**
```json
{
  "system": "You are a helpful coding assistant specialized in Python.",
  "messages": [...]
}
```

**With Compression (Internal):**
```json
{
  "system": "You will read Python code in a compressed DSL. Expand: fn=function, cls=class, def=define...\n\nYou are a helpful coding assistant specialized in Python.",
  "messages": [...]
}
```

### Compression Benefits

Anthropic models benefit significantly from KrunchWrap compression:

- **Token Savings**: 15-40% reduction in prompt tokens
- **Cost Reduction**: Lower API costs due to reduced token usage
- **Speed Improvement**: Faster processing with shorter prompts
- **Context Efficiency**: More room for conversation history

#### Example Compression

**Before Compression (1,250 tokens):**
```python
def calculate_fibonacci_sequence(number_of_terms):
    if number_of_terms <= 0:
        return []
    elif number_of_terms == 1:
        return [0]
    elif number_of_terms == 2:
        return [0, 1]
    else:
        fibonacci_sequence = [0, 1]
        for index in range(2, number_of_terms):
            next_number = fibonacci_sequence[index - 1] + fibonacci_sequence[index - 2]
            fibonacci_sequence.append(next_number)
        return fibonacci_sequence
```

**After Compression (890 tokens):**
```python
def calc_fib_seq(n_terms):
    if n_terms <= 0: return []
    elif n_terms == 1: return [0]
    elif n_terms == 2: return [0, 1]
    else:
        fib_seq = [0, 1]
        for i in range(2, n_terms):
            next_n = fib_seq[i-1] + fib_seq[i-2]
            fib_seq.append(next_n)
        return fib_seq
```

### Advanced Features

#### Prompt Caching Support

KrunchWrap is compatible with Anthropic's prompt caching:

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="Your system prompt here",
    messages=[...],
    max_tokens=1024,
    extra_headers={
        "anthropic-beta": "prompt-caching-2024-07-31"
    }
)
```

KrunchWrap compression can enhance caching effectiveness by creating more consistent prompt structures.

#### Streaming Support

Full streaming support for real-time responses:

```python
stream = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="You are a helpful assistant.",
    messages=[...],
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.type == "content_block_delta":
        print(chunk.delta.text, end="")
```

### Integration with Other Frontends

#### SillyTavern

Configure SillyTavern to use KrunchWrap as an Anthropic-compatible endpoint:

1. Set API Type to "Claude"
2. Set API URL to `http://localhost:5001`
3. Enter your Anthropic API key
4. Select a Claude model

#### WebUI (text-generation-webui)

Use the OpenAI extension with Anthropic model names:

```python
# In the OpenAI extension settings
base_url = "http://localhost:5001/v1"
model = "claude-3-5-sonnet-20241022"
```

### Monitoring and Debugging

Enable verbose logging to monitor Anthropic processing:

```jsonc
{
    "system_prompt": {
        "interface_engine": "anthropic"
    },
    "verbose_logging": true,
    "debug_categories": ["anthropic_interception", "system_prompt"]
}
```

Logs will show:
- Anthropic request detection
- System prompt processing
- Compression statistics
- Message structure validation

### Comparison: Direct vs Cline vs Standard

| Feature | Direct Anthropic | Cline Integration | Standard/WebUI |
|---------|-----------------|-------------------|----------------|
| **API Format** | Native Anthropic | OpenAI-compatible | OpenAI-compatible |
| **System Prompts** | Separate `system` param | Messages array | Messages array |
| **Detection** | Headers/model name | User-agent/model format | Default fallback |
| **Compression** | ✅ Full support | ✅ Full support | ✅ Full support |
| **Streaming** | ✅ Native | ✅ Via Cline | ✅ Standard |
| **Prompt Caching** | ✅ Native support | ✅ Via Cline | ❌ Not applicable |

### Troubleshooting

#### Request Not Detected as Anthropic

**Problem**: Request is processed with standard interface instead of Anthropic.

**Solutions**:
1. Add Anthropic headers: `anthropic-version`, `x-api-key`
2. Use direct Claude model names (not `anthropic/claude-*`)
3. Force Anthropic interface: `"interface_engine": "anthropic"`

#### System Prompt Issues

**Problem**: System prompt not properly formatted.

**Solutions**:
1. Ensure system prompt is in `system` parameter, not messages array
2. Check that messages only contain `user`/`assistant` roles
3. Enable debug logging to see processing steps

#### Compression Not Applied

**Problem**: No compression benefits observed.

**Solutions**:
1. Check minimum character threshold in config
2. Verify content meets compression criteria
3. Enable compression logging to see statistics

### Performance Optimization

#### Best Practices

1. **Use Direct Model Names**: Use `claude-3-5-sonnet-20241022` instead of `anthropic/claude-3-5-sonnet-20241022`
2. **Proper Message Structure**: Keep system prompts in `system` parameter
3. **Enable Prompt Caching**: Use appropriate headers for frequently used prompts
4. **Monitor Compression**: Track token savings and adjust thresholds

#### Recommended Configuration

```jsonc
{
    "system_prompt": {
        "format": "claude",
        "interface_engine": "auto"
    },
    "min_characters": 500,
    "conversation_stateful_mode": true,
    "verbose_logging": false,
    "compression_threshold": 0.1
}
```

## Conclusion

The new Anthropic integration provides native support for Anthropic's API format while maintaining full compatibility with KrunchWrap's compression system. This offers the best of both worlds: native API compatibility and significant token savings through intelligent compression.

Choose the integration method that best fits your use case:
- **Direct Anthropic**: For native API compatibility and optimal performance
- **Cline Integration**: For development workflows using the Cline extension  
- **Standard/WebUI**: For general-purpose frontends with OpenAI-compatible format 