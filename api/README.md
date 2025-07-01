# KrunchWrapper API Reference

Detailed technical reference for the KrunchWrapper compression proxy API.

## Endpoints

### POST /v1/chat/completions

Creates a chat completion with intelligent compression applied to messages.

#### Request Body

```json
{
  "model": "string",                    // Required: Model identifier
  "messages": [                         // Required: Array of chat messages
    {
      "role": "system|user|assistant",  // Required: Message role
      "content": "string",              // Required: Message content
      "name": "string"                  // Optional: Message sender name
    }
  ],
  "temperature": 0.7,                   // Optional: Sampling temperature (0-2)
  "top_p": 1.0,                        // Optional: Nucleus sampling (0-1)
  "n": 1,                              // Optional: Number of completions
  "stream": false,                     // Optional: Enable streaming
  "max_tokens": 4096,                  // Optional: Maximum tokens to generate
  "presence_penalty": 0.0,             // Optional: Presence penalty (-2 to 2)
  "frequency_penalty": 0.0,            // Optional: Frequency penalty (-2 to 2)
  "user": "string",                    // Optional: User identifier
  "filename": "string"                 // Optional: Filename for language hint
}
```

#### Response

**Standard Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response content here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  }
}
```

**Streaming Response:**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: [DONE]
```

### POST /v1/completions

Creates a text completion with compression support.

#### Request Body

```json
{
  "model": "string",              // Required: Model identifier
  "prompt": "string",             // Required: Prompt text
  "max_tokens": 100,              // Optional: Maximum tokens to generate
  "temperature": 0.7,             // Optional: Sampling temperature
  "top_p": 1.0,                  // Optional: Nucleus sampling
  "n": 1,                        // Optional: Number of completions
  "stream": false,               // Optional: Enable streaming
  "echo": false,                 // Optional: Echo prompt in response
  "stop": "string"               // Optional: Stop sequences
}
```

### POST /v1/embeddings

Creates embeddings with optional compression preprocessing.

#### Request Body

```json
{
  "model": "string",              // Required: Model identifier
  "input": "string|array",        // Required: Text(s) to embed
  "encoding_format": "float",     // Optional: float or base64
  "dimensions": 512,              // Optional: Number of dimensions
  "user": "string"               // Optional: User identifier
}
```

### GET /v1/models

Lists available models from the target LLM API.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai"
    }
  ]
}
```

## Compression Features

### Automatic Content Analysis

KrunchWrapper analyzes request content to determine optimal compression:

- **Pattern Recognition**: Identifies repeated tokens, phrases, and code structures
- **Symbol Assignment**: Maps patterns to Unicode symbols from priority-based pools
- **Efficiency Validation**: Uses model-specific tokenizers to ensure token savings
- **Threshold Management**: Only applies compression when efficiency thresholds are met

### Language Detection

Language detection works through multiple methods:

1. **Filename Parameter**: Use the `filename` parameter for explicit language hints
2. **Content Analysis**: Automatic detection based on syntax patterns
3. **Shebang Detection**: Recognition of `#!/usr/bin/python`, `#!/bin/bash`, etc.
4. **Extension Mapping**: File extension analysis when available

### System Prompt Intelligence

Advanced system prompt processing includes:

- **Multi-Source Interception**: Captures system prompts from various request formats
- **Format Detection**: Automatically identifies ChatML, Claude, Gemini, and other formats  
- **Intelligent Merging**: Combines user system prompts with compression instructions
- **Priority-Based Integration**: Ensures compression instructions take precedence while preserving user intent

## Performance Metrics

KrunchWrapper logs detailed performance metrics for every request:

### Metric Categories

**Throughput Metrics:**
- `t/s (avg)`: Average tokens per second (total tokens / total time)
- `pp t/s`: Prompt processing tokens per second
- `gen t/s`: Generation tokens per second

**Compression Metrics:**
- `compression %`: Percentage reduction in content size
- `compression tokens used`: Number of tokens saved through compression
- `dictionary chars used`: Total dictionary characters applied

**Resource Metrics:**
- `total context used`: Total tokens consumed (input + output)
- `input tokens`: Tokens used for the prompt/input
- `output tokens`: Tokens generated in the response

**Timing Metrics:**
- `total time`: Complete request processing time
- `prep`: Preprocessing and compression time
- `llm`: Time spent waiting for LLM response
- `time to first token`: Latency until first response token (streaming only)

### Logging Configuration

Control logging behavior through configuration:

```json
{
  "verbose_logging": true,        // Show original/compressed content
  "file_logging": true,          // Save logs to dated files
  "log_level": "INFO"            // DEBUG, INFO, WARNING, ERROR
}
```

**Verbose Logging Output:**
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

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid JSON, missing required fields)
- `401`: Unauthorized (invalid API key when required)
- `422`: Unprocessable Entity (validation errors)
- `500`: Internal Server Error
- `502`: Bad Gateway (target LLM API error)
- `503`: Service Unavailable (target LLM API unavailable)

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

## Client Examples

### Python with OpenAI Client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:5001/v1",
    api_key="dummy-key"
)

# Chat completion
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    filename="example.py"  # Language hint
)

# Streaming completion
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript with Fetch

```javascript
// Chat completion
const response = await fetch('http://localhost:5001/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'user', content: 'Write a JavaScript function to reverse a string' }
    ],
    filename: 'example.js'
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);

// Streaming completion
const streamResponse = await fetch('http://localhost:5001/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: 'Count to 5' }],
    stream: true
  })
});

const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ') && !line.includes('[DONE]')) {
      const data = JSON.parse(line.slice(6));
      if (data.choices[0].delta.content) {
        process.stdout.write(data.choices[0].delta.content);
      }
    }
  }
}
```

### cURL Examples

```bash
# Chat completion
curl -X POST http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'

# Streaming completion
curl -X POST http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo", 
    "messages": [{"role": "user", "content": "Count to 3"}],
    "stream": true
  }' \
  --no-buffer

# Get models
curl http://localhost:5001/v1/models
```

## Configuration

For server configuration options, see the main [Configuration Guide](../README.md#configuration).

For advanced compression settings, see [`config/README.md`](../config/README.md).

## Compatibility

KrunchWrapper is compatible with:

- **OpenAI Python SDK** (v1.0+)
- **OpenAI Node.js SDK** (v4.0+)
- **LangChain** (OpenAI integration)
- **LlamaIndex** (OpenAI integration)
- **Any OpenAI-compatible client library**

Simply change the `base_url` to point to your KrunchWrapper server and use your existing code unchanged. 