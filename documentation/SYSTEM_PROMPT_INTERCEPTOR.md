# System Prompt Interceptor

## Overview

The System Prompt Interceptor is an advanced component of KrunchWrapper that automatically captures, analyzes, converts, and merges system prompts from incoming API requests with KrunchWrapper's compression instructions.

## Key Features

- **Multi-Format Interception**: Captures system prompts from various sources (system messages, system parameters, system instructions)
- **Intelligent Format Detection**: Automatically detects the format of intercepted system prompts
- **Format Conversion**: Converts between different system prompt formats seamlessly  
- **Smart Merging**: Intelligently merges user system prompts with KrunchWrapper compression instructions
- **Robust Fallback**: Graceful error handling with fallback mechanisms

## Supported Formats

The interceptor supports all formats defined in `config/system-prompts.jsonc`:

- **Claude** (`claude`): Uses separate `system` parameter
- **ChatGPT** (`chatgpt`): Uses system role in messages array (ChatML)
- **ChatML** (`chatml`): Standard ChatML format used by many providers
- **Gemini** (`gemini`): Uses `system_instruction` with parts structure
- **Qwen** (`qwen`): OpenAI-compatible ChatML format
- **DeepSeek** (`deepseek`): OpenAI-compatible ChatML format
- **Gemma** (`gemma`): Turn-based template format with markers
- **Claude Legacy** (`claude_legacy`): Plain text with Human/Assistant markers

## How It Works

### 1. Interception Phase
The interceptor captures system prompts from multiple sources:

```python
# From messages array (ChatML style)
{"role": "system", "content": "You are helpful"}

# From system parameter (Claude style)  
{"system": "You are helpful", "messages": [...]}

# From system_instruction (Gemini style)
{"system_instruction": {"parts": [{"text": "You are helpful"}]}, "messages": [...]}
```

### 2. Format Detection Phase
Analyzes content to detect specific format patterns:

- **Gemma**: Looks for `<start_of_turn>` and `<end_of_turn>` markers
- **Claude Legacy**: Detects `Human:` and `Assistant:` patterns
- **Gemini**: Identifies JSON structures with `parts` arrays
- **ChatML**: Default for role-based message structures

### 3. Content Cleaning Phase
Removes format-specific structural elements:

```python
# Gemma: <start_of_turn>system\nYou are helpful<end_of_turn>
# Becomes: "You are helpful"

# Claude Legacy: "Human: You are helpful\n\nAssistant:"  
# Becomes: "You are helpful"

# Gemini: {"parts": [{"text": "You are helpful"}]}
# Becomes: "You are helpful"
```

### 4. Merging Phase
Combines prompts with prioritization:

1. **KrunchWrapper compression instructions** (highest priority)
2. **User's original system prompt content**
3. **Format-specific structural requirements**

```python
# Result combines both:
"""
You will read python code in a compressed DSL. Expand: fn=function, cls=class.
IMPORTANT: When responding with code, use compressed format...

You are a helpful coding assistant specialized in Python development.
"""
```

### 5. Format Conversion Phase
Converts merged content to target format specified in configuration.

### 6. Application Phase
Applies the processed system prompt appropriately:
- **ChatML formats**: Adds as system message in array
- **Claude format**: Moves to separate `system` parameter
- **Gemini format**: Structures as `system_instruction` with parts

## Configuration

System prompt format is configured in `config/config.jsonc`:

```jsonc
{
  "system_prompt": {
    "format": "claude"  // Target format for system prompts
  }
}
```

## API Integration

The interceptor is automatically integrated into the API server at `/v1/chat/completions`. It processes:

- Standard ChatML requests
- Claude-style requests with system parameter
- Gemini-style requests with system_instruction
- Mixed format requests

## Error Handling

The interceptor includes comprehensive error handling:

- **Format Detection Failure**: Falls back to ChatML format
- **Conversion Errors**: Uses ChatML as universal fallback
- **Processing Errors**: Applies compression-only fallback
- **Complete Failure**: Returns original messages unchanged

## Logging

The interceptor provides detailed logging:

```
INFO: Successfully processed system prompts. Intercepted: 1, Target format: claude
ERROR: Error converting to format 'invalid': Unknown format
WARNING: Applying fallback system prompt processing
```

## Usage Examples  

### Example 1: ChatML to Claude Conversion

**Input:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a Python expert"},
    {"role": "user", "content": "Help with my code"}
  ]
}
```

**Output (for Claude format):**
```json
{
  "system": "You will read python code in compressed DSL...\n\nYou are a Python expert",
  "messages": [
    {"role": "user", "content": "Help with my code"}
  ]
}
```

### Example 2: Gemma Format Detection

**Input:**
```json
{
  "messages": [
    {"role": "system", "content": "<start_of_turn>system\nYou are helpful<end_of_turn>"}
  ]
}
```

**Processed:**
- Detects Gemma format
- Cleans content to "You are helpful"  
- Merges with compression instructions
- Converts to target format

## Implementation Details

### Core Files

- **`core/system_prompt_interceptor.py`**: Main interceptor logic
- **`api/server.py`**: Integration with API server
- **`core/system_prompt.py`**: Format handling utilities
- **`config/system-prompts.jsonc`**: Format definitions

### Key Classes

- **`SystemPromptInterceptor`**: Main interceptor class
- **`SystemPromptFormatter`**: Format conversion utilities

### Integration Points

The interceptor integrates at the message processing stage in `api/server.py`, after compression but before forwarding to the target LLM API.

## Benefits

1. **Seamless Integration**: Works with any system prompt format automatically
2. **Compression Preservation**: Always preserves KrunchWrapper compression functionality  
3. **User Prompt Preservation**: Maintains user's original system prompt instructions
4. **Format Flexibility**: Converts between any supported formats transparently
5. **Robust Operation**: Graceful fallbacks ensure system never fails

## Future Enhancements

- Support for additional custom formats
- Advanced conflict resolution strategies
- Performance optimizations for high-volume usage
- Enhanced logging and debugging capabilities 