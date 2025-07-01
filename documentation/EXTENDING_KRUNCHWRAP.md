# Extending KrunchWrap: Adding New Endpoints and Interface Engines

This guide covers how to extend KrunchWrap to support new API providers and new client interfaces. Whether you want to add support for a new LLM API (like Gemini, DeepSeek, etc.) or integrate a new coding assistant (like Aider, Roo Code, etc.), this document provides step-by-step instructions.

## Table of Contents

1. [Adding New API Endpoints](#adding-new-api-endpoints)
2. [Adding New Interface Engines](#adding-new-interface-engines)
3. [Testing Your Integration](#testing-your-integration)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Adding New API Endpoints

This section covers adding support for new LLM APIs (like Gemini, OpenAI, DeepSeek, etc.) that can be used with existing clients like Cline.

### Overview

When adding a new API endpoint, you need to:
1. Create an endpoint handler that understands the API's request/response format
2. Handle authentication and headers specific to that API
3. Implement streaming if the API supports it
4. Add configuration options
5. Integrate with KrunchWrap's compression pipeline

### Step 1: Create the Endpoint Handler

Create a new file in `server/endpoints/` for your API:

```python
# server/endpoints/gemini.py
import json
import time
from typing import Dict, Any, Optional, List, Union
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from server.config import ServerConfig
from server.logging_utils import log_message, log_performance_metrics
from server.streaming import create_aiohttp_session, create_streaming_timeout
from server.request_utils import (
    detect_and_process_compression, filter_timeout_parameters,
    extract_model_from_provider_format, set_global_model_context
)

# Define API-specific request models
class GeminiMessage(BaseModel):
    role: str
    parts: List[Dict[str, Any]]

class GeminiRequest(BaseModel):
    model: str
    contents: List[GeminiMessage]
    generation_config: Optional[Dict[str, Any]] = None
    safety_settings: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = False
    
    class Config:
        extra = "allow"

async def gemini_generate(request: GeminiRequest, http_request: Request, config: ServerConfig):
    """Handle Gemini API requests"""
    start_time = time.time()
    
    log_message(f"🔧 [GEMINI] Processing Gemini API request", "INFO", config)
    log_message(f"🔧 [GEMINI] Model: {request.model}, Messages: {len(request.contents)}, Stream: {request.stream}", "INFO", config)
    
    # Set model context for tokenizer validation
    if request.model:
        clean_model = extract_model_from_provider_format(request.model)
        set_global_model_context(clean_model)
    
    # Convert to internal format for compression processing
    messages_dict = []
    for content in request.contents:
        # Extract text from Gemini's parts format
        text_parts = []
        for part in content.parts:
            if "text" in part:
                text_parts.append(part["text"])
        
        messages_dict.append({
            'role': content.role,
            'content': ' '.join(text_parts)
        })
    
    # Calculate content size and determine if compression should be applied
    total_content_size = sum(len(msg['content']) for msg in messages_dict)
    should_compress = total_content_size >= config.min_characters
    
    log_message(f"📏 Total content size: {total_content_size} chars, Min threshold: {config.min_characters} chars", config=config)
    log_message(f"🗜️  Compression {'enabled' if should_compress else 'skipped'}", config=config)
    
    # Process through compression pipeline
    rule_union: Dict[str, str] = {}
    compression_ratio = 0.0
    dynamic_chars_used = 0
    compression_tokens_saved = 0
    
    if should_compress:
        log_message(f"🗜️ [GEMINI] Using compression pipeline", "INFO", config)
        
        # Process through Interface Engine
        processed_messages, system_metadata, engine = detect_and_process_compression(
            request=http_request,
            messages=messages_dict,
            rule_union=rule_union,
            config=config,
            model_id=request.model,
            system_param=None,
            system_instruction=None,
            target_format="gemini",
            request_data={"model": request.model, "contents": request.contents}
        )
        
        # Convert back to Gemini format with compression applied
        compressed_contents = []
        for i, msg in enumerate(processed_messages):
            original_parts = request.contents[i].parts if i < len(request.contents) else [{"text": msg['content']}]
            compressed_parts = []
            
            for part in original_parts:
                if "text" in part:
                    # Apply compression to text parts
                    part["text"] = msg['content']
                compressed_parts.append(part)
            
            compressed_contents.append({
                "role": msg['role'],
                "parts": compressed_parts
            })
        
        # Calculate compression metrics
        original_size = sum(len(msg['content']) for msg in messages_dict)
        compressed_size = sum(len(msg['content']) for msg in processed_messages)
        compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0
        compression_tokens_saved = max(0, (original_size - compressed_size) // 4)
        
        log_message(f"🗜️ [GEMINI] Compression complete: {len(rule_union)} rules applied", config=config)
    else:
        compressed_contents = [content.dict() for content in request.contents]
    
    # Prepare payload for Gemini API
    payload = {
        "contents": compressed_contents,
        "generationConfig": request.generation_config or {}
    }
    
    # Add optional parameters
    if request.safety_settings:
        payload["safetySettings"] = request.safety_settings
    if request.stream:
        payload["generationConfig"]["stream"] = True
    
    # Filter timeout parameters
    payload = filter_timeout_parameters(payload, config.filter_timeout_parameters)
    
    # Forward to Gemini API
    llm_start_time = time.time()
    target_url = f"{config.target_url}/v1/models/{request.model}:generateContent"
    
    try:
        async with create_aiohttp_session() as session:
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": config.api_key  # Gemini uses x-goog-api-key
            }
            
            if request.stream:
                # Handle streaming (implement similar to Anthropic)
                return await handle_gemini_streaming_response(
                    target_url, payload, headers, rule_union,
                    start_time, llm_start_time, compression_ratio,
                    compression_tokens_saved, dynamic_chars_used, config
                )
            else:
                # Handle non-streaming
                async with session.post(target_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log_message(f"🚨 [GEMINI ERROR] Target API returned {resp.status}: {error_text}", config=config)
                        raise HTTPException(status_code=resp.status, detail=error_text)
                    
                    data = await resp.json()
                    
                    # Decompress response if needed
                    if rule_union:
                        data = decompress_gemini_response(data, rule_union)
                    
                    # Log metrics and return
                    end_time = time.time()
                    log_message(f"✅ [GEMINI] Successfully processed request", "INFO", config)
                    
                    return data
                    
    except Exception as e:
        log_message(f"🚨 [GEMINI ERROR] Request failed: {e}", config=config)
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {str(e)}")

# Add streaming and decompression functions...
async def handle_gemini_streaming_response(target_url, payload, headers, rule_union, start_time, llm_start_time, compression_ratio, compression_tokens_saved, dynamic_chars_used, config):
    """Handle Gemini-specific streaming format"""
    # Implement Gemini-specific streaming format
    pass

def decompress_gemini_response(data, rule_union):
    """Implement Gemini-specific response decompression"""
    # Implement response decompression for Gemini format
    pass
```

### Step 2: Register the Endpoint

Add your endpoint to the main server file (`api/server.py`):

```python
# In api/server.py
from server.endpoints.gemini import gemini_generate, GeminiRequest

# Add the route
@app.post("/v1/models/{model}:generateContent")
async def gemini_endpoint(model: str, request: GeminiRequest, http_request: Request):
    """Gemini API generateContent endpoint"""
    config = ServerConfig()
    request.model = model  # Set model from URL parameter
    return await gemini_generate(request, http_request, config)
```

### Step 3: Update Configuration

Add API-specific configuration to `config/server.jsonc`:

```jsonc
{
    // Existing config...
    
    // For Gemini API
    "target_host": "generativelanguage.googleapis.com",
    "target_port": 443,
    "target_use_https": true,
    
    // API key will be in x-goog-api-key header
    "api_key": "your-gemini-api-key"
}
```

### Step 4: Add Model Detection

Update interface detection if needed in `core/interface_engine.py`:

```python
def detect_api_from_model(model_name: str) -> str:
    """Detect API provider from model name"""
    if model_name.startswith("gemini-"):
        return "gemini"
    elif model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith("gpt-"):
        return "openai"
    elif model_name.startswith("deepseek-"):
        return "deepseek"
    # Add more model patterns...
    return "unknown"
```

---

## Adding New Interface Engines

This section covers adding support for new client interfaces (like Aider, Roo Code, etc.) that can work with existing APIs.

### Overview

Interface engines handle client-specific request formats and behaviors. Each interface may have:
- Unique request headers or user agents
- Different system prompt formats
- Specific compression requirements
- Custom tool calling conventions

### Step 1: Extend the Interface Engine Enum

Add your interface to `core/interface_engine.py`:

```python
from enum import Enum

class InterfaceEngine(Enum):
    CLINE = "cline"
    ANTHROPIC = "anthropic"
    WEBUI = "webui"
    SILLYTAVERN = "sillytavern"
    STANDARD = "standard"
    AIDER = "aider"  # Add your new interface
    ROO_CODE = "roo_code"  # Add another interface
```

### Step 2: Add Detection Logic

Update the auto-detection logic to recognize your interface:

```python
def detect_interface_engine(request: Request, config) -> InterfaceEngine:
    """Auto-detect interface engine from request characteristics"""
    
    # Check User-Agent header
    user_agent = request.headers.get("user-agent", "").lower()
    
    # Check for Aider
    if "aider" in user_agent or "aider-chat" in user_agent:
        log_message(f"🔧 [ENGINE] Detected Aider via User-Agent: {user_agent}", "DEBUG", config)
        return InterfaceEngine.AIDER
    
    # Check for Roo Code
    if "roo-code" in user_agent or "roo_code" in user_agent:
        log_message(f"🔧 [ENGINE] Detected Roo Code via User-Agent: {user_agent}", "DEBUG", config)
        return InterfaceEngine.ROO_CODE
    
    # Check custom headers
    if request.headers.get("x-client-name") == "aider":
        return InterfaceEngine.AIDER
    
    # Check for Cline (existing)
    if any(header.startswith("claude-") for header in request.headers.keys()):
        return InterfaceEngine.CLINE
    
    # Check for Anthropic API requests
    if ("x-api-key" in request.headers and 
        "anthropic-version" in request.headers):
        return InterfaceEngine.ANTHROPIC
    
    # Default detection logic...
    return InterfaceEngine.STANDARD
```

### Step 3: Create Interface-Specific System Prompt Interceptor

Create `core/aider_system_prompt_interceptor.py`:

```python
import time
from typing import Dict, List, Any, Optional, Tuple
from core.system_prompt_interceptor import SystemPromptInterceptor
from server.logging_utils import log_message

class AiderSystemPromptInterceptor(SystemPromptInterceptor):
    """Aider-specific system prompt processing"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.interface_name = "Aider"
        log_message(f"🔧 AiderSystemPromptInterceptor: Initialized (extends SystemPromptInterceptor)", "DEBUG", self.config)
    
    def process_system_prompt(
        self, 
        messages: List[Dict[str, Any]], 
        rule_union: Dict[str, str], 
        model_id: str = None,
        system_param: str = None,
        system_instruction: str = None,
        target_format: str = "standard",
        request_data: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process system prompts for Aider interface"""
        
        log_message(f"🔧 Starting Aider system prompt interception", "DEBUG", self.config)
        log_message(f"Messages: {len(messages)}, Compression rules: {len(rule_union)}", "DEBUG", self.config)
        
        # Aider-specific logic
        aider_system_prompt = None
        processed_messages = messages.copy()
        
        # Check for Aider-specific system prompt patterns
        if messages and messages[0].get("role") == "system":
            system_content = messages[0].get("content", "")
            
            # Detect Aider system prompts (they often contain specific patterns)
            if "aider" in system_content.lower() or "coding assistant" in system_content.lower():
                log_message(f"🔧 Detected Aider system prompt: {len(system_content)} chars", "DEBUG", self.config)
                aider_system_prompt = system_content
                
                # Process compression on system prompt if applicable
                if len(system_content) >= self.config.min_characters and rule_union:
                    try:
                        from core.compress import compress_with_dynamic_analysis
                        packed = compress_with_dynamic_analysis(system_content, skip_tool_detection=False)
                        rule_union.update(packed.used)
                        aider_system_prompt = packed.text
                        log_message(f"🗜️ Compressed Aider system prompt: {len(system_content)} → {len(packed.text)} chars", "DEBUG", self.config)
                    except Exception as e:
                        log_message(f"⚠️ Aider system prompt compression failed: {e}", "WARNING", self.config)
                
                # Remove from messages if using system parameter
                if target_format in ["anthropic", "gemini"]:
                    processed_messages = messages[1:]
        
        # Add Aider-specific compression optimizations
        if "coding task" in str(messages).lower():
            # Add code-specific compression optimizations
            log_message(f"🔧 Applied Aider coding optimizations", "DEBUG", self.config)
        
        # Prepare metadata
        metadata = {
            "aider_system_prompt": aider_system_prompt,
            "rule_union": rule_union,
            "interface": "aider",
            "processing_time": time.time()
        }
        
        log_message(f"✅ Aider system prompt interception complete", "DEBUG", self.config)
        log_message(f"Result: {len(messages)} → {len(processed_messages)} messages, system prompt: {len(aider_system_prompt) if aider_system_prompt else 0} chars", "DEBUG", self.config)
        
        return processed_messages, metadata
```

### Step 4: Register the Interface Handler

Update `core/interface_engine.py` to use your new interceptor:

```python
def get_interface_compression_handler(config) -> 'InterfaceCompressionHandler':
    """Get interface compression handler with all interceptors"""
    
    # Import all interceptors
    from core.cline_system_prompt_interceptor import ClineSystemPromptInterceptor
    from core.anthropic_system_prompt_interceptor import AnthropicSystemPromptInterceptor
    from core.webui_system_prompt_interceptor import WebuiSystemPromptInterceptor
    from core.aider_system_prompt_interceptor import AiderSystemPromptInterceptor  # Add your import
    
    # Create handler
    handler = InterfaceCompressionHandler(config)
    
    # Register interceptors
    handler.register_interceptor(InterfaceEngine.CLINE, ClineSystemPromptInterceptor(config))
    handler.register_interceptor(InterfaceEngine.ANTHROPIC, AnthropicSystemPromptInterceptor(config))
    handler.register_interceptor(InterfaceEngine.WEBUI, WebuiSystemPromptInterceptor(config))
    handler.register_interceptor(InterfaceEngine.AIDER, AiderSystemPromptInterceptor(config))  # Register yours
    
    return handler
```

### Step 5: Add Interface-Specific Configuration

Update `config/config.jsonc` to include interface-specific settings:

```jsonc
{
    "system_prompt": {
        "use_cline": true,
        "use_aider": true,  // Add your interface
        "use_roo_code": true,
        
        "aider": {  // Interface-specific settings
            "preserve_code_blocks": true,
            "enhance_compression": true,
            "min_characters": 100
        }
    }
}
```

---

## Testing Your Integration

### Unit Testing

Create tests for your new components:

```python
# tests/test_gemini_integration.py
import pytest
import asyncio
from server.endpoints.gemini import gemini_generate, GeminiRequest
from server.config import ServerConfig

@pytest.mark.asyncio
async def test_gemini_endpoint():
    """Test Gemini endpoint handling"""
    request = GeminiRequest(
        model="gemini-pro",
        contents=[{
            "role": "user",
            "parts": [{"text": "Hello, how are you?"}]
        }],
        stream=False
    )
    
    config = ServerConfig()
    # Mock the HTTP request object
    mock_request = MockRequest()
    
    # Test non-streaming
    response = await gemini_generate(request, mock_request, config)
    assert response is not None
```

### Integration Testing

Create end-to-end tests:

```python
# tests/test_aider_interface.py
import pytest
from core.interface_engine import detect_interface_engine, InterfaceEngine
from unittest.mock import Mock

def test_aider_detection():
    """Test Aider interface detection"""
    mock_request = Mock()
    mock_request.headers = {"user-agent": "aider-chat/1.0"}
    
    config = Mock()
    engine = detect_interface_engine(mock_request, config)
    
    assert engine == InterfaceEngine.AIDER
```

### Manual Testing

1. **Start KrunchWrap:**
   ```bash
   python api/server.py
   ```

2. **Test with curl (for new API endpoints):**
   ```bash
   curl -X POST http://localhost:5002/v1/models/gemini-pro:generateContent \
     -H "Content-Type: application/json" \
     -H "x-goog-api-key: your-api-key" \
     -d '{
       "contents": [
         {
           "role": "user", 
           "parts": [{"text": "Hello"}]
         }
       ]
     }'
   ```

3. **Test with your client (for new interfaces):**
   Configure your client (Aider, etc.) to use `http://localhost:5002` as the base URL.

---

## Best Practices

### 1. Error Handling

Always implement comprehensive error handling:

```python
try:
    # API call
    response = await session.post(url, json=payload)
except aiohttp.ClientError as e:
    log_message(f"🚨 Network error: {e}", config=config)
    raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
except Exception as e:
    log_message(f"🚨 Unexpected error: {e}", config=config)
    raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
```

### 2. Logging

Use consistent logging patterns:

```python
log_message(f"🔧 [API_NAME] Processing request", "INFO", config)
log_message(f"📊 [API_NAME] Compression: {compression_ratio:.1f}%", "INFO", config)
log_message(f"✅ [API_NAME] Request completed successfully", "INFO", config)
log_message(f"🚨 [API_NAME ERROR] Something went wrong: {error}", config=config)
```

### 3. Configuration

Make your integration configurable:

```python
class YourConfig:
    def __init__(self, config_dict):
        self.api_key = config_dict.get("api_key")
        self.base_url = config_dict.get("base_url", "https://api.example.com")
        self.timeout = config_dict.get("timeout", 30)
```

### 4. Streaming Support

If the API supports streaming, implement it properly:

```python
async def handle_streaming_response(...):
    """Handle streaming with proper event format"""
    async def event_generator():
        try:
            async with session.post(url, json=payload) as resp:
                async for line in resp.content:
                    # Process and yield events
                    yield format_sse_event(line)
        finally:
            # Cleanup
            pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 5. Compression Integration

Always integrate with KrunchWrap's compression pipeline:

```python
# Process through compression
if should_compress:
    processed_messages, metadata, engine = detect_and_process_compression(
        request=http_request,
        messages=messages_dict,
        rule_union=rule_union,
        config=config,
        model_id=model_name,
        target_format="your_api_format"
    )
```

---

## Troubleshooting

### Common Issues

1. **Port Conflicts:**
   ```bash
   lsof -ti:5002 | xargs kill -9
   ```

2. **Import Errors:**
   - Check Python path
   - Verify all dependencies are installed
   - Check for circular imports

3. **Authentication Failures:**
   - Verify API key format
   - Check header names (x-api-key vs Authorization vs x-goog-api-key)
   - Ensure proper URL construction

4. **Streaming Issues:**
   - Verify SSE format compliance
   - Check session lifecycle management
   - Ensure proper async/await usage

### Debug Mode

Enable debug logging in `config/server.jsonc`:

```jsonc
{
    "logging": {
        "log_level": "DEBUG",
        "verbose_logging": true
    }
}
```

### Testing Checklist

- [ ] Interface detection works correctly
- [ ] Authentication headers are forwarded properly
- [ ] Request/response format conversion is accurate
- [ ] Compression integration functions
- [ ] Streaming works (if supported)
- [ ] Error handling covers edge cases
- [ ] Configuration is properly loaded
- [ ] Logging provides useful information

---

## Summary

This guide provides the foundation for extending KrunchWrap with new APIs and interfaces. The key principles are:

1. **Follow existing patterns** - Use the Anthropic integration as a reference
2. **Integrate with compression** - Always use the compression pipeline
3. **Handle errors gracefully** - Provide meaningful error messages
4. **Test thoroughly** - Unit tests, integration tests, and manual testing
5. **Document configuration** - Make it easy for users to configure

Each API and interface has unique characteristics, so adapt these patterns to fit your specific requirements while maintaining consistency with KrunchWrap's architecture. 