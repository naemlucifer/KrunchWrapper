"""Text completions endpoint module."""
import time
import json
from fastapi import Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import aiohttp

from core.model_context import set_global_model_context, extract_model_from_provider_format
from core.compress import compress_with_dynamic_analysis
from core.cline_server_handler import is_cline_request
from server.logging_utils import (
    log_message, log_performance_metrics, log_verbose_content,
    log_completion_message
)


async def text_completion(request: Request, config):
    """Handle legacy completions endpoint by proxying to target or adapting to chat completions."""
    # Start timing
    start_time = time.time()
    
    # Use unified Interface Engine for detection (replacing manual Cline detection)
    from core.interface_engine import get_interface_compression_handler
    handler = get_interface_compression_handler(config)
    detected_engine = handler.detect_interface_engine(request, {})
    
    # Legacy compatibility - extract is_cline for any remaining cline-specific code
    is_cline = (detected_engine.value == "cline")
    
    if is_cline:
        log_message(f"🔍 [ENGINE] Detected {detected_engine.value} interface via Interface Engine for /v1/completions endpoint", "DEBUG")
    else:
        log_message(f"🔍 [ENGINE] Detected {detected_engine.value} interface via Interface Engine for /v1/completions endpoint", "DEBUG")
    
    body_bytes = await request.body()
    
    # DEBUG: Log the raw request body
    log_message(f"🔍 [DEBUG] Raw request body size: {len(body_bytes)} bytes", "DEBUG")
    log_message(f"🔍 [DEBUG] Raw request body (first 1000 chars): {body_bytes[:1000].decode('utf-8', errors='ignore')}", "DEBUG")
    if len(body_bytes) < 500:  # For smaller requests, show the full body
        log_message(f"🔍 [DEBUG] Full request body: {body_bytes.decode('utf-8', errors='ignore')}", "DEBUG")
    
    # Extract API key from request headers
    auth_header = request.headers.get("authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
    
    # Parse the request body
    try:
        data = json.loads(body_bytes)
        stream = data.get("stream", False)
        
        # Set model context for tokenizer validation
        model_name = data.get("model")
        if model_name:
            clean_model = extract_model_from_provider_format(model_name)
            set_global_model_context(clean_model)
            log_message(f"🔧 Set model context: {clean_model} (from {model_name})", "DEBUG")
        
        # DEBUG: Log the parsed data structure
        log_message(f"🔍 [DEBUG] Parsed request keys: {list(data.keys())}", "DEBUG")
        log_message(f"🔍 [DEBUG] Request data size: {len(json.dumps(data))} chars", "DEBUG")
        
        # DEBUG: Log all fields that might contain conversation content
        for key in ['prompt', 'messages', 'text', 'input', 'conversation', 'chat']:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    log_message(f"🔍 [DEBUG] Field '{key}': {len(value)} chars - '{value}'", "DEBUG")
                elif isinstance(value, list):
                    log_message(f"🔍 [DEBUG] Field '{key}': list with {len(value)} items", "DEBUG")
                    for i, item in enumerate(value[:3]):  # Show first 3 items
                        log_message(f"🔍 [DEBUG]   [{i}]: {str(item)}", "DEBUG")
                else:
                    log_message(f"🔍 [DEBUG] Field '{key}': {type(value)} - {str(value)}", "DEBUG")
        
    except Exception as e:
        log_message(f"❌ [DEBUG] Failed to parse request body: {e}", "ERROR")
        stream = False
        data = {}
    
    # Basic compression for text completions
    prompt = data.get("prompt", "")
    
    # DEBUG: Log the extracted prompt details
    log_message(f"🔍 [DEBUG] Extracted prompt size: {len(prompt)} chars", "DEBUG")
    log_message(f"🔍 [DEBUG] Extracted prompt (first 200 chars): {prompt[:200]}", "DEBUG")
    
    # Check for common SillyTavern issue: empty system template only
    if prompt.strip() in ["<|im_start|>system<|im_end|>", "<|im_start|>system<|im_end|>\n"]:
        log_message("⚠️  [WARNING] Detected SillyTavern sending only system template without conversation content!", "WARNING")
        log_message("💡 [SUGGESTION] Try configuring SillyTavern to use OpenAI Chat Completions format:", "INFO")
        log_message("   • API Type: OpenAI Compatible", "INFO")
        log_message("   • Endpoint: http://127.0.0.1:5002/v1/chat/completions", "INFO")
        log_message("   • This will properly send conversation history as messages array", "INFO")
        
        # Return a helpful error response
        return {
            "id": "cmpl-krunchwrapper-error",
            "object": "text_completion",
            "created": int(time.time()),
            "model": data.get("model", "default-model"),
            "choices": [{
                "text": "\n\n**KrunchWrapper Error**: SillyTavern sent only system template without conversation content.\n\n**Solution**: Configure SillyTavern to use:\n• API Type: OpenAI Compatible\n• Endpoint: http://127.0.0.1:5002/v1/chat/completions\n\nThis will properly send your conversation history.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 50,
                "total_tokens": (len(prompt) // 4) + 50
            }
        }
    
    original_prompt_size = len(prompt)
    compression_ratio = 0.0
    compression_tokens_saved = 0
    dynamic_chars_used = 0
    
    # Apply compression if prompt is large enough
    if original_prompt_size >= config.min_characters:
        log_message(f"🗜️  Applying dynamic compression to prompt")
        
        # Use dynamic compression only
        # CRITICAL FIX: Never skip tool detection - tool calls should never be compressed regardless of client
        packed = compress_with_dynamic_analysis(prompt, skip_tool_detection=False, cline_mode=(detected_engine.value in ["cline", "webui", "sillytavern", "standard"]))
        
        # Update the request data with compressed prompt
        data["prompt"] = packed.text
        body_bytes = json.dumps(data).encode('utf-8')
        
        # Calculate compression metrics
        compressed_size = len(packed.text)
        compression_ratio = (original_prompt_size - compressed_size) / original_prompt_size if original_prompt_size > 0 else 0
        
        # Calculate actual token savings based on character lengths
        original_tokens = original_prompt_size // 4
        compressed_tokens = compressed_size // 4
        compression_tokens_saved = max(0, original_tokens - compressed_tokens)
        dynamic_chars_used = sum(len(value) for value in packed.used.values()) if packed.used else 0
        
        log_message(f"📊 Token calculation: {original_tokens} original → {compressed_tokens} compressed = {compression_tokens_saved} tokens saved", "DEBUG")
        
        log_message(f"🗜️  Applied compression: {compression_ratio*100:.1f}% reduction")
    else:
        log_message(f"🗜️  Compression skipped (prompt too short: {original_prompt_size} < {config.min_characters} chars)")
    
    preprocessing_end_time = time.time()
    preprocessing_time = preprocessing_end_time - start_time
    
    # Try to proxy to target completions endpoint first, then fallback to chat completions
    # This is a simplified version - full implementation would handle streaming and more complex cases
    try:
        target_url = f"{config.target_url}/completions"
        async with aiohttp.ClientSession() as session:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            async with session.post(target_url, headers=headers, data=body_bytes) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    
                    # Calculate and log metrics
                    end_time = time.time()
                    total_time = end_time - start_time
                    llm_time = end_time - preprocessing_end_time
                    
                    # Extract token usage from response
                    usage = response_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    # Log performance metrics
                    log_performance_metrics(
                        "completions",
                        total_time,
                        preprocessing_time,
                        llm_time,
                        compression_ratio,
                        compression_tokens_saved,
                        dynamic_chars_used,
                        total_tokens,
                        prompt_tokens,
                        completion_tokens,
                        None,
                        None
                    )
                    
                    log_verbose_content(
                        "completions",
                        original_content=prompt,
                        compressed_content=data.get("prompt", ""),
                        response_data=response_data
                    )
                    
                    # Log completion message with total compression stats
                    rule_union = {} if not locals().get('packed') else getattr(locals().get('packed'), 'used', {})
                    log_completion_message(
                        "completions",
                        compression_tokens_saved,
                        dynamic_chars_used,
                        rule_union,
                        original_prompt_size,
                        "generic",
                        None,  # Comment stats not available for completions endpoint
                        None   # Timing breakdown not available for completions endpoint
                    )
                    
                    return response_data
    except Exception as e:
        print(f"Error proxying to completions endpoint: {e}")
    
    # If target doesn't support completions, adapt to chat completions
    try:
        prompt = data.get("prompt", "")
        
        # Convert to chat format
        chat_request = {
            "model": data.get("model", "default-model"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens"),
            "stream": False  # Non-streaming for this fallback
        }
        
        # Forward to chat completions endpoint
        target_url = f"{config.target_url}/chat/completions"
        async with aiohttp.ClientSession() as session:
            # Use client's API key if provided, otherwise use configured API key
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            async with session.post(target_url, headers=headers, json=chat_request) as resp:
                if resp.status == 200:
                    # Handle JSON response with conversion
                    chat_response = await resp.json()
                    
                    # Calculate and log metrics
                    end_time = time.time()
                    total_time = end_time - start_time
                    llm_time = end_time - preprocessing_end_time
                    
                    # Extract token usage from response
                    usage = chat_response.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    # Log performance metrics
                    log_performance_metrics(
                        "completions→chat",
                        total_time,
                        preprocessing_time,
                        llm_time,
                        compression_ratio,
                        compression_tokens_saved,
                        dynamic_chars_used,
                        total_tokens,
                        prompt_tokens,
                        completion_tokens,
                        None,
                        None
                    )
                    
                    # Extract content from first choice
                    content = chat_response["choices"][0].get("message", {}).get("content", "")
                    
                    log_verbose_content(
                        "completions→chat",
                        original_content=prompt,
                        compressed_content=data.get("prompt", prompt),
                        response_data=chat_response
                    )
                    
                    # Log completion message with total compression stats
                    rule_union = {} if not locals().get('packed') else getattr(locals().get('packed'), 'used', {})
                    log_completion_message(
                        "completions→chat",
                        compression_tokens_saved,
                        dynamic_chars_used,
                        rule_union,
                        original_prompt_size,
                        "generic",
                        None,  # Comment stats not available for completions endpoint
                        None   # Timing breakdown not available for completions→chat endpoint
                    )
                        
                    return {
                        "id": chat_response.get("id", "cmpl-default"),
                        "object": "text_completion",
                        "created": chat_response.get("created", 0),
                        "model": chat_response.get("model", "default-model"),
                        "choices": [{
                            "text": content,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": chat_response["choices"][0].get("finish_reason", "stop")
                        }],
                        "usage": chat_response.get("usage", {})
                    }
                else:
                    print(f"Error from chat completions endpoint: {resp.status}")
                    error_text = await resp.text()
                    print(f"Error details: {error_text}")
    except Exception as e:
        print(f"Error adapting completions to chat completions: {e}")
    
    # If all attempts fail, return a fallback response
    return {
        "id": "cmpl-fallback",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "default-model",
        "choices": [{
            "text": "I'm a fallback response from KrunchWrapper.",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


async def legacy_completion(request: Request, config):
    """Handle legacy completion endpoint (without /v1/ prefix and singular form)."""
    print("Redirecting /completion request to /v1/completions")
    return await text_completion(request, config)


async def legacy_completions(request: Request, config):
    """Handle legacy completions endpoint (without /v1/ prefix)."""
    print("Redirecting /completions request to /v1/completions")
    return await text_completion(request, config) 