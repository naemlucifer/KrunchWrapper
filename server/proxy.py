"""
Proxy handler for KrunchWrapper server.
Handles all proxy requests with compression, decompression, and interface-specific processing.
"""

import json
import time
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import Request, HTTPException, Response
from fastapi.responses import StreamingResponse

# Import logging functions
from server.logging_utils import (
    log_message, log_proxy_request_content, log_proxy_response_content,
    log_passthrough_request, log_passthrough_response, log_performance_metrics,
    log_verbose_content, log_completion_message
)

# Import models and helper functions  
from server.models import extract_message_content_for_compression

# Import streaming handlers
from server.streaming import (
    create_streaming_timeout, context_aware_decompress
)

# Import request/response utility functions
from server.request_utils import filter_timeout_parameters
from server.response_utils import (
    decompress_response, clean_compression_artifacts, clean_messages_array
)

# Import core compression and system prompt functionality
from core.compress import compress_with_dynamic_analysis
from core.interface_engine import detect_and_process_compression, get_interface_compression_handler
from core.cline_system_prompt_interceptor import ClineSystemPromptInterceptor
from core.cline_server_handler import is_cline_request, should_disable_compression_for_cline
from core.model_context import extract_model_from_provider_format, set_global_model_context


async def _handle_double_slash_chat_completions(request: Request, path: str, body: bytes, config, start_time: float):
    """Handle double slash corrected chat completions requests."""
    log_message("🔧 [FIX] Redirecting to proper chat completions endpoint", "DEBUG")
    try:
        body_str = body.decode('utf-8')
        log_message(f"🔍 [DEBUG] Raw request body (first 500 chars): {body_str[:500]}", "DEBUG")
        
        # Emergency safeguard: Check if body contains compressed symbols that might break JSON parsing
        if any(ord(c) > 127 for c in body_str):
            log_message("⚠️ [WARNING] Request body contains non-ASCII characters, possible compression interference", "WARNING")
        
        try:
            data = json.loads(body_str)
            log_message(f"🔍 [DEBUG] Parsed JSON successfully, keys: {list(data.keys())}", "DEBUG")
        except json.JSONDecodeError as json_error:
            log_message(f"❌ JSON parse error: {json_error}", "ERROR")
            log_message(f"🔍 [DEBUG] Raw body that failed to parse: {body_str}", "DEBUG")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {json_error}")
        
        # Clean up the data
        clean_compression_artifacts(data)
        clean_messages_array(data)
        
        # Detect interface engine before using it
        from core.interface_engine import get_interface_compression_handler
        handler = get_interface_compression_handler(config)
        detected_engine = handler.detect_interface_engine(request, data)
        
        # Process compression and system prompts
        log_message(f"🔍 [DEBUG] Processing chat completion with compression", "DEBUG")
        
        # Apply compression logic (simplified version)
        total_content_size = sum(len(msg.get('content', '')) for msg in data.get('messages', []) if msg.get('role') in {"user", "assistant", "system"})
        rule_union = {}
        
        if total_content_size >= config.min_characters:
            log_message(f"🗜️ Applying compression to {total_content_size} characters", "DEBUG")
            messages = data.get('messages', [])
            
            # Apply compression to messages
            for i, msg in enumerate(messages):
                if msg.get('role') in {"user", "assistant"}:
                    original_content = msg.get('content', '')
                    if original_content:
                        try:
                            exclude_symbols = set(rule_union.keys()) if rule_union else set()
                            packed = compress_with_dynamic_analysis(
                                original_content, 
                                skip_tool_detection=False, 
                                cline_mode=(detected_engine.value in ["cline", "webui", "sillytavern", "standard"]),
                                exclude_symbols=exclude_symbols
                            )
                            messages[i]['content'] = packed.text
                            rule_union.update(packed.used)
                        except Exception as compression_error:
                            log_message(f"⚠️ Compression failed for message {i}: {compression_error}", "WARNING")
            
            # Process system prompts using unified interface engine
            messages, system_metadata, detected_engine = detect_and_process_compression(
                request=request,
                messages=messages,
                rule_union=rule_union,
                config=config,
                model_id=data.get('model'),
                system_param=data.get('system'),
                system_instruction=data.get('system_instruction'),
                target_format=config.system_prompt_format,
                request_data=data
            )
            
            data['messages'] = messages
        
        # Apply timeout parameter filtering
        data = filter_timeout_parameters(data, config.filter_timeout_parameters)
        
        # Forward the request to target
        return await _forward_double_slash_request(data, rule_union, config, path, start_time)
        
    except Exception as outer_e:
        log_message(f"❌ Error handling double slash correction: {outer_e}", "ERROR")
        raise HTTPException(status_code=400, detail=f"Request processing error: {str(outer_e)}")


async def _handle_chat_completions_compression(request: Request, body: bytes, config, start_time: float):
    """Handle compression for chat completions requests."""
    try:
        # Parse request body
        request_data = json.loads(body)
        
        # Check interface and compression requirements
        model = request_data.get("model", "")
        messages = request_data.get("messages", [])
        
        # Calculate total content size
        total_content_size = sum(len(extract_message_content_for_compression(msg.get("content", ""))) 
                               for msg in messages if msg.get("role") in {"user", "assistant", "system"})
        log_message(f"🔍 [PROXY] Request content analysis: {len(messages)} messages, {total_content_size:,} chars total", "DEBUG")
        
        # Set model context
        if model:
            clean_model = extract_model_from_provider_format(model)
            set_global_model_context(clean_model)
            log_message(f"🔧 Set model context: {clean_model} (from {model})", "DEBUG")
        
        # Detect interface engine
        handler = get_interface_compression_handler(config)
        detected_engine = handler.detect_interface_engine(request, request_data)
        is_cline = (detected_engine.value == "cline")
        log_message(f"🔍 [PROXY] Detected interface: {detected_engine.value}", "DEBUG")
        
        # Check if compression should be applied
        should_compress = True
        rule_union = {}
        
        if total_content_size < config.min_characters:
            should_compress = False
            log_message(f"🚫 [PROXY] Content size ({total_content_size:,} chars) below minimum threshold ({config.min_characters})", "INFO")
        elif is_cline and should_disable_compression_for_cline(request, config):
            should_compress = False
            log_message("🚫 [PROXY] Compression disabled for Cline compatibility", "INFO")
        elif handler.should_disable_compression(detected_engine, request):
            should_compress = False
            log_message(f"🚫 [PROXY] Compression disabled for {detected_engine.value} interface", "INFO")
        
        if should_compress:
            # Apply compression
            log_message(f"🗜️ [PROXY] Using standard compression (new conversation, {detected_engine.value} interface)", "INFO")
            
            for i, msg in enumerate(messages):
                if msg.get("role") in {"user", "assistant", "system"} and should_compress:
                    raw_content = msg.get("content", "")
                    original_content = extract_message_content_for_compression(raw_content)
                    
                    if original_content is None:
                        original_content = ""
                    
                    if original_content:
                        exclude_symbols = set(rule_union.keys()) if rule_union else set()
                        packed = compress_with_dynamic_analysis(
                            original_content, 
                            skip_tool_detection=False, 
                            cline_mode=(detected_engine.value in ["cline", "webui", "sillytavern", "standard"]),
                            exclude_symbols=exclude_symbols
                        )
                        
                        msg["content"] = packed.text
                        rule_union.update(packed.used)
                        log_message(f"🗜️ [PROXY] Compressed {msg.get('role')} message {i}: {len(original_content)} → {len(packed.text)} chars", "DEBUG")
            
            # Update request data with compressed messages
            request_data["messages"] = messages
            
            # Process through unified interface engine system (FIXED!)
            log_message(f"🔧 [PROXY] Processing through {detected_engine.value} interface system prompt handler", "INFO")
            
            # Use the unified Interface Engine system for ALL interfaces
            processed_messages, system_metadata, engine = detect_and_process_compression(
                request=request,
                messages=messages,
                rule_union=rule_union,
                config=config,
                model_id=model,
                system_param=request_data.get('system'),
                system_instruction=request_data.get('system_instruction'),
                target_format=getattr(config, 'system_prompt_format', 'chatml'),
                request_data=request_data
            )
            
            # Update request data with processed messages
            request_data["messages"] = processed_messages
            
            # Handle Anthropic-specific system prompt format
            if engine.value == "anthropic" and "anthropic_system_prompt" in system_metadata:
                # For Anthropic API, move the system prompt to the 'system' parameter
                request_data["system"] = system_metadata["anthropic_system_prompt"]
                log_message(f"🔧 [PROXY] Applied Anthropic system prompt: {len(system_metadata['anthropic_system_prompt'])} chars", "DEBUG")
            
            # Update rule_union with any new system prompt rules from the interface engine
            if 'rule_union' in system_metadata:
                additional_rules = system_metadata['rule_union']
                if additional_rules:
                    rule_union.update(additional_rules)
                    log_message(f"🗣️ [PROXY] Updated rule_union with {len(additional_rules)} system prompt rules from {engine.value} interface", "DEBUG")
            
            # Apply timeout parameter filtering
            processed_request = filter_timeout_parameters(request_data, config.filter_timeout_parameters)
            body = json.dumps(processed_request).encode('utf-8')
            log_message(f"✅ [PROXY] Applied {detected_engine.value} system prompt processing - compressed request for model: {model}", "INFO")
        else:
            # Compression disabled - still use Interface Engine for system prompt processing
            log_message(f"🔄 [PROXY] Processing {detected_engine.value} request without compression", "INFO")
            
            # Even without compression, we still need interface-specific system prompt handling
            processed_messages, system_metadata, engine = detect_and_process_compression(
                request=request,
                messages=messages,
                rule_union=rule_union,  # Empty rule_union since no compression
                config=config,
                model_id=model,
                system_param=request_data.get('system'),
                system_instruction=request_data.get('system_instruction'),
                target_format=getattr(config, 'system_prompt_format', 'chatml'),
                request_data=request_data
            )
            
            # Update request data with processed messages (system prompts may have been added)
            request_data["messages"] = processed_messages
            
            # Handle Anthropic-specific system prompt format (even without compression)
            if engine.value == "anthropic" and "anthropic_system_prompt" in system_metadata:
                # For Anthropic API, move the system prompt to the 'system' parameter
                request_data["system"] = system_metadata["anthropic_system_prompt"]
                log_message(f"🔧 [PROXY] Applied Anthropic system prompt (no compression): {len(system_metadata['anthropic_system_prompt'])} chars", "DEBUG")
            
            # Apply timeout parameter filtering
            processed_request = filter_timeout_parameters(request_data, config.filter_timeout_parameters)
            body = json.dumps(processed_request).encode('utf-8')
            log_message(f"✅ [PROXY] Applied {detected_engine.value} system prompt processing without compression", "INFO")
                
        return body, rule_union
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_message(f"❌ [PROXY] Error processing chat completions request: {e}", "ERROR")
        log_message(f"❌ [PROXY] Full traceback: {error_details}", "ERROR")
        return body, {}


async def proxy_request(request: Request, path: str, config):
    """
    Handle all proxy requests with compression, decompression, and interface-specific processing.
    
    This function handles:
    - Request logging and content analysis
    - Double slash fixes for SillyTavern compatibility
    - Chat completions compression and system prompt processing
    - Streaming and non-streaming responses
    - Response decompression and metrics calculation
    """
    log_message(f"🔄 Request received at proxy endpoint: /{path}")
    
    # REMOVED: Special Cline routing that bypassed compression
    # Now all chat/completions requests (including Cline) go through the main compression pipeline
    # at /v1/chat/completions where selective tool call compression is implemented
    
    # Start timing
    start_time = time.time()
    
    # Capture request body for logging
    body = await request.body()
    
    # Log request content for all requests when verbose logging is enabled
    try:
        if body and request.method == "POST":
            body_str = body.decode('utf-8')
            try:
                request_data = json.loads(body_str)
                total_content_size = 0
                if "messages" in request_data:
                    total_content_size = sum(len(msg.get('content', '')) for msg in request_data.get('messages', []))
                
                log_proxy_request_content(
                    f"/{path}",
                    request_data=request_data,
                    content_size=total_content_size,
                    config=config
                )
            except json.JSONDecodeError:
                log_proxy_request_content(
                    f"/{path}",
                    request_body=body_str[:2000],  # Limit to first 2000 chars for non-JSON
                    content_size=len(body_str),
                    config=config
                )
    except Exception as e:
        log_message(f"⚠️ Error logging request content: {e}", "DEBUG")
    
    # Import the endpoints module to use its functions
    from server.endpoints.models import list_models
    
    # Fix double slash issues from SillyTavern
    if path.startswith("v1//"):
        corrected_path = path.replace("v1//", "v1/")
        log_message(f"🔧 [FIX] Correcting double slash: /{path} → /{corrected_path}", "DEBUG")
        
        # Handle corrected models endpoint
        if corrected_path == "v1/models":
            log_message("🔧 [FIX] Redirecting to proper models endpoint", "DEBUG")
            return await list_models(request, config)
        
        # Handle corrected chat completions endpoint
        if corrected_path == "v1/chat/completions":
            return await _handle_double_slash_chat_completions(request, path, body, config, start_time)
    
    # Forward all other requests to the target LLM API
    # Continue with the timing started earlier
    
    headers = {k: v for k, v in request.headers.items() 
              if k.lower() not in ["host", "content-length"]}
    
    # Extract API key from request headers if not present
    if "authorization" not in {k.lower() for k in headers} and config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    
    # Initialize metrics variables
    compression_ratio = 0.0
    compression_tokens_saved = 0
    dynamic_chars_used = 0
    rule_union = {}
    original_content_size = 0
    compressed_content_size = 0
    
    # Special handling for chat completions requests (use compression regardless of interface engine)
    if request.method == "POST" and "chat/completions" in path:
        # Calculate original content size before compression
        try:
            request_data = json.loads(body.decode('utf-8'))
            original_content_size = sum(len(msg.get('content', '')) for msg in request_data.get('messages', []))
        except:
            original_content_size = 0
        
        body, rule_union = await _handle_chat_completions_compression(request, body, config, start_time)
        headers.pop('content-length', None)
        headers.pop('Content-Length', None)
        
        # Calculate compressed content size and token savings
        try:
            compressed_request_data = json.loads(body.decode('utf-8'))
            compressed_content_size = sum(len(msg.get('content', '')) for msg in compressed_request_data.get('messages', []))
            
            # Calculate token savings based on character reduction
            if original_content_size > 0 and compressed_content_size < original_content_size:
                original_tokens = original_content_size // 4
                compressed_tokens = compressed_content_size // 4
                compression_tokens_saved = max(0, original_tokens - compressed_tokens)
                compression_ratio = (original_content_size - compressed_content_size) / original_content_size
                dynamic_chars_used = sum(len(value) for value in rule_union.values()) if rule_union else 0
                
                log_message(f"📊 [PROXY] Token calculation: {original_tokens} original → {compressed_tokens} compressed = {compression_tokens_saved} tokens saved", "DEBUG")
        except:
            pass
    
    # Minimal intervention - just pass requests through for non-double-slash cases
    
    preprocessing_end_time = time.time()
    preprocessing_time = preprocessing_end_time - start_time
    
    llm_start_time = time.time()
    target_url = f"{config.target_url}/{path}"
    
    return await _forward_request_to_target(
        request, target_url, headers, body, rule_union, path, 
        start_time, preprocessing_time, llm_start_time, config,
        compression_tokens_saved, dynamic_chars_used, compression_ratio, original_content_size
    )


async def _forward_double_slash_request(data: dict, rule_union: dict, config, path: str, start_time: float):
    """Forward double slash corrected request to target."""
    target_url = f"{config.target_url}/chat/completions"
    
    # Check if streaming is requested
    is_streaming = data.get('stream', False)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            target_url,
            json=data,
            headers={"Content-Type": "application/json"}
        ) as resp:
            # Handle streaming responses
            if is_streaming or resp.headers.get("content-type", "").startswith("text/event-stream"):
                return await _handle_streaming_response(resp, rule_union, target_url, data, path, start_time, config)
            else:
                # Handle non-streaming JSON responses
                response_data = await resp.json()
                
                # Decompress the response if needed
                if rule_union:
                    response_data = decompress_response(response_data, rule_union)
                
                return Response(
                    content=json.dumps(response_data).encode('utf-8'),
                    status_code=resp.status,
                    headers={"Content-Type": "application/json"}
                )


async def _handle_streaming_response(resp, rule_union: dict, target_url: str, data: dict, path: str, start_time: float, config=None):
    """Handle streaming responses with decompression."""
    async def stream_generator():
        timeout = create_streaming_timeout(data, config)
        async with aiohttp.ClientSession(timeout=timeout) as stream_session:
            try:
                async with stream_session.post(
                    target_url,
                    json=data,
                    headers={"Content-Type": "application/json"}
                ) as stream_resp:
                    async for chunk in stream_resp.content.iter_chunked(8192):
                        if rule_union:
                            try:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith('data: ') and not chunk_str.startswith('data: [DONE]'):
                                    json_str = chunk_str[6:].strip()
                                    if json_str:
                                        chunk_data = json.loads(json_str)
                                        chunk = f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass
                        yield chunk
            except Exception as stream_error:
                log_message(f"⚠️ Streaming connection error: {stream_error}", "WARNING")
                yield b"data: [DONE]\n\n"
    
    return StreamingResponse(
        stream_generator(),
        status_code=resp.status,
        headers=dict(resp.headers),
        media_type="text/event-stream"
    )


async def _forward_request_to_target(request: Request, target_url: str, headers: dict, body: bytes, 
                                   rule_union: dict, path: str, start_time: float, 
                                   preprocessing_time: float, llm_start_time: float, config,
                                   compression_tokens_saved: int = 0, dynamic_chars_used: int = 0, 
                                   compression_ratio: float = 0.0, original_content_size: int = 0):
    """Forward request to target LLM API and handle response."""
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body
        ) as resp:
            # Check if this is a streaming response
            is_streaming = resp.headers.get("content-type", "").startswith("text/event-stream")
            
            if is_streaming:
                return await _handle_proxy_streaming_response(
                    request, target_url, headers, body, rule_union, path, config
                )
            else:
                return await _handle_proxy_non_streaming_response(
                    resp, rule_union, path, start_time, preprocessing_time, llm_start_time, config, request, body,
                    compression_tokens_saved, dynamic_chars_used, compression_ratio, original_content_size
                )


async def _handle_proxy_streaming_response(request: Request, target_url: str, headers: dict, body: bytes, 
                                         rule_union: dict, path: str, config):
    """Handle streaming responses for proxy requests."""
    log_message(f"🌊 Streaming response detected for /{path}", "DEBUG")
    
    async def stream_generator():
        try:
            request_payload = json.loads(body.decode('utf-8')) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_payload = {}
        timeout = create_streaming_timeout(request_payload, config)
        
        # For streaming, we'll capture the response pieces for logging
        captured_response_parts = []
        async with aiohttp.ClientSession(timeout=timeout) as stream_session:
            try:
                async with stream_session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body
                ) as stream_resp:
                    async for chunk in stream_resp.content.iter_chunked(8192):
                        # Capture chunk for logging
                        try:
                            chunk_str = chunk.decode('utf-8')
                            if chunk_str.startswith('data: ') and not chunk_str.startswith('data: [DONE]'):
                                json_str = chunk_str[6:].strip()
                                if json_str:
                                    chunk_data = json.loads(json_str)
                                    if "choices" in chunk_data:
                                        for choice in chunk_data["choices"]:
                                            if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"] is not None:
                                                captured_response_parts.append(choice["delta"]["content"])
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                        
                        if rule_union and request.method == "POST" and "chat/completions" in path:
                            # DEBUG: Log rule_union status at streaming decompression point
                            log_message(f"🔍 [PROXY STREAMING DEBUG] rule_union has {len(rule_union)} rules available for decompression", "DEBUG")
                            
                            # For streaming chat completions, decompress each chunk if it contains JSON
                            try:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith('data: ') and not chunk_str.startswith('data: [DONE]'):
                                    # Extract JSON from SSE format
                                    json_str = chunk_str[6:].strip()  # Remove 'data: ' prefix
                                    if json_str:
                                        chunk_data = json.loads(json_str)
                                        
                                        # CRITICAL: Actually decompress the streaming content
                                        for choice in chunk_data.get("choices", []):
                                            if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                                original_content = choice["delta"]["content"]
                                                symbols_found = [symbol for symbol in rule_union.keys() if symbol in original_content]
                                                if symbols_found:
                                                    log_message(f"🔍 [PROXY STREAMING DEBUG] Found symbols {symbols_found} in chunk: '{original_content[:50]}...'", "DEBUG")
                                                    decompressed_content = context_aware_decompress(original_content, rule_union)
                                                    choice["delta"]["content"] = decompressed_content
                                                    if decompressed_content != original_content:
                                                        log_message(f"🔍 [PROXY STREAMING DEBUG] Decompressed chunk: '{original_content}' → '{decompressed_content}'", "DEBUG")
                                        
                                        # Reconstruct SSE format with decompressed content
                                        chunk = f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # If decompression fails, pass chunk as-is
                                pass
                            yield chunk
                        else:
                            # CRITICAL FIX: Always yield chunks when no compression is applied
                            yield chunk
            except Exception as stream_error:
                log_message(f"⚠️ Streaming connection error: {stream_error}", "WARNING")
                # Send a final [DONE] message to properly close the stream
                yield b"data: [DONE]\n\n"
            finally:
                # Log the complete streaming response
                if captured_response_parts:
                    full_response = "".join(captured_response_parts)
                    end_time = time.time()
                    processing_time = end_time - time.time()  # This would need start_time passed in
                    
                    log_proxy_response_content(
                        f"/{path} (streaming)",
                        response_data={"choices": [{"message": {"content": full_response}}]},
                        processing_time=processing_time,
                        status_code=stream_resp.status if 'stream_resp' in locals() else 200,
                        config=config
                    )
    
    return StreamingResponse(
        stream_generator(),
        status_code=200,  # We'll use 200 as default since we don't have resp.status here
        headers={"Content-Type": "text/event-stream"},
        media_type="text/event-stream"
    )


async def _handle_proxy_non_streaming_response(resp, rule_union: dict, path: str, start_time: float,
                                             preprocessing_time: float, llm_start_time: float, 
                                             config, request: Request, body: bytes,
                                             compression_tokens_saved: int = 0, dynamic_chars_used: int = 0,
                                             compression_ratio: float = 0.0, original_content_size: int = 0):
    """Handle non-streaming responses for proxy requests."""
    # Handle non-streaming responses
    response_body = await resp.read()
    
    # Log response content for all requests
    end_time = time.time()
    processing_time = end_time - start_time
    
    try:
        # Try to parse response as JSON for better logging
        response_data = json.loads(response_body)
        log_proxy_response_content(
            f"/{path}",
            response_data=response_data,
            processing_time=processing_time,
            status_code=resp.status,
            config=config
        )
    except json.JSONDecodeError:
        # Log as text if not JSON
        response_text = response_body.decode('utf-8', errors='ignore')
        log_proxy_response_content(
            f"/{path}",
            response_body=response_text,
            processing_time=processing_time,
            status_code=resp.status,
            config=config
        )
    except Exception as e:
        log_message(f"⚠️ Error logging response content: {e}", "DEBUG")
    
    # Handle metrics and decompression for chat completions
    if request.method == "POST" and "chat/completions" in path:
        try:
            data = json.loads(response_body)
            if "choices" in data:
                # DEBUG: Log rule_union before decompression
                log_message(f"🔍 [PROXY DECOMPRESSION DEBUG] rule_union has {len(rule_union)} entries:")
                for symbol, pattern in list(rule_union.items())[:10]:  # Show first 10
                    log_message(f"🔍 [PROXY DECOMPRESSION DEBUG]   '{symbol}' → '{pattern}'")
                if len(rule_union) > 10:
                    log_message(f"🔍 [PROXY DECOMPRESSION DEBUG]   ... and {len(rule_union) - 10} more")
                
                # Decompress response
                data = decompress_response(data, rule_union)
                response_body = json.dumps(data).encode('utf-8')
                
                # Calculate and log metrics
                end_time = time.time()
                total_time = end_time - start_time
                llm_time = end_time - llm_start_time
                
                # Extract token usage from response
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Log performance metrics
                log_performance_metrics(
                    path,
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
                    llm_start_time,
                    config
                )
                
                # Log verbose content for proxy requests
                if config.verbose_logging:
                    try:
                        request_data = json.loads(body.decode('utf-8'))
                        original_messages = request_data.get("messages", [])
                        compressed_messages = request_data.get("messages", [])
                        log_verbose_content(
                            path,
                            original_messages=original_messages,
                            compressed_messages=compressed_messages,
                            response_data=data,
                            config=config
                        )
                    except:
                        pass
                
                # Log completion message
                try:
                    log_completion_message(
                        f"proxy/{path}",
                        compression_tokens_saved,
                        dynamic_chars_used,
                        rule_union,
                        original_content_size,
                        "generic",  # language
                        None,  # comment_stats
                        None,  # timing_breakdown
                        config
                    )
                    
                    # Log passthrough response if compression was minimal
                    if len(rule_union) == 0:
                        log_passthrough_response(
                            f"proxy/{path}",
                            response_data=data,
                            processing_time=total_time,
                            config=config
                        )
                except:
                    pass
        except Exception:
            # If decompression fails, return original response
            pass
            
    # CRITICAL FIX: Handle headers properly after potential decompression
    response_headers = dict(resp.headers)
    
    # Always remove Content-Length headers to avoid mismatches
    response_headers.pop('content-length', None)
    response_headers.pop('Content-Length', None)
    log_message(f"🔧 [PROXY] Removed Content-Length headers to prevent mismatches", "DEBUG")
    
    return Response(
        content=response_body,
        status_code=resp.status,
        headers=response_headers
    ) 