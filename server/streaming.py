"""
Streaming response handlers for KrunchWrapper server.

This module contains all streaming-specific logic extracted from server.py:
- Streaming timeout configuration
- aiohttp session management for streaming
- Context-aware decompression for streaming chunks
- Complete streaming response handler with metrics

All functions are optimized for streaming performance and Cline compatibility.
"""

import json
import time
import logging
from typing import Dict, Optional

import aiohttp
from fastapi.responses import StreamingResponse

# Import decompression function from core
from core.compress import decompress

# Import logging functions  
from server.logging_utils import (
    log_message, log_performance_metrics, log_completion_message,
    log_passthrough_response
)


def create_streaming_timeout(request_payload: dict = None, config=None) -> aiohttp.ClientTimeout:
    """
    Create ClientTimeout object for streaming connections.
    
    Args:
        request_payload: Optional request payload that may contain timeout overrides
        config: ServerConfig instance to use (avoids creating new instances)
    
    Returns:
        aiohttp.ClientTimeout object with appropriate settings
    """
    # Import config here only if not provided (avoid circular imports)
    if config is None:
        from server.config import ServerConfig
        config = ServerConfig()
    
    # Start with config defaults
    connect_timeout = config.streaming_connect_timeout
    total_timeout = config.streaming_total_timeout
    
    # Apply overrides from request payload if enabled
    if config.enable_timeout_override and request_payload:
        # Check for various timeout parameter names that might be in the request
        timeout_override_keys = {
            'timeout', 'request_timeout', 'total_timeout', 'client_timeout',
            'connect_timeout', 'connection_timeout', 'stream_timeout'
        }
        
        for key in timeout_override_keys:
            if key in request_payload:
                override_value = request_payload[key]
                if isinstance(override_value, (int, float)) and override_value > 0:
                    if 'connect' in key.lower():
                        connect_timeout = override_value
                        log_message(f"⏱️  Override: connect timeout set to {override_value}s from request", "DEBUG", config)
                    else:
                        total_timeout = override_value
                        log_message(f"⏱️  Override: total timeout set to {override_value}s from request", "DEBUG", config)
                    break  # Use first valid override found
    
    return aiohttp.ClientTimeout(
        total=total_timeout,
        connect=connect_timeout
    )


def create_aiohttp_session(timeout: aiohttp.ClientTimeout = None) -> aiohttp.ClientSession:
    """
    Create a properly configured aiohttp ClientSession with larger buffer sizes 
    to handle large streaming chunks and avoid "Chunk too big" errors.
    """
    connector = aiohttp.TCPConnector(
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    return aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=1024 * 1024,  # 1MB read buffer
        max_line_size=8192 * 4,    # 32KB max line size (default is 8KB)
        max_field_size=8192 * 8    # 64KB max field size (default is 8KB)
    )


def context_aware_decompress(chunk_content: str, rule_union: dict) -> str:
    """
    Context-aware decompression that only decompresses when compression symbols are actually present.
    """
    if not chunk_content or not rule_union:
        return chunk_content
    
    # CRITICAL: Only decompress if compression symbols are actually present in the content
    if not any(symbol in chunk_content for symbol in rule_union.keys()):
        return chunk_content
    
    # Apply decompression since symbols are present
    return decompress(chunk_content, rule_union)


async def handle_streaming_response_with_metrics(
    target_url: str, 
    payload: dict, 
    rule_union: dict, 
    start_time: float, 
    llm_start_time: float, 
    preprocessing_time: float,
    compression_ratio: float, 
    compression_tokens_saved: int, 
    dynamic_chars_used: int, 
    original_messages: list = None, 
    compressed_messages: list = None, 
    comment_stats: dict = None, 
    should_compress: bool = True,
    config=None
):
    """Handle streaming response from LLM API with decompression and metrics."""
    # Import config here only if not provided (avoid circular imports)
    if config is None:
        from server.config import ServerConfig
        config = ServerConfig()
    
    # Track streaming metrics
    total_tokens = 0
    completion_tokens = 0
    prompt_tokens = 0
    first_token_time = None
    
    # Capture streaming response content for verbose logging
    captured_response_content = []
    
    async def event_generator():
        nonlocal total_tokens, completion_tokens, prompt_tokens, first_token_time
        
        try:
            # Create session inside the generator to ensure proper lifecycle
            timeout = create_streaming_timeout(payload, config)
            async with create_aiohttp_session(timeout) as session:
                # Use explicit JSON encoding to handle Unicode properly  
                headers = {"Content-Type": "application/json; charset=utf-8"}
                
                async with session.post(target_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log_message(f"🚨 [ERROR] Target API returned {resp.status}: {error_text}", config=config)
                        # Yield error response in SSE format
                        error_response = {
                            "error": {
                                "message": f"Target API error: {error_text}",
                                "type": "api_error",
                                "code": resp.status
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield f"data: [DONE]\n\n"
                        return
                    
                    # Process the SSE stream
                    async for line in resp.content:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                # Calculate final metrics for streaming
                                end_time = time.time()
                                total_time = end_time - start_time
                                llm_time = end_time - llm_start_time
                                
                                # Calculate rates
                                avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
                                prompt_tokens_per_sec = prompt_tokens / preprocessing_time if preprocessing_time > 0 else 0
                                gen_tokens_per_sec = completion_tokens / llm_time if llm_time > 0 else 0
                                
                                # Log performance metrics
                                log_performance_metrics(
                                    "chat/completions",
                                    total_time,
                                    preprocessing_time,
                                    llm_time,
                                    compression_ratio,
                                    compression_tokens_saved,
                                    dynamic_chars_used,
                                    total_tokens,
                                    prompt_tokens,
                                    completion_tokens,
                                    first_token_time,
                                    llm_start_time,
                                    config
                                )
                                
                                # Enhanced verbose logging with actual content
                                if config.verbose_logging:
                                    log_message(f"\n🔍 Verbose Logging (streaming):", config=config)
                                    log_message("=" * 80, config=config)
                                    log_message("📝 ORIGINAL MESSAGES:", config=config)
                                    if original_messages:
                                        for i, msg in enumerate(original_messages):
                                            content = msg.get("content", "")
                                            log_message(f"   [{msg.get('role', 'unknown')}] {content}", config=config)
                                    else:
                                        log_message("   [No original messages available]", config=config)
                                    
                                    log_message("\n🗜️  COMPRESSED MESSAGES:", config=config)
                                    if compressed_messages:
                                        for i, msg in enumerate(compressed_messages):
                                            content = msg.get("content", "")
                                            log_message(f"   [{msg.get('role', 'unknown')}] {content}", config=config)
                                    else:
                                        log_message("   [No compressed messages available]", config=config)
                                    
                                    # Display captured streaming response
                                    full_response = "".join(captured_response_content)
                                    if full_response.strip():
                                        log_message(f"\n🤖 LLM RESPONSE:", config=config)
                                        log_message(f"   {full_response}", config=config)
                                    else:
                                        log_message(f"\n🤖 LLM RESPONSE: [No content captured]", config=config)
                                    
                                    log_message("=" * 80 + "\n", config=config)
                                
                                # Prepare timing breakdown for streaming
                                current_time = time.time()
                                # For streaming, we don't have separate decompression timing since it happens per chunk
                                streaming_timing_breakdown = {
                                    'ingest_to_prompt': llm_start_time - start_time,  # Preprocessing time
                                    'prompt_to_llm_send': 0.01,  # Minimal time for streaming setup
                                    'llm_processing': current_time - llm_start_time,
                                    'llm_response_to_decompress': 0.01,  # Smart decompression per chunk (only when symbols present)
                                    'total_time': current_time - start_time
                                }
                                
                                # Log completion message with total compression stats and timing
                                # Calculate original content size from original messages
                                original_content_size = sum(len(msg.get("content", "")) for msg in original_messages) if original_messages else 0
                                log_completion_message(
                                    "chat/completions (streaming)",
                                    compression_tokens_saved,
                                    dynamic_chars_used,
                                    rule_union,
                                    original_content_size,
                                    "generic",  # Language detection context may not be available here
                                    comment_stats,
                                    streaming_timing_breakdown,
                                    config
                                )
                                
                                # Log passthrough response if compression was skipped for streaming
                                if compression_ratio <= config.min_compression_ratio or not should_compress:
                                    # For streaming, construct response data from captured content
                                    full_response = "".join(captured_response_content)
                                    streaming_response_data = {
                                        "choices": [{"message": {"content": full_response}}],
                                        "usage": {
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens
                                        }
                                    }
                                    log_passthrough_response(
                                        "chat/completions (streaming)",
                                        response_data=streaming_response_data,
                                        processing_time=current_time - start_time,
                                        config=config
                                    )
                                
                                yield f"data: [DONE]\n\n"
                                continue
                            
                            try:
                                data = json.loads(data_str)
                                
                                # Track token usage from streaming chunks
                                if "usage" in data:
                                    usage = data["usage"]
                                    total_tokens = usage.get("total_tokens", total_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                
                                # Track first token timing and capture content
                                for choice in data.get("choices", []):
                                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                                        if first_token_time is None:
                                            first_token_time = time.time()
                                        
                                        # Get the original content
                                        original_content = choice["delta"]["content"]
                                        
                                        # IMPORTANT: Only decompress if we have a complete JSON structure
                                        # and compression symbols are present
                                        if rule_union and any(symbol in original_content for symbol in rule_union.keys()):
                                            # Only decompress complete content, not partial chunks
                                            try:
                                                decompressed_content = decompress(original_content, rule_union)
                                                choice["delta"]["content"] = decompressed_content
                                                log_message(f"🔍 [STREAMING DEBUG] Decompressed: '{original_content}' → '{decompressed_content}'", "DEBUG", config)
                                            except Exception as e:
                                                # If decompression fails, keep original content
                                                log_message(f"⚠️ Decompression failed in streaming chunk: {e}", "INFO", config)
                                                decompressed_content = original_content
                                        else:
                                            # No compression symbols present - pass through unchanged
                                            decompressed_content = original_content
                                        
                                        # Capture decompressed content for verbose logging 
                                        captured_response_content.append(decompressed_content)
                                
                                # Validate JSON structure before yielding (for better Cline compatibility)
                                try:
                                    # Ensure the data has the expected structure for Cline compatibility
                                    if "choices" in data and isinstance(data["choices"], list):
                                        for choice in data["choices"]:
                                            if "delta" in choice:
                                                # Ensure content is string or None, not corrupted
                                                if "content" in choice["delta"]:
                                                    content = choice["delta"]["content"]
                                                    if content is not None and not isinstance(content, str):
                                                        choice["delta"]["content"] = str(content)
                                    
                                    # CRITICAL: Ensure required fields are present for Cline
                                    if "id" not in data:
                                        data["id"] = f"chatcmpl-{int(time.time())}"
                                    if "object" not in data:
                                        data["object"] = "chat.completion.chunk"
                                    if "created" not in data:
                                        data["created"] = int(time.time())
                                    if "model" not in data and payload.get("model"):
                                        data["model"] = payload["model"]
                                    
                                    # CRITICAL: Ensure we yield the properly formatted SSE line
                                    yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                                except Exception as e:
                                    log_message(f"⚠️ JSON validation error in streaming: {e}", "ERROR", config)
                                    log_message(f"⚠️ Problematic data: {json.dumps(data, indent=2) if 'data' in locals() else 'No data'}", "ERROR", config)
                                    # Yield original line if validation fails
                                    yield line + "\n"
                            except json.JSONDecodeError:
                                # Pass through any non-JSON data
                                yield f"data: {data_str}\n\n"
                        else:
                            # Pass through any non-data lines
                            yield line + "\n"
        except aiohttp.ClientError as e:
            log_message(f"🚨 [ERROR] Network error in streaming: {e}", config=config)
            # Yield error response in SSE format
            error_response = {
                "error": {
                    "message": f"Network error: {str(e)}",
                    "type": "connection_error", 
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
            yield f"data: [DONE]\n\n"
        except json.JSONDecodeError as e:
            log_message(f"🚨 [ERROR] JSON decoding error in streaming: {e}", config=config)
            # Yield error response in SSE format
            error_response = {
                "error": {
                    "message": f"JSON decoding error: {str(e)}",
                    "type": "decoding_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
            yield f"data: [DONE]\n\n"
        except Exception as e:
            log_message(f"🚨 [ERROR] Unexpected streaming error: {e}", config=config)
            # Yield error response in SSE format
            error_response = {
                "error": {
                    "message": f"Streaming error: {str(e)}",
                    "type": "unknown_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
            yield f"data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    ) 