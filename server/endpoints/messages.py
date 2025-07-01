"""Anthropic Messages endpoint module."""
import time
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import Request, HTTPException
from pydantic import BaseModel, Field
import aiohttp

from server.logging_utils import (
    log_message, log_performance_metrics, log_verbose_content, 
    log_passthrough_request, log_passthrough_response, log_completion_message
)
from server.streaming import (
    create_aiohttp_session, handle_streaming_response_with_metrics, create_streaming_timeout
)
from server.request_utils import (
    filter_timeout_parameters, apply_interface_specific_fixes, 
    _is_conversation_continuation
)
from server.response_utils import (
    decompress_response, fix_payload_encoding
)

from core.compress import compress_with_dynamic_analysis
from core.model_context import set_global_model_context, extract_model_from_provider_format
from core.interface_engine import detect_and_process_compression


class AnthropicContentBlock(BaseModel):
    """Anthropic content block format."""
    type: str = Field(..., description="Content block type (e.g., 'text')")
    text: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional content block types


class AnthropicMessage(BaseModel):
    """Anthropic message format."""
    role: str
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic /v1/messages request format."""
    model: str
    messages: List[AnthropicMessage] 
    system: Optional[Union[str, List[AnthropicContentBlock]]] = None
    max_tokens: int
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional Anthropic-specific fields


async def anthropic_messages(request: AnthropicMessagesRequest, http_request: Request, config):
    """Handle Anthropic Messages API requests with compression."""
    start_time = time.time()
    
    # Check for cache control in messages
    has_cache_control = False
    for msg in request.messages:
        if isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, 'cache_control') and block.cache_control:
                    has_cache_control = True
                    break
                elif isinstance(block, dict) and block.get('cache_control'):
                    has_cache_control = True
                    break
    if request.system and isinstance(request.system, list):
        for block in request.system:
            if hasattr(block, 'cache_control') and block.cache_control:
                has_cache_control = True
                break
            elif isinstance(block, dict) and block.get('cache_control'):
                has_cache_control = True
                break
    
    log_message(f"🔧 [ANTHROPIC] Processing Anthropic Messages API request", "INFO", config)
    log_message(f"🔧 [ANTHROPIC] Model: {request.model}, Messages: {len(request.messages)}, Max tokens: {request.max_tokens}, Stream: {request.stream}", "INFO", config)
    log_message(f"🗄️ [ANTHROPIC] Prompt caching: {'Enabled' if has_cache_control else 'Not detected'}", "INFO", config)
    
    # Set model context for tokenizer validation
    model_name = request.model
    if model_name:
        clean_model = extract_model_from_provider_format(model_name)
        set_global_model_context(clean_model)
        log_message(f"🔧 Set model context: {clean_model} (from {model_name})", "DEBUG", config)
    
    # Helper function to extract text from content (handles both string and content blocks)
    def extract_text_from_content(content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if hasattr(block, 'text') and block.text:
                    text_parts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
            return ' '.join(text_parts)
        return ""
    
    # Check if content meets minimum character threshold for compression
    total_content_size = sum(len(extract_text_from_content(msg.content)) for msg in request.messages if msg.role in {"user", "assistant"})
    if request.system:
        total_content_size += len(extract_text_from_content(request.system))
    should_compress = total_content_size >= config.min_characters
    
    # Estimate token usage for rate limiting awareness
    estimated_tokens = total_content_size // 4  # Rough estimate: 4 chars = 1 token
    log_message(f"📏 Total content size: {total_content_size} chars (~{estimated_tokens} tokens), Min threshold: {config.min_characters} chars", config=config)
    log_message(f"🗜️  Compression {'enabled' if should_compress else 'skipped'}", config=config)
    if estimated_tokens > 15000:
        log_message(f"⚠️ [ANTHROPIC] Large request: ~{estimated_tokens} tokens may approach rate limits (20,000/min)", "WARNING", config)
    
    # Force Anthropic interface detection for this endpoint
    from core.interface_engine import get_interface_compression_handler, InterfaceEngine
    handler = get_interface_compression_handler(config)
    detected_engine = InterfaceEngine.ANTHROPIC  # Force Anthropic interface
    
    # Helper function to preserve content block structure while applying existing compression rules
    def preserve_content_blocks_with_compression(content, apply_compression=False, compression_rules=None):
        """Preserve content block structure including cache_control while applying existing compression rules."""
        if isinstance(content, str):
            if apply_compression and compression_rules and len(compression_rules) > 0:
                # Apply existing compression rules to string content
                compressed_text = content
                log_message(f"🔍 [ANTHROPIC DEBUG] Applying {len(compression_rules)} rules to string of {len(content)} chars", "DEBUG", config)
                for original, symbol in compression_rules.items():
                    old_len = len(compressed_text)
                    compressed_text = compressed_text.replace(original, symbol)
                    if len(compressed_text) != old_len:
                        log_message(f"🔍 [ANTHROPIC DEBUG] Rule applied: '{original}' → '{symbol}' (saved {old_len - len(compressed_text)} chars)", "DEBUG", config)
                log_message(f"🗜️ [ANTHROPIC] Applied rules to string: {len(content)} → {len(compressed_text)} chars", "DEBUG", config)
                return compressed_text
            else:
                log_message(f"🔍 [ANTHROPIC DEBUG] No compression applied: apply_compression={apply_compression}, rules={len(compression_rules) if compression_rules else 0}", "DEBUG", config)
            return content
        elif isinstance(content, list):
            # Preserve content blocks structure and apply compression rules to text within blocks
            processed_blocks = []
            for block in content:
                if hasattr(block, 'model_dump'):  # Pydantic model
                    block_dict = block.model_dump()
                elif isinstance(block, dict):
                    block_dict = block.copy()
                else:
                    continue
                
                # Apply compression rules to text content while preserving cache_control
                if 'text' in block_dict and block_dict['text']:
                    original_text = block_dict['text']
                    if apply_compression and compression_rules and len(compression_rules) > 0:
                        # Apply existing compression rules
                        compressed_text = original_text
                        for original, symbol in compression_rules.items():
                            compressed_text = compressed_text.replace(original, symbol)
                        block_dict['text'] = compressed_text
                        log_message(f"🗜️ [ANTHROPIC] Applied rules to content block: {len(original_text)} → {len(compressed_text)} chars, preserved cache_control: {bool(block_dict.get('cache_control'))}", "DEBUG", config)
                    else:
                        # No compression, keep original text
                        pass
                
                processed_blocks.append(block_dict)
            return processed_blocks
        return content
    
    # Convert AnthropicMessagesRequest to dict format for Interface Engine
    messages_dict = []
    for msg in request.messages:
        msg_dict = {
            'role': msg.role,
            'content': extract_text_from_content(msg.content),  # Still needed for Interface Engine
        }
        messages_dict.append(msg_dict)
    
    rule_union: Dict[str, str] = {}
    compression_ratio = 0.0
    dynamic_chars_used = 0
    compression_tokens_saved = 0
    
    if should_compress:
        log_message(f"🗜️ [ANTHROPIC] Using Anthropic interface compression pipeline with cache_control preservation", "INFO", config)
        
        # First: Process all message content through main compression pipeline
        all_content_text = ""
        for msg in messages_dict:
            all_content_text += msg.get('content', '') + "\n"
        
        # Add system prompt to compression if present
        system_text = extract_text_from_content(request.system) if request.system else ""
        if system_text:
            all_content_text += system_text + "\n"
        
        log_message(f"🗜️ [ANTHROPIC] Compressing {len(all_content_text)} chars of content", "DEBUG", config)
        
        # Run main compression to generate rule_union
        try:
            from core.compress import compress_with_dynamic_analysis
            packed = compress_with_dynamic_analysis(all_content_text, skip_tool_detection=False)
            # CRITICAL FIX: packed.used contains decompression rules (symbol → long_text)
            # For compression, we need the inverse (long_text → symbol)
            compression_rules = {long_text: symbol for symbol, long_text in packed.used.items()}
            rule_union.update(compression_rules)
            log_message(f"🗜️ [ANTHROPIC] Main compression generated {len(rule_union)} rules", "INFO", config)
            # Debug: Show what compression rules were generated
            if len(rule_union) > 0:
                sample_rules = list(rule_union.items())[:3]  # Show first 3 rules
                log_message(f"🔍 [ANTHROPIC DEBUG] Sample compression rules: {sample_rules}", "DEBUG", config)
        except Exception as e:
            log_message(f"⚠️ [ANTHROPIC] Main compression failed: {e}", "WARNING", config)
        
        # Second: Process through Interface Engine for system prompt handling (no additional compression)
        processed_messages, system_metadata, engine = detect_and_process_compression(
            request=http_request,
            messages=messages_dict,
            rule_union=rule_union,  # Pass existing rules from main compression
            config=config,
            model_id=request.model,
            system_param=extract_text_from_content(request.system) if request.system else None,
            system_instruction=None,
            target_format="claude",
            request_data={"model": request.model, "messages": messages_dict}
        )
        
        # Update rule_union with any additional system prompt compression rules
        if 'rule_union' in system_metadata:
            additional_rules = system_metadata['rule_union']
            if additional_rules:
                rule_union.update(additional_rules)
                log_message(f"🗣️ [ANTHROPIC] Updated rule_union with {len(additional_rules)} system prompt rules", "DEBUG", config)
        
        # Get the processed system prompt from metadata
        final_system_prompt = system_metadata.get("anthropic_system_prompt")
        if final_system_prompt is None and request.system:
            final_system_prompt = extract_text_from_content(request.system)
        
        log_message(f"🗜️ [ANTHROPIC] Compression complete: {len(rule_union)} total rules applied", config=config)
        log_message(f"✅ [ANTHROPIC] Applied Anthropic system prompt processing - ready for LLM", "INFO", config)
    else:
        # No compression, but still process through Interface Engine for system prompts
        log_message(f"🔄 [ANTHROPIC] Processing Anthropic request without compression", "INFO", config)
        
        # Still use Interface Engine for system prompt processing
        processed_messages, system_metadata, engine = detect_and_process_compression(
            request=http_request,
            messages=messages_dict,
            rule_union=rule_union,  # Empty since no compression
            config=config,
            model_id=request.model,
            system_param=extract_text_from_content(request.system) if request.system else None,
            system_instruction=None,
            target_format="claude",
            request_data={"model": request.model, "messages": messages_dict}
        )
        
        packed_msgs = processed_messages
        compression_tokens_saved = 0
        final_system_prompt = system_metadata.get("anthropic_system_prompt")
        if final_system_prompt is None and request.system:
            final_system_prompt = extract_text_from_content(request.system)
        
        log_message(f"✅ [ANTHROPIC] Applied Anthropic system prompt processing without compression", "INFO", config)
    
    # Prepare payload for the Anthropic API in native format with cache_control preservation
    # Process original messages with compression while preserving cache_control
    final_messages = []
    for msg in request.messages:
        processed_content = preserve_content_blocks_with_compression(
            msg.content, 
            apply_compression=should_compress, 
            compression_rules=rule_union
        )
        final_messages.append({
            "role": msg.role,
            "content": processed_content
        })
    
    # Process system prompt with cache_control preservation
    final_system = None
    if request.system:
        final_system = preserve_content_blocks_with_compression(
            request.system,
            apply_compression=should_compress,
            compression_rules=rule_union
        )
        # If Interface Engine provided a processed system prompt, prefer that for compression rules
        if final_system_prompt and final_system_prompt != extract_text_from_content(request.system):
            # System prompt was processed by Interface Engine, but we need to preserve cache_control
            if isinstance(request.system, list):
                # Update the text in the first content block with the processed system prompt
                final_system = preserve_content_blocks_with_compression(request.system, apply_compression=False)
                if final_system and len(final_system) > 0 and 'text' in final_system[0]:
                    final_system[0]['text'] = final_system_prompt
            else:
                final_system = final_system_prompt
    
    # Calculate compression metrics including token savings
    # Only count user/assistant message content for compression metrics (exclude system prompt decoder)
    original_content_size = sum(len(extract_text_from_content(msg.content)) for msg in request.messages if msg.role in {"user", "assistant"})
    original_system_size = len(extract_text_from_content(request.system)) if request.system else 0
    
    # Calculate compressed size from final payload (excluding compression decoder)
    compressed_content_size = 0
    for msg in final_messages:
        if msg.get("role") in {"user", "assistant"}:
            compressed_content_size += len(extract_text_from_content(msg.get("content", "")))
    
    # For system prompt, only count the original system content size if it exists
    # The compression decoder is metadata, not user content, so exclude it from metrics
    compressed_system_size = original_system_size  # System size doesn't change for compression ratio calculation
    
    # Total content sizes for metrics calculation
    total_original_size = original_content_size + original_system_size
    total_compressed_size = compressed_content_size + compressed_system_size
    
    compression_ratio = (total_original_size - total_compressed_size) / total_original_size if total_original_size > 0 else 0
    dynamic_chars_used = sum(len(value) for value in rule_union.values())
    compression_tokens_saved = max(0, (total_original_size - total_compressed_size) // 4)  # Rough estimate
    
    # Add debug logging for metrics calculation
    log_message(f"🗜️ [ANTHROPIC] Applied rules to string: {total_original_size} → {total_compressed_size} chars", "DEBUG", config)
    
    log_message(f"📊 [ANTHROPIC] Final metrics: {compression_tokens_saved} tokens saved, {compression_ratio*100:.1f}% compression, Rules: {len(rule_union)}", "INFO", config)
    if len(rule_union) > 0:
        log_message(f"🔧 [ANTHROPIC] Cache control preserved in content blocks", "INFO", config)
    
    payload = {
        "model": request.model,
        "messages": final_messages,
        "max_tokens": request.max_tokens,
    }
    
    # Add the processed system prompt
    if final_system:
        payload["system"] = final_system
    
    # Add optional parameters
    for field in ["temperature", "top_p", "top_k", "stream", "stop_sequences", "metadata"]:
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value
    
    # Apply timeout parameter filtering (though Anthropic may have different timeouts)
    payload = filter_timeout_parameters(payload, config.filter_timeout_parameters)
    
    # Forward request to target Anthropic API
    llm_start_time = time.time()
    # Remove /v1 suffix from target_url if present, then add /v1/messages
    base_url = config.target_url.rstrip('/v1').rstrip('/')
    target_url = f"{base_url}/v1/messages"
    
    # Check if this is a streaming request
    is_streaming = payload.get("stream", False)
    
    # Handle request - both streaming and non-streaming
    try:
        async with create_aiohttp_session() as session:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            
            # Add Anthropic-specific headers from the original request
            anthropic_headers = ["x-api-key", "anthropic-version", "anthropic-beta"]
            for header_name in anthropic_headers:
                if header_name in http_request.headers:
                    headers[header_name] = http_request.headers[header_name]
            
            # Add API key if configured
            if not headers.get("x-api-key") and config.api_key:
                headers["x-api-key"] = config.api_key
            
            log_message(f"🔧 [ANTHROPIC] Forwarding to: {target_url}", "DEBUG", config)
            log_message(f"🔧 [ANTHROPIC] Headers: {list(headers.keys())}", "DEBUG", config)
            log_message(f"🔧 [ANTHROPIC] Streaming: {is_streaming}", "DEBUG", config)
            log_message(f"🔧 [ANTHROPIC] Payload: {json.dumps(payload, indent=2)[:500]}...", "DEBUG", config)
            
            if is_streaming:
                # Handle streaming response for Anthropic Messages API
                # For streaming, we need a self-contained session that doesn't depend on the outer context
                return await handle_anthropic_streaming_response(
                    target_url, payload, headers, rule_union, 
                    start_time, llm_start_time, compression_ratio, 
                    compression_tokens_saved, dynamic_chars_used, config
                )
            else:
                # Handle non-streaming response
                async with session.post(target_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log_message(f"🚨 [ANTHROPIC ERROR] Target API returned {resp.status}: {error_text}", config=config)
                        raise HTTPException(status_code=resp.status, detail=error_text)
                    
                    # Handle Anthropic response format
                    data = await resp.json()
                    llm_response_received_time = time.time()
                    
                    # Decompress response (Anthropic format)
                    decompressed_data = decompress_anthropic_response(data, rule_union)
                    decompression_complete_time = time.time()
                    
                    # Calculate metrics
                    total_time = decompression_complete_time - start_time
                    llm_time = llm_response_received_time - llm_start_time
                    
                    # Extract token usage from Anthropic response
                    usage = data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    total_tokens = input_tokens + output_tokens
                    
                    # Log performance metrics
                    log_performance_metrics(
                        "messages",
                        total_time,
                        time.time() - start_time,
                        llm_time,
                        compression_ratio,
                        compression_tokens_saved,
                        dynamic_chars_used,
                        total_tokens,
                        input_tokens,
                        output_tokens,
                        None,
                        llm_start_time,
                        config
                    )
                    
                    log_message(f"✅ [ANTHROPIC] Successfully processed request: {input_tokens} input + {output_tokens} output = {total_tokens} total tokens", "INFO", config)
                    
                    return decompressed_data
                
    except aiohttp.ClientError as e:
        log_message(f"🚨 [ANTHROPIC ERROR] Network error when sending to target API: {e}", config=config)
        raise HTTPException(status_code=500, detail=f"Failed to connect to target Anthropic API: {str(e)}")
    except Exception as e:
        log_message(f"🚨 [ANTHROPIC ERROR] Unexpected error when sending to target API: {e}", config=config)
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")


async def handle_anthropic_streaming_response(
    target_url: str, payload: dict, headers: dict, rule_union: dict,
    start_time: float, llm_start_time: float, compression_ratio: float,
    compression_tokens_saved: int, dynamic_chars_used: int, config
):
    """Handle streaming response specifically for Anthropic Messages API following official SSE format."""
    from fastapi.responses import StreamingResponse
    
    async def anthropic_event_generator():
        # Create session that will live for the entire streaming duration
        timeout = create_streaming_timeout(payload, config)
        
        # Create a more robust session for streaming with explicit timeout handling
        connector = aiohttp.TCPConnector(
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=120,  # Keep connections alive longer
            enable_cleanup_closed=False  # Don't auto-cleanup closed connections
        )
        
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            read_bufsize=1024 * 1024,  # 1MB read buffer
            max_line_size=8192 * 4,    # 32KB max line size
            max_field_size=8192 * 8,   # 64KB max field size
            auto_decompress=False      # Disable auto-decompression to avoid issues
        )
        
        log_message(f"🔧 [ANTHROPIC DEBUG] Created streaming session", "DEBUG", config)
        
        try:
            resp = None
            try:
                log_message(f"🔧 [ANTHROPIC DEBUG] Starting request to {target_url}", "DEBUG", config)
                resp = await session.post(target_url, json=payload, headers=headers)
                log_message(f"🔧 [ANTHROPIC DEBUG] Got response status: {resp.status}", "DEBUG", config)
                
                if resp.status != 200:
                    error_text = await resp.text()
                    log_message(f"🚨 [ANTHROPIC ERROR] Target API returned {resp.status}: {error_text}", config=config)
                    # Send error in proper SSE format
                    yield f"event: error\n"
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_text, 'type': 'api_error', 'code': resp.status}})}\n\n"
                    return
                
                # Track metrics
                total_tokens = 0
                input_tokens = 0
                output_tokens = 0
                
                log_message(f"🔧 [ANTHROPIC DEBUG] Starting to read streaming content", "DEBUG", config)
                
                # Process Server-Sent Events stream
                try:
                    async for line_bytes in resp.content:
                        try:
                            line = line_bytes.decode('utf-8').rstrip()
                            
                            # Skip empty lines
                            if not line:
                                yield "\n"
                                continue
                            
                            # Handle SSE event lines
                            if line.startswith('event: '):
                                event_name = line[7:].strip()
                                yield f"event: {event_name}\n"
                                continue
                                
                            elif line.startswith('data: '):
                                data_str = line[6:].strip()
                                
                                # Parse JSON data for Anthropic events
                                try:
                                    data = json.loads(data_str)
                                    event_type = data.get("type")
                                    
                                    # Handle different Anthropic streaming events
                                    if event_type == "message_start":
                                        log_message(f"🔧 [ANTHROPIC STREAM] Message started", "DEBUG", config)
                                        
                                    elif event_type == "content_block_start":
                                        log_message(f"🔧 [ANTHROPIC STREAM] Content block started", "DEBUG", config)
                                        
                                    elif event_type == "content_block_delta":
                                        # This is where text content appears - apply decompression here
                                        if (rule_union and 
                                            "delta" in data and 
                                            data["delta"].get("type") == "text_delta" and 
                                            "text" in data["delta"]):
                                            
                                            original_text = data["delta"]["text"]
                                            # CRITICAL FIX: rule_union contains compression rules (long_text → symbol)
                                            # For decompression, we need the inverse (symbol → long_text) 
                                            decompression_rules = {symbol: long_text for long_text, symbol in rule_union.items()}
                                            # Only decompress if compression symbols are present
                                            if any(symbol in original_text for symbol in decompression_rules.keys()):
                                                try:
                                                    from server.streaming import context_aware_decompress
                                                    decompressed_text = context_aware_decompress(original_text, decompression_rules)
                                                    data["delta"]["text"] = decompressed_text
                                                    log_message(f"🔍 [ANTHROPIC STREAMING] Decompressed: '{original_text}' → '{decompressed_text}'", "DEBUG", config)
                                                except Exception as e:
                                                    log_message(f"⚠️ [ANTHROPIC] Streaming decompression failed: {e}", "DEBUG", config)
                                    
                                    elif event_type == "content_block_stop":
                                        log_message(f"🔧 [ANTHROPIC STREAM] Content block stopped", "DEBUG", config)
                                        
                                    elif event_type == "message_delta":
                                        # Extract token usage and check for stop_reason
                                        if "usage" in data:
                                            usage = data["usage"]
                                            if "input_tokens" in usage:
                                                input_tokens = usage["input_tokens"]
                                            if "output_tokens" in usage:
                                                output_tokens = usage["output_tokens"]
                                            total_tokens = input_tokens + output_tokens
                                        
                                        # Check if this contains stop_reason (end of stream)
                                        if "delta" in data and "stop_reason" in data["delta"]:
                                            stop_reason = data["delta"]["stop_reason"]
                                            log_message(f"🔧 [ANTHROPIC STREAM] Stream ending with stop_reason: {stop_reason}", "DEBUG", config)
                                            
                                    elif event_type == "message_stop":
                                        # Stream is complete - calculate final metrics
                                        end_time = time.time()
                                        total_time = end_time - start_time
                                        llm_time = end_time - llm_start_time
                                        
                                        # Log performance metrics
                                        log_performance_metrics(
                                            "messages",
                                            total_time,
                                            time.time() - start_time,
                                            llm_time,
                                            compression_ratio,
                                            compression_tokens_saved,
                                            dynamic_chars_used,
                                            total_tokens,
                                            input_tokens,
                                            output_tokens,
                                            None,
                                            llm_start_time,
                                            config
                                        )
                                        
                                        log_message(f"✅ [ANTHROPIC] Successfully streamed response: {input_tokens} input + {output_tokens} output = {total_tokens} total tokens", "INFO", config)
                                        
                                        # Send the message_stop event and return
                                        yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                                        return
                                    
                                    # Send the processed data for all event types
                                    yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                                    
                                except json.JSONDecodeError:
                                    # Pass through non-JSON data as-is
                                    yield f"data: {data_str}\n\n"
                            
                            else:
                                # Pass through other SSE lines (like comments, empty lines, etc.)
                                yield f"{line}\n"
                                
                        except UnicodeDecodeError:
                            # Skip lines that can't be decoded
                            continue
                            
                except Exception as stream_read_error:
                    log_message(f"🚨 [ANTHROPIC DEBUG] Error reading stream: {stream_read_error}", config=config)
                    raise stream_read_error
                        
            except Exception as e:
                log_message(f"🚨 [ANTHROPIC ERROR] Streaming error: {e}", config=config)
                # Send error in proper SSE format
                yield f"event: error\n"
                yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e), 'type': 'streaming_error', 'code': 500}})}\n\n"
            
            finally:
                # Ensure response is properly closed
                if resp and not resp.closed:
                    log_message(f"🔧 [ANTHROPIC DEBUG] Closing response", "DEBUG", config)
                    resp.close()
                    
        finally:
            # Always close the session when streaming is done
            if session and not session.closed:
                log_message(f"🔧 [ANTHROPIC DEBUG] Closing session", "DEBUG", config)
                await session.close()
    
    return StreamingResponse(
        anthropic_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


def decompress_anthropic_response(data: Dict[str, Any], rule_union: Dict[str, str]) -> Dict[str, Any]:
    """Decompress Anthropic response format."""
    if not rule_union:
        return data
    
    # CRITICAL FIX: rule_union now contains compression rules (long_text → symbol)
    # For decompression, we need the inverse (symbol → long_text)
    decompression_rules = {symbol: long_text for long_text, symbol in rule_union.items()}
    
    # Handle Anthropic response format
    if "content" in data and isinstance(data["content"], list):
        for content_block in data["content"]:
            if "text" in content_block:
                # Apply decompression with inverted rules
                from server.streaming import context_aware_decompress
                original_text = content_block["text"]
                decompressed_text = context_aware_decompress(original_text, decompression_rules)
                content_block["text"] = decompressed_text
    
    return data 