"""Chat completion endpoint module."""
import time
import json
import hashlib
import logging
from typing import Dict
from fastapi import Request, HTTPException
import aiohttp

from server.models import ChatCompletionRequest
from server.logging_utils import (
    log_message, log_performance_metrics, log_verbose_content, 
    log_passthrough_request, log_passthrough_response, log_completion_message
)
from server.streaming import (
    create_aiohttp_session, handle_streaming_response_with_metrics
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
from core.cline_server_handler import get_cline_handler, is_cline_request, should_disable_compression_for_cline


async def chat_completion(request: ChatCompletionRequest, http_request: Request, config):
    """Handle OpenAI-compatible chat completions with compression."""
    start_time = time.time()
    
    # Initialize Cline handler and detect Cline requests
    cline_handler = get_cline_handler(config)
    
    # TEMPORARY DEBUG: Log ALL request headers to see what Cline sends
    log_message(f"🔍 [ALL REQUESTS] Headers: {dict(http_request.headers)}", "DEBUG", config)
    log_message(f"🔍 [ALL REQUESTS] Method: {http_request.method}", "DEBUG", config)
    log_message(f"🔍 [ALL REQUESTS] URL: {http_request.url}", "DEBUG", config)
    
    # Use unified Interface Engine for detection (replacing manual Cline detection)
    # This provides consistent interface handling across all endpoints
    from core.interface_engine import get_interface_compression_handler
    handler = get_interface_compression_handler(config)
    detected_engine = handler.detect_interface_engine(http_request, {})
    
    # Legacy compatibility - extract is_cline for any remaining cline-specific code
    is_cline = (detected_engine.value == "cline")
    
    if is_cline:
        log_message(f"🔍 [ENGINE] Detected {detected_engine.value} interface via Interface Engine", "DEBUG", config)
        cline_handler.log_cline_request_details(http_request, None)
    else:
        log_message(f"🔍 [ENGINE] Detected {detected_engine.value} interface via Interface Engine", "INFO", config)
    
    # Extract session ID from headers for conversation identification
    session_id = http_request.headers.get("x-session-id") or http_request.headers.get("X-Session-ID")

    # ENHANCED session ID generation for conversation continuity
    # Generate session ID for multi-message requests even if not provided
    if not session_id and len(request.messages) > 1:
        # Generate from first 2 messages for consistency
        signature = "".join([f"{msg.role}:{msg.content[:30]}" for msg in request.messages[:2]])
        session_id = f"auto_{hashlib.md5(signature.encode()).hexdigest()[:12]}"
        log_message(f"🔧 [SESSION] Generated session ID for conversation continuity: {session_id}", "INFO", config)
    elif not session_id and is_cline:
        # For Cline requests, always generate a session ID to support conversation compression
        signature = f"cline_{int(time.time())}"
        session_id = f"cline_{hashlib.md5(signature.encode()).hexdigest()[:8]}"
        log_message(f"🔧 [SESSION] Generated Cline session ID: {session_id}", "INFO", config)
    
    # Set model context for tokenizer validation
    model_name = request.model
    if model_name:
        # Extract clean model name from provider/model format if needed
        clean_model = extract_model_from_provider_format(model_name)
        set_global_model_context(clean_model)
        log_message(f"🔧 Set model context: {clean_model} (from {model_name})", "DEBUG", config)
    
    # Check if content meets minimum character threshold for compression
    total_content_size = sum(len(msg.content or "") for msg in request.messages if msg.role in {"user", "assistant", "system"})
    should_compress = total_content_size >= config.min_characters
    
    # Check for Cline-specific compression disabling
    if should_compress and is_cline:
        if should_disable_compression_for_cline(http_request, config):
            should_compress = False
            log_message("🚫 [CLINE] Compression disabled for Cline compatibility", "INFO", config)
    
    log_message(f"📏 Total content size: {total_content_size} chars, Min threshold: {config.min_characters} chars", config=config)
    log_message(f"🗜️  Compression {'enabled' if should_compress else 'skipped'}", config=config)
    
    # Use unified Interface Engine for compression and system prompt processing
    # This provides full conversation compression, system prompt handling, and interface-specific processing
    
    rule_union: Dict[str, str] = {}
    compression_ratio = 0.0
    dynamic_chars_used = 0
    compression_tokens_saved = 0
    
    if should_compress:
        log_message(f"🗜️ [ENGINE] Using {detected_engine.value} interface compression pipeline", "INFO", config)
        
        # Convert ChatCompletionRequest messages to dict format for Interface Engine
        messages_dict = []
        for msg in request.messages:
            msg_dict = msg.dict(exclude_none=True) if hasattr(msg, 'dict') else {
                'role': msg.role,
                'content': msg.content or '',
            }
            if msg_dict.get("content") is None:
                msg_dict["content"] = ""
            messages_dict.append(msg_dict)
        
        # Apply compression to messages
        original_total_tokens = 0
        compressed_total_tokens = 0
        
        for i, msg in enumerate(messages_dict):
            if msg.get("role") in {"user", "assistant", "system"} and should_compress:
                original_content = msg.get("content", "")
                
                if original_content:
                    # Track original tokens
                    original_msg_tokens = len(original_content) // 4
                    original_total_tokens += original_msg_tokens
                    
                    # Apply compression
                    exclude_symbols = set(rule_union.keys()) if rule_union else set()
                    packed = compress_with_dynamic_analysis(
                        original_content, 
                        skip_tool_detection=False, 
                        cline_mode=(detected_engine.value in ["cline", "webui", "sillytavern", "standard"]),
                        exclude_symbols=exclude_symbols
                    )
                    
                    messages_dict[i]["content"] = packed.text
                    rule_union.update(packed.used)
                    
                    # Track compressed tokens
                    compressed_msg_tokens = len(packed.text) // 4
                    compressed_total_tokens += compressed_msg_tokens
                    
                    log_message(f"🗜️ [ENGINE] Compressed {msg.get('role')} message {i}: {len(original_content)} → {len(packed.text)} chars", "DEBUG", config)
                else:
                    # Empty content, track tokens as 0
                    original_total_tokens += 0
                    compressed_total_tokens += 0
            else:
                # Non-compressible message, track tokens normally
                msg_content = msg.get("content", "")
                msg_tokens = len(msg_content) // 4
                original_total_tokens += msg_tokens
                compressed_total_tokens += msg_tokens
        
        # Use Interface Engine for system prompt processing
        log_message(f"🔧 [ENGINE] Processing through {detected_engine.value} interface system prompt handler", "INFO", config)
        
        processed_messages, system_metadata, engine = detect_and_process_compression(
            request=http_request,
            messages=messages_dict,
            rule_union=rule_union,
            config=config,
            model_id=request.model,
            target_format=getattr(config, 'system_prompt_format', 'chatml'),
            request_data={"model": request.model, "messages": messages_dict}
        )
        
        # Update rule_union with any system prompt compression rules
        if 'rule_union' in system_metadata:
            additional_rules = system_metadata['rule_union']
            if additional_rules:
                rule_union.update(additional_rules)
                log_message(f"🗣️ [ENGINE] Updated rule_union with {len(additional_rules)} system prompt rules from {engine.value} interface", "DEBUG", config)
        
        packed_msgs = processed_messages
        compression_tokens_saved = max(0, original_total_tokens - compressed_total_tokens)
        
        log_message(f"🗜️ [ENGINE] Compression complete: {len(rule_union)} total rules applied", config=config)
        log_message(f"✅ [ENGINE] Applied {detected_engine.value} system prompt processing - ready for LLM", "INFO", config)
    else:
        # No compression, but still process through Interface Engine for system prompts
        log_message(f"🔄 [ENGINE] Processing {detected_engine.value} request without compression", "INFO", config)
        
        # Convert messages and process system prompts even without compression
        messages_dict = []
        for msg in request.messages:
            msg_dict = msg.dict(exclude_none=True) if hasattr(msg, 'dict') else {
                'role': msg.role,
                'content': msg.content or '',
            }
            if msg_dict.get("content") is None:
                msg_dict["content"] = ""
            messages_dict.append(msg_dict)
        
        # Still use Interface Engine for system prompt processing
        processed_messages, system_metadata, engine = detect_and_process_compression(
            request=http_request,
            messages=messages_dict,
            rule_union=rule_union,  # Empty since no compression
            config=config,
            model_id=request.model,
            target_format=getattr(config, 'system_prompt_format', 'chatml'),
            request_data={"model": request.model, "messages": messages_dict}
        )
        
        packed_msgs = processed_messages
        compression_tokens_saved = 0
        
        log_message(f"✅ [ENGINE] Applied {detected_engine.value} system prompt processing without compression", "INFO", config)
    
    # Calculate compression metrics including token savings
    original_content_size = sum(len(msg.content or "") for msg in request.messages if msg.role in {"user", "assistant", "system"})
    compressed_content_size = sum(len(msg.get("content", "")) for msg in packed_msgs if msg.get("role") in {"user", "assistant", "system"})
    compression_ratio = (original_content_size - compressed_content_size) / original_content_size if original_content_size > 0 else 0
    dynamic_chars_used = sum(len(value) for value in rule_union.values())
    
    log_message(f"📊 [ENGINE] Final metrics: {compression_tokens_saved} tokens saved, {compression_ratio*100:.1f}% compression", "DEBUG", config)
    
    # Prepare payload for the LLM API
    payload = {
        "model": request.model,
        "messages": packed_msgs,
        "stream": request.stream,
    }
    
    # Add optional parameters
    for field in ["temperature", "top_p", "n", "max_tokens", "presence_penalty", "frequency_penalty", "user"]:
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value
    
    # Apply timeout parameter filtering
    payload = filter_timeout_parameters(payload, config.filter_timeout_parameters)
    
    # Forward request to target LLM API
    llm_start_time = time.time()
    target_url = f"{config.target_url}/chat/completions"
    
    # Handle streaming vs non-streaming
    if request.stream:
        # For streaming, use the existing streaming handler
        return await handle_streaming_response_with_metrics(
            target_url, payload, rule_union, start_time, llm_start_time, 
            time.time() - start_time, compression_ratio, compression_tokens_saved, 
            dynamic_chars_used,
            original_messages=[msg.dict() for msg in request.messages],
            compressed_messages=packed_msgs,
            comment_stats=None,
            should_compress=should_compress,
            config=config
        )
    else:
        # Handle non-streaming response
        try:
            async with create_aiohttp_session() as session:
                headers = {"Content-Type": "application/json; charset=utf-8"}
                
                async with session.post(target_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log_message(f"🚨 [ERROR] Target API returned {resp.status}: {error_text}", config=config)
                        raise HTTPException(status_code=resp.status, detail=error_text)
                    
                    # Handle normal response
                    data = await resp.json()
                    llm_response_received_time = time.time()
                    
                    # Decompress response
                    decompressed_data = decompress_response(data, rule_union)
                    decompression_complete_time = time.time()
                    
                    # Calculate metrics
                    total_time = decompression_complete_time - start_time
                    llm_time = llm_response_received_time - llm_start_time
                    
                    # Extract token usage from response
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    # Log performance metrics
                    log_performance_metrics(
                        "chat/completions",
                        total_time,
                        time.time() - start_time,
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
                    
                    # Log completion message with correct token savings
                    timing_breakdown = {
                        'ingest_to_prompt': llm_start_time - start_time,
                        'prompt_to_llm_send': 0.01,  # Minimal for non-streaming
                        'llm_processing': llm_time,
                        'llm_response_to_decompress': decompression_complete_time - llm_response_received_time,
                        'total_time': total_time
                    }
                    
                    log_completion_message(
                        "chat/completions",
                        compression_tokens_saved,
                        dynamic_chars_used,
                        rule_union,
                        original_content_size,
                        "generic",
                        None,  # comment_stats would need to be extracted from packed results
                        timing_breakdown,
                        config
                    )
                    
                    return decompressed_data
                    
        except aiohttp.ClientError as e:
            log_message(f"🚨 [ERROR] Network error when sending to target LLM: {e}", config=config)
            raise HTTPException(status_code=500, detail=f"Failed to connect to target LLM API: {str(e)}")
        except Exception as e:
            log_message(f"🚨 [ERROR] Unexpected error when sending to target LLM: {e}", config=config)
            raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}") 