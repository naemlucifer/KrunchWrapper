"""
Logging utilities for KrunchWrapper server.
Contains all logging functions extracted from server.py for better organization.

All functions use the async logging system exclusively:
- No print() fallbacks - everything goes through Python's logging system
- Benefits from setup_global_async_logging() for non-blocking performance
- Uses background worker threads with batching for optimal performance
- Maintains compatibility with existing file and console handlers
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Buffer for early configuration messages that occur before file logging is set up
_config_message_buffer = []

def _extract_debug_category_from_message(message: str) -> str:
    """
    Extract debug category from message content based on common patterns.
    
    Returns the appropriate debug category name for the message content.
    """
    # Convert to uppercase for consistent matching
    msg_upper = message.upper()
    
    # Check most specific patterns first to avoid mismatches
    
    # System Operations & Infrastructure - check specific tags first
    if "[FIX]" in msg_upper:
        return "system_fixes"
    if "[TIMEOUT FILTER]" in msg_upper:
        return "request_filtering"
    if "[CLEANUP]" in msg_upper:
        return "cleanup_operations"
    if "[ERROR]" in msg_upper:
        return "error_handling"
    
    # Request Processing & Client Detection
    if "[ALL REQUESTS]" in msg_upper or "[DEBUG]" in msg_upper:
        return "request_processing"
    if "[CLINE]" in msg_upper:
        return "cline_integration"
    if "[SESSION]" in msg_upper:
        return "session_management"
    
    # Response Processing & Decompression - check specific patterns first
    if any(term in msg_upper for term in ["[STREAMING DEBUG]", "[PROXY STREAMING DEBUG]"]):
        return "streaming_responses"
    if any(term in msg_upper for term in ["[NON-STREAMING DEBUG]", "[PROXY DECOMPRESSION DEBUG]"]):
        return "response_decompression"
    if "TOKEN CALCULATION" in msg_upper:
        return "token_calculations"
        
    # Compression Operations & Analysis - check specific patterns first
    if "[CONVERSATION DETECTION]" in msg_upper:
        return "conversation_detection"
    if any(term in msg_upper for term in ["[KV DEBUG]", "[KV CACHE]"]):
        return "kv_cache_optimization"
    if "[CONVERSATION COMPRESS]" in msg_upper:
        return "conversation_compression"
    if "[PROXY]" in msg_upper:
        return "compression_proxy"
        
    # Server Communication & Forwarding - check model context first
    if "SET MODEL CONTEXT" in msg_upper:
        return "model_context"
    if any(term in msg_upper for term in ["SERVER", "FORWARDED", "TARGET", "API"]):
        return "server_communication"
        
    # Development & Testing
    if "[PAYLOAD DEBUG]" in msg_upper:
        return "payload_debugging"
    if any(term in msg_upper for term in ["TEST", "DEMO", "EXPERIMENT"]):
        return "test_utilities"
    
    # More general patterns (checked after specific ones)
    if any(term in msg_upper for term in ["OVERRIDE", "TIMEOUT"]):
        return "system_fixes"  # Timeout overrides are system fixes
    if any(term in msg_upper for term in ["COMPRESSION", "DYNAMIC DICTIONARY", "ANALYSIS"]):
        return "compression_core"
    if any(term in msg_upper for term in ["SYMBOL", "MODEL CONTEXT"]):
        return "symbol_management"
        
    # Default to request_processing for unmatched debug messages
    return "request_processing"

def log_message(message: str, level: str = "INFO", config=None, debug_category: str = None):
    """
    Log message using optimized async logging system with debug category filtering.
    
    Args:
        message: The message to log
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        config: Server configuration object
        debug_category: Debug category for filtering (only applies to DEBUG level messages)
    """
    # Apply debug category filtering for DEBUG level messages
    if level.upper() == "DEBUG" and config:
        # If no debug category is explicitly provided, try to extract it from the message
        if not debug_category:
            debug_category = _extract_debug_category_from_message(message)
        
        # Check if this debug category is enabled
        if hasattr(config, 'is_debug_category_enabled'):
            if not config.is_debug_category_enabled(debug_category):
                return  # Skip this debug message - category is disabled
    
    # Try to use async logging first for performance benefits
    try:
        from core.async_logger import get_optimized_logger
        async_logger = get_optimized_logger()
        if async_logger and async_logger.enable_verbose:
            # Use async logging for better performance - non-blocking
            async_logger.log_phase("APP_LOG", message, level=level.upper())
            return
    except Exception:
        pass
    
    # Fallback to standard logging (still benefits from async handler if enabled)
    if config and hasattr(config, 'logger'):
        log_level = getattr(logging, level.upper(), logging.INFO)
        config.logger.log(log_level, message)
        return
    
    # Final fallback to standard logger
    logger = logging.getLogger('krunchwrapper')
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)

def log_config_message(message: str, level: str = "INFO"):
    """Log configuration message during server setup using async logging."""
    # Check if file logging is set up by looking for FileHandler in root logger
    root_logger = logging.getLogger()
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    
    if has_file_handler:
        # File logging is ready, log directly (async logging system handles both console and file output)
        logger = logging.getLogger('server.config')
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message)
    else:
        # File logging not ready yet, buffer the message but still log it async
        _config_message_buffer.append((message, level))
        # Use async logging even during early startup
        logger = logging.getLogger('server.config')
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, f"[EARLY] {message}")

def flush_config_message_buffer():
    """Flush buffered configuration messages to the logging system."""
    global _config_message_buffer
    if _config_message_buffer:
        logger = logging.getLogger('server.config')
        for message, level in _config_message_buffer:
            try:
                log_level = getattr(logging, level.upper(), logging.INFO)
                logger.log(log_level, f"[EARLY] {message}")
            except Exception:
                pass
        _config_message_buffer.clear()

def log_performance_metrics(
    endpoint_type: str,
    total_time: float,
    preprocessing_time: float,
    llm_time: float,
    compression_ratio: float,
    compression_tokens_saved: int,
    dynamic_chars_used: int,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    first_token_time: float = None,
    llm_start_time: float = None,
    config=None
):
    """Log performance metrics in a consistent format across all endpoints."""
    # Calculate rates
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    prompt_tokens_per_sec = prompt_tokens / preprocessing_time if preprocessing_time > 0 else 0
    
    # Handle different endpoint types for generation rate
    if endpoint_type.lower() == "embeddings":
        gen_tokens_per_sec_str = "N/A (embeddings)"
    elif endpoint_type.lower() == "placeholder":
        gen_tokens_per_sec_str = "N/A (placeholder)"
        avg_tokens_per_sec = 0  # Override for placeholder
        prompt_tokens_per_sec = 0  # Override for placeholder
    else:
        gen_tokens_per_sec = completion_tokens / llm_time if llm_time > 0 else 0
        gen_tokens_per_sec_str = f"{gen_tokens_per_sec:.1f}"
    
    # Handle output tokens display
    if endpoint_type.lower() == "embeddings" or endpoint_type.lower() == "placeholder":
        output_tokens_str = "0 (embeddings)" if "embeddings" in endpoint_type.lower() else "0 (placeholder)"
    else:
        output_tokens_str = str(completion_tokens)
    
    # Handle total context display for placeholders
    if endpoint_type.lower() == "placeholder":
        total_context_str = "0 (placeholder)"
        input_tokens_str = "0 (placeholder)"
    else:
        total_context_str = str(total_tokens)
        input_tokens_str = str(prompt_tokens)
    
    # Handle timing display
    if endpoint_type.lower() == "placeholder":
        llm_time_str = "N/A"
    else:
        llm_time_str = f"{llm_time:.2f}s"
    
    # Log performance metrics
    log_message(f"📊 Performance Metrics ({endpoint_type}):", config=config)
    log_message(f"   t/s (avg): {avg_tokens_per_sec:.1f}", config=config)
    log_message(f"   pp t/s: {prompt_tokens_per_sec:.1f}", config=config)
    log_message(f"   gen t/s: {gen_tokens_per_sec_str}", config=config)
    log_message(f"   compression %: {compression_ratio*100:.1f}%", config=config)
    log_message(f"   compression tokens used: {compression_tokens_saved}", config=config)
    log_message(f"   dynamic chars used: {dynamic_chars_used}", config=config)
    log_message(f"   total context used: {total_context_str}", config=config)
    log_message(f"   input tokens: {input_tokens_str}", config=config)
    log_message(f"   output tokens: {output_tokens_str}", config=config)
    log_message(f"   total time: {total_time:.2f}s (prep: {preprocessing_time:.2f}s, llm: {llm_time_str})", config=config)
    
    # Add first token time for streaming
    if first_token_time is not None and llm_start_time is not None:
        log_message(f"   time to first token: {first_token_time - llm_start_time:.2f}s", config=config)

def log_verbose_content(
    endpoint_type: str,
    original_content: str = None,
    compressed_content: str = None,
    llm_response: str = None,
    original_messages: list = None,
    compressed_messages: list = None,
    response_data: dict = None,
    config=None
):
    """Log verbose content showing before compression, after compression, and LLM response."""
    if not config or not config.verbose_logging:
        return
    
    log_message(f"\n🔍 Verbose Logging ({endpoint_type}):", config=config)
    log_message("=" * 80, config=config)
    
    # Handle different content types
    if original_messages and compressed_messages:
        # For chat completions - show message differences
        log_message("📝 ORIGINAL MESSAGES:", config=config)
        for i, msg in enumerate(original_messages):
            content = msg.get("content", "")
            log_message(f"   [{msg.get('role', 'unknown')}] {content}", config=config)
        
        log_message("\n🗜️  COMPRESSED MESSAGES:", config=config)
        for i, msg in enumerate(compressed_messages):
            content = msg.get("content", "")
            log_message(f"   [{msg.get('role', 'unknown')}] {content}", config=config)
    
    elif original_content and compressed_content:
        # For text completions - show prompt differences
        log_message(f"📝 ORIGINAL PROMPT:", config=config)
        log_message(f"   {original_content}", config=config)
        
        log_message(f"\n🗜️  COMPRESSED PROMPT:", config=config)
        log_message(f"   {compressed_content}", config=config)
    
    # Show LLM response
    if llm_response:
        log_message(f"\n🤖 LLM RESPONSE:", config=config)
        log_message(f"   {llm_response}", config=config)
    elif response_data:
        # Extract response from different response formats
        if "choices" in response_data:
            for choice in response_data["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    log_message(f"\n🤖 LLM RESPONSE:", config=config)
                    log_message(f"   {content}", config=config)
                    break
                elif "text" in choice:
                    content = choice["text"]
                    log_message(f"\n🤖 LLM RESPONSE:", config=config)
                    log_message(f"   {content}", config=config)
                    break
    
    log_message("=" * 80 + "\n", config=config)

def log_passthrough_request(
    endpoint_type: str,
    reason: str,
    messages: list = None,
    content: str = None,
    content_size: int = 0,
    compression_ratio: float = 0.0,
    min_characters: int = 0,
    min_compression_ratio: float = 0.0,
    config=None
):
    """Log detailed information about requests that are passed through without compression."""
    if not config or not config.show_passthrough_requests:
        return
    
    # Use async logging via Python logger instead of print() 
    import logging
    logger = logging.getLogger('krunchwrapper')
    
    # Build complete log message for async processing
    lines = [
        f"\n📤 Passthrough Request ({endpoint_type}):",
        "=" * 80,
        f"🚫 Reason: {reason}"
    ]
    
    # Show size information
    if content_size > 0:
        lines.append(f"📏 Content size: {content_size:,} characters")
        if min_characters > 0:
            lines.append(f"📐 Min characters threshold: {min_characters:,}")
    
    # Show compression ratio information
    if compression_ratio > 0:
        lines.append(f"🗜️  Compression ratio achieved: {compression_ratio*100:.2f}%")
        lines.append(f"📊 Min compression ratio threshold: {min_compression_ratio*100:.2f}%")
    
    # Show message content
    if messages:
        lines.append("📝 REQUEST MESSAGES:")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # Truncate long content for readability
            if len(content) > 300:
                display_content = content[:300] + "... [truncated]"
            else:
                display_content = content
            lines.append(f"   [{role}] {display_content}")
    elif content:
        lines.append("📝 REQUEST CONTENT:")
        # Truncate long content for readability
        if len(content) > 300:
            display_content = content[:300] + "... [truncated]"
        else:
            display_content = content
        lines.append(f"   {display_content}")
    
    lines.append("=" * 80 + "\n")
    
    # Send complete message to async logger (benefits from setup_global_async_logging)
    complete_message = "\n".join(lines)
    logger.info(complete_message)

def log_passthrough_response(
    endpoint_type: str,
    response_data: dict = None,
    response_text: str = None,
    processing_time: float = 0.0,
    config=None
):
    """Log the response for requests that were passed through without compression."""
    if not config or not config.show_passthrough_requests:
        return
    
    # Use async logging via Python logger
    import logging
    logger = logging.getLogger('krunchwrapper')
    
    lines = [
        f"\n📥 Passthrough Response ({endpoint_type}):",
        "=" * 80
    ]
    
    if processing_time > 0:
        lines.append(f"⏱️  Processing time: {processing_time:.3f}s")
    
    # Extract and show response content
    if response_data:
        # Handle different response formats
        if "choices" in response_data:
            lines.append("🤖 RESPONSE CONTENT:")
            for i, choice in enumerate(response_data["choices"]):
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    # Truncate long responses for readability
                    if len(content) > 500:
                        display_content = content[:500] + "... [truncated]"
                    else:
                        display_content = content
                    lines.append(f"   [choice {i}] {display_content}")
                elif "text" in choice:
                    content = choice["text"]
                    # Truncate long responses for readability
                    if len(content) > 500:
                        display_content = content[:500] + "... [truncated]"
                    else:
                        display_content = content
                    lines.append(f"   [choice {i}] {display_content}")
        
        # Show token usage if available
        if "usage" in response_data:
            usage = response_data["usage"]
            lines.append("📊 TOKEN USAGE:")
            lines.append(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            lines.append(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            lines.append(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
    
    elif response_text:
        lines.append("🤖 RESPONSE CONTENT:")
        # Truncate long responses for readability
        if len(response_text) > 500:
            display_content = response_text[:500] + "... [truncated]"
        else:
            display_content = response_text
        lines.append(f"   {display_content}")
    
    lines.append("=" * 80 + "\n")
    
    # Send complete message to async logger
    complete_message = "\n".join(lines)
    logger.info(complete_message)

def log_proxy_request_content(
    endpoint_type: str,
    request_data: dict = None,
    request_body: str = None,
    content_size: int = 0,
    config=None
):
    """Log information about proxy requests - abbreviated for INFO level, detailed for DEBUG level."""
    # Always log proxy requests when verbose logging is enabled
    if not config or not (config.verbose_logging or config.show_passthrough_requests):
        return
    
    # Use async logging via Python logger
    import logging
    logger = logging.getLogger('krunchwrapper')
    
    # Check current log levels to determine what content to generate
    console_log_level = getattr(config, 'log_level', 'INFO').upper()
    file_log_level = getattr(config, 'file_log_level', console_log_level).upper()
    
    console_needs_details = console_log_level == 'DEBUG'
    file_needs_details = file_log_level == 'DEBUG'
    
    # Always send abbreviated content at INFO level (for console when console_log_level is INFO)
    # Send detailed content at DEBUG level (for file when file_log_level is DEBUG)
    
    # Generate abbreviated content for INFO level
    abbreviated_lines = [f"🔄 Proxy Request ({endpoint_type}):"]
    abbreviated_lines.append("=" * 80)
    
    # Show size information
    if content_size > 0:
        abbreviated_lines.append(f"📏 Content size: {content_size:,} characters")
    
    # Show abbreviated request content
    if request_data:
        abbreviated_lines.append("📝 REQUEST DATA:")
        
        # Show model
        if "model" in request_data:
            abbreviated_lines.append(f"   🤖 Model: {request_data['model']}")
        
        # Show message count and roles only
        if "messages" in request_data:
            message_count = len(request_data["messages"])
            roles = [msg.get('role', 'unknown') for msg in request_data["messages"]]
            role_summary = ", ".join(roles)
            abbreviated_lines.append(f"   💬 Messages: {message_count} messages ({role_summary})")
            
            # Show first 200 chars of each message as preview
            for i, msg in enumerate(request_data["messages"]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str):
                    preview = content[:200].replace('\n', ' ').strip()
                    if len(content) > 200:
                        preview += "..."
                    abbreviated_lines.append(f"      [{role}] {preview} [total: {len(content)} chars]")
                elif isinstance(content, list):
                    # Handle multi-part content (like Cline's structured messages)
                    total_chars = sum(len(str(part.get('text', ''))) if isinstance(part, dict) else len(str(part)) for part in content)
                    abbreviated_lines.append(f"      [{role}] [structured content with {len(content)} parts, total: {total_chars} chars]")
                else:
                    abbreviated_lines.append(f"      [{role}] [{type(content).__name__} content]")
        
        # Show other parameters (abbreviated)
        other_params = {k: v for k, v in request_data.items() if k not in ['messages', 'model']}
        if other_params:
            param_summary = ", ".join(f"{k}: {v}" for k, v in list(other_params.items())[:3])
            if len(other_params) > 3:
                param_summary += f" ... (+{len(other_params)-3} more)"
            abbreviated_lines.append(f"   ⚙️  Parameters: {param_summary}")
    
    elif request_body:
        abbreviated_lines.append("📝 REQUEST BODY:")
        # Show just first 300 chars for INFO level
        preview = request_body[:300].replace('\n', ' ').strip()
        if len(request_body) > 300:
            preview += "..."
        abbreviated_lines.append(f"   {preview} [total: {len(request_body)} chars]")
    
    abbreviated_lines.append("=" * 80 + "\n")
    
    # Send abbreviated content at INFO level
    abbreviated_message = "\n".join(abbreviated_lines)
    logger.info(abbreviated_message)
    
    # If file needs detailed content, also send detailed content at DEBUG level
    if file_needs_details:
        # DEBUG level: Show full detailed content (original behavior)
        lines = [
            f"\n🔄 Proxy Request ({endpoint_type}):",
            "=" * 80
        ]
        
        # Show size information
        if content_size > 0:
            lines.append(f"📏 Content size: {content_size:,} characters")
        
        # Show request content
        if request_data:
            lines.append("📝 REQUEST DATA:")
            
            # Show model
            if "model" in request_data:
                lines.append(f"   🤖 Model: {request_data['model']}")
            
            # Show messages
            if "messages" in request_data:
                lines.append("   💬 Messages:")
                for i, msg in enumerate(request_data["messages"]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    # Show full content for short messages, truncate long ones
                    if len(content) > 1000:
                        display_content = content[:1000] + f"... [truncated, total: {len(content)} chars]"
                    else:
                        display_content = content
                    lines.append(f"      [{role}] {display_content}")
            
            # Show other parameters
            other_params = {k: v for k, v in request_data.items() if k not in ['messages', 'model']}
            if other_params:
                lines.append("   ⚙️  Parameters:")
                for key, value in other_params.items():
                    lines.append(f"      {key}: {value}")
        
        elif request_body:
            lines.append("📝 REQUEST BODY:")
            # Show truncated body for very long requests
            if len(request_body) > 2000:
                display_content = request_body[:2000] + f"... [truncated, total: {len(request_body)} chars]"
            else:
                display_content = request_body
            lines.append(f"   {display_content}")
        
        lines.append("=" * 80 + "\n")
        
        # Send detailed content at DEBUG level (will only go to file if file_log_level is DEBUG)
        detailed_message = "\n".join(lines)
        logger.debug(detailed_message)

def log_proxy_response_content(
    endpoint_type: str,
    response_data: dict = None,
    response_body: str = None,
    processing_time: float = 0.0,
    status_code: int = 200,
    config=None
):
    """Log information about proxy responses - abbreviated for INFO level, detailed for DEBUG level."""
    # Always log proxy responses when verbose logging is enabled
    if not config or not (config.verbose_logging or config.show_passthrough_requests):
        return
    
    # Use async logging via Python logger
    import logging
    logger = logging.getLogger('krunchwrapper')
    
    # Check current log levels to determine what content to generate
    console_log_level = getattr(config, 'log_level', 'INFO').upper()
    file_log_level = getattr(config, 'file_log_level', console_log_level).upper()
    
    console_needs_details = console_log_level == 'DEBUG'
    file_needs_details = file_log_level == 'DEBUG'
    
    # Generate abbreviated content for INFO level
    abbreviated_lines = [f"📥 Proxy Response ({endpoint_type}):"]
    abbreviated_lines.append("=" * 80)
    abbreviated_lines.append(f"🌐 Status: {status_code}")
    
    if processing_time > 0:
        abbreviated_lines.append(f"⏱️  Processing time: {processing_time:.3f}s")
    
    # Show abbreviated response content
    if response_data:
        # Handle different response formats
        if "choices" in response_data:
            abbreviated_lines.append("🤖 RESPONSE CONTENT:")
            choice_count = len(response_data["choices"])
            abbreviated_lines.append(f"   Choices: {choice_count}")
            
            # Show preview of first choice only
            if choice_count > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    preview = content[:200].replace('\n', ' ').strip() if content else ""
                    if len(content or "") > 200:
                        preview += "..."
                    abbreviated_lines.append(f"   [choice 0] {preview} [total: {len(content or '')} chars]")
                elif "text" in choice:
                    content = choice["text"]
                    preview = content[:200].replace('\n', ' ').strip() if content else ""
                    if len(content or "") > 200:
                        preview += "..."
                    abbreviated_lines.append(f"   [choice 0] {preview} [total: {len(content or '')} chars]")
        
        # Show token usage summary if available
        if "usage" in response_data:
            usage = response_data["usage"]
            abbreviated_lines.append("📊 TOKEN USAGE:")
            abbreviated_lines.append(f"   Total: {usage.get('total_tokens', 'N/A')} ({usage.get('prompt_tokens', 'N/A')} + {usage.get('completion_tokens', 'N/A')})")
    
    elif response_body:
        abbreviated_lines.append("🤖 RESPONSE BODY:")
        preview = response_body[:200].replace('\n', ' ').strip()
        if len(response_body) > 200:
            preview += "..."
        abbreviated_lines.append(f"   {preview} [total: {len(response_body)} chars]")
    
    abbreviated_lines.append("=" * 80 + "\n")
    
    # Send abbreviated content at INFO level
    abbreviated_message = "\n".join(abbreviated_lines)
    logger.info(abbreviated_message)
    
    # If file needs detailed content, also send detailed content at DEBUG level
    if file_needs_details:
        # DEBUG level: Show full detailed content (original behavior)
        lines = [
            f"\n📥 Proxy Response ({endpoint_type}):",
            "=" * 80,
            f"🌐 Status: {status_code}"
        ]
        
        if processing_time > 0:
            lines.append(f"⏱️  Processing time: {processing_time:.3f}s")
        
        # Extract and show response content
        if response_data:
            # Handle different response formats
            if "choices" in response_data:
                lines.append("🤖 RESPONSE CONTENT:")
                for i, choice in enumerate(response_data["choices"]):
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        # Show full content for short responses, truncate long ones
                        if len(content) > 1000:
                            display_content = content[:1000] + f"... [truncated, total: {len(content)} chars]"
                        else:
                            display_content = content
                        lines.append(f"   [choice {i}] {display_content}")
                    elif "text" in choice:
                        content = choice["text"]
                        # Show full content for short responses, truncate long ones
                        if len(content) > 1000:
                            display_content = content[:1000] + f"... [truncated, total: {len(content)} chars]"
                        else:
                            display_content = content
                        lines.append(f"   [choice {i}] {display_content}")
            
            # Show token usage if available
            if "usage" in response_data:
                usage = response_data["usage"]
                lines.append("📊 TOKEN USAGE:")
                lines.append(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                lines.append(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                lines.append(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        elif response_body:
            lines.append("🤖 RESPONSE BODY:")
            # Show truncated body for very long responses
            if len(response_body) > 2000:
                display_content = response_body[:2000] + f"... [truncated, total: {len(response_body)} chars]"
            else:
                display_content = response_body
            lines.append(f"   {display_content}")
        
        lines.append("=" * 80 + "\n")
        
        # Send detailed content at DEBUG level (will only go to file if file_log_level is DEBUG)
        detailed_message = "\n".join(lines)
        logger.debug(detailed_message)

def log_completion_message(
    endpoint_type: str,
    compression_tokens_saved: int,
    dynamic_chars_used: int,
    rule_union: dict,
    original_content_size: int = 0,
    lang: str = "generic",
    comment_stats: dict = None,
    timing_breakdown: dict = None,
    config=None
):
    """
    Log the final completion message showing total compression % and context saved.
    This accounts for compression savings offset by system prompt overhead.
    Always displayed regardless of verbose settings.
    
    Args:
        endpoint_type: Type of endpoint (chat/completions, completions, etc.)
        compression_tokens_saved: Raw tokens saved through compression
        dynamic_chars_used: Characters used in dynamic compression
        rule_union: Dictionary of substitutions used
        original_content_size: Original content size in characters
        lang: Programming language detected
        comment_stats: Comment stripping statistics
        timing_breakdown: Dictionary with detailed timing metrics
    """
    # Calculate system prompt overhead
    system_prompt_overhead_tokens = 0
    
    # Always calculate overhead if there's a rule_union or if we might have added system prompts
    # The system prompt interceptor can add prompts even with empty rule_union (e.g., KV cache scenarios)
    if rule_union or compression_tokens_saved > 0:
        # Use the actual overhead calculation from conversation_compress.py for accuracy
        try:
            from core.conversation_compress import _estimate_system_prompt_overhead
            system_prompt_overhead_tokens = _estimate_system_prompt_overhead(rule_union)
            log_message(f"📊 Using accurate system prompt overhead calculation: {system_prompt_overhead_tokens} tokens", "DEBUG", config=config)
        except ImportError:
            # Fallback to legacy calculation if import fails
            if rule_union and len(rule_union) > 0:
                # Full compression system prompt (legacy fallback)
                if lang and lang != "generic":
                    pairs = ", ".join(f"{k}={v}" for k, v in list(rule_union.items())[:5])  # Sample first 5 for estimation
                    sample_prompt = (
                        f"You will read {lang} code in a compressed DSL. "
                        f"Apply these symbol substitutions when understanding and responding: {pairs}. "
                        f"This reduces token usage."
                    )
                else:
                    sample_prompt = "You will read code. Reason about it as-is."
                
                # Estimate tokens in system prompt (rough estimation: chars/4)
                # Add overhead for the complete rule_union (not just sample)
                base_prompt_tokens = len(sample_prompt) // 4
                
                # Add token cost for each substitution pair beyond the sample
                if len(rule_union) > 5:
                    additional_pairs = len(rule_union) - 5
                    # Each additional pair costs roughly: "key=value, " = ~3-8 tokens depending on length
                    avg_pair_length = sum(len(k) + len(v) + 3 for k, v in rule_union.items()) // len(rule_union) if rule_union else 0
                    additional_tokens = additional_pairs * (avg_pair_length // 4)
                    system_prompt_overhead_tokens = base_prompt_tokens + additional_tokens
                else:
                    system_prompt_overhead_tokens = base_prompt_tokens
                log_message(f"📊 Using legacy system prompt overhead calculation: {system_prompt_overhead_tokens} tokens", "DEBUG", config=config)
            else:
                # KV cache or minimal system prompt scenario
                # Even without rule_union, system prompt interceptor may add basic instructions
                if compression_tokens_saved > 0:
                    # Estimate minimal system prompt for decompression context
                    minimal_prompt = "You will read code. Reason about it as-is."
                    system_prompt_overhead_tokens = len(minimal_prompt) // 4
                else:
                    system_prompt_overhead_tokens = 0
    
    # Calculate net savings
    net_tokens_saved = compression_tokens_saved - system_prompt_overhead_tokens
    
    # Calculate total compression percentage accounting for system prompt overhead
    if original_content_size > 0:
        # Net compression % = (tokens saved - system overhead) / original tokens
        original_tokens_estimate = original_content_size // 4
        if original_tokens_estimate > 0:
            total_compression_percentage = (net_tokens_saved / original_tokens_estimate) * 100
        else:
            total_compression_percentage = 0.0
    else:
        total_compression_percentage = 0.0
    
    # Ensure we don't show negative percentages for display (though net can be negative)
    display_compression_percentage = max(0.0, total_compression_percentage)
    
    # Always log completion message (not dependent on verbose setting)
    log_message("="*60, config=config)
    log_message("🏁 REQUEST COMPLETION SUCCESS", config=config)
    log_message("="*60, config=config)
    
    # Include comment stripping statistics if available
    if comment_stats and comment_stats.get("enabled", False):
        comment_chars_saved = comment_stats.get("chars_saved", 0)
        comment_tokens_saved = comment_stats.get("tokens_saved", 0)
        comment_language = comment_stats.get("language", "unknown")
        if comment_chars_saved > 0:
            log_message(f"📝 Comment Stripping: {comment_chars_saved:,} chars, {comment_tokens_saved} tokens saved ({comment_language})", config=config)
    
    log_message(f"📊 Total Compression: {display_compression_percentage:.1f}% (net after system prompt overhead)", config=config)
    log_message(f"💾 Total Context Saved: {net_tokens_saved} tokens (net: {compression_tokens_saved} saved - {system_prompt_overhead_tokens} overhead)", config=config)
    
    # Additional details
    if rule_union:
        log_message(f"🗂️  Dynamic Entries Used: {len(rule_union)}", config=config)
        log_message(f"📝 Dynamic Characters: {dynamic_chars_used}", config=config)
    
    if net_tokens_saved < 0:
        log_message("⚠️  Note: System prompt overhead exceeded compression savings for this request", config=config)
    
    # Add cache metrics from optimized validator
    try:
        from core.optimized_model_validator import get_optimized_validator
        validator = get_optimized_validator()
        cache_stats = validator.get_cache_stats()
        
        if cache_stats['cache_hits'] > 0 or cache_stats['cache_misses'] > 0:
            cache_mode = "model-specific" if cache_stats.get('model_specific_cache', True) else "model-agnostic"
            log_message(f"🚀 Cache Performance: {cache_stats['hit_rate']:.1%} hit rate ({cache_mode} mode)", config=config)
            log_message(f"📈 Cache Stats: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses, {cache_stats['cached_validations']} entries", config=config)
            
            if cache_stats.get('avg_validation_time', 0) > 0:
                log_message(f"⚡ Avg Validation Time: {cache_stats['avg_validation_time']*1000:.2f}ms", config=config)
    except Exception as e:
        # Silently ignore cache stats errors to avoid breaking the completion flow
        log_message(f"⚠️  Cache stats unavailable: {str(e)}", "DEBUG", config=config)
    
    # Add detailed timing breakdown if available
    if timing_breakdown:
        log_message("⏱️  Timing Breakdown:", config=config)
        
        # Phase timings
        ingest_to_prompt = timing_breakdown.get('ingest_to_prompt', 0)
        prompt_to_llm_send = timing_breakdown.get('prompt_to_llm_send', 0)
        llm_processing = timing_breakdown.get('llm_processing', 0)
        llm_response_to_decompress = timing_breakdown.get('llm_response_to_decompress', 0)
        total_time = timing_breakdown.get('total_time', 0)
        
        # Calculate percentages of total time
        if total_time > 0:
            ingest_pct = (ingest_to_prompt / total_time) * 100
            prompt_pct = (prompt_to_llm_send / total_time) * 100
            llm_pct = (llm_processing / total_time) * 100
            decompress_pct = (llm_response_to_decompress / total_time) * 100
        else:
            ingest_pct = prompt_pct = llm_pct = decompress_pct = 0
        
        log_message(f"   📥 Ingest → Prompt Ready: {ingest_to_prompt:.3f}s ({ingest_pct:.1f}%)", config=config)
        log_message(f"   🚀 Prompt → LLM Send: {prompt_to_llm_send:.3f}s ({prompt_pct:.1f}%)", config=config)
        log_message(f"   🤖 LLM Processing: {llm_processing:.3f}s ({llm_pct:.1f}%)", config=config)
        log_message(f"   🔄 Response → Decompressed: {llm_response_to_decompress:.3f}s ({decompress_pct:.1f}%)", config=config)
        log_message(f"   🏁 Total Request Time: {total_time:.3f}s", config=config)
    
    log_message("="*60, config=config)

def log_startup_messages():
    """Log a clean startup banner without duplicating shell script messages."""
    logger = logging.getLogger('server.startup')
    
    # Don't duplicate the shell script messages - they're already printed
    # Instead, log a clean banner to mark the transition from shell to Python logging
    logger.info("=" * 80)
    logger.info("🚀 KrunchWrapper Compression Proxy - Python Server Starting")
    logger.info("=" * 80) 