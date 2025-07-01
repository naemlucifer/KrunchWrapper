"""
Request utility functions for KrunchWrapper server.

This module contains utility functions for timeout filtering, parameter 
normalization, and conversation detection that are used by multiple endpoints.
"""

from typing import Dict, List, Any, Optional
from core.interface_engine import InterfaceEngine

# Import logging functions from the extracted module
from server.logging_utils import log_message


def filter_timeout_parameters(payload: dict, filter_enabled: bool = True) -> dict:
    """
    Filter timeout-related parameters from request payload.
    
    Args:
        payload: The request payload dictionary
        filter_enabled: Whether to filter timeout parameters (from config)
    
    Returns:
        Filtered payload dictionary
    """
    if not filter_enabled:
        return payload
    
    # List of timeout-related parameters to filter
    timeout_params = {
        'timeout', 'request_timeout', 'response_timeout', 'connection_timeout',
        'read_timeout', 'write_timeout', 'stream_timeout', 'api_timeout',
        'client_timeout', 'server_timeout', 'connect_timeout', 'total_timeout'
    }
    
    filtered_payload = {}
    filtered_count = 0
    
    for key, value in payload.items():
        if key.lower() in timeout_params:
            log_message(f"🚫 [TIMEOUT FILTER] Removed timeout parameter: {key}={value}", "DEBUG")
            filtered_count += 1
        else:
            filtered_payload[key] = value
    
    if filtered_count > 0:
        log_message(f"⏱️  Filtered {filtered_count} timeout parameter(s) from request", "INFO")
    
    return filtered_payload


def apply_interface_specific_fixes(payload: dict, detected_engine, target_url: str) -> dict:
    """Apply parameter fixes based on the detected interface engine."""
    payload = payload.copy()
    
    # Apply max_tokens: -1 fix based on interface type and target URL patterns
    if payload.get("max_tokens") == -1:
        # Check for known llama.cpp server patterns
        is_llama_cpp = any(indicator in target_url.lower() for indicator in [
            'llama', 'llamacpp', 'llama.cpp', 'localhost:5001', 'localhost:8080'
        ])
        
        # Apply fix for interfaces known to have issues with max_tokens: -1
        if (detected_engine in [InterfaceEngine.CLINE, InterfaceEngine.WEBUI] and is_llama_cpp):
            # Cline/WebUI + llama.cpp commonly have this issue
            del payload["max_tokens"]
            log_message(f"🔧 [FIX] Converted max_tokens: -1 to unlimited (removed for {detected_engine.value} + llama.cpp)", "DEBUG")
        elif detected_engine == InterfaceEngine.STANDARD and is_llama_cpp:
            # Standard interface with llama.cpp may also have this issue
            del payload["max_tokens"]
            log_message(f"🔧 [FIX] Converted max_tokens: -1 to unlimited (removed for standard interface + llama.cpp)", "DEBUG")
        else:
            # Unknown combination - log but preserve to avoid breaking working setups
            log_message(f"ℹ️ [INFO] Found max_tokens: -1 with {detected_engine.value} interface - preserving original parameter")
    
    # Apply cache_prompt fix for multi-turn conversations
    if "messages" in payload and len(payload["messages"]) > 2:
        if payload.get("cache_prompt", False):
            # Check for llama.cpp server patterns  
            is_llama_cpp = any(indicator in target_url.lower() for indicator in [
                'llama', 'llamacpp', 'llama.cpp', 'localhost:5001', 'localhost:8080'
            ])
            
            if detected_engine == InterfaceEngine.CLINE and is_llama_cpp:
                # Cline + llama.cpp is known to have cache_prompt response loop issues
                payload["cache_prompt"] = False
                log_message("🔧 [FIX] Disabled cache_prompt for multi-turn conversation (Cline + llama.cpp detected)", "DEBUG")
            else:
                # Other combinations - preserve to avoid breaking working setups
                log_message(f"ℹ️ [INFO] Found cache_prompt=true in multi-turn with {detected_engine.value} - preserving original parameter")
    
    return payload


def _is_conversation_continuation(messages: List[Any], session_id: str = None) -> bool:
    """
    Enhanced conversation continuation detection with detailed logging.
    A conversation continuation is detected when:
    1. Multiple user messages (indicating back-and-forth)
    2. Any assistant responses (indicating conversation history)
    3. A session ID is provided (indicating ongoing session)
    
    Args:
        messages: List of chat messages (can be Pydantic models or dicts)
        session_id: Optional session ID for ongoing conversations
        
    Returns:
        bool: True if this appears to be a conversation continuation
    """
    def get_message_role(msg):
        """Safely extract role from message, handling both Pydantic models and dicts."""
        if hasattr(msg, 'role'):
            return msg.role
        elif hasattr(msg, 'get'):
            return msg.get('role', '')
        else:
            return ''
    
    user_count = sum(1 for msg in messages if get_message_role(msg) == "user")
    assistant_count = sum(1 for msg in messages if get_message_role(msg) == "assistant") 
    system_count = sum(1 for msg in messages if get_message_role(msg) == "system")
    total_messages = len(messages)
    
    # Enhanced detection logic
    is_continuation = (
        user_count > 1 or           # Multiple user messages = back-and-forth
        assistant_count > 0 or      # Any assistant response = conversation history
        (session_id is not None)    # Session ID provided = ongoing conversation
    )
    
    # ENHANCED DEBUG logging for conversation detection
    log_message(f"🔍 [CONVERSATION DETECTION] Message analysis:", "DEBUG")
    log_message(f"    - Total messages: {total_messages}", "INFO")
    log_message(f"    - User messages: {user_count}", "INFO")
    log_message(f"    - Assistant messages: {assistant_count}", "INFO") 
    log_message(f"    - System messages: {system_count}", "INFO")
    log_message(f"    - Session ID provided: {session_id is not None} ({session_id if session_id else 'None'})", "INFO")
    log_message(f"    - Detection result: {'CONTINUATION' if is_continuation else 'NEW CONVERSATION'}", "INFO")
    
    # Log detection reasoning
    if is_continuation:
        reasons = []
        if user_count > 1:
            reasons.append(f"multiple user messages ({user_count})")
        if assistant_count > 0:
            reasons.append(f"assistant responses present ({assistant_count})")
        if session_id is not None:
            reasons.append(f"session ID provided ({session_id})")
        log_message(f"🔍 [CONVERSATION DETECTION] Detected as continuation because: {', '.join(reasons)}", "DEBUG")
    else:
        log_message(f"🔍 [CONVERSATION DETECTION] Detected as new conversation: single user message, no assistant responses, no session ID", "DEBUG")
    
    return is_continuation 