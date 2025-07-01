"""
Response processing utilities for KrunchWrapper server.

This module contains utilities for processing and decompressing responses
from the target LLM API, specifically for non-streaming responses.
"""

import json
from typing import Dict, List, Any, Optional

from server.logging_utils import log_message
from server.streaming import context_aware_decompress


def decompress_response(data, rule_union, original_messages=None):
    """Decompress the response content for non-streaming responses only."""
    for choice in data.get("choices", []):
        # Handle regular message content (non-streaming) ONLY
        if "message" in choice and "content" in choice["message"]:
            original_content = choice["message"]["content"]
            
            # DEBUG: Log what we're trying to decompress
            if rule_union:
                symbols_found = [symbol for symbol in rule_union.keys() if symbol in original_content]
                if symbols_found:
                    log_message(f"🔍 [NON-STREAMING DEBUG] Found symbols {symbols_found} in response: '{original_content[:100]}...'")
                    log_message(f"🔍 [NON-STREAMING DEBUG] rule_union mappings: {dict(list(rule_union.items())[:5])}")
            
            # Check if this is multimodal content that needs reconstruction
            if original_messages and hasattr(original_messages[0], 'is_multimodal'):
                # Use multimodal-aware decompression for responses if original was multimodal
                # Note: This is mainly for completeness - responses are typically plain text
                decompressed_content = context_aware_decompress(original_content, rule_union)
            else:
                decompressed_content = context_aware_decompress(original_content, rule_union)
            
            if decompressed_content != original_content:
                log_message(f"🔍 [NON-STREAMING DEBUG] Decompressed: '{original_content[:50]}...' → '{decompressed_content[:50]}...'")
            
            choice["message"]["content"] = decompressed_content
            
        # NOTE: Do NOT handle streaming delta content here - streaming responses have smart decompression
        # in the streaming loop that only decompresses when compression symbols are actually present
        # elif "delta" in choice and "content" in choice["delta"]:
        #     choice["delta"]["content"] = context_aware_decompress(choice["delta"]["content"], rule_union)
    return data


def fix_payload_encoding(obj):
    """Fix encoding issues by ensuring all strings are properly encoded."""
    if isinstance(obj, dict):
        return {k: fix_payload_encoding(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_payload_encoding(item) for item in obj]
    elif isinstance(obj, str):
        # Ensure string is properly encoded and can be JSON serialized
        try:
            return obj.encode('utf-8').decode('utf-8')
        except UnicodeError:
            # If still fails, escape problematic characters
            return obj.encode('unicode_escape').decode('ascii')
    else:
        return obj


def clean_compression_artifacts(obj):
    """Recursively clean any compression artifacts from nested structures."""
    if isinstance(obj, dict):
        # Remove any single-character keys that might be compression symbols
        keys_to_remove = [k for k in obj.keys() if len(str(k)) == 1 and ord(str(k)[0]) > 127]
        for key in keys_to_remove:
            log_message(f"🧹 [CLEANUP] Removing compression artifact key '{key}'", "DEBUG")
            obj.pop(key)
        # Recursively clean nested objects
        for value in obj.values():
            clean_compression_artifacts(value)
    elif isinstance(obj, list):
        for item in obj:
            clean_compression_artifacts(item)


def clean_messages_array(data):
    """Clean up messages array by removing invalid fields."""
    if 'messages' in data and isinstance(data['messages'], list):
        for i, msg in enumerate(data['messages']):
            if isinstance(msg, dict):
                # Remove any fields that don't belong in messages
                allowed_message_fields = {'role', 'content', 'name'}
                fields_to_remove = [k for k in msg.keys() if k not in allowed_message_fields]
                for field in fields_to_remove:
                    log_message(f"🧹 [CLEANUP] Removing field '{field}' from message {i}", "DEBUG")
                    msg.pop(field) 