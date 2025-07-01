"""
Anthropic System Prompt Interceptor

This module provides Anthropic-specific system prompt interception and processing for KrunchWrapper.
It handles the Anthropic API's native format where system prompts are sent as a separate 'system' parameter
rather than included in the messages array.

Key Features:
- Handles Anthropic's system parameter format
- Supports Claude's message structure requirements
- Integrates with KrunchWrapper's compression system
- Maintains compatibility with Anthropic SDK patterns
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from .system_prompt_interceptor import SystemPromptInterceptor, _log_verbose_system_prompt_phase
from .system_prompt import build_system_prompt
from .async_logger import get_performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('anthropic_intercept')


class AnthropicSystemPromptInterceptor(SystemPromptInterceptor):
    """
    Anthropic-specific system prompt interceptor that handles the Anthropic API's native format.
    
    The Anthropic API uses a different structure:
    - System prompts are sent as a separate 'system' parameter
    - Messages array contains only user/assistant messages
    - Supports Anthropic-specific features like prompt caching
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        logger.info("🔧 AnthropicSystemPromptInterceptor: Initialized (extends SystemPromptInterceptor)")
        logger.info("🎯 Anthropic API formatting:         Ready")
        
    def intercept_and_process_anthropic(self, 
                                      messages: List[Dict[str, Any]], 
                                      rule_union: Dict[str, str], 
                                      lang: str = "generic",
                                      system_param: Optional[str] = None,
                                      target_format: str = "claude") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for Anthropic-specific system prompt interception and processing.
        
        Handles the Anthropic API's native format where system prompts are separate from messages.
        
        Args:
            messages: List of chat messages (should only contain user/assistant messages for Anthropic)
            rule_union: Compression substitutions that were used
            lang: Programming language detected
            system_param: System parameter from request (Anthropic's system prompt)
            target_format: Target format (should be "claude" for Anthropic)
            
        Returns:
            Tuple of (processed_messages, metadata)
        """
        try:
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"🔧 Starting Anthropic system prompt interception")
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"Messages: {len(messages)}, Compression rules: {len(rule_union)}")
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"System param length: {len(system_param) if system_param else 0} chars")
            
            # Validate that this is a proper Anthropic request structure
            if not self._validate_anthropic_message_structure(messages):
                _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                    "⚠️ Invalid Anthropic message structure, falling back to standard processing")
                return self.intercept_and_process(messages, rule_union, lang, target_format, system_param, None)
            
            # Extract user content for context analysis
            user_content = self._extract_user_content(messages)
            
            # Build compression decoder instructions if we have compression rules
            compression_decoder = ""
            if rule_union:
                # Use configured stateful mode from server config for optimal KV cache behavior
                stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
                compression_decoder, decoder_metadata = build_system_prompt(
                    rule_union, lang, target_format, user_content, cline_mode=False, stateful_mode=stateful_mode, new_symbols_only=None
                )
                
                _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                    f"Built compression decoder: {len(compression_decoder)} chars")
            
            # Create Anthropic-optimized system prompt
            final_system_prompt = self._create_anthropic_system_prompt(
                system_param, compression_decoder, rule_union, lang
            )
            
            # For Anthropic API, we return the messages as-is (without system messages)
            # and put the final system prompt in metadata to be used as the 'system' parameter
            processed_messages = [msg for msg in messages if msg.get("role") != "system"]
            
            metadata = {
                "anthropic_system_prompt": final_system_prompt,
                "original_system_preserved": bool(system_param),
                "compression_decoder_added": bool(rule_union),
                "final_system_prompt_length": len(final_system_prompt),
                "rules_count": len(rule_union),
                "format": target_format,
                "interface": "anthropic"
            }
            
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"✅ Anthropic system prompt interception complete")
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"Result: {len(messages)} → {len(processed_messages)} messages, system prompt: {len(final_system_prompt)} chars")
            
            return processed_messages, metadata
            
        except Exception as e:
            _log_verbose_system_prompt_phase("ANTHROPIC_INTERCEPTION", 
                f"❌ Error in Anthropic system prompt interception: {e}")
            logger.error(f"Error in Anthropic system prompt interception: {e}")
            # Fallback to standard processing
            return self.intercept_and_process(messages, rule_union, lang, target_format, system_param, None)
    
    def _validate_anthropic_message_structure(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate that the message structure is appropriate for Anthropic API.
        
        Anthropic messages should:
        - Only contain user/assistant roles (no system messages)
        - Have alternating user/assistant pattern (mostly)
        - Start with a user message
        
        Args:
            messages: List of messages to validate
            
        Returns:
            bool: True if structure is valid for Anthropic
        """
        if not messages:
            return True  # Empty is valid
        
        # Check for system messages (which shouldn't be in Anthropic message array)
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        if system_messages:
            _log_verbose_system_prompt_phase("ANTHROPIC_VALIDATION", 
                f"Found {len(system_messages)} system messages in message array - not ideal for Anthropic")
            # This is okay, we'll just filter them out, but note the issue
        
        # Check for valid roles
        valid_roles = {"user", "assistant"}
        invalid_messages = [msg for msg in messages if msg.get("role") not in valid_roles]
        if invalid_messages:
            _log_verbose_system_prompt_phase("ANTHROPIC_VALIDATION", 
                f"Found {len(invalid_messages)} messages with invalid roles for Anthropic")
            return False
        
        # Filter out system messages for the rest of the validation
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        if not non_system_messages:
            return True  # All system messages is still valid
        
        # Should typically start with user message
        if non_system_messages[0].get("role") != "user":
            _log_verbose_system_prompt_phase("ANTHROPIC_VALIDATION", 
                "First non-system message is not from user - unusual for Anthropic but allowed")
        
        _log_verbose_system_prompt_phase("ANTHROPIC_VALIDATION", 
            f"✅ Message structure valid for Anthropic: {len(non_system_messages)} user/assistant messages")
        
        return True
    
    def _extract_user_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract user content for context analysis."""
        user_content = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    user_content.append(str(content))
        return "\n\n".join(user_content)
    
    def _create_anthropic_system_prompt(self, 
                                      original_system: Optional[str], 
                                      compression_decoder: str, 
                                      rule_union: Dict[str, str], 
                                      lang: str) -> str:
        """
        Create an Anthropic-optimized system prompt by merging compression instructions
        with the original system prompt.
        
        Args:
            original_system: Original system prompt from the request
            compression_decoder: Compression decoder instructions
            rule_union: Compression rules that were applied
            lang: Programming language context
            
        Returns:
            str: Final system prompt for Anthropic API
        """
        _log_verbose_system_prompt_phase("ANTHROPIC_SYSTEM_CREATION", 
            "🔧 Creating Anthropic-optimized system prompt")
        
        prompt_parts = []
        
        # Add compression decoder first if we have compression rules
        if compression_decoder and compression_decoder.strip():
            prompt_parts.append(compression_decoder.strip())
            _log_verbose_system_prompt_phase("ANTHROPIC_SYSTEM_CREATION", 
                f"Added compression decoder: {len(compression_decoder)} chars")
        
        # Add original system prompt if provided
        if original_system and original_system.strip():
            prompt_parts.append(original_system.strip())
            _log_verbose_system_prompt_phase("ANTHROPIC_SYSTEM_CREATION", 
                f"Added original system prompt: {len(original_system)} chars")
        
        # Join with double newlines for clear separation
        final_prompt = "\n\n".join(prompt_parts)
        
        # Ensure we have some content
        if not final_prompt.strip():
            final_prompt = "You are Claude, a helpful AI assistant created by Anthropic."
            _log_verbose_system_prompt_phase("ANTHROPIC_SYSTEM_CREATION", 
                "No system content provided, using default Claude prompt")
        
        _log_verbose_system_prompt_phase("ANTHROPIC_SYSTEM_CREATION", 
            f"✅ Final Anthropic system prompt: {len(final_prompt)} chars")
        
        return final_prompt


# Global instance for use across the application
_anthropic_interceptor = None

def get_anthropic_system_prompt_interceptor(config=None):
    """Get or create the global Anthropic system prompt interceptor instance."""
    global _anthropic_interceptor
    if _anthropic_interceptor is None:
        _anthropic_interceptor = AnthropicSystemPromptInterceptor(config=config)
    return _anthropic_interceptor 