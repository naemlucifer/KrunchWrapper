"""
WebUI-specific System Prompt Interceptor

This interceptor is specifically designed to handle WebUI system messages better

```core/webui_system_prompt_interceptor.py
<code_block_to_apply_changes_from>
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from .system_prompt import build_system_prompt
from .conciseness_instructions import ConcisenessInstructionsHandler

logger = logging.getLogger(__name__)


class WebUISystemPromptInterceptor:
    """
    WebUI-specific system prompt interceptor that handles WebUI's system message patterns
    and integrates compression decoders more effectively.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.conciseness_handler = ConcisenessInstructionsHandler()
        
    def intercept_and_process_webui(self, 
                                   messages: List[Dict[str, Any]], 
                                   rule_union: Dict[str, str], 
                                   lang: str = "generic",
                                   target_format: str = "chatml") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process WebUI messages with compression decoder integration.
        
        Args:
            messages: List of messages from WebUI
            rule_union: Compression rules that were applied
            lang: Programming language detected
            target_format: Target system prompt format
            
        Returns:
            Tuple of (processed_messages, metadata)
        """
        logger.info("🔧 [WEBUI INTERCEPTOR] Starting WebUI-specific system prompt processing")
        logger.info(f"🔧 [WEBUI INTERCEPTOR] Input: {len(messages)} messages, {len(rule_union)} compression rules")
        
        # Extract the WebUI system message if present
        webui_system_message = self._extract_webui_system_message(messages)
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        if not rule_union:
            # No compression applied - just return messages as-is
            logger.info("🔧 [WEBUI INTERCEPTOR] No compression rules - returning original messages")
            return messages, {"webui_system_preserved": bool(webui_system_message)}
        
        # Extract user content for context
        user_content = self._extract_user_content(messages)
        
        # Use configured stateful mode from server config for optimal KV cache behavior
        stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
        
        # Build compression decoder instructions
        compression_decoder, decoder_metadata = build_system_prompt(
            rule_union, lang, target_format, user_content, cline_mode=False, stateful_mode=stateful_mode, new_symbols_only=None
        )
        
        # Create WebUI-optimized system prompt
        final_system_prompt = self._create_webui_optimized_system_prompt(
            webui_system_message, compression_decoder, rule_union, lang
        )
        
        # Create the final message structure
        processed_messages = []
        
        # Add the merged system message
        if final_system_prompt.strip():
            processed_messages.append({
                "role": "system",
                "content": final_system_prompt
            })
        
        # Add all non-system messages
        processed_messages.extend(non_system_messages)
        
        metadata = {
            "webui_system_preserved": bool(webui_system_message),
            "compression_decoder_added": bool(rule_union),
            "final_system_prompt_length": len(final_system_prompt),
            "rules_count": len(rule_union),
            "format": target_format
        }
        
        logger.info("🔧 [WEBUI INTERCEPTOR] Processing complete")
        logger.info(f"🔧 [WEBUI INTERCEPTOR] Final: {len(processed_messages)} messages, system prompt: {len(final_system_prompt)} chars")
        
        return processed_messages, metadata
    
    def _extract_webui_system_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the WebUI system message if present."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content and content.strip():
                    logger.debug(f"🔧 [WEBUI INTERCEPTOR] Found WebUI system message: {len(content)} chars")
                    return content.strip()
        logger.debug("🔧 [WEBUI INTERCEPTOR] No WebUI system message found")
        return None
    
    def _extract_user_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract user content for context analysis."""
        user_content = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    user_content.append(str(content))
        return "\n\n".join(user_content)
    
    def _create_webui_optimized_system_prompt(self, 
                                            webui_system_message: Optional[str],
                                            compression_decoder: str,
                                            rule_union: Dict[str, str],
                                            lang: str) -> str:
        """
        Create a WebUI-optimized system prompt that properly integrates:
        1. WebUI user's system message
        2. Compression decoder instructions  
        3. Clear guidance on how to handle compressed content
        """
        parts = []
        
        # 1. Start with user's original system message if present
        if webui_system_message:
            parts.append(webui_system_message)
            logger.debug("🔧 [WEBUI INTERCEPTOR] Added user's original system message")
        
        # 2. Add compression decoder with WebUI-specific formatting
        if rule_union:
            # Create a more explicit decoder section for WebUI
            decoder_section = self._format_decoder_for_webui(compression_decoder, rule_union, lang)
            parts.append(decoder_section)
            logger.debug("🔧 [WEBUI INTERCEPTOR] Added compression decoder section")
        
        # 3. Add conciseness instructions if enabled
        if self.conciseness_handler.should_inject_instructions(bool(rule_union)):
            user_content = ""  # We don't have access to user content here, but that's ok
            conciseness_instructions = self.conciseness_handler.generate_instructions(
                user_content=user_content, language=lang
            )
            if conciseness_instructions:
                parts.append(conciseness_instructions)
                logger.debug("🔧 [WEBUI INTERCEPTOR] Added conciseness instructions")
        
        # Join with clear separation
        return "\n\n".join(parts)
    
    def _format_decoder_for_webui(self, compression_decoder: str, rule_union: Dict[str, str], lang: str) -> str:
        """
        Format the compression decoder specifically for WebUI with clear, explicit instructions.
        """
        # Create a more explicit, WebUI-friendly decoder format
        decoder_lines = [
            "IMPORTANT: The user's message contains compressed symbols to save tokens.",
            f"This is {lang} code that has been compressed using the following symbol substitutions:",
            ""
        ]
        
        # Show the decoder in a clear, tabular format
        decoder_lines.append("SYMBOL DECODER:")
        sorted_symbols = sorted(rule_union.items(), key=lambda x: (len(x[0]), x[0]))
        
        for symbol, original in sorted_symbols[:10]:  # Show first 10 to avoid overwhelming
            # Truncate very long patterns for readability
            display_original = original[:50] + "..." if len(original) > 50 else original
            decoder_lines.append(f"  {symbol} = {display_original}")
        
        if len(rule_union) > 10:
            decoder_lines.append(f"  ... and {len(rule_union) - 10} more symbols")
        
        decoder_lines.extend([
            "",
            "INSTRUCTIONS:",
            "1. When you see these symbols in the user's message, treat them as the full expanded text",
            "2. Analyze and respond to the EXPANDED meaning, not the symbols themselves", 
            "3. In your response, you may use the same symbols to keep responses concise",
            "4. Do NOT refer to these as 'errors' or 'invalid symbols' - they are intentional compression"
        ])
        
        return "\n".join(decoder_lines)


def detect_webui_request(request_headers: Dict[str, str], request_data: Dict[str, Any]) -> bool:
    """
    Detect if a request is coming from the WebUI.
    
    Args:
        request_headers: HTTP request headers
        request_data: Request payload data
        
    Returns:
        True if this appears to be a WebUI request
    """
    # Check for WebUI-specific patterns
    user_agent = request_headers.get("user-agent", "").lower()
    
    # WebUI typically sends requests from browsers
    browser_patterns = ["mozilla", "chrome", "safari", "firefox", "edge"]
    is_browser = any(pattern in user_agent for pattern in browser_patterns)
    
    # Check for localhost/local network origins (WebUI typically runs locally)
    origin = request_headers.get("origin", "")
    referer = request_headers.get("referer", "")
    is_local = any("localhost" in header or "127.0.0.1" in header or "192.168." in header 
                   for header in [origin, referer])
    
    # Check request structure (WebUI has specific patterns)
    messages = request_data.get("messages", [])
    has_webui_structure = False
    
    if messages:
        # WebUI often sends a system message first
        first_msg = messages[0] if messages else {}
        if first_msg.get("role") == "system":
            has_webui_structure = True
        
        # WebUI often has specific parameter patterns
        webui_params = ["cache_prompt", "samplers", "dynatemp_range", "timings_per_token"]
        has_webui_params = any(param in request_data for param in webui_params)
        if has_webui_params:
            has_webui_structure = True
    
    # Combine indicators
    webui_indicators = sum([is_browser, is_local, has_webui_structure])
    is_webui = webui_indicators >= 2  # Require at least 2 indicators
    
    if is_webui:
        logger.debug(f"🔧 [WEBUI DETECTION] Detected WebUI request (indicators: {webui_indicators}/3)")
        logger.debug(f"🔧 [WEBUI DETECTION] Browser: {is_browser}, Local: {is_local}, Structure: {has_webui_structure}")
    
    return is_webui


# Global instance for use across the application
_webui_interceptor = None

def get_webui_system_prompt_interceptor(config=None):
    """Get or create the global WebUI system prompt interceptor instance."""
    global _webui_interceptor
    if _webui_interceptor is None:
        _webui_interceptor = WebUISystemPromptInterceptor(config=config)
    return _webui_interceptor 