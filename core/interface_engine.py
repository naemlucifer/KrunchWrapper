"""
Interface Engine System for KrunchWrapper

This module provides a unified, modular system for handling different client interfaces
(Cline, WebUI, SillyTavern, etc.) with their specific compression and system prompt requirements.

The system is designed to be easily extensible for future interfaces like roo, aider, etc.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import Request

from .cline_system_prompt_interceptor import ClineSystemPromptInterceptor
from .webui_system_prompt_interceptor import get_webui_system_prompt_interceptor, detect_webui_request
from .anthropic_system_prompt_interceptor import get_anthropic_system_prompt_interceptor
from .system_prompt_interceptor import SystemPromptInterceptor

logger = logging.getLogger(__name__)


class InterfaceEngine(Enum):
    """Supported interface engines for compression handling."""
    CLINE = "cline"
    WEBUI = "webui" 
    SILLYTAVERN = "sillytavern"
    ANTHROPIC = "anthropic"  # Direct Anthropic API support
    ROO = "roo"  # Future support
    AIDER = "aider"  # Future support
    STANDARD = "standard"  # Default/fallback


class InterfaceCompressionHandler:
    """
    Unified compression handler that routes to appropriate interface-specific processors.
    
    This replaces the complex branching logic in server.py with a clean, modular system
    that can be easily extended for new interfaces.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger('interface_handler')
        
        # Initialize interface-specific interceptors
        self._cline_interceptor = None
        self._webui_interceptor = None
        self._anthropic_interceptor = None
        self._standard_interceptor = None
        
        self.logger.info("🔧 InterfaceCompressionHandler:  Initialized")
    
    def detect_interface_engine(self, request: Request, request_data: Dict[str, Any] = None) -> InterfaceEngine:
        """
        Detect which interface engine is being used based on request characteristics.
        
        Args:
            request: FastAPI Request object
            request_data: Optional parsed request data
            
        Returns:
            InterfaceEngine enum value
        """
        # First check explicit configuration - new interface_engine setting
        if self.config and hasattr(self.config, 'system_prompt') and isinstance(self.config.system_prompt, dict):
            interface_engine = self.config.system_prompt.get('interface_engine', 'auto').lower()
            
            if interface_engine != 'auto':
                # Explicit interface engine specified - OVERRIDE auto-detection
                try:
                    engine = InterfaceEngine(interface_engine)
                    self.logger.debug(f"🔍 [ENGINE] OVERRIDE: Forcing {engine.value} interface (configured)")
                    return engine
                except ValueError:
                    self.logger.warning(f"🚨 [ENGINE] Unknown interface_engine '{interface_engine}', falling back to auto-detection")
            
            # Check legacy use_cline setting for backward compatibility
            if self.config.system_prompt.get('use_cline', False):
                self.logger.debug("🔍 [ENGINE] OVERRIDE: Forcing Cline interface (legacy use_cline=true)")
                return InterfaceEngine.CLINE
        
        # Check legacy config structure for backward compatibility (removed - use_cline no longer exists)
        # elif self.config and hasattr(self.config, 'use_cline') and self.config.use_cline:
        #     self.logger.debug("🔍 [ENGINE] OVERRIDE: Forcing Cline interface (legacy config.use_cline=true)")
        #     return InterfaceEngine.CLINE
        
        # Check new interface_engine setting directly on config object (server config)
        elif self.config and hasattr(self.config, 'interface_engine') and self.config.interface_engine != 'auto':
            try:
                engine = InterfaceEngine(self.config.interface_engine)
                self.logger.debug(f"🔍 [ENGINE] OVERRIDE: Forcing {engine.value} interface (server config)")
                return engine
            except ValueError:
                self.logger.warning(f"🚨 [ENGINE] Unknown interface_engine '{self.config.interface_engine}', falling back to auto-detection")
        
        # Auto-detection based on request characteristics
        request_headers = dict(request.headers) if request else {}
        if request_data is None and request:
            request_data = {}
        
        # Check for WebUI characteristics
        if detect_webui_request(request_headers, request_data):
            self.logger.debug("🔍 [ENGINE] Auto-detected WebUI interface")
            return InterfaceEngine.WEBUI
        
        # Check for SillyTavern characteristics
        user_agent = request_headers.get("user-agent", "").lower() if request_headers else ""
        if "sillytavern" in user_agent or "silly" in user_agent:
            self.logger.debug("🔍 [ENGINE] Auto-detected SillyTavern interface")
            return InterfaceEngine.SILLYTAVERN
        
        # Check for Anthropic API characteristics
        # Look for Anthropic-specific headers and API patterns
        anthropic_headers = ["x-api-key", "anthropic-version", "anthropic-beta"]
        for header in anthropic_headers:
            if header in request_headers:
                self.logger.debug(f"🔍 [ENGINE] Auto-detected Anthropic interface via header: {header}")
                return InterfaceEngine.ANTHROPIC
        
        # Check for direct Claude model requests (not via Cline)
        if request_data and "model" in request_data:
            model = request_data["model"].lower()
            # Direct Claude model calls (not through Cline's provider/model format)
            if "claude" in model and "/" not in model:
                self.logger.debug(f"🔍 [ENGINE] Auto-detected Anthropic interface via direct Claude model: {model}")
                return InterfaceEngine.ANTHROPIC
        
        # Check for Anthropic SDK user agents
        if "anthropic" in user_agent and "sdk" in user_agent:
            self.logger.debug("🔍 [ENGINE] Auto-detected Anthropic interface via SDK user-agent")
            return InterfaceEngine.ANTHROPIC
        
        # Check for Cline characteristics (only during auto-detection)
        cline_patterns = [
            "cline", 
            "so/js"  # Cline actually uses this User-Agent (so/js 4.83.0)
        ]
        for pattern in cline_patterns:
            if pattern in user_agent:
                self.logger.debug(f"🔍 [ENGINE] Auto-detected Cline interface via user-agent pattern: {pattern}")
                return InterfaceEngine.CLINE
        
        # Check for Cline-specific headers
        cline_headers = ["x-task-id", "x-cline-session", "x-cline-version", "x-anthropic-version"]
        for header in cline_headers:
            if header in request_headers:
                self.logger.debug(f"🔍 [ENGINE] Auto-detected Cline interface via header: {header}")
                return InterfaceEngine.CLINE
        
        # Check for other interfaces in the future
        # if "roo" in user_agent:
        #     self.logger.debug("🔍 [ENGINE] Auto-detected Roo interface")
        #     return InterfaceEngine.ROO
        # if "aider" in user_agent:
        #     self.logger.debug("🔍 [ENGINE] Auto-detected Aider interface")
        #     return InterfaceEngine.AIDER
        
        self.logger.debug("🔍 [ENGINE] Using standard interface (no specific interface detected)")
        return InterfaceEngine.STANDARD
    
    def get_interceptor_for_engine(self, engine: InterfaceEngine):
        """Get the appropriate system prompt interceptor for the given engine."""
        if engine == InterfaceEngine.CLINE:
            if self._cline_interceptor is None:
                self._cline_interceptor = ClineSystemPromptInterceptor(config=self.config)
            return self._cline_interceptor
        
        elif engine == InterfaceEngine.WEBUI:
            if self._webui_interceptor is None:
                self._webui_interceptor = get_webui_system_prompt_interceptor(config=self.config)
            return self._webui_interceptor
        
        elif engine == InterfaceEngine.ANTHROPIC:
            if self._anthropic_interceptor is None:
                self._anthropic_interceptor = get_anthropic_system_prompt_interceptor(config=self.config)
            return self._anthropic_interceptor
        
        elif engine == InterfaceEngine.SILLYTAVERN:
            # SillyTavern uses standard interceptor but with specific handling
            if self._standard_interceptor is None:
                self._standard_interceptor = SystemPromptInterceptor(config=self.config)
            return self._standard_interceptor
        
        else:  # STANDARD, ROO, AIDER, or any future engines
            if self._standard_interceptor is None:
                self._standard_interceptor = SystemPromptInterceptor(config=self.config)
            return self._standard_interceptor
    
    def process_compression(
        self,
        engine: InterfaceEngine,
        messages: List[Dict[str, Any]],
        rule_union: Dict[str, str],
        request: Request = None,
        model_id: str = None,
        system_param: str = None,
        system_instruction: str = None,
        target_format: str = "chatml"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process compression using the appropriate interface engine.
        
        Args:
            engine: Interface engine to use
            messages: List of chat messages
            rule_union: Compression rules that were applied
            request: Optional FastAPI request object
            model_id: Model ID for the request
            system_param: Claude-style system parameter
            system_instruction: Gemini-style system instruction
            target_format: Target system prompt format
            
        Returns:
            Tuple of (processed_messages, metadata)
        """
        self.logger.info(f"🔧 [ENGINE] Processing compression with {engine.value} interface")
        
        interceptor = self.get_interceptor_for_engine(engine)
        
        # Route to appropriate processing method based on engine
        if engine == InterfaceEngine.CLINE:
            if hasattr(interceptor, 'intercept_and_process_cline'):
                return interceptor.intercept_and_process_cline(
                    messages=messages,
                    rule_union=rule_union,
                    lang="generic",
                    model_id=model_id,
                    system_param=system_param
                )
            else:
                self.logger.warning("🚨 [ENGINE] Cline interceptor missing intercept_and_process_cline method")
                return self._fallback_to_standard(messages, rule_union, target_format, system_param, system_instruction)
        
        elif engine == InterfaceEngine.WEBUI:
            if hasattr(interceptor, 'intercept_and_process_webui'):
                return interceptor.intercept_and_process_webui(
                    messages=messages,
                    rule_union=rule_union,
                    lang="generic",
                    target_format=target_format
                )
            else:
                self.logger.warning("🚨 [ENGINE] WebUI interceptor missing intercept_and_process_webui method")
                return self._fallback_to_standard(messages, rule_union, target_format, system_param, system_instruction)
        
        elif engine == InterfaceEngine.ANTHROPIC:
            if hasattr(interceptor, 'intercept_and_process_anthropic'):
                return interceptor.intercept_and_process_anthropic(
                    messages=messages,
                    rule_union=rule_union,
                    lang="generic",
                    system_param=system_param,
                    target_format="claude"  # Anthropic uses Claude format
                )
            else:
                self.logger.warning("🚨 [ENGINE] Anthropic interceptor missing intercept_and_process_anthropic method")
                return self._fallback_to_standard(messages, rule_union, "claude", system_param, system_instruction)
        
        elif engine == InterfaceEngine.SILLYTAVERN:
            # SillyTavern uses standard processing but with specific considerations
            self.logger.debug("🔧 [ENGINE] Using standard processing for SillyTavern")
            return interceptor.intercept_and_process(
                messages=messages,
                rule_union=rule_union,
                lang="generic",
                target_format=target_format,
                system_param=system_param,
                system_instruction=system_instruction
            )
        
        else:  # STANDARD, ROO, AIDER, or future engines
            self.logger.debug(f"🔧 [ENGINE] Using standard processing for {engine.value}")
            return interceptor.intercept_and_process(
                messages=messages,
                rule_union=rule_union,
                lang="generic",
                target_format=target_format,
                system_param=system_param,
                system_instruction=system_instruction
            )
    
    def _fallback_to_standard(
        self,
        messages: List[Dict[str, Any]],
        rule_union: Dict[str, str],
        target_format: str,
        system_param: str = None,
        system_instruction: str = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fallback to standard processing when interface-specific processing fails."""
        self.logger.warning("🔄 [ENGINE] Falling back to standard processing")
        if not hasattr(self, '_fallback_interceptor') or self._fallback_interceptor is None:
            self._fallback_interceptor = SystemPromptInterceptor(config=self.config)
        return self._fallback_interceptor.intercept_and_process(
            messages=messages,
            rule_union=rule_union,
            lang="generic",
            target_format=target_format,
            system_param=system_param,
            system_instruction=system_instruction
        )
    
    def should_disable_compression(self, engine: InterfaceEngine, request: Request = None) -> bool:
        """
        Check if compression should be disabled for the given interface engine.
        
        Args:
            engine: Interface engine to check
            request: Optional FastAPI request object
            
        Returns:
            True if compression should be disabled
        """
        if not self.config:
            return False
        
        # Check engine-specific compression disabling
        if engine == InterfaceEngine.CLINE:
            # Check if Cline compression is globally disabled
            if hasattr(self.config, 'disable_compression_for_cline') and self.config.disable_compression_for_cline:
                self.logger.info("🚫 [ENGINE] Compression disabled for Cline via global config")
                return True
        
        # Check general client-based disabling
        if hasattr(self.config, 'disable_compression_clients') and request:
            user_agent = request.headers.get("user-agent", "").lower()
            for client in self.config.disable_compression_clients:
                if client.lower() in user_agent:
                    self.logger.info(f"🚫 [ENGINE] Compression disabled for client: {client}")
                    return True
        
        return False


# Global handler instance
_global_handler = None


def get_interface_compression_handler(config=None) -> InterfaceCompressionHandler:
    """Get or create the global interface compression handler."""
    global _global_handler
    if _global_handler is None:
        _global_handler = InterfaceCompressionHandler(config=config)
    return _global_handler


def detect_and_process_compression(
    request: Request,
    messages: List[Dict[str, Any]],
    rule_union: Dict[str, str],
    config=None,
    model_id: str = None,
    system_param: str = None,
    system_instruction: str = None,
    target_format: str = "chatml",
    request_data: Dict[str, Any] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], InterfaceEngine]:
    """
    Unified function to detect interface and process compression.
    
    This is the main entry point that replaces the complex branching logic in server.py.
    
    Returns:
        Tuple of (processed_messages, metadata, detected_engine)
    """
    handler = get_interface_compression_handler(config)
    
    # Detect which interface engine to use
    engine = handler.detect_interface_engine(request, request_data)
    
    # Check if compression should be disabled for this engine
    if handler.should_disable_compression(engine, request):
        logger.info(f"🚫 [ENGINE] Compression disabled for {engine.value} interface")
        return messages, {"engine": engine.value, "compression_disabled": True}, engine
    
    # Process compression with the detected engine
    processed_messages, metadata = handler.process_compression(
        engine=engine,
        messages=messages,
        rule_union=rule_union,
        request=request,
        model_id=model_id,
        system_param=system_param,
        system_instruction=system_instruction,
        target_format=target_format
    )
    
    # Add engine info to metadata
    metadata["engine"] = engine.value
    metadata["compression_disabled"] = False
    
    return processed_messages, metadata, engine 