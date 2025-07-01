"""
Configuration module for KrunchWrapper server.
Contains the ServerConfig class and related configuration functionality.
"""

import os
import logging
from datetime import datetime
import pathlib
import sys

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jsonc_parser import load_jsonc
from core.system_prompt_interceptor import SystemPromptInterceptor
from core.cline_system_prompt_interceptor import ClineSystemPromptInterceptor

# Import logging functions
from server.logging_utils import (
    log_config_message, flush_config_message_buffer, log_startup_messages
)


class ServerConfig:
    """Configuration for the KrunchWrapper proxy server."""
    def __init__(self):
        # EARLY SETUP: Initialize file logging first to capture all startup messages
        self._early_setup_logging()
        
        # Try to load from config file first
        config_path = pathlib.Path(__file__).parents[1] / "config" / "server.jsonc"
        config_data = {}
        
        if config_path.exists():
            try:
                config_data = load_jsonc(str(config_path))
                log_config_message(f"📋 Loaded configuration from {config_path}")
            except Exception as e:
                log_config_message(f"❌ Error loading config from {config_path}: {e}", "ERROR")
        
        # Load main config to check for Anthropic mode triggers
        app_config_path = pathlib.Path(__file__).parents[1] / "config" / "config.jsonc"
        app_config = {}
        if app_config_path.exists():
            try:
                app_config = load_jsonc(str(app_config_path))
                log_config_message(f"📋 Loaded application configuration from {app_config_path}")
            except Exception as e:
                log_config_message(f"❌ Error loading app config from {app_config_path}: {e}", "ERROR")
        
        # 🎯 AUTOMATIC ANTHROPIC MODE DETECTION
        # Check if both conditions are met for Anthropic mode
        system_prompt_config = app_config.get("system_prompt", {})
        format_is_claude = system_prompt_config.get("format", "") == "claude"
        interface_is_anthropic = system_prompt_config.get("interface_engine", "") == "anthropic"
        
        anthropic_mode_enabled = format_is_claude and interface_is_anthropic
        
        if anthropic_mode_enabled:
            # Use Anthropic configuration if available
            anthropic_config = config_data.get("anthropic", {})
            if anthropic_config:
                log_config_message("🎯 ANTHROPIC MODE DETECTED: Switching to Anthropic server configuration")
                log_config_message(f"   ✅ format: '{system_prompt_config.get('format')}'")
                log_config_message(f"   ✅ interface_engine: '{system_prompt_config.get('interface_engine')}'")
                log_config_message("   🔄 Using anthropic section from server.jsonc")
                
                # Override server config with Anthropic settings
                for key, value in anthropic_config.items():
                    if key != "conversation_compression":  # Handle this separately
                        config_data[key] = value
                        log_config_message(f"   🔧 {key}: {value}")
                
                # Handle conversation compression separately (merge with existing)
                if "conversation_compression" in anthropic_config:
                    existing_conv_config = config_data.get("conversation_compression", {})
                    existing_conv_config.update(anthropic_config["conversation_compression"])
                    config_data["conversation_compression"] = existing_conv_config
                    log_config_message(f"   🔧 conversation_compression: Updated with Anthropic settings")
            else:
                log_config_message("⚠️  ANTHROPIC MODE DETECTED but no 'anthropic' section found in server.jsonc", "WARNING")
                log_config_message("   Using default server configuration", "WARNING")
        else:
            log_config_message("📍 Standard mode: Using default server configuration")
            if format_is_claude:
                log_config_message(f"   ℹ️  format=claude detected, but interface_engine={system_prompt_config.get('interface_engine', 'auto')}")
            if interface_is_anthropic:
                log_config_message(f"   ℹ️  interface_engine=anthropic detected, but format={system_prompt_config.get('format', 'claude')}")
        
        # Set values with priority: env vars > config file > defaults
        self.port = int(os.environ.get("KRUNCHWRAPPER_PORT", config_data.get("port", 5001)))
        self.host = os.environ.get("KRUNCHWRAPPER_HOST", config_data.get("host", "0.0.0.0"))
        
        # Build target URL from components or use full URL if provided
        self.target_host = os.environ.get("LLM_API_HOST", config_data.get("target_host", "localhost"))
        self.target_port = int(os.environ.get("LLM_API_PORT", config_data.get("target_port", 5002)))
        self.target_use_https = config_data.get("target_use_https", False)
        target_url = os.environ.get("LLM_API_URL", None)
        
        if target_url:
            self.target_url = target_url
        else:
            # Use HTTPS if specified in config, otherwise HTTP
            protocol = "https" if self.target_use_https else "http"
            # Don't include port in URL if it's the default port for the protocol
            if (self.target_use_https and self.target_port == 443) or (not self.target_use_https and self.target_port == 80):
                self.target_url = f"{protocol}://{self.target_host}/v1"
            else:
                self.target_url = f"{protocol}://{self.target_host}:{self.target_port}/v1"
            
        # API key settings
        self.api_key = os.environ.get("LLM_API_KEY", config_data.get("api_key", ""))
        self.require_api_key = os.environ.get("LLM_REQUIRE_API_KEY", config_data.get("require_api_key", False))
        
        self.min_compression_ratio = float(os.environ.get(
            "MIN_COMPRESSION_RATIO", 
            config_data.get("min_compression_ratio", 0.05)
        ))
        
        # Use the already-loaded app config from Anthropic mode detection
        self.app_config = app_config
        
        # Language detection removed - dynamic compression works on any content
        
        # Get compression settings
        compression_config = self.app_config.get("compression", {})
        self.min_characters = compression_config.get("min_characters", 250)
        
        # Get client-specific compression settings
        self.disable_compression_clients = compression_config.get("disable_for_clients", [])
        self.disable_compression_for_cline = compression_config.get("disable_compression_for_cline", False)
        self.selective_tool_call_compression = compression_config.get("selective_tool_call_compression", True)
        self.tool_call_min_content_size = compression_config.get("tool_call_min_content_size", 300)
        self.cline_preserve_system_prompt = compression_config.get("cline_preserve_system_prompt", True)
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🗜️  COMPRESSION CONFIGURATION")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"🔄 Engine:                       Dynamic compression")
        log_config_message(f"📏 Min characters:               {self.min_characters}")
        if self.disable_compression_clients:
            log_config_message(f"🚫 Disabled for clients:         {', '.join(self.disable_compression_clients)}")
        if self.disable_compression_for_cline:
            log_config_message("🚫 Cline compression:            Disabled")
        log_config_message(f"🔧 Tool call compression:        {'Enabled' if self.selective_tool_call_compression else 'Disabled'}")
        if self.selective_tool_call_compression:
            log_config_message(f"📐 Tool call min size:           {self.tool_call_min_content_size} chars")
        log_config_message(f"🛡️  Cline system preservation:    {'Enabled' if self.cline_preserve_system_prompt else 'Disabled'}")
        
        # Get conversation compression settings
        conv_config = self.app_config.get("conversation_compression", {})
        self.conversation_compression_enabled = conv_config.get("enabled", True)
        self.conversation_stateful_mode = conv_config.get("mode", "stateful") == "stateful"
        self.conversation_max_conversations = conv_config.get("max_conversations", 1000)
        self.conversation_cleanup_interval = conv_config.get("cleanup_interval", 3600)
        self.conversation_min_net_efficiency = conv_config.get("min_net_efficiency", 0.01)
        self.conversation_efficiency_trend_window = conv_config.get("efficiency_trend_window", 3)
        self.conversation_long_threshold = conv_config.get("long_conversation_threshold", 20)
        self.conversation_long_min_efficiency = conv_config.get("long_conversation_min_efficiency", 0.02)
        self.conversation_force_compression = conv_config.get("force_compression", False)
        self.conversation_kv_cache_threshold = conv_config.get("kv_cache_threshold", 20)
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🗣️  CONVERSATION COMPRESSION")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"🗣️  Status:                       {'Enabled' if self.conversation_compression_enabled else 'Disabled'}")
        if self.conversation_compression_enabled:
            mode_display = "Stateful (KV Cache Optimized)" if self.conversation_stateful_mode else "Stateless (Standard APIs)"
            log_config_message(f"🧠 Compression mode:             {mode_display}")
            log_config_message(f"🚀 KV cache threshold:           {self.conversation_kv_cache_threshold} chars")
            log_config_message(f"📈 Max conversations:            {self.conversation_max_conversations}")
            log_config_message(f"📊 Min net efficiency:           {self.conversation_min_net_efficiency}")
        
        # Check comment stripping status
        comment_stripping_config = self.app_config.get("comment_stripping", {})
        comment_stripping_enabled = comment_stripping_config.get("enabled", False)
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("📝 COMMENT STRIPPING")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        if comment_stripping_enabled:
            supported_languages = [lang for lang, enabled in comment_stripping_config.get("languages", {}).items() if enabled]
            log_config_message(f"📝 Status:                       Enabled ({len(supported_languages)} languages)")
            log_config_message(f"🔤 Languages:                    {', '.join(supported_languages)}")
        else:
            log_config_message("📝 Status:                       Disabled")
        
        # Get logging configuration from unified logging section in server config
        logging_config = config_data.get("logging", {})
        
        # Get verbose logging setting from unified config
        self.verbose_logging = os.environ.get("KRUNCHWRAPPER_VERBOSE", logging_config.get("verbose_logging", False))
        if isinstance(self.verbose_logging, str):
            self.verbose_logging = self.verbose_logging.lower() in ['true', '1', 'yes', 'on']
        
        # Get Cline streaming content logging setting (can be very verbose!)
        # Parse cline_stream_content_logging with new options
        cline_stream_config = os.environ.get("KRUNCHWRAPPER_CLINE_STREAM_LOGGING", logging_config.get("cline_stream_content_logging", False))
        
        # Handle different configuration types
        if isinstance(cline_stream_config, bool):
            self.cline_stream_content_logging = cline_stream_config
            self.cline_stream_logging_target = "both" if cline_stream_config else "disabled"
        elif isinstance(cline_stream_config, str):
            cline_stream_config = cline_stream_config.lower()
            if cline_stream_config in ['true', '1', 'yes', 'on']:
                self.cline_stream_content_logging = True
                self.cline_stream_logging_target = "both"
            elif cline_stream_config in ['false', '0', 'no', 'off']:
                self.cline_stream_content_logging = False
                self.cline_stream_logging_target = "disabled"
            elif cline_stream_config in ['terminal', 'console']:
                self.cline_stream_content_logging = True
                self.cline_stream_logging_target = "terminal"
            elif cline_stream_config == 'file':
                self.cline_stream_content_logging = True
                self.cline_stream_logging_target = "file"
            elif cline_stream_config == 'both':
                self.cline_stream_content_logging = True
                self.cline_stream_logging_target = "both"
            else:
                # Invalid value, default to disabled
                self.cline_stream_content_logging = False
                self.cline_stream_logging_target = "disabled"
        else:
            # Invalid type, default to disabled
            self.cline_stream_content_logging = False
            self.cline_stream_logging_target = "disabled"
        
        # Terminal output formatting setting was already loaded in _early_setup_logging()
        
        # Get passthrough request logging setting from unified config
        self.show_passthrough_requests = os.environ.get("KRUNCHWRAPPER_SHOW_PASSTHROUGH", logging_config.get("show_passthrough_requests", False))
        if isinstance(self.show_passthrough_requests, str):
            self.show_passthrough_requests = self.show_passthrough_requests.lower() in ['true', '1', 'yes', 'on']
        
        # Get file logging settings from unified config
        self.file_logging = os.environ.get("KRUNCHWRAPPER_FILE_LOGGING", logging_config.get("file_logging", False))
        if isinstance(self.file_logging, str):
            self.file_logging = self.file_logging.lower() in ['true', '1', 'yes', 'on']
        
        # Load console and file log levels separately from unified config
        self.log_level = os.environ.get("KRUNCHWRAPPER_LOG_LEVEL", logging_config.get("log_level", "INFO")).upper()
        self.file_log_level = os.environ.get("KRUNCHWRAPPER_FILE_LOG_LEVEL", logging_config.get("file_log_level", self.log_level)).upper()
        
        # Load debug category toggles from unified config (full setup)
        self.debug_categories = logging_config.get("debug_categories", {})
        
        # Setup full file logging configuration if enabled
        if self.file_logging:
            self._configure_full_logging()
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("📊 LOGGING CONFIGURATION")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"📝 File logging:                 {'Enabled' if self.file_logging else 'Disabled'}")
        if self.file_logging:
            log_config_message(f"📺 Console log level:            {self.log_level}")
            log_config_message(f"📁 File log level:               {self.file_log_level}")
            log_config_message(f"📺 Terminal format:              {'Simplified' if self.simplify_terminal_output else 'Full (same as file)'}")
        else:
            log_config_message(f"📺 Console log level:            {self.log_level}")
        log_config_message(f"🔍 Verbose logging:              {'Enabled' if self.verbose_logging else 'Disabled'}")
        log_config_message(f"🤫 Suppress debug modules:       {self.suppress_debug_modules if self.suppress_debug_modules else 'None'}")
        
        # Only show Cline stream logging message when DEBUG is enabled
        if self.log_level == "DEBUG":
            if self.cline_stream_content_logging:
                target_display = {
                    "terminal": "Terminal only",
                    "file": "File only", 
                    "both": "Terminal + File",
                    "disabled": "Disabled"
                }.get(self.cline_stream_logging_target, "Unknown")
                log_config_message(f"🔍 Cline stream logging:         Enabled ({target_display})")
            else:
                log_config_message(f"🔍 Cline stream logging:         Disabled")
        
        log_config_message(f"📤 Passthrough logging:          {'Enabled' if self.show_passthrough_requests else 'Disabled'}")
        
        # Show debug category status if DEBUG level is enabled
        if self.log_level == "DEBUG":
            enabled_categories = [name for name, enabled in self.debug_categories.items() if enabled]
            disabled_categories = [name for name, enabled in self.debug_categories.items() if not enabled]
            log_config_message(f"🎯 Debug categories enabled:     {len(enabled_categories)}/{len(self.debug_categories)}")
            if disabled_categories:
                log_config_message(f"🚫 Debug categories disabled:    {', '.join(disabled_categories)}")
        
        # Get system prompt settings
        system_prompt_config = self.app_config.get("system_prompt", {})
        self.system_prompt_format = system_prompt_config.get("format", "claude")
        # New interface engine system - replaces use_cline
        self.interface_engine = system_prompt_config.get("interface_engine", "auto")
        
        # Legacy support for use_cline (backward compatibility)
        legacy_use_cline = system_prompt_config.get("use_cline", False)
        if legacy_use_cline and self.interface_engine == "auto":
            self.interface_engine = "cline"
            log_config_message("🔄 Converted legacy use_cline=true to interface_engine=cline")
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🤖 SYSTEM PROMPT & API")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"🌐 Target API URL:               {self.target_url}")
        log_config_message(f"📝 Prompt format:                {self.system_prompt_format}")
        log_config_message(f"🔧 Interface engine:             {self.interface_engine}")
        log_config_message(f"✅ Parameter pass-through:       Enabled (legacy params cleaned)")
        log_config_message(f"🎯 Parameter validation:         Target server")
        
        # Initialize the new modular interface engine system
        from core.interface_engine import get_interface_compression_handler
        self.interface_handler = get_interface_compression_handler(config=self)
        
        # For backward compatibility, still create individual interceptors
        # but the main processing will use the new interface engine system
        if self.interface_engine == "cline":
            self.system_prompt_interceptor = ClineSystemPromptInterceptor(config=self)
            interceptor_type = "Cline (via Interface Engine)"
        else:
            self.system_prompt_interceptor = SystemPromptInterceptor(config=self)
            interceptor_type = f"Interface Engine ({self.interface_engine})"
        
        # Log interceptor type after initialization (to come after module init messages)
        log_config_message(f"🔍 System interceptor:           {interceptor_type}")
        
        # Log the interface engine system initialization
        log_config_message(f"🔧 Interface engine handler:     Initialized")
        
        # Get proxy settings
        proxy_config = self.app_config.get("proxy", {})
        self.filter_timeout_parameters = proxy_config.get("filter_timeout_parameters", True)
        
        # Get timeout settings for streaming connections
        timeout_config = proxy_config.get("timeout", {})
        self.streaming_connect_timeout = timeout_config.get("streaming_connect_timeout", 30)
        self.streaming_total_timeout = timeout_config.get("streaming_total_timeout", None)  # None = no timeout
        self.enable_timeout_override = timeout_config.get("enable_timeout_override", True)
        
        # Get streaming configuration
        streaming_config = self.app_config.get("streaming", {})
        self.preserve_sse_format = streaming_config.get("preserve_sse_format", True)
        self.validate_json_chunks = streaming_config.get("validate_json_chunks", True)
        self.smart_decompression = streaming_config.get("smart_decompression", True)
        self.cline_compatibility_mode = streaming_config.get("cline_compatibility_mode", True)
        self.chunk_processing_timeout = streaming_config.get("chunk_processing_timeout", 5.0)
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🌊 STREAMING & PROXY")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"⏱️  Timeout filtering:            {'Enabled' if self.filter_timeout_parameters else 'Disabled'}")
        log_config_message(f"⏱️  Connect timeout:              {self.streaming_connect_timeout}s")
        log_config_message(f"⏱️  Total timeout:                {'Disabled' if self.streaming_total_timeout is None else f'{self.streaming_total_timeout}s'}")
        log_config_message(f"⏱️  Timeout override:             {'Enabled' if self.enable_timeout_override else 'Disabled'}")
        log_config_message(f"🔄 SSE format preservation:      {'Enabled' if self.preserve_sse_format else 'Disabled'}")
        log_config_message(f"✅ JSON chunk validation:        {'Enabled' if self.validate_json_chunks else 'Disabled'}")
        log_config_message(f"🔍 Smart decompression:          {'Enabled' if self.smart_decompression else 'Disabled'}")
        log_config_message(f"🔧 Cline compatibility:          {'Enabled' if self.cline_compatibility_mode else 'Disabled'}")
        
        # Language detection removed - no longer needed

    def setup_file_logging(self):
        """Setup file logging with datetime-based filenames."""
        # Create logs directory
        log_dir = pathlib.Path(__file__).parents[1] / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Generate filename with current datetime
        now = datetime.now()
        log_filename = f"krunchwrapper_{now.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        
        # Configure the root logger to capture all module logs
        # Use the lower of the two log levels to ensure all messages reach handlers
        min_level = min(
            getattr(logging, self.log_level, logging.INFO),
            getattr(logging, self.file_log_level, logging.INFO)
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(min_level)
        
        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create file handler with file-specific log level
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, self.file_log_level, logging.INFO))
        
        # Create console handler with console-specific log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
        
        # Create formatters - file always uses full format, console can be simplified
        file_formatter = logging.Formatter('%(asctime)s - %(name)-16s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Check if simplified terminal output is enabled
        simplify_terminal = getattr(self, 'simplify_terminal_output', False)
        
        if simplify_terminal:
            # Simplified console format: level (padded) and message
            console_formatter = logging.Formatter('%(levelname)-5s - %(message)s')
            console_handler.setFormatter(console_formatter)
        else:
            # Full console format: same as file
            console_formatter = logging.Formatter('%(asctime)s - %(name)-16s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Apply selective debug suppression to specific modules
        if self.suppress_debug_modules:
            log_config_message(f"🤫 Applying selective debug suppression to {len(self.suppress_debug_modules)} modules:")
            for module_name in self.suppress_debug_modules:
                module_logger = logging.getLogger(module_name)
                # Set the suppressed module to INFO level instead of DEBUG
                # This will prevent DEBUG messages from these modules while allowing INFO and above
                module_logger.setLevel(logging.INFO)
                log_config_message(f"   📵 {module_name}: DEBUG → INFO")
        
        # Also get the specific krunchwrapper logger for compatibility
        self.logger = logging.getLogger('krunchwrapper')
        # Use the same minimum level as root logger
        self.logger.setLevel(min_level)
        
        # Store log path for later use (will be shown in async logging section)
        self.log_file_path = log_path
        
        # Flush any buffered configuration messages now that file logging is set up
        flush_config_message_buffer()
    
    def _early_setup_logging(self):
        """Early logging setup to capture startup messages."""
        # Get basic configuration early for file logging determination
        config_path = pathlib.Path(__file__).parents[1] / "config" / "server.jsonc"
        config_data = {}
        if config_path.exists():
            try:
                config_data = load_jsonc(str(config_path))
            except Exception:
                pass
        
        # Get unified logging configuration from server config
        logging_config = config_data.get("logging", {})
        
        # Determine if file logging should be enabled from unified config
        self.file_logging = os.environ.get("KRUNCHWRAPPER_FILE_LOGGING", logging_config.get("file_logging", False))
        if isinstance(self.file_logging, str):
            self.file_logging = self.file_logging.lower() in ['true', '1', 'yes', 'on']
        
        # Load console and file log levels separately (early setup) from unified config
        self.log_level = os.environ.get("KRUNCHWRAPPER_LOG_LEVEL", logging_config.get("log_level", "INFO")).upper()
        self.file_log_level = os.environ.get("KRUNCHWRAPPER_FILE_LOG_LEVEL", logging_config.get("file_log_level", self.log_level)).upper()
        
        # Load selective debug suppression modules from unified config
        suppress_debug_env = os.environ.get("KRUNCHWRAPPER_SUPPRESS_DEBUG_MODULES", "")
        if suppress_debug_env:
            self.suppress_debug_modules = [module.strip() for module in suppress_debug_env.split(",") if module.strip()]
        else:
            self.suppress_debug_modules = logging_config.get("suppress_debug_modules", [])
        
        # Get simplified terminal setting from unified config
        self.simplify_terminal_output = logging_config.get("simplify_terminal_output", False)
        
        # Load debug category toggles from unified config (early setup)
        self.debug_categories = logging_config.get("debug_categories", {})
        
        # If file logging is enabled, set it up immediately
        if self.file_logging:
            self.setup_file_logging()
    
    def _configure_full_logging(self):
        """Configure additional logging features after full configuration is loaded."""
        # No additional configuration needed - simplified terminal format is now
        # applied from the beginning in setup_file_logging()
        pass
    
    def is_debug_category_enabled(self, category: str) -> bool:
        """
        Check if a specific debug category is enabled.
        
        Args:
            category: The debug category name to check
            
        Returns:
            True if the category is enabled (or if no categories are configured),
            False if explicitly disabled
        """
        # If no debug categories are configured, allow all debug messages (backward compatibility)
        if not self.debug_categories:
            return True
            
        # Return the specific category setting, defaulting to True if not found
        return self.debug_categories.get(category, True) 