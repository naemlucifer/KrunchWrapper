import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path

from .system_prompt import SystemPromptFormatter, build_system_prompt
from .dynamic_config_parser import DynamicConfigManager
from .compress import compress_with_dynamic_analysis
from .async_logger import log_verbose_system_prompt_phase_fast, get_performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('system_prompt')

def _log_verbose_system_prompt_phase(phase: str, message: str, data: Any = None):
    """
    Optimized system prompt logging with async processing.
    Drop-in replacement for the original synchronous logging function.
    """
    # Use the fast async logger for most cases
    log_verbose_system_prompt_phase_fast(phase, message, data)
    
    # Maintain backward compatibility with server module logging
    try:
        import sys
        
        # Check multiple possible module names for the server
        server_module = None
        for module_name in ['api.server', 'server', 'main']:
            if module_name in sys.modules:
                server_module = sys.modules[module_name]
                break
        
        if server_module:
            # Try to access config and verbose logging setting
            config = getattr(server_module, 'config', None)
            if config and hasattr(config, 'verbose_logging') and config.verbose_logging:
                # Use the server's log_message function if available for critical phases
                if hasattr(server_module, 'log_message') and phase in ['ERROR', 'COMPLETION']:
                    server_module.log_message(f"🔧 [SYSTEM PROMPT {phase}] {message}", "INFO")
                    
    except Exception as e:
        logger.debug(f"Could not access server config for verbose logging: {e}")
    
    # Keep critical phase logging synchronous for immediate visibility
    if phase in ['ERROR', 'COMPLETION']:
        logger.info(f"🔧 [SYSTEM PROMPT {phase}] {message}")
        if data is not None:
            if isinstance(data, (dict, list)):
                try:
                    formatted_data = json.dumps(data, indent=2, ensure_ascii=False)[:1000]
                    if len(str(data)) > 1000:
                        formatted_data += "..."
                    logger.info(f"🔧 [SYSTEM PROMPT {phase}] Data: {formatted_data}")
                except:
                    logger.info(f"🔧 [SYSTEM PROMPT {phase}] Data: {str(data)[:500]}")
            else:
                logger.info(f"🔧 [SYSTEM PROMPT {phase}] Data: {str(data)[:500]}")

class SystemPromptInterceptor:
    """
    Advanced system prompt interceptor that captures, analyzes, converts, and merges
    system prompts from incoming API requests with KrunchWrapper compression instructions.
    Now includes dynamic dictionary analysis capability with config-based settings.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.formatter = SystemPromptFormatter()
        self.supported_formats = self._load_supported_formats()
        self.dynamic_config_manager = DynamicConfigManager()
        
        # Add initialization logging
        logger.info("🔧 SystemPromptInterceptor:      Initialized")
        logger.info(f"📋 Loaded formats:               {len(self.supported_formats)} supported formats")
        logger.info(f"⚙️  Dynamic config manager:       {'enabled' if self.dynamic_config_manager.is_enabled() else 'disabled'}")
        
    def _load_supported_formats(self) -> Dict[str, Any]:
        """Load supported formats from system-prompts.jsonc"""
        return self.formatter.formats
    
    def intercept_and_process(self, 
                            messages: List[Dict[str, Any]], 
                            rule_union: Dict[str, str], 
                            lang: str, 
                            target_format: str,
                            system_param: Optional[str] = None,
                            system_instruction: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point: Intercept, analyze, convert, and merge system prompts.
        Now includes config-based dynamic dictionary analysis and performance monitoring.
        
        Args:
            messages: List of chat messages
            rule_union: Compression substitutions that were used
            lang: Programming language detected
            target_format: Target system prompt format
            system_param: System parameter from request (for Claude-style APIs)
            system_instruction: System instruction field (for Gemini-style APIs)
            
        Returns:
            Tuple of (processed_messages, metadata)
        """
        # Add entry point logging
        logger.info("🔧 [SYSTEM PROMPT INTERCEPTOR] ===== ENTRY POINT =====")
        logger.info(f"🔧 [SYSTEM PROMPT INTERCEPTOR] Processing request with {len(messages)} messages")
        logger.info(f"🔧 [SYSTEM PROMPT INTERCEPTOR] Target format: {target_format}, Language: {lang}")
        logger.info(f"🔧 [SYSTEM PROMPT INTERCEPTOR] Rule union size: {len(rule_union)}, System param: {'present' if system_param else 'none'}")
        logger.info(f"🔧 [SYSTEM PROMPT INTERCEPTOR] System instruction: {'present' if system_instruction else 'none'}")
        
        # CRITICAL FIX: If no compression rules, return original messages unchanged (like WebUI interceptor)
        if not rule_union:
            logger.info("🔧 [SYSTEM PROMPT INTERCEPTOR] No compression rules - returning original messages unchanged")
            logger.info("🔧 [SYSTEM PROMPT INTERCEPTOR] ===== COMPLETED WITH PASSTHROUGH =====")
            return messages, {
                "intercepted_prompts": 0,
                "target_format": target_format,
                "dynamic_compression": False,
                "final_system_prompt_length": 0,
                "passthrough_reason": "no_compression_rules"
            }
        
        # Initialize performance monitoring
        perf_monitor = get_performance_monitor()
        
        try:
            with perf_monitor.time_operation("system_prompt_interception_total"):
                _log_verbose_system_prompt_phase("INTERCEPTION", 
                    f"Starting system prompt interception with target format: {target_format}")
                _log_verbose_system_prompt_phase("INTERCEPTION", 
                    f"Input: {len(messages)} messages, compression rules: {len(rule_union)}, lang: {lang}")
                
                # Step 1: Intercept existing system prompts
                with perf_monitor.time_operation("system_prompt_interception"):
                    _log_verbose_system_prompt_phase("INTERCEPTION", "Step 1: Intercepting existing system prompts")
                    intercepted_prompts = self._intercept_system_prompts(
                        messages, system_param, system_instruction
                    )
                    _log_verbose_system_prompt_phase("INTERCEPTION", 
                        f"Found {len(intercepted_prompts)} system prompts from various sources", 
                        [{"source": p["source"], "content_length": len(p["content"]), "content_preview": p["content"][:100]} 
                         for p in intercepted_prompts])
                            
                # Step 2: Check if compression was already applied by looking at rule_union
                with perf_monitor.time_operation("compression_analysis"):
                    compression_already_applied = bool(rule_union)
                    _log_verbose_system_prompt_phase("COMPRESSION", 
                        f"Step 2: Compression status - already applied: {compression_already_applied}")
                    
                    if compression_already_applied:
                        # Compression was already applied in the main request handler
                        # Extract metadata from existing compression
                        user_content = self._extract_user_content(messages)
                        dynamic_metadata = {
                            "dynamic_enabled": True,
                            "dynamic_analyzed": True,
                            "dynamic_dict_used": True,
                            "dynamic_compression_ratio": self._estimate_compression_ratio(messages, rule_union),
                            "dynamic_dict_path": "external_compression",
                            "dynamic_tokens_used": len(rule_union),
                        }
                        compressed_messages = None  # Use the already-compressed messages
                        logger.debug(f"Using existing compression with {len(rule_union)} tokens")
                        _log_verbose_system_prompt_phase("COMPRESSION", 
                            f"Using existing compression: {len(rule_union)} rule tokens, ratio: {dynamic_metadata['dynamic_compression_ratio']:.1%}")
                    else:
                        # No prior compression, perform dynamic dictionary analysis
                        _log_verbose_system_prompt_phase("COMPRESSION", 
                            "No prior compression detected, performing dynamic dictionary analysis")
                        user_content = self._extract_user_content(messages)
                        dynamic_metadata, compressed_messages = self._process_dynamic_dictionary_analysis(messages, user_content, lang)
                        _log_verbose_system_prompt_phase("COMPRESSION", 
                            f"Dynamic analysis complete", dynamic_metadata)
                
                # Step 3: Detect format of intercepted prompts
                with perf_monitor.time_operation("format_detection"):
                    _log_verbose_system_prompt_phase("FORMAT_DETECTION", "Step 3: Detecting format of intercepted prompts")
                    detected_formats = self._detect_formats(intercepted_prompts)
                    _log_verbose_system_prompt_phase("FORMAT_DETECTION", 
                        f"Format detection complete", detected_formats)
                
                # Step 4: Build KrunchWrapper compression instructions
                with perf_monitor.time_operation("decoder_generation"):
                    _log_verbose_system_prompt_phase("DECODER_GENERATION", "Step 4: Building KrunchWrapper compression instructions/decoder")
                    # Use configured stateful mode from server config for optimal KV cache behavior
                    stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
                    compression_prompt, compression_metadata = build_system_prompt(
                        rule_union, lang, target_format, user_content, cline_mode=False, stateful_mode=stateful_mode, new_symbols_only=None
                    )
                    _log_verbose_system_prompt_phase("DECODER_GENERATION", 
                        f"Generated compression decoder prompt ({len(compression_prompt)} chars)", 
                        {"compression_prompt_preview": compression_prompt[:200], "metadata": compression_metadata})
                
                # Step 5: Add dynamic dictionary information if applicable
                with perf_monitor.time_operation("decoder_enhancement"):
                    if dynamic_metadata.get("dynamic_dict_used"):
                        _log_verbose_system_prompt_phase("DECODER_ENHANCEMENT", 
                            "Step 5: Enhancing compression prompt with dynamic dictionary information")
                        original_length = len(compression_prompt)
                        compression_prompt = self._enhance_compression_prompt_with_dynamic_info(
                            compression_prompt, dynamic_metadata
                        )
                        _log_verbose_system_prompt_phase("DECODER_ENHANCEMENT", 
                            f"Enhanced compression prompt: {original_length} → {len(compression_prompt)} chars")
                    else:
                        _log_verbose_system_prompt_phase("DECODER_ENHANCEMENT", 
                            "Step 5: Skipping dynamic dictionary enhancement (no dynamic compression)")
                
                # Step 6: Merge all system prompts intelligently
                with perf_monitor.time_operation("system_prompt_merging"):
                    _log_verbose_system_prompt_phase("MERGING", "Step 6: Merging system prompts intelligently")
                    merged_content = self._merge_system_prompts(
                        intercepted_prompts, compression_prompt, detected_formats
                    )
                    _log_verbose_system_prompt_phase("MERGING", 
                        f"Merged system prompt content ({len(merged_content)} chars)", 
                        {"merged_content_preview": merged_content[:300]})
                
                # Step 7: Convert to target format
                with perf_monitor.time_operation("format_conversion"):
                    _log_verbose_system_prompt_phase("FORMAT_CONVERSION", f"Step 7: Converting to target format: {target_format}")
                    final_content, final_metadata = self._convert_to_target_format(
                        merged_content, target_format
                    )
                    _log_verbose_system_prompt_phase("FORMAT_CONVERSION", 
                        f"Format conversion complete", 
                        {"final_metadata": final_metadata, "final_content_preview": final_content[:200]})
                
                # Step 8: Apply to messages (use compressed messages if available, otherwise original)
                with perf_monitor.time_operation("system_prompt_application"):
                    _log_verbose_system_prompt_phase("APPLICATION", "Step 8: Applying processed system prompt to messages")
                    base_messages = compressed_messages if compressed_messages is not None else messages
                    processed_messages = self._apply_system_prompt(
                        base_messages, final_content, final_metadata, target_format
                    )
                    _log_verbose_system_prompt_phase("APPLICATION", 
                        f"Applied system prompt to {len(processed_messages)} messages (was {len(messages)})")
                
                # Merge metadata
                final_metadata.update(dynamic_metadata)
                
                # Log successful processing with detailed summary and performance stats
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"✅ System prompt interception COMPLETE - Summary:")
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"   • Intercepted {len(intercepted_prompts)} system prompts")
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"   • Target format: {target_format}")
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"   • Dynamic compression: {dynamic_metadata.get('dynamic_dict_used', False)}")
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"   • Final system prompt length: {len(final_content)} chars")
                _log_verbose_system_prompt_phase("COMPLETION", 
                    f"   • Processing metadata", final_metadata)
                
                # Log performance statistics
                perf_stats = perf_monitor.get_stats()
                if perf_stats:
                    _log_verbose_system_prompt_phase("COMPLETION", 
                        f"   • Performance stats", perf_stats)
                
                logger.info(f"Successfully processed system prompts. "
                           f"Intercepted: {len(intercepted_prompts)}, "
                           f"Target format: {target_format}, "
                           f"Dynamic dict: {dynamic_metadata.get('dynamic_dict_used', False)}")
                
                logger.info("🔧 [SYSTEM PROMPT INTERCEPTOR] ===== COMPLETED SUCCESSFULLY =====")
                return processed_messages, final_metadata
            
        except Exception as e:
            _log_verbose_system_prompt_phase("ERROR", f"❌ Error in system prompt interception: {e}")
            logger.error(f"Error in system prompt interception: {e}")
            logger.info("🔧 [SYSTEM PROMPT INTERCEPTOR] ===== FAILED WITH ERROR =====")
            # Fallback: Apply compression prompt only
            return self._apply_fallback(messages, rule_union, lang, target_format)
    
    def _extract_user_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract all user content from messages for dynamic analysis."""
        user_content = []
        
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    user_content.append(str(content))
        
        return "\n\n".join(user_content)
    
    def _process_dynamic_dictionary_analysis(self, messages: List[Dict[str, Any]], user_content: str, lang: str) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        """
        Process dynamic dictionary analysis and apply compression to user messages.
        
        Args:
            messages: Original messages list
            user_content: Combined user content from messages
            lang: Programming language hint
            
        Returns:
            Tuple of (metadata, compressed_messages_or_None)
        """
        metadata = {
            "dynamic_enabled": False,
            "dynamic_analyzed": False,
            "dynamic_dict_used": False,
            "dynamic_compression_ratio": 0.0,
            "dynamic_dict_path": None,
            "dynamic_tokens_used": 0,
        }
        
        # Check if dynamic dictionary is enabled in config
        if not self.dynamic_config_manager.is_enabled():
            logger.debug("Dynamic dictionary analysis is disabled in configuration")
            return metadata, None
        
        metadata["dynamic_enabled"] = True
        
        if not user_content.strip():
            return metadata, None
        
        # Import here to avoid circular imports
        from .dynamic_dictionary import get_dynamic_dictionary_analyzer
        
        try:
            # Use cached analyzer instance (it will load config internally)
            analyzer = get_dynamic_dictionary_analyzer()
            
            # Check if this prompt should be analyzed
            should_analyze, reason = analyzer.should_analyze_prompt(user_content)
            
            logger.debug(f"Dynamic analysis decision: {should_analyze} - {reason}")
            
            if should_analyze:
                # Perform dynamic compression analysis
                result = compress_with_dynamic_analysis(user_content)
                
                metadata.update({
                    "dynamic_analyzed": True,
                    "dynamic_dict_used": result.dynamic_dict_used is not None,
                    "dynamic_compression_ratio": (len(user_content) - len(result.text)) / len(user_content) if user_content else 0,
                    "dynamic_dict_path": result.dynamic_dict_used,
                    "dynamic_tokens_used": len(result.used),
                })
                
                logger.debug(f"Dynamic compression applied: {metadata['dynamic_compression_ratio']*100:.1f}% compression")
                
                # CRITICAL FIX: Actually apply compression to user messages
                if result.used and metadata['dynamic_compression_ratio'] > 0:
                    compressed_messages = self._apply_compression_to_messages(messages, result.used)
                    logger.debug(f"Compression applied to {len([m for m in messages if m.get('role') == 'user'])} user messages")
                    return metadata, compressed_messages
            
        except Exception as e:
            logger.error(f"Error in dynamic dictionary processing: {e}")
            metadata["dynamic_error"] = str(e)
        
        return metadata, None
    
    def _apply_compression_to_messages(self, messages: List[Dict[str, Any]], used_dict: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Apply compression dictionary to user messages.
        
        Args:
            messages: Original messages
            used_dict: Dictionary of substitutions (symbol -> original_text)
            
        Returns:
            Messages with compressed user content
        """
        compressed_messages = []
        
        # Sort replacements by original text length (longest first) to prevent substring conflicts
        sorted_replacements = sorted(used_dict.items(), key=lambda x: len(x[1]), reverse=True)
        
        for msg in messages:
            if msg.get("role") == "user":
                # Apply compression to user content
                compressed_content = msg.get("content", "")
                if compressed_content:
                    # Apply all substitutions in order
                    for symbol, original_text in sorted_replacements:
                        compressed_content = compressed_content.replace(original_text, symbol)
                    
                    # Create new message with compressed content
                    compressed_msg = msg.copy()
                    compressed_msg["content"] = compressed_content
                    compressed_messages.append(compressed_msg)
                else:
                    compressed_messages.append(msg.copy())
            else:
                # Keep non-user messages unchanged
                compressed_messages.append(msg.copy())
        
        return compressed_messages
    
    def _enhance_compression_prompt_with_dynamic_info(self, 
                                                    compression_prompt: str, 
                                                    dynamic_metadata: Dict[str, Any]) -> str:
        """
        Enhance the compression prompt with dynamic dictionary information.
        
        Args:
            compression_prompt: Original compression prompt
            dynamic_metadata: Dynamic processing metadata
            
        Returns:
            Enhanced prompt with dynamic dictionary information
        """
        # Dynamic dictionary functionality is active but we don't add verbose messages to the system prompt
        # The compression is still applied to the messages, just without the verbose notification
        return compression_prompt

    def _intercept_system_prompts(self, 
                                messages: List[Dict[str, Any]], 
                                system_param: Optional[str] = None,
                                system_instruction: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Intercept system prompts from various sources in the request.
        
        Returns list of intercepted prompts with metadata about their source and format.
        """
        intercepted = []
        
        _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", "🔍 Scanning for system prompts in request...")
        
        # Intercept from system parameter (Claude-style)
        if system_param:
            _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                f"Found system parameter (Claude-style): {len(system_param)} chars")
            intercepted.append({
                "content": system_param,
                "source": "system_parameter",
                "original_format": "system_parameter"
            })
            _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                f"System parameter content preview: {system_param[:150]}...")
        
        # Intercept from system_instruction field (Gemini-style)  
        if system_instruction:
            _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                f"Found system_instruction (Gemini-style): {len(str(system_instruction))} chars")
            intercepted.append({
                "content": system_instruction,
                "source": "system_instruction", 
                "original_format": "system_instruction"
            })
            _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                f"System instruction content preview: {str(system_instruction)[:150]}...")
        
        # Intercept from messages array
        system_messages_found = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_messages_found += 1
                # Ensure content is always a string
                content = msg.get("content", "")
                if content is None:
                    content = ""
                _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                    f"Found system message #{system_messages_found} at index {i}: {len(content)} chars")
                intercepted.append({
                    "content": content,
                    "source": "messages_array",
                    "original_format": "chatml",
                    "message_index": i
                })
                _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                    f"System message #{system_messages_found} preview: {content[:150]}...")
        
        _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
            f"✅ System prompt interception complete: {len(intercepted)} prompts found")
        if not intercepted:
            _log_verbose_system_prompt_phase("INTERCEPTION_DETAILS", 
                "ℹ️  No existing system prompts found - will use compression decoder only")
        
        return intercepted
    
    def _detect_formats(self, intercepted_prompts: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Detect the format of each intercepted system prompt.
        
        Returns mapping of prompt index to detected format name.
        """
        format_detection = {}
        
        for i, prompt in enumerate(intercepted_prompts):
            content = prompt["content"]
            source = prompt["source"]
            
            # Start with source-based detection
            if source == "system_parameter":
                detected_format = "claude"
            elif source == "system_instruction":
                detected_format = "gemini"  
            elif source == "messages_array":
                detected_format = "chatgpt"  # Default for ChatML
            else:
                detected_format = "chatgpt"  # Safe fallback
            
            # Try to refine detection with content analysis
            refined_format = self._analyze_content_format(content)
            if refined_format:
                detected_format = refined_format
                
            format_detection[str(i)] = detected_format
            logger.debug(f"Detected format '{detected_format}' for prompt {i} from {source}")
        
        return format_detection
    
    def _analyze_content_format(self, content: str) -> Optional[str]:
        """
        Analyze content to detect specific format patterns.
        
        Returns format name if detected, None if unclear.
        """
        # Look for Gemma-style turn markers
        if "<start_of_turn>" in content and "<end_of_turn>" in content:
            return "gemma"
        
        # Look for Claude legacy format patterns
        if content.startswith("Human:") and "Assistant:" in content:
            return "claude_legacy"
        
        # Look for structured formats (JSON-like)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "parts" in parsed:
                return "gemini"
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Look for other format indicators
        if "role:" in content and ("system" in content or "user" in content):
            return "chatgpt"
        
        return None
    
    def _merge_system_prompts(self, 
                            intercepted_prompts: List[Dict[str, Any]], 
                            compression_prompt: str,
                            detected_formats: Dict[str, str]) -> str:
        """
        Intelligently merge intercepted system prompts with compression instructions.
        
        Priority order:
        1. KrunchWrapper compression instructions (highest priority)
        2. User's original system prompt content
        3. Format-specific structural requirements
        """
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"🔀 Starting intelligent system prompt merging...")
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"Input: {len(intercepted_prompts)} user prompts + compression decoder ({len(compression_prompt)} chars)")
        
        if not intercepted_prompts:
            _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                "No user system prompts to merge - using compression decoder only")
            return compression_prompt
        
        # Extract and clean intercepted content
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            "🧹 Cleaning and extracting user system prompt content...")
        user_prompts = []
        for i, prompt in enumerate(intercepted_prompts):
            content = prompt["content"]
            format_name = detected_formats.get(str(i), "chatgpt")
            
            _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                f"Processing user prompt #{i+1} (format: {format_name}, {len(content)} chars)")
            
            # Clean content based on detected format
            cleaned_content = self._clean_content_for_format(content, format_name)
            if cleaned_content.strip():
                user_prompts.append(cleaned_content.strip())
                _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                    f"✅ User prompt #{i+1} cleaned: {len(content)} → {len(cleaned_content)} chars")
                _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                    f"User prompt #{i+1} preview: {cleaned_content[:150]}...")
            else:
                _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                    f"⚠️  User prompt #{i+1} is empty after cleaning - skipping")
        
        # Merge with prioritization
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"🏗️  Building merged system prompt with priority ordering...")
        merged_parts = []
        
        # 1. KrunchWrapper compression instructions (highest priority)
        merged_parts.append(compression_prompt.strip())
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"[PRIORITY 1] Added compression decoder: {len(compression_prompt)} chars")
        
        # 2. User's original system prompts
        for i, user_prompt in enumerate(user_prompts):
            if user_prompt and user_prompt not in merged_parts[0]:  # Avoid duplication
                merged_parts.append(user_prompt)
                _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                    f"[PRIORITY 2] Added user prompt #{i+1}: {len(user_prompt)} chars")
            else:
                _log_verbose_system_prompt_phase("MERGING_DETAILS", 
                    f"[PRIORITY 2] Skipped user prompt #{i+1}: duplicate or empty")
        
        # Join with proper spacing
        final_merged = "\n\n".join(merged_parts)
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"✅ System prompt merging complete: {len(merged_parts)} parts → {len(final_merged)} chars total")
        _log_verbose_system_prompt_phase("MERGING_DETAILS", 
            f"Final merged preview: {final_merged[:300]}...")
        
        return final_merged
    
    def _clean_content_for_format(self, content: str, format_name: str) -> str:
        """
        Clean content by removing format-specific structural elements.
        """
        # Ensure content is always a string
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
            
        if format_name == "gemma":
            # Remove Gemma turn markers
            content = re.sub(r'<start_of_turn>.*?\n', '', content)
            content = re.sub(r'<end_of_turn>', '', content)
            
        elif format_name == "claude_legacy":
            # Remove legacy Claude format markers
            content = re.sub(r'^Human:\s*', '', content)
            content = re.sub(r'\n\nAssistant:\s*$', '', content)
            
        elif format_name == "gemini":
            # Handle structured Gemini format
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "parts" in parsed:
                    parts = parsed["parts"]
                    if isinstance(parts, list) and len(parts) > 0:
                        first_part = parts[0]
                        if isinstance(first_part, dict) and "text" in first_part:
                            content = first_part["text"]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass  # Keep original content if parsing fails
        
        # Ensure we always return a string
        cleaned = content.strip() if content else ""
        return cleaned
    
    def _convert_to_target_format(self, content: str, target_format: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert merged content to the target format.
        """
        try:
            self.formatter.set_format(target_format)
            return self.formatter.format_system_prompt(content)
        except Exception as e:
            logger.error(f"Error converting to format '{target_format}': {e}")
            # Fallback to ChatML format
            self.formatter.set_format("chatgpt")
            return self.formatter.format_system_prompt(content)
    
    def _apply_system_prompt(self, 
                           messages: List[Dict[str, Any]], 
                           content: str, 
                           metadata: Dict[str, Any],
                           target_format: str) -> List[Dict[str, Any]]:
        """
        Apply the processed system prompt to the messages array.
        """
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"📝 Applying processed system prompt to message array...")
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"Input: {len(messages)} messages, system prompt: {len(content)} chars, target format: {target_format}")
        
        # Ensure content is always a valid string
        if content is None:
            content = ""
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                "⚠️  System prompt content was None - using empty string")
        if not isinstance(content, str):
            content = str(content)
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"⚠️  System prompt content was not string - converted to: {type(content).__name__}")
            
        processed_messages = messages.copy()
        
        # Count and log existing system messages before removal
        existing_system_msgs = [msg for msg in processed_messages if msg.get("role") == "system"]
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"Found {len(existing_system_msgs)} existing system messages to replace")
        if existing_system_msgs:
            for i, msg in enumerate(existing_system_msgs):
                existing_content = msg.get("content", "")
                _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                    f"Existing system message #{i+1}: {len(existing_content)} chars - {existing_content[:100]}...")
        
        # Remove existing system messages
        processed_messages = [msg for msg in processed_messages if msg.get("role") != "system"]
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"Removed existing system messages: {len(messages)} → {len(processed_messages)} messages")
        
        # Handle different format requirements
        format_type = metadata.get("format", "chatml")
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"Applying system prompt using format type: {format_type}")
        
        if format_type in ["chatml", "messages", "contents"]:
            # Standard system message approach
            system_role = metadata.get("system_role", "system")
            system_msg = {"role": system_role, "content": content}
            processed_messages.insert(0, system_msg)
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"✅ Added system message with role '{system_role}' at position 0")
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"System message preview: {content[:200]}...")
            
        elif format_type == "template":
            # Template formats like Gemma - handled at API level typically
            # For now, add as system message for compatibility
            system_msg = {"role": "system", "content": content}
            processed_messages.insert(0, system_msg)
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"✅ Added template-format system message at position 0 (will be handled by API layer)")
            
        elif format_type == "plain":
            # Plain formats like legacy Claude - content already formatted
            system_msg = {"role": "system", "content": content}
            processed_messages.insert(0, system_msg)
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"✅ Added plain-format system message at position 0")
        
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"🚀 System prompt application complete - ready for forwarding to target LLM")
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            f"Final message array: {len(processed_messages)} messages")
        
        # Log final message structure for debugging
        _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
            "Final message structure:")
        for i, msg in enumerate(processed_messages[:3]):  # Show first 3 messages
            role = msg.get("role", "unknown")
            content_length = len(msg.get("content", ""))
            content_preview = msg.get("content", "")[:100]
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"  Message {i}: [{role}] {content_length} chars - {content_preview}...")
        if len(processed_messages) > 3:
            _log_verbose_system_prompt_phase("APPLICATION_DETAILS", 
                f"  ... and {len(processed_messages) - 3} more messages")
        
        return processed_messages
    
    def _apply_fallback(self, 
                       messages: List[Dict[str, Any]], 
                       rule_union: Dict[str, str], 
                       lang: str, 
                       target_format: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fallback mechanism when interception fails.
        Applies only KrunchWrapper compression instructions.
        """
        logger.warning("Applying fallback system prompt processing")
        
        try:
            # Extract user content for conciseness instructions
            user_content = self._extract_user_content(messages)
            
            # Build compression prompt only
            # Use configured stateful mode for consistency
            stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
            compression_prompt, metadata = build_system_prompt(rule_union, lang, target_format, user_content, cline_mode=False, stateful_mode=stateful_mode, new_symbols_only=None)
            
            # Apply to messages
            processed_messages = self._apply_system_prompt(
                messages, compression_prompt, metadata, target_format
            )
            
            return processed_messages, metadata
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            # Ultimate fallback: return original messages
            return messages, {}
    
    def detect_request_format(self, request_data: Dict[str, Any]) -> str:
        """
        Detect the format of an incoming API request.
        
        This can help determine what format the client expects in the response.
        """
        # Check for format-specific request structures
        if "system" in request_data:
            return "claude"  # Separate system parameter
            
        if "system_instruction" in request_data:
            return "gemini"  # Gemini-style system instruction
            
        messages = request_data.get("messages", [])
        if messages:
            # Look for system message patterns
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    detected = self._analyze_content_format(content)
                    if detected:
                        return detected
                    # Check if this is explicitly ChatML format
                    if self._is_chatml_format(content):
                        return "chatml"
                    return "chatgpt"  # Default ChatML
        
        return "chatgpt"  # Safe default 
        
    def _is_chatml_format(self, content: str) -> bool:
        """
        Check if content appears to be in ChatML format.
        
        ChatML typically has messages with roles and content.
        """
        # Check for ChatML indicators
        if '"role"' in content and '"content"' in content:
            return True
        
        # Check for structured message format
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        return True
        except (json.JSONDecodeError, TypeError):
            pass
            
        return False 

    def _estimate_compression_ratio(self, messages: List[Dict[str, Any]], rule_union: Dict[str, str]) -> float:
        """
        Estimate compression ratio from already-compressed messages and rule_union.
        
        Args:
            messages: List of compressed messages
            rule_union: Dictionary of substitutions used
            
        Returns:
            Estimated compression ratio as a float
        """
        if not rule_union:
            return 0.0
        
        # Extract current compressed content
        compressed_content = ""
        for msg in messages:
            if msg.get("role") in {"user", "assistant"}:
                content = msg.get("content", "")
                if content:
                    compressed_content += content + "\n"
        
        # Estimate original size by expanding the compressed content
        estimated_original_content = compressed_content
        # Sort by original text length (longest first) to prevent substring conflicts
        sorted_replacements = sorted(rule_union.items(), key=lambda x: len(x[1]), reverse=True)
        for symbol, original_text in sorted_replacements:
            estimated_original_content = estimated_original_content.replace(symbol, original_text)
        
        # Calculate estimated compression ratio
        if len(estimated_original_content) > 0:
            compression_ratio = (len(estimated_original_content) - len(compressed_content)) / len(estimated_original_content)
            return max(0.0, min(1.0, compression_ratio))  # Clamp between 0 and 1
        
        return 0.0 