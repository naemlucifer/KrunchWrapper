import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from .system_prompt_interceptor import SystemPromptInterceptor, _log_verbose_system_prompt_phase
from .system_prompt import SystemPromptFormatter, build_system_prompt
from .async_logger import get_performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cline_intercept')

class ClineSystemPromptInterceptor(SystemPromptInterceptor):
    """
    Cline-specific system prompt interceptor that handles the way cline sends API requests
    to various LLM providers. Cline uses a unified OpenAI-compatible format for all providers
    and handles provider-specific formatting internally.
    
    IMPORTANT: This class ONLY differs in system prompt format detection and handling.
    All compression logic is inherited from SystemPromptInterceptor, which uses the same
    core dynamic compression (compress_with_dynamic_analysis) as standard mode.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.last_compression_rules = {}  # Store last compression rules for conversation persistence
        self.request_used_symbols = set()  # CRITICAL FIX: Track symbols used within current request
        # Add Cline-specific initialization logging
        logger.info("🔧 ClineSystemPromptInterceptor:  Initialized (extends SystemPromptInterceptor)")
        logger.info("🎯 Cline API formatting:         Ready")
        
    def intercept_and_process_cline(self, 
                            messages: List[Dict[str, Any]], 
                            rule_union: Dict[str, str], 
                            lang: str,
                            model_id: str = None,
                            system_param: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for cline-specific system prompt interception and processing.
        
        This method can use either:
        1. Cline-compatible approach (preserves original system prompt, adds compression decoder separately)
        2. Standard merging approach (merges compression decoder with existing system prompts)
        
        The behavior is controlled by the 'cline_preserve_system_prompt' config option.
        
        Args:
            messages: List of chat messages
            rule_union: Compression substitutions that were used
            lang: Programming language detected
            model_id: The model ID in cline format (provider/model)
            system_param: System parameter from request
            
        Returns:
            Tuple of (processed_messages, metadata)
        """
        try:
            # Check config option to determine which approach to use
            preserve_system_prompt = True  # Default to preserving for safety
            if self.config and hasattr(self.config, 'cline_preserve_system_prompt'):
                preserve_system_prompt = getattr(self.config, 'cline_preserve_system_prompt', True)
            
            approach_name = "preserve + separate" if preserve_system_prompt else "merge (standard)"
            _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                f"🔧 Starting Cline system prompt interception using '{approach_name}' approach")
            _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                f"Model ID: {model_id}, Messages: {len(messages)}, Compression rules: {len(rule_union)}")
            
            # Determine the target format based on the model_id
            target_format = self._determine_target_format(model_id)
            _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                f"Determined target format: {target_format} (based on model_id: {model_id})")
            
            if preserve_system_prompt:
                # NEW APPROACH: Preserve Cline's system prompt, add compression decoder separately
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"Using Cline-compatible approach: preserving original system prompt")
                
                # Build ONLY the compression decoder prompt (no merging with Cline's prompt)
                from .system_prompt import build_system_prompt
                user_content = self._extract_user_content(messages)
                
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"🔧 DEBUG: About to call build_system_prompt with user_content type: {type(user_content).__name__}")
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"🔧 DEBUG: user_content preview: {str(user_content)[:100]}...")
                
                # Use configured stateful mode from server config for optimal KV cache behavior
                stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
                compression_prompt, metadata = build_system_prompt(rule_union, lang, target_format, user_content, cline_mode=True, stateful_mode=stateful_mode, new_symbols_only=None)
                
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"🔧 DEBUG: build_system_prompt returned compression_prompt type: {type(compression_prompt).__name__}")
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"🔧 DEBUG: compression_prompt value: {str(compression_prompt)[:200]}...")
                
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"Built compression decoder prompt: {len(compression_prompt) if isinstance(compression_prompt, str) else 'NON-STRING'} chars")
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"Compression prompt preview: {str(compression_prompt)[:200]}...")
                
                # Apply Cline-compatible system prompt (preserve + add approach)
                processed_messages = self._apply_cline_compatible_system_prompt(
                    messages, compression_prompt, metadata, target_format
                )
                
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"✅ Cline-compatible system prompt interception complete")
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"Result: {len(messages)} → {len(processed_messages)} messages")
                
                return processed_messages, metadata
                
            else:
                # OLD APPROACH: Use standard interception process (merging behavior)
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"Using standard merging approach: compression decoder + Cline prompt merged")
                
                # Use the standard interception process with the determined target format
                result = self.intercept_and_process(
                    messages, rule_union, lang, target_format, system_param, None
                )
                
                _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                    f"✅ Standard merging system prompt interception complete")
                
                return result
            
        except Exception as e:
            _log_verbose_system_prompt_phase("CLINE_INTERCEPTION", 
                f"❌ Error in Cline system prompt interception: {e}")
            logger.error(f"Error in cline system prompt interception: {e}")
            # Fallback: Apply compression prompt only
            return self._apply_fallback(messages, rule_union, lang, "chatgpt")
    
    def _determine_target_format(self, model_id: str) -> str:
        """
        Determine the appropriate system prompt format based on the cline model ID.
        
        Args:
            model_id: The model ID in cline format (provider/model)
            
        Returns:
            The target format to use
        """
        _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
            f"🎯 Determining target format for model: {model_id}")
        
        if not model_id:
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "No model_id provided - defaulting to ChatML format")
            return "chatml"  # Default to ChatML format
            
        # Extract provider from model_id (format is typically provider/model)
        parts = model_id.split('/')
        provider = parts[0].lower() if len(parts) > 0 else ""
        model_name = parts[1] if len(parts) > 1 else ""
        
        _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
            f"Parsed model_id - Provider: '{provider}', Model: '{model_name}'")
        
        # Map provider to appropriate format
        target_format = "chatml"  # Default fallback
        
        if provider == "anthropic":
            target_format = "claude"
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Anthropic provider → Claude format")
        elif provider in ["google", "gemini"]:
            target_format = "gemini"
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Google/Gemini provider → Gemini format")
        elif provider == "openai" or provider == "openai-native":
            target_format = "chatml"  # OpenAI uses ChatML format
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected OpenAI provider → ChatML format")
        elif provider == "deepseek":
            if model_id.startswith("deepseek/deepseek-r1"):
                # Special case for DeepSeek Reasoner models
                target_format = "deepseek"
                _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                    "Detected DeepSeek R1 model → DeepSeek format")
            else:
                target_format = "chatml"  # Standard DeepSeek uses ChatML
                _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                    "Detected DeepSeek (non-R1) → ChatML format")
        elif provider == "qwen":
            target_format = "qwen"
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Qwen provider → Qwen format")
        elif provider == "gemma":
            target_format = "gemma"
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Gemma provider → Gemma format")
        elif provider == "mistral":
            target_format = "chatml"  # Mistral uses ChatML format
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Mistral provider → ChatML format")
        elif provider == "x-ai" or provider == "xai":
            target_format = "chatml"  # Grok uses ChatML format
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected xAI/Grok provider → ChatML format")
        elif provider == "ollama":
            target_format = "chatml"  # Ollama uses ChatML format
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                "Detected Ollama provider → ChatML format")
        elif provider == "bedrock":
            # For AWS Bedrock, we need to check the model name
            if len(parts) > 1 and "claude" in parts[1].lower():
                target_format = "claude"  # Claude models on Bedrock
                _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                    "Detected Bedrock Claude model → Claude format")
            else:
                target_format = "chatml"  # Default for other Bedrock models
                _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                    "Detected Bedrock (non-Claude) → ChatML format")
        else:
            _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
                f"Unknown provider '{provider}' → defaulting to ChatML format")
        
        _log_verbose_system_prompt_phase("CLINE_FORMAT_DETECTION", 
            f"✅ Target format determined: {target_format}")
        
        return target_format
    
    def process_cline_request(self, request_data: Dict[str, Any], rule_union: Dict[str, str], lang: str) -> Dict[str, Any]:
        """
        Process a cline API request.
        
        Args:
            request_data: The original request data
            rule_union: Compression substitutions that were used
            lang: Programming language detected
            
        Returns:
            Modified request data with processed system prompt
        """
        try:
            # CRITICAL FIX: Don't clear compression rules here - they need to persist for decompression
            # self.last_compression_rules = {}  # REMOVED - This was causing decompression failures
            
            # CRITICAL FIX: Clear request-scoped symbol tracking for new request
            self.request_used_symbols = set()
            # Add any existing symbols from rule_union to prevent collisions
            if rule_union:
                self.request_used_symbols.update(rule_union.keys())
                logger.debug(f"🔧 Initialized request with {len(rule_union)} existing symbols to prevent collisions")
            
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"🚀 Processing Cline API request")
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Incoming rule_union: {len(rule_union)} rules")
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Request keys: {list(request_data.keys())}")
            
            # Extract model_id if present
            model_id = request_data.get("model", "")
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Model ID: {model_id}")
            
            # Extract messages and system prompt
            messages = request_data.get("messages", [])
            system_prompt = None
            
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Found {len(messages)} messages in request")
            
            # Check if there's a separate system parameter
            if "system" in request_data:
                system_prompt = request_data["system"]
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"Found separate system parameter: {len(system_prompt)} chars")
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"System parameter preview: {system_prompt[:150]}...")
            else:
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    "No separate system parameter found")
            
            # Process messages and system prompt
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Calling Cline system prompt interception...")
            processed_messages, metadata = self.intercept_and_process_cline(
                messages, rule_union, lang, model_id, system_prompt
            )
            
            # Update request with processed data
            result = request_data.copy()
            result["messages"] = processed_messages
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Updated request messages: {len(messages)} → {len(processed_messages)}")
            
            # If using Claude format and there's a system message at the beginning,
            # move it to the system parameter
            target_format = self._determine_target_format(model_id)
            if target_format == "claude" and processed_messages and processed_messages[0].get("role") == "system":
                system_content = processed_messages[0].get("content", "")
                result["system"] = system_content
                result["messages"] = processed_messages[1:]
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"🔄 Claude format detected - moved system message to system parameter")
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"System parameter: {len(system_content)} chars")
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"Messages after system extraction: {len(result['messages'])}")
            
            # CRITICAL FIX: Ensure system prompt compression rules are available for decompression
            # Update the rule_union that was passed in with any new system prompt rules
            if self.last_compression_rules:
                rule_union.update(self.last_compression_rules)
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"🔧 Updated rule_union with {len(self.last_compression_rules)} system prompt rules")
                _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                    f"🔧 Total rule_union now has {len(rule_union)} rules for decompression")
                # CRITICAL: Clear last_compression_rules AFTER updating rule_union
                self.last_compression_rules = {}
            
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"✅ Cline request processing complete - ready for forwarding")
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"Final request structure: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"❌ Error processing Cline request: {e}")
            logger.error(f"Error processing cline request: {e}")
            # Return original request on error
            _log_verbose_system_prompt_phase("CLINE_REQUEST_PROCESSING", 
                f"⚠️  Returning original request due to error")
            return request_data 

    def _apply_cline_compatible_system_prompt(self, 
                                           messages: List[Dict[str, Any]], 
                                           compression_prompt: str, 
                                           metadata: Dict[str, Any], 
                                           target_format: str) -> List[Dict[str, Any]]:
        """
        Apply Cline-compatible system prompt processing.
        
        This preserves Cline's original system prompt and adds compression decoder separately,
        ensuring compatibility with Cline's expected message structure.
        
        Args:
            messages: List of chat messages
            compression_prompt: The compression decoder prompt
            metadata: System prompt metadata
            target_format: Target format for the system prompt
            
        Returns:
            Processed messages with Cline-compatible system prompt
        """
        try:
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"🛡️  Applying Cline-compatible system prompt processing")
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"Input: {len(messages)} messages, compression prompt: {len(compression_prompt) if isinstance(compression_prompt, str) else type(compression_prompt).__name__} chars")
            
            processed_messages = messages.copy()
            
            # CRITICAL FIX: Ensure compression_prompt is always a string
            if not isinstance(compression_prompt, str):
                _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                    f"⚠️ WARNING: compression_prompt is {type(compression_prompt).__name__}, converting to string")
                if compression_prompt is None:
                    compression_prompt = ""
                elif isinstance(compression_prompt, (list, tuple)):
                    compression_prompt = " ".join(str(item) for item in compression_prompt)
                else:
                    compression_prompt = str(compression_prompt)
            
            # If there's no compression prompt, just return original messages
            if not compression_prompt or not compression_prompt.strip():
                _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                    "No compression prompt - returning original messages unchanged")
                return processed_messages
            
            # Find existing system messages
            existing_system_messages = []
            non_system_messages = []
            
            for msg in processed_messages:
                if msg.get("role") == "system":
                    existing_system_messages.append(msg)
                else:
                    non_system_messages.append(msg)
            
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"Found {len(existing_system_messages)} existing system messages")
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"Found {len(non_system_messages)} non-system messages")
            
            # Create the final message array
            final_messages = []
            
            # 1. Preserve Cline's original system prompts first (but compress them!)
            for i, sys_msg in enumerate(existing_system_messages):
                original_content = sys_msg.get("content", "")
                _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                    f"Processing original system message #{i+1}: {len(original_content)} chars")
                
                # CRITICAL: Compress the Cline system prompt content itself
                if original_content and len(original_content) > 100:  # Only compress substantial system prompts
                    try:
                        from core.compress import compress_with_dynamic_analysis
                        # CRITICAL FIX: Pass excluded symbols to prevent collisions within same request
                        packed = compress_with_dynamic_analysis(
                            original_content, 
                            skip_tool_detection=False, 
                            cline_mode=True,
                            exclude_symbols=self.request_used_symbols
                        )
                        compressed_content = packed.text
                        # Update the rule union with new compressions
                        for symbol, pattern in packed.used.items():
                            if symbol not in metadata.get('rule_union', {}):  # Avoid conflicts
                                metadata.setdefault('rule_union', {})[symbol] = pattern
                                # CRITICAL FIX: Store for conversation persistence AND immediate decompression
                                self.last_compression_rules[symbol] = pattern
                                # CRITICAL FIX: Track symbol usage within this request
                                self.request_used_symbols.add(symbol)
                        
                        _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                            f"✅ Compressed system message #{i+1}: {len(original_content)} → {len(compressed_content)} chars")
                        _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                            f"🔧 Added {len(packed.used)} new symbols to request exclusion list")
                        
                        # Use compressed content
                        compressed_sys_msg = sys_msg.copy()
                        compressed_sys_msg["content"] = compressed_content
                        final_messages.append(compressed_sys_msg)
                    except Exception as e:
                        _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                            f"⚠️ System prompt compression failed for message #{i+1}: {e}")
                        # Use original if compression fails
                        final_messages.append(sys_msg.copy())
                else:
                    # Too short to compress or empty
                    _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                        f"Preserving system message #{i+1} (too short to compress): {len(original_content)} chars")
                    final_messages.append(sys_msg.copy())
            
            # 2. Add compression decoder as a separate system message (updated with all compression rules)
            # Rebuild the compression prompt with any new rules from system prompt compression
            updated_rule_union = metadata.get('rule_union', {})
            if updated_rule_union:
                try:
                    from .system_prompt import build_system_prompt
                    user_content = self._extract_user_content([m for m in final_messages if m.get('role') != 'system'])
                    # Use configured stateful mode for consistency
                    stateful_mode = getattr(self.config, 'conversation_stateful_mode', False) if self.config else False
                    updated_compression_prompt, updated_metadata = build_system_prompt(updated_rule_union, "generic", target_format, user_content, cline_mode=True, stateful_mode=stateful_mode, new_symbols_only=None)
                    
                    # CRITICAL FIX: Ensure updated_compression_prompt is always a string
                    if not isinstance(updated_compression_prompt, str):
                        _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                            f"⚠️ WARNING: updated_compression_prompt is {type(updated_compression_prompt).__name__}, converting to string")
                        if updated_compression_prompt is None:
                            updated_compression_prompt = ""
                        elif isinstance(updated_compression_prompt, (list, tuple)):
                            updated_compression_prompt = " ".join(str(item) for item in updated_compression_prompt)
                        else:
                            updated_compression_prompt = str(updated_compression_prompt)
                    
                    compression_system_msg = {
                        "role": "system", 
                        "content": updated_compression_prompt.strip()
                    }
                    _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                        f"Added updated compression decoder with {len(updated_rule_union)} rules: {len(updated_compression_prompt)} chars")
                except Exception as e:
                    # Fallback to original compression prompt
                    compression_system_msg = {
                        "role": "system", 
                        "content": compression_prompt.strip()
                    }
                    _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                        f"⚠️ Failed to update compression decoder ({e}), using original: {len(compression_prompt)} chars")
            else:
                # Use original compression prompt
                compression_system_msg = {
                    "role": "system", 
                    "content": compression_prompt.strip()
                }
                _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                    f"Added original compression decoder: {len(compression_prompt)} chars")
            
            final_messages.append(compression_system_msg)
            
            # 3. Add all non-system messages
            final_messages.extend(non_system_messages)
            
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"✅ Cline-compatible processing complete: {len(messages)} → {len(final_messages)} messages")
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"Final structure: {len([m for m in final_messages if m.get('role') == 'system'])} system + {len([m for m in final_messages if m.get('role') != 'system'])} other messages")
            
            return final_messages
            
        except Exception as e:
            _log_verbose_system_prompt_phase("CLINE_COMPATIBLE_APPLICATION", 
                f"❌ Error in Cline-compatible system prompt processing: {e}")
            logger.error(f"Error in Cline-compatible system prompt processing: {e}")
            # Fallback: return original messages
            return messages 