from typing import Dict, NamedTuple, Optional, Any
import os
import pathlib
import logging
import sys
import json

from .dynamic_dictionary import DynamicDictionaryAnalyzer
from .model_tokenizer_validator import get_model_tokenizer_validator, validate_with_model
from .model_context import get_effective_model, normalize_model_name
from .comment_stripper import CommentStripper
from .tool_identifier import contains_tool_calls, compress_tool_call_content, decompress_tool_call_content
from .markdown_identifier import contains_markdown_content

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc

__all__ = ["Packed", "compress_with_dynamic_analysis", "compress_with_selective_tool_call_analysis", "decompress"]

logger = logging.getLogger(__name__)






def validate_tokenization_efficiency(original_text: str, compressed_text: str, used_dict: Dict[str, str], model_name: Optional[str] = None) -> float:
    """
    Validate that compression actually saves tokens by testing tokenization.
    
    Args:
        original_text: Original text
        compressed_text: Compressed text with Unicode symbols
        used_dict: Dictionary of substitutions used
        model_name: Optional model name for model-specific validation
        
    Returns:
        Actual token compression ratio (negative means compression made it worse)
    """
    # Try to get model from context if not provided
    if not model_name:
        model_name = get_effective_model()
    
    # Load configuration for model tokenizer settings
    try:
        config_path = pathlib.Path(__file__).parents[1] / "config" / "config.jsonc"
        config = load_jsonc(str(config_path))
        model_config = config.get("model_tokenizer", {})
        use_model_specific = model_config.get("enabled", True)
        fallback_method = model_config.get("fallback_method", "tiktoken")
        default_family = model_config.get("default_model_family", "gpt-4")
    except Exception as e:
        logger.warning(f"Failed to load model tokenizer config: {e}")
        use_model_specific = True
        fallback_method = "tiktoken"
        default_family = "gpt-4"
    
    # Try model-specific validation first if enabled and model is available
    if use_model_specific and model_name:
        try:
            normalized_model = normalize_model_name(model_name)
            logger.debug(f"Attempting model-specific validation for: {normalized_model}")
            
            validator = get_model_tokenizer_validator()
            result = validator.validate_token_efficiency(original_text, compressed_text, normalized_model)
            
            if result.get("method") == "model_specific":
                token_compression_ratio = result.get("token_ratio", 0.0)
                
                # Consolidated logging: all validation info in one place
                logger.debug(f"Token validation - Model: {result.get('model_family', 'unknown')}, "
                           f"Tokenizer: {result.get('tokenizer_type', 'unknown')}")
                logger.debug(f"Tokens: {result.get('original_tokens', 0)} → {result.get('compressed_tokens', 0)} "
                           f"({token_compression_ratio*100:.2f}% compression)")
                logger.debug(f"Characters: {len(original_text)} → {len(compressed_text)} chars, "
                           f"{len(used_dict)} substitutions")
                
                # Additional debug: Show first few replacements to understand what's happening
                if used_dict and token_compression_ratio < 0:
                    logger.warning("Negative compression detected! Showing first few replacements:")
                    for i, (symbol, token) in enumerate(list(used_dict.items())[:5]):
                        logger.warning(f"  '{token}' → '{symbol}' (lengths: {len(token)} → {len(symbol)})")
                
                return token_compression_ratio
            else:
                logger.debug(f"Model-specific validation failed, result method: {result.get('method')}")
        except Exception as e:
            logger.warning(f"Model-specific validation failed for {model_name}: {e}")
    
    # Fallback to generic tiktoken validation
    if fallback_method == "tiktoken":
        try:
            import tiktoken
            
            # Use a common tokenizer for validation - GPT-4 tokenizer is widely compatible
            tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Count tokens in original and compressed text
            original_tokens = len(tokenizer.encode(original_text))
            compressed_tokens = len(tokenizer.encode(compressed_text))
            
            # Calculate actual token compression ratio
            if original_tokens > 0:
                token_compression_ratio = (original_tokens - compressed_tokens) / original_tokens
                
                # Consolidated logging: all validation info in one place  
                logger.debug(f"Token validation - Fallback: tiktoken, "
                           f"Tokens: {original_tokens} → {compressed_tokens} ({token_compression_ratio*100:.2f}% compression)")
                logger.debug(f"Characters: {len(original_text)} → {len(compressed_text)} chars, "
                           f"{len(used_dict)} substitutions")
                
                # Additional debug: Show first few replacements to understand what's happening
                if used_dict and token_compression_ratio < 0:
                    logger.warning("Negative compression detected! Showing first few replacements:")
                    for i, (symbol, token) in enumerate(list(used_dict.items())[:5]):
                        logger.warning(f"  '{token}' → '{symbol}' (lengths: {len(token)} → {len(symbol)})")
                
                return token_compression_ratio
            else:
                return 0.0
                
        except ImportError:
            logger.warning("tiktoken not available for token validation, using character estimation")
        except Exception as e:
            logger.error(f"Error during tiktoken validation: {e}")
    
    # Final fallback to character-based estimation
    if len(original_text) > 0:
        char_ratio = (len(original_text) - len(compressed_text)) / len(original_text)
        logger.info(f"Character-based validation fallback - Character compression ratio: {char_ratio*100:.2f}%")
        return char_ratio
    else:
        return 0.0

class Packed(NamedTuple):
    """A container returned by compression functions."""
    text: str
    used: Dict[str, str]  # mapping of *short* -> *long*
    language: Optional[str]  # language used for compression
    fallback_used: bool  # whether fallback multi-language compression was used
    dynamic_dict_used: Optional[str] = None  # path to dynamic dictionary if used
    comment_stats: Optional[Dict] = None  # comment stripping statistics
    original_content_structure: Optional[Any] = None  # preserve original multimodal structure


def compress_with_dynamic_analysis(src: str, skip_tool_detection: bool = False, cline_mode: bool = False, exclude_symbols: set = None) -> Packed:
    """
    Compress using dynamic analysis with potential tool call awareness.
    
    Args:
        src: Source text to compress
        skip_tool_detection: Skip tool call detection for performance
        cline_mode: Enable Cline-specific optimizations  
        exclude_symbols: Set of symbols to exclude from assignment (prevents collisions)
    
    Returns:
        Packed result with compressed text and substitutions
    """
    logger.debug("Starting dynamic dictionary analysis and compression")
    
    # CRITICAL SAFETY: Validate input to prevent compression of server code
    # This prevents the compression system from corrupting the server's own execution
    if not src or not isinstance(src, str):
        logger.warning("Invalid input to compression: not a string")
        return Packed(src or "", {}, None, False, None, None, None)
    
    # No safety restrictions - users can compress anything they want!
    # This is a text compression service, not a security-critical system
    
    # Step 0: Check for tool calls and skip compression if found (unless disabled for Cline compatibility)
    if not skip_tool_detection and not cline_mode and contains_tool_calls(src):
        logger.debug("Tool calls detected, skipping compression to preserve JSON structure")
        return Packed(src, {}, None, False, None, None, None)
    elif not skip_tool_detection and cline_mode and contains_tool_calls(src):
        logger.debug("Cline mode: allowing compression despite tool call examples in system prompt")
    
    # Step 0.5: Check for markdown content and skip compression if found
    # UNLESS we're in Cline mode, where we allow compression of system prompts with markdown
    if not cline_mode and contains_markdown_content(src):
        logger.debug("Markdown content detected, skipping compression to preserve formatting")
        return Packed(src, {}, None, False, None, None, None)
    elif cline_mode and contains_markdown_content(src):
        logger.debug("Markdown content detected in Cline mode - allowing compression with structure preservation")
    
    # Step 1: Strip comments if enabled (before compression analysis)
    comment_stripper = CommentStripper()
    processed_src, comment_stats = comment_stripper.strip_comments(src)
    
    # Use the processed source for the rest of the compression pipeline
    working_src = processed_src
    
    # Initialize dynamic dictionary analyzer (loads config internally) - use cached instance
    from .dynamic_dictionary import get_dynamic_dictionary_analyzer
    analyzer = get_dynamic_dictionary_analyzer()
    
    # Check if dynamic analysis is enabled and worthwhile
    if not analyzer.is_enabled():
        logger.debug("Dynamic dictionary analysis is disabled, returning uncompressed")
        # Return with comment stripping applied if it was enabled
        if comment_stats.get("enabled", False):
            return Packed(working_src, {}, None, False, None, comment_stats, None)
        else:
            return Packed(src, {}, None, False, None, comment_stats, None)
    
    should_analyze, reason = analyzer.should_analyze_prompt(working_src, skip_tool_detection, cline_mode=cline_mode)
    if not should_analyze:
        logger.debug(f"🚫 Dynamic analysis skipped: {reason}, returning uncompressed")
        logger.debug(f"📊 Content length: {len(working_src)} chars, Skip tool detection: {skip_tool_detection}")
        logger.debug(f"📝 Content preview: {working_src[:200]}...")
        # Return with comment stripping applied if it was enabled
        if comment_stats.get("enabled", False):
            return Packed(working_src, {}, None, False, None, comment_stats, None)
        else:
            return Packed(src, {}, None, False, None, comment_stats, None)
    
    # Analyze the source text (no language hint needed)
    # CRITICAL FIX: Pass exclude_symbols to prevent symbol collisions within same request
    analysis_result = analyzer.analyze_prompt(working_src, exclude_symbols=exclude_symbols)
    
    # Check if dynamic compression is worthwhile
    compression_ratio = analysis_result["compression_analysis"]["compression_ratio"]
    threshold = analyzer.config.get("compression_threshold", 0.01)  # Use config threshold or 1% default
    # Add small epsilon for floating point comparison tolerance
    epsilon = 1e-6
    if compression_ratio < (threshold - epsilon):
        logger.debug(f"Dynamic analysis shows minimal benefit ({compression_ratio*100:.2f}% < {threshold*100:.1f}%), returning uncompressed")
        # Log comment stripping savings even if compression is skipped
        if comment_stats.get("enabled", False):
            comment_chars_saved = comment_stats.get("chars_saved", 0)
            comment_tokens_saved = comment_stats.get("tokens_saved", 0)
            comment_language = comment_stats.get("language", "unknown")
            if comment_chars_saved > 0:
                logger.info(f"📝 Comment stripping saved {comment_chars_saved:,} chars, {comment_tokens_saved} tokens ({comment_language})")
            return Packed(working_src, {}, None, False, None, comment_stats, None)
        else:
            return Packed(src, {}, None, False, None, comment_stats, None)
    
    # Create temporary dictionary
    temp_dict_path = analyzer.create_temporary_dictionary(analysis_result)
    
    if not temp_dict_path:
        logger.debug("No temporary dictionary created, returning uncompressed")
        # Return with comment stripping applied if it was enabled
        if comment_stats.get("enabled", False):
            return Packed(working_src, {}, None, False, None, comment_stats, None)
        else:
            return Packed(src, {}, None, False, None, comment_stats, None)
    
    try:
        # Apply dynamic dictionary compression
        compressed_text, dynamic_used = _compress_with_temp_dictionary(working_src, temp_dict_path)
        
        # CRITICAL: Validate that compression actually saves tokens
        token_compression_ratio = validate_tokenization_efficiency(working_src, compressed_text, dynamic_used)
        
        # If token compression is poor, fall back to uncompressed
        token_threshold = analyzer.config.get("compression_threshold", 0.01)
        # Add small epsilon for floating point comparison tolerance
        epsilon = 1e-6
        
        # Make threshold more lenient if content appears to already be compressed
        # (detected by presence of Unicode symbols or very short average word length)
        avg_word_length = len(working_src.replace(' ', '')) / max(1, len(working_src.split()))
        unicode_symbol_count = sum(1 for char in working_src if ord(char) > 127)
        unicode_ratio = unicode_symbol_count / max(1, len(working_src))
        
        # If content appears already compressed, use a more lenient threshold
        if unicode_ratio > 0.01 or avg_word_length < 4:
            # Use half the normal threshold for potentially pre-compressed content
            effective_threshold = token_threshold * 0.5
            logger.info(f"Detected potentially compressed content (unicode_ratio={unicode_ratio:.3f}, avg_word_len={avg_word_length:.1f}), using lenient threshold: {effective_threshold*100:.1f}%")
        else:
            effective_threshold = token_threshold
        
        if token_compression_ratio < (effective_threshold - epsilon):
            logger.debug(f"Token validation failed: {token_compression_ratio*100:.2f}% < {effective_threshold*100:.1f}%, using standard compression")
            # Return with comment stripping applied if it was enabled
            if comment_stats.get("enabled", False):
                return Packed(working_src, {}, None, False, None, comment_stats, None)
            else:
                return Packed(src, {}, None, False, None, comment_stats, None)
        
        # Calculate final compression ratio (character-based for logging)
        final_ratio = (len(working_src) - len(compressed_text)) / len(working_src) if len(working_src) > 0 else 0
        
        # Include comment stripping statistics in the summary
        if comment_stats.get("enabled", False):
            comment_chars_saved = comment_stats.get("chars_saved", 0)
            comment_tokens_saved = comment_stats.get("tokens_saved", 0)
            comment_language = comment_stats.get("language", "unknown")
            
            # Calculate total savings including both comment stripping and compression
            total_chars_saved = (len(src) - len(compressed_text))
            total_original_chars = len(src)
            total_compression_ratio = total_chars_saved / total_original_chars if total_original_chars > 0 else 0
            
            logger.info(f"📝 Comment stripping saved {comment_chars_saved:,} chars, {comment_tokens_saved} tokens ({comment_language})")
            logger.info(f"🗜️  Dynamic compression achieved {final_ratio*100:.1f}% character compression, {token_compression_ratio*100:.1f}% token compression")
            logger.info("=" * 60)
            logger.info("📊 MESSAGE COMPRESSION SUMMARY")
            logger.info(f"📊 Total pipeline savings: {total_compression_ratio*100:.1f}% ({total_chars_saved:,} chars from {total_original_chars:,})")
            logger.info("=" * 60)
            logger.debug(f"Used {len(dynamic_used)} dynamic tokens")
        else:
            logger.info(f"Dynamic compression achieved {final_ratio*100:.1f}% character compression, {token_compression_ratio*100:.1f}% token compression")
            logger.debug(f"Used {len(dynamic_used)} dynamic tokens")
        
        return Packed(
            text=compressed_text,
            used=dynamic_used,
            language=None,  # No language detection needed
            fallback_used=False,
            dynamic_dict_used=temp_dict_path,
            comment_stats=comment_stats,
            original_content_structure=None
        )
        
    except Exception as e:
        logger.error(f"Error in dynamic compression: {e}")
        logger.debug("Returning uncompressed due to error")
        # Return with comment stripping applied if it was enabled
        if comment_stats.get("enabled", False):
            return Packed(working_src, {}, None, False, None, comment_stats, None)
        else:
            return Packed(src, {}, None, False, None, comment_stats, None)
    
    finally:
        # CRITICAL: Always cleanup temporary dictionaries to prevent persistence
        try:
            analyzer.cleanup_old_dictionaries()
            # CRITICAL: Also immediately clean up the current temporary dictionary
            if temp_dict_path and os.path.exists(temp_dict_path):
                os.unlink(temp_dict_path)
                logger.debug(f"🧹 Immediately cleaned up temporary dictionary: {temp_dict_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary dictionary: {cleanup_error}")
            # This is critical - if we can't clean up, at least log it
            logger.warning(f"CRITICAL: Temporary dictionary may persist: {temp_dict_path}")


def _compress_with_temp_dictionary(src: str, dict_path: str) -> tuple[str, Dict[str, str]]:
    """
    Compress text with a temporary dictionary with strict isolation.
    CRITICAL: This function MUST NOT affect global state or other parts of the system.
    """
    import json
    import copy
    
    # Create a completely isolated copy of the source to prevent any side effects
    isolated_src = str(src)  # Ensure we have a completely separate string object
    
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            table = data.get("tokens", {})
    except Exception as e:
        logger.error(f"Failed to load temporary dictionary {dict_path}: {e}")
        return isolated_src, {}
    
    # CRITICAL: Create completely isolated copies to prevent global contamination
    used: Dict[str, str] = {}
    out = isolated_src
    
    # CRITICAL FIX: Dictionary is in symbol -> pattern format
    # Sort by pattern length (longest first) to prevent substring interference
    # IMPORTANT: We need to sort by the VALUE (pattern) length, not key (symbol) length
    sorted_tokens = sorted(table.items(), key=lambda x: len(x[1]), reverse=True)
    
    # CRITICAL SAFETY: Apply replacements ONLY to our isolated string with proper validation
    replacement_count = 0
    for symbol, pattern in sorted_tokens:
        if pattern in out and len(pattern) >= 3:  # Safety: only replace meaningful patterns
            # Count occurrences before replacement for validation
            occurrences = out.count(pattern)
            if occurrences > 0:
                # Use string replace on the isolated copy only
                new_out = out.replace(pattern, symbol)
                if new_out != out and len(new_out) < len(out):  # Validate compression occurred
                    out = new_out
                    used[symbol] = pattern  # Store symbol -> pattern for decompression
                    replacement_count += 1
                    logger.debug(f"Replaced '{pattern}' with '{symbol}' ({occurrences} times)")
    
    logger.debug(f"Applied {replacement_count} replacements from dictionary")
    
    # CRITICAL: Ensure we return completely new objects with no shared references
    return str(out), dict(used)


def decompress(dst: str, used: Dict[str, str]) -> str:
    """
    Reverse a previous compression call given its *used* table.
    Handles both standard compression and selective tool call compression.
    CRITICAL: This function MUST NOT affect global state or other parts of the system.
    """
    # Create completely isolated copy to prevent any side effects
    out = str(dst)  # Ensure we have a completely separate string object
    
    # CRITICAL: Create isolated copy of the used dictionary
    isolated_used = dict(used) if used else {}
    
    if not isolated_used:
        return out
    
    # Step 1: Handle selective tool call decompression first
    # Check if there are any tool call field rules (prefixed with field names)
    tool_call_rules = {k: v for k, v in isolated_used.items() if '_' in k and any(
        field in k.split('_')[0].lower() for field in 
        ['content', 'file_content', 'text', 'data', 'body', 'source', 'code', 'output', 'result', 'response']
    )}
    
    if tool_call_rules:
        try:
            out = decompress_tool_call_content(out, tool_call_rules)
            logger.debug(f"Applied tool call content decompression: {len(tool_call_rules)} rules")
            # Remove tool call rules from the main decompression
            for key in tool_call_rules.keys():
                isolated_used.pop(key, None)
        except Exception as e:
            logger.debug(f"Tool call content decompression failed: {e}")
    
    # Step 2: Apply standard decompression for remaining rules
    if not isolated_used:
        return out
    
    # Sort by symbol length (shortest first) then by pattern length (longest first)
    # This prevents shorter symbols from being replaced inside longer patterns
    sorted_used = sorted(isolated_used.items(), 
                        key=lambda x: (len(x[0]), -len(x[1])))
    
    # CRITICAL SAFETY: Apply replacements ONLY to our isolated string with validation
    replacement_count = 0
    for symbol, pattern in sorted_used:
        if symbol in out:
            # Count occurrences for validation
            occurrences = out.count(symbol)
            if occurrences > 0:
                # Use string replace on the isolated copy only
                new_out = out.replace(symbol, pattern)
                if new_out != out and len(new_out) > len(out):  # Validate decompression occurred
                    out = new_out
                    replacement_count += 1
                    logger.debug(f"Restored '{symbol}' to '{pattern[:50]}...' ({occurrences} times)")
    
    logger.debug(f"Applied {replacement_count} standard decompression replacements")
    
    # CRITICAL: Return completely new string object with no shared references
    return str(out)


def compress_with_selective_tool_call_analysis(src: str, skip_tool_detection: bool = False, cline_mode: bool = False) -> Packed:
    """
    Compress text using selective tool call compression + dynamic dictionary analysis.
    
    This function:
    1. Detects tool calls and selectively compresses large content fields within them
    2. Applies standard dynamic compression to non-tool-call content
    3. Preserves tool call JSON structure for client compatibility
    
    Args:
        src: Source text to compress
        skip_tool_detection: Skip tool call detection entirely (for compatibility)
        cline_mode: Enable Cline-specific compression behaviors (allows markdown compression for system prompts)
    
    Returns:
        Packed object with compression results
    """
    logger.debug("Starting selective tool call + dynamic compression analysis")
    
    # Validate input
    if not src or not isinstance(src, str):
        logger.warning("Invalid input to compression: not a string")
        return Packed(src or "", {}, None, False, None, None, None)
    
    # Step 1: Strip comments if enabled (before compression analysis)
    comment_stripper = CommentStripper()
    processed_src, comment_stats = comment_stripper.strip_comments(src)
    working_src = processed_src
    
    # Initialize compression tracking
    rule_union = {}
    
    # Step 2: Handle tool calls selectively (if not skipped)
    if not skip_tool_detection and contains_tool_calls(working_src):
        logger.debug("Tool calls detected - applying selective content compression")
        
        # Create a simple compression function for tool call content
        def tool_content_compress(content: str):
            try:
                # Use the existing dynamic analysis but only on the content - use cached instance
                from .dynamic_dictionary import get_dynamic_dictionary_analyzer
                analyzer = get_dynamic_dictionary_analyzer()
                if analyzer.is_enabled():
                    should_analyze, reason = analyzer.should_analyze_prompt(content, skip_tool_detection=True)
                    if should_analyze:
                        analysis_result = analyzer.analyze_prompt(content)
                        compression_ratio = analysis_result["compression_analysis"]["compression_ratio"]
                        threshold = analyzer.config.get("compression_threshold", 0.01)
                        
                        if compression_ratio >= threshold:
                            temp_dict_path = analyzer.create_temporary_dictionary(analysis_result)
                            if temp_dict_path:
                                try:
                                    compressed_text, used_dict = _compress_with_temp_dictionary(content, temp_dict_path)
                                    # Cleanup immediately
                                    if os.path.exists(temp_dict_path):
                                        os.unlink(temp_dict_path)
                                    return compressed_text, used_dict
                                except Exception as e:
                                    logger.debug(f"Tool content compression failed: {e}")
                                    if temp_dict_path and os.path.exists(temp_dict_path):
                                        os.unlink(temp_dict_path)
                
                # Fallback: return original content
                return content, {}
            except Exception as e:
                logger.debug(f"Error in tool content compression: {e}")
                return content, {}
        
        # Apply selective tool call compression
        try:
            working_src, tool_call_rules = compress_tool_call_content(
                working_src, 
                tool_content_compress,
                min_content_size=300  # Compress content fields >= 300 chars
            )
            rule_union.update(tool_call_rules)
            
            if tool_call_rules:
                logger.info(f"Applied selective tool call compression: {len(tool_call_rules)} rules generated")
            
        except Exception as e:
            logger.debug(f"Selective tool call compression failed: {e}")
    
    # Step 3: Apply standard dynamic compression to remaining content
    # (This will skip tool calls due to the normal tool detection unless cline_mode allows it)
    analyzer = get_dynamic_dictionary_analyzer()
    if analyzer.is_enabled():
        # Pass cline_mode to the should_analyze_prompt method
        should_analyze, reason = analyzer.should_analyze_prompt(working_src, skip_tool_detection=False, cline_mode=cline_mode)
        if should_analyze:
            logger.debug("Applying standard dynamic compression to non-tool-call content")
            
            analysis_result = analyzer.analyze_prompt(working_src)
            compression_ratio = analysis_result["compression_analysis"]["compression_ratio"]
            threshold = analyzer.config.get("compression_threshold", 0.01)
            
            if compression_ratio >= threshold:
                temp_dict_path = analyzer.create_temporary_dictionary(analysis_result)
                if temp_dict_path:
                    try:
                        compressed_text, dynamic_used = _compress_with_temp_dictionary(working_src, temp_dict_path)
                        rule_union.update(dynamic_used)
                        working_src = compressed_text
                        
                        if dynamic_used:
                            logger.info(f"Applied standard dynamic compression: {len(dynamic_used)} additional rules")
                        
                        # Cleanup
                        if os.path.exists(temp_dict_path):
                            os.unlink(temp_dict_path)
                            
                    except Exception as e:
                        logger.debug(f"Standard dynamic compression failed: {e}")
                        if temp_dict_path and os.path.exists(temp_dict_path):
                            os.unlink(temp_dict_path)
    
    # Step 4: Validate compression effectiveness
    if rule_union:
        token_compression_ratio = validate_tokenization_efficiency(processed_src, working_src, rule_union)
        token_threshold = 0.01  # 1% minimum token savings
        
        if token_compression_ratio < token_threshold:
            logger.debug(f"Token validation failed: {token_compression_ratio*100:.2f}% < {token_threshold*100:.1f}%, returning uncompressed")
            if comment_stats.get("enabled", False):
                return Packed(processed_src, {}, None, False, None, comment_stats, None)
            else:
                return Packed(src, {}, None, False, None, comment_stats, None)
    
    # Step 5: Calculate final metrics
    final_ratio = (len(processed_src) - len(working_src)) / len(processed_src) if len(processed_src) > 0 else 0
    
    # Log results
    if comment_stats.get("enabled", False):
        comment_chars_saved = comment_stats.get("chars_saved", 0)
        total_chars_saved = (len(src) - len(working_src))
        total_compression_ratio = total_chars_saved / len(src) if len(src) > 0 else 0
        
        logger.info(f"📝 Comment stripping saved {comment_chars_saved:,} chars")
        logger.info(f"🔧 Selective tool + dynamic compression achieved {final_ratio*100:.1f}% compression")
        logger.info("=" * 60)
        logger.info("📊 MESSAGE COMPRESSION SUMMARY")
        logger.info(f"📊 Total pipeline savings: {total_compression_ratio*100:.1f}% ({total_chars_saved:,} chars)")
        logger.info("=" * 60)
        logger.debug(f"Used {len(rule_union)} compression rules")
    else:
        logger.info(f"Selective tool + dynamic compression achieved {final_ratio*100:.1f}% compression")
        logger.debug(f"Used {len(rule_union)} compression rules")
    
    return Packed(
        text=working_src,
        used=rule_union,
        language=None,
        fallback_used=False,
        dynamic_dict_used=None,  # Multiple temp dicts used, cleaned up
        comment_stats=comment_stats,
        original_content_structure=None
    ) 


def compress_multimodal_aware(content: Any, skip_tool_detection: bool = False, cline_mode: bool = False) -> Packed:
    """
    Compress content while preserving multimodal structure for proper reconstruction.
    
    This function:
    1. Extracts text from multimodal content for compression
    2. Preserves the original structure for later reconstruction
    3. Applies standard compression to the extracted text
    
    Args:
        content: The original content (string, list, dict, or other)
        skip_tool_detection: Skip tool call detection (for Cline compatibility)
        cline_mode: Enable Cline-specific compression behaviors
    
    Returns:
        Packed object with compression results and preserved structure
    """
    logger.debug("Starting multimodal-aware compression")
    
    # Import here to avoid circular imports
    try:
        from api.server import extract_message_content_for_compression
    except ImportError:
        logger.error("Cannot import extract_message_content_for_compression - falling back to string conversion")
        extract_message_content_for_compression = lambda x: str(x) if x is not None else ""
    
    # If content is already a string, use standard compression
    if isinstance(content, str):
        result = compress_with_dynamic_analysis(content, skip_tool_detection, cline_mode)
        return result._replace(original_content_structure=None)  # No structure to preserve
    
    # Extract text content for compression
    extracted_text = extract_message_content_for_compression(content)
    
    # Apply standard compression to the extracted text
    compressed_result = compress_with_dynamic_analysis(extracted_text, skip_tool_detection, cline_mode)
    
    # Return result with preserved original structure
    return compressed_result._replace(original_content_structure=content)


def decompress_multimodal_aware(compressed_text: str, used: Dict[str, str], original_structure: Any = None) -> Any:
    """
    Decompress text and reconstruct original multimodal structure if available.
    
    Args:
        compressed_text: The compressed text to decompress
        used: Dictionary of compression rules to reverse
        original_structure: The original multimodal structure to reconstruct (if any)
    
    Returns:
        Reconstructed content in original format, or plain string if no structure
    """
    logger.debug("Starting multimodal-aware decompression")
    
    # First decompress the text
    decompressed_text = decompress(compressed_text, used)
    
    # If no original structure, return the decompressed text
    if original_structure is None:
        return decompressed_text
    
    # Reconstruct the original multimodal structure
    return reconstruct_multimodal_content(decompressed_text, original_structure)


def reconstruct_multimodal_content(decompressed_text: str, original_structure: Any) -> Any:
    """
    Reconstruct multimodal content from decompressed text using the original structure.
    
    This function attempts to intelligently map decompressed text back to the original
    multimodal format by understanding common patterns and structures.
    
    Args:
        decompressed_text: The decompressed text content
        original_structure: The original multimodal structure
    
    Returns:
        Reconstructed content in original format
    """
    # Import here to avoid circular imports
    try:
        from api.server import extract_message_content_for_compression
    except ImportError:
        logger.warning("Cannot import extract_message_content_for_compression for reconstruction")
        return decompressed_text
    
    # If original was a simple string, return decompressed text
    if isinstance(original_structure, str):
        return decompressed_text
    
    # If original was a list (multimodal array)
    elif isinstance(original_structure, list):
        return reconstruct_multimodal_list(decompressed_text, original_structure)
    
    # If original was a dictionary
    elif isinstance(original_structure, dict):
        return reconstruct_multimodal_dict(decompressed_text, original_structure)
    
    # For other types, return the decompressed text
    else:
        logger.debug(f"Cannot reconstruct type {type(original_structure)}, returning decompressed text")
        return decompressed_text


def reconstruct_multimodal_list(decompressed_text: str, original_list: list) -> list:
    """
    Reconstruct a multimodal list structure from decompressed text.
    
    This handles common patterns like:
    - [{"type": "text", "text": "content"}, {"type": "image_url", "image_url": {...}}]
    - Mixed content arrays with text and attachments
    """
    reconstructed = []
    remaining_text = decompressed_text
    
    for item in original_list:
        if isinstance(item, dict):
            item_type = item.get('type', '')
            
            if item_type == 'text':
                # Find the text portion that corresponds to this item
                original_text = item.get('text', '')
                if original_text and original_text in remaining_text:
                    # Use the original text length to extract the corresponding portion
                    start_pos = remaining_text.find(original_text)
                    if start_pos >= 0:
                        # Extract text of similar length from decompressed content
                        text_lines = remaining_text[start_pos:].split('\n')
                        extracted_text = '\n'.join(text_lines[:len(original_text.split('\n'))])
                        remaining_text = remaining_text[start_pos + len(extracted_text):].lstrip('\n')
                    else:
                        # Fallback: use first portion of remaining text
                        lines = remaining_text.split('\n')
                        extracted_text = '\n'.join(lines[:max(1, len(original_text.split('\n')))])
                        remaining_text = '\n'.join(lines[len(extracted_text.split('\n')):])
                    
                    # Create reconstructed text item
                    reconstructed_item = item.copy()
                    reconstructed_item['text'] = extracted_text.strip()
                    reconstructed.append(reconstructed_item)
                else:
                    # If we can't match, use remaining text
                    reconstructed_item = item.copy()
                    reconstructed_item['text'] = remaining_text.strip()
                    remaining_text = ""
                    reconstructed.append(reconstructed_item)
            
            elif item_type == 'attachment':
                # Look for attachment marker in decompressed text
                name = item.get('name', 'attachment')
                marker = f"[Attachment: {name}]"
                if marker in remaining_text:
                    # Extract the content after the marker
                    marker_pos = remaining_text.find(marker)
                    after_marker = remaining_text[marker_pos + len(marker):].lstrip('\n')
                    
                    # Try to extract the attachment content
                    lines = after_marker.split('\n')
                    # Look for next attachment or end of content
                    content_lines = []
                    for line in lines:
                        if line.startswith('[Attachment:') or line == '[Image attachment]':
                            break
                        content_lines.append(line)
                    
                    extracted_content = '\n'.join(content_lines).strip()
                    
                    # Update remaining text
                    consumed_length = len(marker) + len('\n'.join(content_lines))
                    remaining_text = remaining_text[marker_pos + consumed_length:].lstrip('\n')
                    
                    # Create reconstructed attachment item
                    reconstructed_item = item.copy()
                    if extracted_content:
                        reconstructed_item['data'] = extracted_content
                    reconstructed.append(reconstructed_item)
                else:
                    # Keep original attachment unchanged
                    reconstructed.append(item.copy())
            
            elif item_type in ['image_url', 'image']:
                # Image items are preserved as-is (they weren't compressed)
                reconstructed.append(item.copy())
            
            else:
                # Generic dict - try to find text in common fields
                reconstructed_item = item.copy()
                for field in ['text', 'content', 'data']:
                    if field in item and isinstance(item[field], str):
                        # Assign remaining text to this field
                        if remaining_text:
                            lines = remaining_text.split('\n')
                            extracted_text = '\n'.join(lines[:max(1, len(str(item[field]).split('\n')))])
                            remaining_text = '\n'.join(lines[len(extracted_text.split('\n')):])
                            reconstructed_item[field] = extracted_text.strip()
                        break
                reconstructed.append(reconstructed_item)
        else:
            # Non-dict item - convert to string and use remaining text
            if remaining_text:
                lines = remaining_text.split('\n')
                extracted_text = lines[0] if lines else ""
                remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                reconstructed.append(extracted_text)
            else:
                reconstructed.append(str(item))
    
    # If there's remaining text and no items processed it, add it as a text item
    if remaining_text.strip() and not reconstructed:
        reconstructed.append({"type": "text", "text": remaining_text.strip()})
    
    return reconstructed


def reconstruct_multimodal_dict(decompressed_text: str, original_dict: dict) -> dict:
    """
    Reconstruct a multimodal dictionary structure from decompressed text.
    
    This handles dictionaries with text fields that need to be updated.
    """
    reconstructed = original_dict.copy()
    
    # Look for common text fields and update them
    text_fields = ['text', 'content', 'data']
    for field in text_fields:
        if field in original_dict and isinstance(original_dict[field], str):
            # Replace with decompressed text
            reconstructed[field] = decompressed_text
            break
    
    return reconstructed 