import pathlib
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import re

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc
from .conciseness_instructions import ConcisenessInstructionsHandler

__all__ = ["SystemPromptFormatter", "build_system_prompt", "encode_dictionary_minimal", "decode_dictionary_minimal"]


class SystemPromptFormatter:
    """Handles system prompt formatting based on configured format from system-prompts.jsonc"""
    
    def __init__(self):
        self.formats = self._load_system_prompt_formats()
        self.current_format = None
        self.format_config = None
    
    def _load_system_prompt_formats(self) -> Dict[str, Any]:
        """Load system prompt formats from system-prompts.jsonc"""
        config_path = pathlib.Path(__file__).parents[1] / "config" / "system-prompts.jsonc"
        
        if not config_path.exists():
            raise FileNotFoundError(f"System prompts configuration not found at {config_path}")
        
        try:
            return load_jsonc(str(config_path))
        except Exception as e:
            raise RuntimeError(f"Error loading system prompts from {config_path}: {e}")
    
    def set_format(self, format_name: str):
        """Set the current system prompt format"""
        if format_name not in self.formats:
            available = list(self.formats.keys())
            raise ValueError(f"Unknown system prompt format '{format_name}'. Available formats: {available}")
        
        self.current_format = format_name
        self.format_config = self.formats[format_name]
    
    def format_system_prompt(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Format system prompt content according to the current format.
        
        Returns:
            Tuple of (formatted_content, metadata) where metadata contains
            information about how to structure the prompt in the API call
        """
        if not self.current_format:
            raise RuntimeError("No system prompt format set. Call set_format() first.")
        
        config = self.format_config
        format_type = config.get("format", "chatml")
        
        metadata = {
            "format": format_type,
            "system_method": config.get("system_method"),
            "system_role": config.get("system_role", "system"),
            "system_structure": config.get("system_structure"),
            "prompt_template": config.get("prompt_template"),
            "prepend": config.get("prepend"),
            "system_prefix": config.get("system_prefix"),
            "system_suffix": config.get("system_suffix")
        }
        
        # Format the content based on the format type
        if format_type == "template" and "prompt_template" in config:
            # For template-based formats like Gemma
            formatted_content = config["prompt_template"].replace("{system_prompt}", content)
        elif format_type == "plain" and config.get("system_prefix") and config.get("system_suffix"):
            # For plain text formats like legacy Claude
            formatted_content = f"{config['system_prefix']}{content}{config['system_suffix']}"
        else:
            # For most formats (chatml, messages, contents), content stays as-is
            # The metadata will tell the caller how to structure it
            formatted_content = content
        
        return formatted_content, metadata
    
    def should_use_system_parameter(self) -> bool:
        """Check if this format uses a separate system parameter instead of message structure"""
        return self.format_config and self.format_config.get("system_method") == "system_parameter"
    
    def should_use_system_instruction(self) -> bool:
        """Check if this format uses system_instruction field"""
        return self.format_config and self.format_config.get("system_method") == "system_instruction"
    
    def get_system_structure(self) -> Optional[Dict[str, Any]]:
        """Get the system structure for formats that need it (like Gemini)"""
        return self.format_config.get("system_structure") if self.format_config else None


def build_system_prompt(used: Dict[str, str], lang: str, format_name: str = "claude", user_content: str = "", cline_mode: bool = False, stateful_mode: bool = False, new_symbols_only: Dict[str, str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Build a system prompt with compression instructions formatted for the specified format.
    Now includes configurable conciseness instructions when enabled.
    
    Args:
        used: Dictionary of substitutions that were actually used (for decompression)
        lang: Programming language
        format_name: System prompt format to use
        user_content: Content from user messages for context analysis
        cline_mode: Enable Cline-specific system prompt optimizations
        stateful_mode: When True, only include new symbols in system prompt for persistent KV cache servers
        new_symbols_only: Dictionary of new symbols only (used when stateful_mode=True)
    
    Returns:
        Tuple of (formatted_prompt, metadata) where metadata contains formatting information
    """
    # Ensure lang is never None or empty
    if not lang or lang is None:
        lang = "generic"
    
    # Build the core compression instruction content
    # For stateful mode, use new symbols only if provided and in stateful mode
    symbols_for_prompt = new_symbols_only if (stateful_mode and new_symbols_only is not None) else used
    
    if not symbols_for_prompt:
        if stateful_mode and new_symbols_only is not None:
            # Stateful mode with no new symbols - minimal prompt
            content = f"Continue processing {lang} code using existing compression context. No new symbols this turn."
        else:
            # Standard case with no symbols
            content = f"You will read {lang} code. Reason about it as-is."
    else:
        # Load dictionary format configuration
        dict_config = _load_dictionary_format_config()
        
        # Determine which format to use
        style = dict_config.get("style", "minimal")
        threshold = dict_config.get("minimal_format_threshold", 3)
        
        # Use minimal format if style is minimal and we have enough entries
        if style == "minimal" and len(symbols_for_prompt) >= threshold:
            if stateful_mode and new_symbols_only is not None and len(new_symbols_only) > 0:
                # Stateful mode: format new symbols with context
                content = _format_dictionary_stateful(new_symbols_only, lang, dict_config, "new")
            else:
                content = _format_dictionary_minimal(symbols_for_prompt, lang, dict_config)
        else:
            if stateful_mode and new_symbols_only is not None and len(new_symbols_only) > 0:
                # Stateful mode: format new symbols with context (verbose)
                content = _format_dictionary_stateful_verbose(new_symbols_only, lang)
            else:
                content = _format_dictionary_verbose(symbols_for_prompt, lang)
            
        # Log format choice for debugging if enabled
        if dict_config.get("enable_debug_view", False):
            import logging
            logger = logging.getLogger(__name__)
            
            # Calculate token savings estimate
            verbose_format = _format_dictionary_verbose(symbols_for_prompt, lang)
            minimal_format = _format_dictionary_minimal(symbols_for_prompt, lang, dict_config)
            
            char_savings = len(verbose_format) - len(minimal_format)
            token_savings_estimate = char_savings // 4  # Rough estimate: 4 chars per token
            
            stateful_note = " (STATEFUL - new symbols only)" if (stateful_mode and new_symbols_only is not None) else ""
            logger.debug(f"Dictionary format choice: {style} (entries: {len(symbols_for_prompt)}, threshold: {threshold}){stateful_note}")
            logger.debug(f"Verbose format: {len(verbose_format)} chars")
            logger.debug(f"Minimal format: {len(minimal_format)} chars") 
            logger.debug(f"Estimated token savings: {token_savings_estimate} tokens ({char_savings} chars)")
            
            if style == "minimal" and len(symbols_for_prompt) >= threshold:
                logger.debug(f"Using minimal format - savings: {char_savings} chars (~{token_savings_estimate} tokens){stateful_note}")
            else:
                logger.debug(f"Using verbose format - reason: style={style}, entries={len(symbols_for_prompt)}, threshold={threshold}{stateful_note}")
    
    # Handle conciseness instructions injection
    conciseness_handler = ConcisenessInstructionsHandler()
    has_compression = bool(used)
    
    if conciseness_handler.should_inject_instructions(has_compression):
        conciseness_instructions = conciseness_handler.generate_instructions(
            user_content=user_content,
            language=lang
        )
        
        if conciseness_instructions:
            injection_position = conciseness_handler.get_injection_position()
            
            if injection_position == "before":
                content = conciseness_instructions + "\n\n" + content
            elif injection_position == "after":
                content = content + "\n" + conciseness_instructions
            elif injection_position == "separate_section":
                # Add as a distinct section with clear separation
                content = content + "\n\n" + "=" * 50 + conciseness_instructions + "\n" + "=" * 50
    
    # Format according to the specified format
    formatter = SystemPromptFormatter()
    formatter.set_format(format_name)
    
    return formatter.format_system_prompt(content)


def _needs_alternative_delimiters(pairs: List[Tuple[str, str]]) -> bool:
    """Check if any phrase contains standard delimiters that would break parsing."""
    for symbol, phrase in pairs:
        if ';' in phrase or ':' in phrase:
            return True
    return False


def _escape_delimiters(text: str, use_alternative: bool = False) -> str:
    """Escape delimiter characters in text to prevent parsing conflicts."""
    if use_alternative:
        # If using alternative delimiters, escape those instead
        text = text.replace('‖', '\\‖').replace('⟦', '\\⟦')
    else:
        # Escape standard delimiters
        text = text.replace(';', '\\;').replace(':', '\\:')
    return text


def encode_dictionary_minimal(pairs: List[Tuple[str, str]], config: Optional[Dict[str, Any]] = None) -> str:
    """
    Encode dictionary pairs using minimal mapping syntax.
    
    Args:
        pairs: List of (symbol, phrase) tuples
        config: Dictionary format configuration (optional)
    
    Returns:
        Encoded dictionary string in format: #DICT symbol:phrase;symbol:phrase;#DICT_END
    """
    if not pairs:
        return ""
    
    # Use alternative delimiters if any phrase contains standard ones
    use_alternative = _needs_alternative_delimiters(pairs)
    
    if use_alternative:
        if config and 'alternative_delimiters' in config:
            pair_sep = config['alternative_delimiters'].get('pair_separator', '‖')
            kv_sep = config['alternative_delimiters'].get('key_value_separator', '⟦')
        else:
            # Default alternative delimiters if no config provided
            pair_sep = '‖'
            kv_sep = '⟦'
    else:
        pair_sep = ';'
        kv_sep = ':'
    
    # Build the encoded pairs
    encoded_pairs = []
    for symbol, phrase in pairs:
        # Escape any conflicting delimiters in the phrase
        escaped_phrase = _escape_delimiters(phrase, use_alternative)
        encoded_pairs.append(f"{symbol}{kv_sep}{escaped_phrase}")
    
    # Join pairs and wrap with sentinels
    blob = pair_sep.join(encoded_pairs)
    return f"#DICT {blob}{pair_sep}#DICT_END"


def decode_dictionary_minimal(encoded: str) -> Dict[str, str]:
    """
    Decode minimal mapping syntax back to dictionary.
    
    Args:
        encoded: Encoded dictionary string
    
    Returns:
        Dictionary of symbol -> phrase mappings
    """
    # First, detect which delimiters are being used by checking the content
    if '‖' in encoded:
        # Using alternative delimiters
        match = re.search(r"#DICT (.+?)‖#DICT_END", encoded)
        pair_sep = '‖'
        kv_sep = '⟦' if '⟦' in encoded else ':'
    else:
        # Using standard delimiters
        match = re.search(r"#DICT (.+?);#DICT_END", encoded)
        pair_sep = ';'
        kv_sep = ':'
    
    if not match:
        return {}
    
    blob = match.group(1)
    
    # Split pairs and decode
    result = {}
    pairs = blob.split(pair_sep)
    
    for pair in pairs:
        if not pair.strip():
            continue
        
        # Split on first occurrence of key-value separator
        parts = pair.split(kv_sep, 1)
        if len(parts) == 2:
            symbol = parts[0].strip()
            phrase = parts[1].strip()
            
            # Unescape delimiters
            if pair_sep == '‖':
                phrase = phrase.replace('\\‖', '‖').replace('\\⟦', '⟦')
            else:
                phrase = phrase.replace('\\;', ';').replace('\\:', ':')
            
            result[symbol] = phrase
    
    return result


def _load_dictionary_format_config() -> Dict[str, Any]:
    """Load dictionary format configuration from system-prompts.jsonc"""
    config_path = pathlib.Path(__file__).parents[1] / "config" / "system-prompts.jsonc"
    
    if not config_path.exists():
        # Return default configuration if file doesn't exist
        return {
            "style": "minimal",
            "minimal_format_threshold": 3,
            "minimal_header": "You are given a symbol dictionary. Use the compressed symbols in your responses to save tokens. Expand symbols only when explaining their meaning.",
            "verbose_header": "You will read {lang} code in a compressed DSL format. Symbol substitutions used: {pairs}. When responding, use the same compressed symbols to save tokens. Only expand symbols when you need to explain their full meaning.",
            "alternative_delimiters": {
                "pair_separator": "‖",
                "key_value_separator": "⟦"
            },
            "enable_debug_view": True
        }
    
    try:
        config = load_jsonc(str(config_path))
        return config.get("dictionary_format", {
            "style": "minimal",
            "minimal_format_threshold": 3,
            "minimal_header": "You are given a symbol dictionary. Use the compressed symbols in your responses to save tokens. Expand symbols only when explaining their meaning.",
            "verbose_header": "You will read {lang} code in a compressed DSL format. Symbol substitutions used: {pairs}. When responding, use the same compressed symbols to save tokens. Only expand symbols when you need to explain their full meaning.",
            "alternative_delimiters": {
                "pair_separator": "‖",
                "key_value_separator": "⟦"
            },
            "enable_debug_view": True
        })
    except Exception:
        # Return default on error
        return {
            "style": "minimal",
            "minimal_format_threshold": 3,
            "minimal_header": "You are given a symbol dictionary. Use the compressed symbols in your responses to save tokens. Expand symbols only when explaining their meaning.",
            "verbose_header": "You will read {lang} code in a compressed DSL format. Symbol substitutions used: {pairs}. When responding, use the same compressed symbols to save tokens. Only expand symbols when you need to explain their full meaning.",
            "alternative_delimiters": {
                "pair_separator": "‖",
                "key_value_separator": "⟦"
            },
            "enable_debug_view": True
        }


def _format_dictionary_verbose(used: Dict[str, str], lang: str) -> str:
    """Format dictionary using verbose/traditional style with full decoder mappings."""
    if not used:
        return f"You will read {lang} code. Reason about it as-is."
    
    # FIXED: Include the actual symbol -> pattern mappings so the LLM can understand the compression
    # Create the decoder dictionary in a clear, readable format
    decoder_lines = []
    decoder_lines.append(f"You will read {lang} code in a compressed format using symbols. Below is the decoder:")
    decoder_lines.append("")
    decoder_lines.append("COMPRESSION DECODER:")
    
    # Sort symbols for consistent output (shorter symbols first, then alphabetical)
    sorted_symbols = sorted(used.items(), key=lambda x: (len(x[0]), x[0]))
    
    for symbol, pattern in sorted_symbols:
        # Escape any special characters in the pattern for display
        escaped_pattern = pattern.replace('\\', '\\\\').replace('"', '\\"')
        decoder_lines.append(f"  {symbol}: {escaped_pattern}")
    
    decoder_lines.append("")
    decoder_lines.append("When responding, use the same compressed symbols to save tokens. Only expand symbols to their full meaning when specifically asked to explain them.")
    
    return "\n".join(decoder_lines)


def _format_dictionary_minimal(used: Dict[str, str], lang: str, config: Dict[str, Any]) -> str:
    """Format dictionary using minimal mapping syntax."""
    # CRITICAL BUG FIX: used is symbol -> pattern format
    # Convert to (symbol, pattern) tuples for encoding
    pairs = [(symbol, pattern) for symbol, pattern in used.items()]
    
    # Get the minimal header instruction
    header = config.get("minimal_header", "You are given a symbol dictionary. Use the compressed symbols in your responses to save tokens. Expand symbols only when explaining their meaning.")
    
    # Encode the dictionary
    encoded_dict = encode_dictionary_minimal(pairs, config)
    
    return f"{header}\n{encoded_dict}"


def _format_dictionary_stateful(new_symbols: Dict[str, str], lang: str, config: Dict[str, Any], mode: str = "new") -> str:
    """Format dictionary for stateful mode using minimal mapping syntax."""
    if not new_symbols:
        return f"Continue processing {lang} code using existing compression context. No new symbols this turn."
    
    # Convert to (symbol, pattern) tuples for encoding
    pairs = [(symbol, pattern) for symbol, pattern in new_symbols.items()]
    
    # Special header for stateful mode
    if mode == "new":
        header = f"NEW compression symbols for this turn (existing symbols remain active). Use ALL symbols in responses."
    else:
        header = f"Additional compression symbols for {lang} code. Use ALL symbols in responses."
    
    # Encode the dictionary
    encoded_dict = encode_dictionary_minimal(pairs, config)
    
    return f"{header}\n{encoded_dict}"


def _format_dictionary_stateful_verbose(new_symbols: Dict[str, str], lang: str) -> str:
    """Format dictionary for stateful mode using verbose style."""
    if not new_symbols:
        return f"Continue processing {lang} code using existing compression context. No new symbols this turn."
    
    # Create the decoder dictionary in a clear, readable format
    decoder_lines = []
    decoder_lines.append(f"NEW compression symbols for this turn (existing symbols remain active):")
    decoder_lines.append("")
    decoder_lines.append("ADDITIONAL DECODER MAPPINGS:")
    
    # Sort symbols for consistent output (shorter symbols first, then alphabetical)
    sorted_symbols = sorted(new_symbols.items(), key=lambda x: (len(x[0]), x[0]))
    
    for symbol, pattern in sorted_symbols:
        # Escape any special characters in the pattern for display
        escaped_pattern = pattern.replace('\\', '\\\\').replace('"', '\\"')
        decoder_lines.append(f"  {symbol}: {escaped_pattern}")
    
    decoder_lines.append("")
    decoder_lines.append("Use ALL symbols (existing + new) in responses to save tokens. Only expand symbols to their full meaning when specifically asked to explain them.")
    
    return "\n".join(decoder_lines) 