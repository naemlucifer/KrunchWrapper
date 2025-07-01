import pathlib
import re
from typing import Dict, List, Any, Optional
import sys
import os
import logging

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc

logger = logging.getLogger(__name__)

class ConcisenessInstructionsHandler:
    """
    Handles loading and processing of configurable conciseness instructions
    that can be injected into system prompts to encourage brief, focused responses.
    """
    
    def __init__(self):
        self.config = None
        self.instructions_config = None
        self._load_configurations()
    
    def _load_configurations(self):
        """Load main config and conciseness instructions config"""
        try:
            # Load main config
            config_path = pathlib.Path(__file__).parents[1] / "config" / "config.jsonc"
            if config_path.exists():
                self.config = load_jsonc(str(config_path))
            
            # Load conciseness instructions if enabled
            if self.is_enabled():
                instructions_file = self.config["conciseness_instructions"]["instructions_file"]
                instructions_path = pathlib.Path(__file__).parents[1] / "config" / instructions_file
                
                if instructions_path.exists():
                    self.instructions_config = load_jsonc(str(instructions_path))
                    logger.debug(f"Loaded conciseness instructions from {instructions_path}")
                else:
                    logger.warning(f"Conciseness instructions file not found: {instructions_path}")
                    
        except Exception as e:
            logger.error(f"Error loading conciseness instructions configuration: {e}")
            self.config = None
            self.instructions_config = None
    
    def is_enabled(self) -> bool:
        """Check if conciseness instructions are enabled in configuration"""
        return (self.config and 
                self.config.get("conciseness_instructions", {}).get("enabled", False))
    
    def should_inject_instructions(self, has_compression: bool = False) -> bool:
        """
        Determine if conciseness instructions should be injected based on configuration
        
        Args:
            has_compression: Whether compression decoder is present in the system prompt
            
        Returns:
            True if instructions should be injected, False otherwise
        """
        if not self.is_enabled() or not self.instructions_config:
            return False
        
        # Check if instructions should only be applied with compression
        only_with_compression = self.config["conciseness_instructions"].get("only_with_compression", False)
        
        if only_with_compression and not has_compression:
            return False
        
        return True
    
    def generate_instructions(self, 
                            user_content: str = "", 
                            language: str = "", 
                            context_hints: Optional[List[str]] = None) -> str:
        """
        Generate formatted conciseness instructions based on context
        
        Args:
            user_content: Content from user messages to analyze for context
            language: Programming language detected (if any)
            context_hints: Additional context hints for instruction selection
            
        Returns:
            Formatted conciseness instructions string
        """
        if not self.instructions_config:
            return ""
        
        try:
            # Collect instructions based on criteria
            instructions = self._collect_instructions(user_content, language, context_hints)
            
            # Format the instructions
            return self._format_instructions(instructions)
            
        except Exception as e:
            logger.error(f"Error generating conciseness instructions: {e}")
            return ""
    
    def _collect_instructions(self, 
                            user_content: str, 
                            language: str, 
                            context_hints: Optional[List[str]] = None) -> List[str]:
        """Collect relevant instructions based on context and configuration"""
        instructions = []
        config = self.instructions_config
        selection_criteria = config.get("selection_criteria", {})
        
        # Always include core instructions if configured and enabled
        if selection_criteria.get("always_include_core", True):
            core_config = config.get("core_instructions", {})
            if isinstance(core_config, dict) and core_config.get("enabled", True):
                core_instructions = core_config.get("instructions", [])
                instructions.extend(core_instructions)
            elif isinstance(core_config, list):
                # Backward compatibility: if it's still a list, use it directly
                instructions.extend(core_config)
        
        # Include code instructions if language is detected and enabled
        if (language and language != "generic" and 
            selection_criteria.get("include_code_when_lang_detected", True)):
            code_config = config.get("code_instructions", {})
            if isinstance(code_config, dict) and code_config.get("enabled", True):
                code_instructions = code_config.get("instructions", [])
                instructions.extend(code_instructions)
            elif isinstance(code_config, list):
                # Backward compatibility: if it's still a list, use it directly
                instructions.extend(code_config)
        
        # Check conditional instructions based on content analysis
        conditional = config.get("conditional_instructions", {})
        user_content_lower = user_content.lower()
        
        for condition_name, condition_config in conditional.items():
            if not condition_config.get("enabled", True):
                continue
                
            trigger_keywords = condition_config.get("trigger_keywords", [])
            if any(keyword in user_content_lower for keyword in trigger_keywords):
                condition_instructions = condition_config.get("instructions", [])
                instructions.extend(condition_instructions)
                logger.debug(f"Added conditional instructions for: {condition_name}")
        
        # Check custom instruction sets
        custom_sets = config.get("custom_instruction_sets", {})
        for set_name, set_config in custom_sets.items():
            if set_config.get("enabled", False):
                custom_instructions = set_config.get("instructions", [])
                instructions.extend(custom_instructions)
                logger.debug(f"Added custom instruction set: {set_name}")
        
        # Apply selection limits
        max_instructions = selection_criteria.get("max_instructions", 0)
        if max_instructions > 0 and len(instructions) > max_instructions:
            if selection_criteria.get("prioritize_by_order", True):
                # Keep first N instructions (highest priority)
                instructions = instructions[:max_instructions]
            else:
                # Could implement other prioritization schemes here
                instructions = instructions[:max_instructions]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_instructions = []
        for instruction in instructions:
            if instruction not in seen:
                seen.add(instruction)
                unique_instructions.append(instruction)
        
        return unique_instructions
    
    def _format_instructions(self, instructions: List[str]) -> str:
        """Format instructions according to configuration"""
        if not instructions:
            return ""
        
        config = self.instructions_config
        formatting = config.get("formatting", {})
        
        # Get formatting options
        prefix = formatting.get("prefix", "")
        suffix = formatting.get("suffix", "")
        separator = formatting.get("instruction_separator", "\n• ")
        use_wrapper = formatting.get("use_section_wrapper", True)
        
        # Build formatted string
        if prefix:
            formatted = f"{prefix}{separator}"
        else:
            formatted = separator
        
        formatted += separator.join(instructions)
        
        if suffix:
            formatted += suffix
        
        # Apply section wrapper if configured
        if use_wrapper:
            wrapper = formatting.get("section_wrapper", {})
            start = wrapper.get("start", "\n\n📝 CONCISENESS GUIDELINES:\n")
            end = wrapper.get("end", "\n")
            formatted = f"{start}{formatted}{end}"
        
        return formatted
    
    def get_injection_position(self) -> str:
        """Get configured injection position relative to compression decoder"""
        if not self.config:
            return "after"
        
        return self.config["conciseness_instructions"].get("injection_position", "after") 