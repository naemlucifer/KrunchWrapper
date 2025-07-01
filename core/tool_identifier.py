"""
Tool Call Identifier Module

This module provides functionality to detect and identify tool calls within text,
allowing the compression system to skip compression for structured data that
should be preserved exactly.

Tool calls typically appear as JSON structures with specific patterns that
need to be preserved to maintain functionality.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class ToolCallIdentifier:
    """
    Identifies various types of tool calls and structured data that should
    not be compressed to preserve functionality.
    """
    
    def __init__(self):
        """Initialize the tool call identifier with default patterns."""
        # Common tool call field names that indicate structured data
        self.tool_indicators = {
            "tool",
            "function", 
            "action",
            "command",
            "operation",
            "method"
        }
        
        # Path-like fields that often accompany tool calls
        self.path_indicators = {
            "path",
            "file",
            "filename",
            "filepath",
            "directory",
            "location"
        }
        
        # Additional structural indicators
        self.structural_indicators = {
            "operationIsLocatedInWorkspace",
            "content",
            "arguments",
            "params",
            "parameters",
            "options",
            "config"
        }
    
    def contains_tool_calls(self, text: str) -> bool:
        """
        Detect if text contains tool call JSON structures or XML-style tool calls.
        
        Tool calls typically have structure like:
        {
          "tool": "readFile",
          "path": "...",
          "content": "...",
          "operationIsLocatedInWorkspace": true
        }
        
        Or XML-style calls like:
        <read_file>
        <path>...</path>
        </read_file>
        
        Args:
            text: Text to analyze
            
        Returns:
            True if tool calls are detected, False otherwise
        """
        try:
            # Check for XML-style tool calls (Cline format)
            if self._has_xml_tool_calls(text):
                logger.debug("XML-style tool call detected")
                return True
            
            # Quick pre-check for JSON-like content with tool indicators
            if not self._has_potential_tool_structure(text):
                return False
            
            # Try to find and parse JSON objects
            json_objects = self._extract_json_objects(text)
            
            for json_obj in json_objects:
                if self._is_tool_call_object(json_obj):
                    tool_name = json_obj.get("tool", json_obj.get("function", "unknown"))
                    logger.debug(f"JSON tool call detected: {tool_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in tool call detection: {e}")
            return False
    
    def _has_xml_tool_calls(self, text: str) -> bool:
        """Check for XML-style tool calls commonly used by Cline."""
        # AGGRESSIVE FIX: Temporarily disable XML tool call detection for Cline content
        # The current patterns are too broad and preventing compression of valuable content
        # 
        # Only block compression for VERY specific, unambiguous tool execution patterns
        # that are clearly active commands, not content references or documentation
        
        # Ultra-specific patterns for active tool execution only
        # These must contain both the tag AND clear execution context
        active_execution_patterns = [
            r'<tool_use[^>]*>\s*<name>[^<]+</name>\s*<parameters>',  # Anthropic tool use format
            r'<function_calls[^>]*>\s*<invoke[^>]*>',  # Function invocation format
        ]
        
        # Count only clearly active tool executions
        for pattern in active_execution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                logger.debug(f"Found {len(matches)} active tool executions, skipping compression")
                for i, match in enumerate(matches[:2]):
                    preview = match[:100].replace('\n', '\\n')
                    logger.debug(f"Active execution match {i+1}: {preview}...")
                return True
        
        # Allow compression of ALL other XML-style content including:
        # - Cline structured content (<task>, <environment_details>)
        # - Tool call references or documentation
        # - Historical tool call examples
        # - Any content that isn't actively executing tools
        logger.debug("No active tool executions detected, allowing compression")
        return False

    def _has_potential_tool_structure(self, text: str) -> bool:
        """Quick check for potential tool call indicators."""
        # Must have both a tool indicator and some structural element
        has_tool_indicator = any(f'"{indicator}"' in text for indicator in self.tool_indicators)
        has_structural_element = (
            any(f'"{indicator}"' in text for indicator in self.path_indicators) or
            any(f'"{indicator}"' in text for indicator in self.structural_indicators)
        )
        
        return has_tool_indicator and (has_structural_element or '{' in text)
    
    def _extract_json_objects(self, text: str) -> List[Dict]:
        """Extract all JSON objects from text."""
        json_objects = []
        
        # Method 1: Simple regex for single-line JSON objects
        json_pattern = r'\{[^{}]*"(?:' + '|'.join(self.tool_indicators) + r')"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # CRITICAL FIX: Normalize JSON by escaping raw newlines in string values
                normalized_match = self._normalize_json_newlines(match)
                obj = json.loads(normalized_match)
                if isinstance(obj, dict):
                    json_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        # Method 2: Multi-line JSON parsing
        json_objects.extend(self._extract_multiline_json_objects(text))
        
        return json_objects
    
    def _extract_multiline_json_objects(self, text: str) -> List[Dict]:
        """Extract multi-line JSON objects that might contain tool calls."""
        json_objects = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for lines that start a JSON object
            if line.startswith('{'):
                json_content, end_index = self._parse_json_from_lines(lines, i)
                
                if json_content:
                    try:
                        # CRITICAL FIX: Normalize JSON before parsing
                        normalized_content = self._normalize_json_newlines(json_content)
                        obj = json.loads(normalized_content)
                        if isinstance(obj, dict):
                            json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                
                i = end_index + 1
            else:
                i += 1
        
        return json_objects
    
    def _parse_json_from_lines(self, lines: List[str], start_index: int) -> tuple[Optional[str], int]:
        """Parse a complete JSON object starting from a given line index."""
        json_content = ""
        brace_count = 0
        i = start_index
        
        while i < len(lines):
            line = lines[i].strip()
            json_content += line if i == start_index else "\n" + line
            
            # Count braces to find the end of the JSON object
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and i > start_index:
                # Complete JSON object found
                return json_content, i
            elif brace_count < 0:
                # Malformed JSON
                return None, i
            
            i += 1
        
        # Incomplete JSON object
        return None, len(lines) - 1
    
    def _is_tool_call_object(self, obj: Dict) -> bool:
        """Determine if a JSON object represents a tool call."""
        if not isinstance(obj, dict):
            return False
        
        # Check for primary tool indicators
        has_tool_indicator = any(key in obj for key in self.tool_indicators)
        
        if not has_tool_indicator:
            return False
        
        # Additional validation - tool calls usually have some structural elements
        has_structural_elements = (
            any(key in obj for key in self.path_indicators) or
            any(key in obj for key in self.structural_indicators) or
            len(obj) >= 2  # At minimum, tool calls have multiple fields
        )
        
        return has_structural_elements
    
    def identify_tool_type(self, text: str) -> Optional[str]:
        """
        Identify the specific type of tool call if present.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tool type/name if detected, None otherwise
        """
        try:
            json_objects = self._extract_json_objects(text)
            
            for json_obj in json_objects:
                if self._is_tool_call_object(json_obj):
                    # Try different common tool identifier fields
                    for field in ["tool", "function", "action", "command", "operation"]:
                        if field in json_obj:
                            return json_obj[field]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error identifying tool type: {e}")
            return None
    
    def get_tool_calls(self, text: str) -> List[Dict]:
        """
        Extract all tool call objects from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries representing tool calls
        """
        try:
            json_objects = self._extract_json_objects(text)
            tool_calls = []
            
            for json_obj in json_objects:
                if self._is_tool_call_object(json_obj):
                    tool_calls.append(json_obj)
            
            return tool_calls
            
        except Exception as e:
            logger.debug(f"Error extracting tool calls: {e}")
            return []

    def compress_tool_call_content(self, text: str, compress_func, min_content_size: int = 500) -> tuple[str, Dict[str, str]]:
        """
        Selectively compress content within tool calls while preserving structure.
        
        This compresses large content fields (like file contents) within tool calls
        while keeping the tool call JSON structure intact for client compatibility.
        
        Args:
            text: Text that may contain tool calls
            compress_func: Function to use for compression (should return (compressed_text, rules_dict))
            min_content_size: Minimum size of content to compress (default 500 chars)
            
        Returns:
            Tuple of (processed_text, combined_rules_dict)
        """
        try:
            if not self.contains_tool_calls(text):
                return text, {}
                
            # Extract JSON objects that look like tool calls
            json_objects = self._extract_json_objects(text)
            combined_rules = {}
            processed_text = text
            
            for json_obj in json_objects:
                if self._is_tool_call_object(json_obj):
                    # Find content fields that could benefit from compression
                    content_fields = self._identify_compressible_content_fields(json_obj, min_content_size)
                    
                    if content_fields:
                        # Create a modified version with compressed content
                        modified_obj = json_obj.copy()
                        obj_rules = {}
                        
                        for field_name in content_fields:
                            original_content = modified_obj[field_name]
                            
                            # Apply compression to this content field
                            try:
                                compressed_result = compress_func(original_content)
                                if hasattr(compressed_result, 'text') and hasattr(compressed_result, 'used'):
                                    # Handle Packed object
                                    compressed_content = compressed_result.text
                                    field_rules = compressed_result.used
                                elif isinstance(compressed_result, tuple) and len(compressed_result) == 2:
                                    # Handle tuple (compressed_text, rules_dict)
                                    compressed_content, field_rules = compressed_result
                                else:
                                    continue  # Skip if compression format is unexpected
                                
                                # Only apply if compression was beneficial
                                if len(compressed_content) < len(original_content) * 0.9:  # At least 10% savings
                                    modified_obj[field_name] = compressed_content
                                    # Prefix field rules to avoid conflicts
                                    for symbol, pattern in field_rules.items():
                                        prefixed_symbol = f"{field_name}_{symbol}"
                                        obj_rules[prefixed_symbol] = pattern
                                        combined_rules[prefixed_symbol] = pattern
                                        
                                    logger.debug(f"Compressed tool call field '{field_name}': {len(original_content)} → {len(compressed_content)} chars")
                                
                            except Exception as e:
                                logger.debug(f"Failed to compress tool call field '{field_name}': {e}")
                                continue
                        
                        # Replace the original JSON in the text with the modified version
                        if obj_rules:  # Only if we actually compressed something
                            original_json_str = json.dumps(json_obj, separators=(',', ':'))
                            modified_json_str = json.dumps(modified_obj, separators=(',', ':'))
                            
                            # Replace in the text (be careful about multiple occurrences)
                            if original_json_str in processed_text:
                                processed_text = processed_text.replace(original_json_str, modified_json_str, 1)
                                logger.debug(f"Applied selective compression to tool call with {len(obj_rules)} compressed fields")
            
            return processed_text, combined_rules
            
        except Exception as e:
            logger.debug(f"Error in selective tool call compression: {e}")
            return text, {}
    
    def _identify_compressible_content_fields(self, tool_call_obj: Dict, min_size: int) -> List[str]:
        """
        Identify fields in a tool call that contain large content suitable for compression.
        
        Args:
            tool_call_obj: The tool call JSON object
            min_size: Minimum size in characters to consider for compression
            
        Returns:
            List of field names that contain compressible content
        """
        compressible_fields = []
        
        # Common field names that often contain large content
        content_field_candidates = [
            "content", "file_content", "text", "data", "body", 
            "source", "code", "output", "result", "response",
            "description", "documentation", "readme", "log"
        ]
        
        for field_name, field_value in tool_call_obj.items():
            if isinstance(field_value, str) and len(field_value) >= min_size:
                # Check if this field name suggests it contains content
                field_lower = field_name.lower()
                if any(candidate in field_lower for candidate in content_field_candidates):
                    compressible_fields.append(field_name)
                    logger.debug(f"Identified compressible field: {field_name} ({len(field_value)} chars)")
        
        return compressible_fields

    def decompress_tool_call_content(self, text: str, rules_dict: Dict[str, str]) -> str:
        """
        Decompress content within tool calls that was selectively compressed.
        
        Args:
            text: Text containing tool calls with compressed content
            rules_dict: Dictionary of compression rules to reverse
            
        Returns:
            Text with decompressed tool call content
        """
        if not rules_dict:
            return text
            
        try:
            # Group rules by field prefix
            field_rules = {}
            for symbol, pattern in rules_dict.items():
                if '_' in symbol:
                    field_name = symbol.split('_', 1)[0]
                    clean_symbol = symbol.split('_', 1)[1]
                    if field_name not in field_rules:
                        field_rules[field_name] = {}
                    field_rules[field_name][clean_symbol] = pattern
            
            # Apply decompression field by field
            processed_text = text
            for field_name, field_rules_dict in field_rules.items():
                # Apply decompression rules for this field
                for symbol, pattern in field_rules_dict.items():
                    if symbol in processed_text:
                        processed_text = processed_text.replace(symbol, pattern)
                        logger.debug(f"Decompressed tool call field '{field_name}': '{symbol}' → '{pattern[:50]}...'")
            
            return processed_text
            
        except Exception as e:
            logger.debug(f"Error in tool call content decompression: {e}")
            return text

    def _normalize_json_newlines(self, json_str: str) -> str:
        """
        Normalize JSON string by escaping raw newlines within string values.
        
        This fixes JSON that contains raw newlines in string values, which is invalid JSON
        but sometimes sent by clients like Cline in tool call content fields.
        
        Args:
            json_str: Potentially malformed JSON string with raw newlines
            
        Returns:
            Normalized JSON string with escaped newlines
        """
        try:
            # Quick check: if it's already valid JSON, return as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        # Strategy: Find string values that contain raw newlines and escape them
        # This is a simplified approach that handles the common case of content fields
        
        # Use regex to find quoted strings that contain raw newlines
        def escape_newlines_in_match(match):
            quoted_string = match.group(0)
            # Escape newlines, carriage returns, and tabs within the quoted string
            escaped = quoted_string.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return escaped
        
        # Find all quoted strings and escape newlines within them
        # This pattern matches strings that are properly quoted but may contain raw newlines
        normalized = re.sub(r'"[^"]*"', escape_newlines_in_match, json_str, flags=re.DOTALL)
        
        return normalized


# Global instance for easy import and use
_identifier = ToolCallIdentifier()

# Convenience functions for backward compatibility and ease of use
def contains_tool_calls(text: str) -> bool:
    """Check if text contains tool calls - convenience function."""
    return _identifier.contains_tool_calls(text)

def identify_tool_type(text: str) -> Optional[str]:
    """Identify tool type - convenience function."""
    return _identifier.identify_tool_type(text)

def get_tool_calls(text: str) -> List[Dict]:
    """Get all tool calls - convenience function."""
    return _identifier.get_tool_calls(text)

def compress_tool_call_content(text: str, compress_func, min_content_size: int = 500) -> tuple[str, Dict[str, str]]:
    """Selectively compress content within tool calls - convenience function."""
    return _identifier.compress_tool_call_content(text, compress_func, min_content_size)

def decompress_tool_call_content(text: str, rules_dict: Dict[str, str]) -> str:
    """Decompress tool call content - convenience function."""
    return _identifier.decompress_tool_call_content(text, rules_dict) 