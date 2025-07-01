"""
Markdown Identifier Module

This module provides functionality to detect and identify markdown content within text,
allowing the compression system to skip compression for formatted content that
should be preserved exactly to maintain readability and functionality.
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["MarkdownIdentifier", "contains_markdown_content", "contains_ai_reasoning_content"]

class MarkdownIdentifier:
    """
    Identifies various types of markdown content and AI-specific formatting that should
    not be compressed to preserve proper rendering and functionality.
    """
    
    def __init__(self):
        """Initialize the markdown identifier with default patterns."""
        
        # AI-specific thinking and reasoning tags - REMOVED FROM MARKDOWN DETECTION
        # These are NOT formatting-critical and CAN be compressed safely
        # Moving these to a separate category that doesn't block compression
        self.ai_reasoning_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<thought>.*?</thought>',
            r'<analysis>.*?</analysis>',
            r'<reasoning>.*?</reasoning>',
        ]
        
        # Code block patterns - KEEP for formatting preservation
        self.code_block_patterns = [
            r'```[\s\S]*?```',  # Fenced code blocks
        ]
        
        # Inline code patterns - KEEP for formatting preservation
        self.inline_code_patterns = [
            r'`[^`\n]+`',       # Single backtick inline code
        ]
        
        # Markdown formatting patterns - KEEP for formatting preservation
        self.formatting_patterns = [
            r'\*\*\*[^*\n]+\*\*\*',     # Bold + Italic text
            r'\*\*[^*\n]+\*\*',         # Bold text
            r'__[^_\n]+__',             # Bold text (underscore)
            r'\*[^*\n\d][^*\n]*\*',     # Italic text (avoid math expressions)
            r'_[a-zA-Z][^_\n]*_',       # Italic text (underscore, starts with letter)
            r'~~[^~\n]+~~',             # Strikethrough
            r'==.+?==',                 # Highlight (MkDocs)
            r'\^\^.+?\^\^',             # Insert/Underline (MkDocs)
            r'\[[^\]]+\]\([^)]+\)',     # Links [text](url)
            r'!\[[^\]]*\]\([^)]+\)',    # Images ![alt](url)
        ]
        
        # Header patterns
        self.header_patterns = [
            r'^#{1,6}\s+.*$',       # ATX headers
        ]
        
        # List patterns
        self.list_patterns = [
            r'^\s*[-*+]\s+.*$',     # Unordered lists
            r'^\s*\d+\.\s+.*$',     # Ordered lists
            r'^\s*\[[ x]\]\s+.*$',  # Task lists
        ]
        
        # Quote patterns
        self.quote_patterns = [
            r'^>\s+.*$',            # Blockquotes
        ]
        
        # HTML elements patterns
        self.html_patterns = [
            r'<[^>]+>',             # HTML tags (general)
            r'<b>.*?</b>',          # Bold HTML
            r'<i>.*?</i>',          # Italic HTML
            r'<em>.*?</em>',        # Emphasis HTML
            r'<strong>.*?</strong>', # Strong HTML
            r'<br\s*/?>',           # Line break HTML
            r'<pre>.*?</pre>',      # Preformatted HTML
        ]
        
        # Extended formatting patterns (MkDocs, etc.)
        self.extended_formatting_patterns = [
            r'[A-Za-z0-9]+~[^~\s]+~',       # Subscript H~2~O
            r'[A-Za-z0-9]+\^[^\^\s]+\^',    # Superscript X^2^
            r'\+\+.+?\+\+',                 # Keyboard keys ++ctrl+alt+del++
        ]
        
        # Critic markup patterns
        self.critic_markup_patterns = [
            r'\{--.*?--\}',         # Deletion {--del--}
            r'\{\+\+.*?\+\+\}',     # Addition {++add++}
            r'\{==.*?==\}',         # Highlight {==mark==}
            r'\{>>.*?<<\}',         # Comment {>>comment<<}
        ]
        
        # Admonition patterns (various flavors)
        self.admonition_patterns = [
            r'^\s*!!!\s+\w+.*$',        # MkDocs Material: !!! type
            r'^\s*:::\w+.*$',           # Docusaurus/MDX: :::type
            r'^\s*>\s*\[!\w+\].*$',     # GitHub/Obsidian: > [!NOTE]
        ]
        
        # Table patterns
        self.table_patterns = [
            r'^\s*\|.*\|.*$',           # Table rows with pipes
            r'^\s*\|[-:\s]+\|.*$',      # Table separator rows
        ]
        
        # Horizontal rule patterns
        self.horizontal_rule_patterns = [
            r'^---+\s*$',               # Dashes
            r'^\*\*\*+\s*$',            # Asterisks
            r'^___+\s*$',               # Underscores
        ]
    
    def contains_ai_reasoning_content(self, text: str) -> bool:
        """
        Detect if text contains AI reasoning tags that can be safely compressed.
        
        This is separate from markdown detection - these tags don't need formatting
        preservation and can be compressed to save tokens.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if AI reasoning content is detected, False otherwise
        """
        try:
            for pattern in self.ai_reasoning_patterns:
                if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                    logger.debug(f"AI reasoning pattern matched: {pattern}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"Error in AI reasoning detection: {e}")
            return False
    
    def contains_markdown_content(self, text: str) -> bool:
        """
        Detect if text contains markdown content that should be preserved.
        
        CRITICAL CHANGE: This now excludes AI reasoning tags like <thinking>
        which can be safely compressed without losing functionality.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if formatting-critical markdown content is detected, False otherwise
        """
        try:
            # PRIORITY CHECK: If this contains AI reasoning content, allow compression
            # AI reasoning tags can be safely compressed even if they contain code blocks
            has_ai_reasoning = self.contains_ai_reasoning_content(text)
            if has_ai_reasoning:
                logger.debug("AI reasoning content detected - allowing compression (reasoning tags can be compressed)")
                # For AI reasoning content, only block compression if there's CRITICAL formatting
                # like complex tables, that would break without proper formatting
                critical_formatting_patterns = [
                    r'^\s*\|.*\|.*\|.*$',           # Tables with multiple pipes (critical formatting)
                    r'^---+\s*$',                   # Horizontal rules (critical formatting)
                    r'^\*\*\*+\s*$',                # Horizontal rules (critical formatting)
                    r'^___+\s*$',                   # Horizontal rules (critical formatting)
                    r'\[.*?\]\(https?://.*?\)',     # HTTP links (preserve for functionality)
                ]
                
                for pattern in critical_formatting_patterns:
                    if re.search(pattern, text, re.MULTILINE):
                        logger.debug(f"Critical formatting pattern found in AI reasoning content: {pattern}")
                        return True
                
                # No critical formatting found - allow compression of AI reasoning content
                logger.debug("AI reasoning content has no critical formatting - allowing compression")
                return False
            
            # Quick heuristics to avoid false positives with code
            lines = text.split('\n')
            
            # If content looks like mostly code (high ratio of lines starting with whitespace or common code patterns)
            code_indicators = 0
            markdown_indicators = 0
            
            for line in lines[:50]:  # Check first 50 lines for performance
                stripped = line.strip()
                if not stripped:
                    continue
                    
                # Code indicators
                if (stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')) or
                    line.startswith(('    ', '\t')) or  # Indented code
                    stripped.endswith((':', '{', '}', ';')) or
                    any(keyword in stripped for keyword in ['=', '()', '[]', '{}',' -> ', ' in ', ' and ', ' or '])):
                    code_indicators += 1
                
                # Markdown indicators (only count clear markdown, excluding AI reasoning)
                if (stripped.startswith('#') and not stripped.startswith('#!/') or  # Headers but not shebangs
                    stripped.startswith(('- ', '* ', '+ ')) or  # Lists
                    stripped.startswith('> ') or  # Quotes
                    '**' in stripped and stripped.count('**') >= 2 or  # Bold
                    stripped.startswith('```') or  # Code blocks
                    '[' in stripped and '](' in stripped):  # Links
                    markdown_indicators += 1
            
            # If it looks like mostly code, be more conservative with markdown detection
            if code_indicators > markdown_indicators * 2:
                logger.debug(f"Content appears to be mostly code ({code_indicators} code vs {markdown_indicators} markdown indicators), using strict markdown detection")
                # Only check for very specific markdown patterns that are unambiguous and shouldn't appear in code
                # REMOVED: AI reasoning patterns - these can be safely compressed
                strict_patterns = (
                    self.code_block_patterns +
                    # Remove header and list patterns since they can appear in code comments/docstrings
                    # Only keep patterns that are very unlikely to appear in legitimate code
                    [r'^\s*>\s+[A-Za-z].*$'] +       # Blockquotes with letters (rare in code)
                    [r'\[.*?\]\(https?://.*?\)'] +   # HTTP links (clear markdown)
                    [r'^\s*\|.*\|.*\|.*$'] +         # Tables with multiple pipes (clear markdown)
                    [r'^---+\s*$'] +                 # Horizontal rules (clear markdown)
                    [r'^\*\*\*+\s*$'] +              # Horizontal rules (clear markdown)
                    [r'^___+\s*$']                   # Horizontal rules (clear markdown)
                )
                
                for pattern in strict_patterns:
                    if re.search(pattern, text, re.MULTILINE):
                        logger.debug(f"Strict markdown pattern matched: {pattern}")
                        return True
                        
                # No strict markdown patterns found - allow compression
                logger.debug("No strict markdown patterns found - allowing compression")
                return False
            
            # Standard detection for non-code content
            # Exclude problematic patterns that often match code
            # REMOVED: AI reasoning patterns - these can be safely compressed
            safe_patterns = (
                self.code_block_patterns + 
                self.inline_code_patterns + 
                [r'\*\*\*[^*\n]+\*\*\*'] +      # Bold + Italic (keep)
                [r'\*\*[^*\n]+\*\*'] +          # Bold (keep)  
                [r'__[^_\n]+__'] +              # Bold underscore (keep)
                # Remove problematic italic patterns that match code
                [r'~~[^~\n]+~~'] +              # Strikethrough (keep)
                [r'==.+?=='] +                  # Highlight (keep)
                [r'\^\^.+?\^\^'] +              # Insert/Underline (keep)
                [r'\[[^\]]+\]\([^)]+\)'] +      # Links (keep)
                [r'!\[[^\]]*\]\([^)]+\)'] +     # Images (keep)
                [r'^#{1,6}\s+.*$'] +            # Headers (keep)
                [r'^\s*[-*+]\s+.*$'] +          # Unordered lists (keep)
                [r'^\s*\d+\.\s+.*$'] +          # Ordered lists (keep)
                [r'^\s*\[[ x]\]\s+.*$'] +       # Task lists (keep)
                [r'^>\s+.*$'] +                 # Blockquotes (keep)
                # Remove general HTML pattern, only keep specific ones
                [r'<b>.*?</b>'] +               # Bold HTML
                [r'<i>.*?</i>'] +               # Italic HTML
                [r'<em>.*?</em>'] +             # Emphasis HTML
                [r'<strong>.*?</strong>'] +     # Strong HTML
                [r'<br\s*/?>'] +                # Line break HTML
                [r'<pre>.*?</pre>'] +           # Preformatted HTML
                self.extended_formatting_patterns +
                self.critic_markup_patterns +
                self.admonition_patterns +
                self.table_patterns +
                self.horizontal_rule_patterns
            )
            
            for pattern in safe_patterns:
                if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                    logger.debug(f"Standard markdown pattern matched: {pattern}")
                    return True
            
            # No markdown patterns found - allow compression
            logger.debug("No formatting-critical markdown patterns found - allowing compression")
            return False
            
        except Exception as e:
            logger.debug(f"Error in markdown detection: {e}")
            return False


# Global instance for easy import and use
_identifier = MarkdownIdentifier()

# Convenience functions
def contains_markdown_content(text: str) -> bool:
    """Check if text contains formatting-critical markdown content - convenience function."""
    return _identifier.contains_markdown_content(text)

def contains_ai_reasoning_content(text: str) -> bool:
    """Check if text contains AI reasoning content that can be compressed - convenience function."""
    return _identifier.contains_ai_reasoning_content(text) 