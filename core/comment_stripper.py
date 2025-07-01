#!/usr/bin/env python3
"""
Comment Stripper Module for KrunchWrap
Strips comments from code while preserving functionality and providing detailed logging.
"""

import re
import logging
import os
import sys
from typing import Tuple, Dict, Optional
from pathlib import Path

# Add utils to path for jsonc_parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

class CommentStripper:
    """
    Strips comments from code in multiple programming languages.
    Provides detailed logging and token savings calculations.
    """
    
    def __init__(self):
        """Initialize the comment stripper with configuration."""
        self.config = self._load_config()
        
        # Language-specific comment patterns
        self.comment_patterns = {
            # Python
            'python': {
                'single_line': [r'#.*?$'],
                'multi_line': [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
                'file_extensions': ['.py', '.pyw']
            },
            
            # JavaScript / TypeScript
            'javascript': {
                'single_line': [r'//.*?$'],
                'multi_line': [r'/\*[\s\S]*?\*/'],
                'file_extensions': ['.js', '.jsx', '.ts', '.tsx', '.mjs']
            },
            
            # C / C++
            'c_cpp': {
                'single_line': [r'//.*?$'],
                'multi_line': [r'/\*[\s\S]*?\*/'],
                'file_extensions': ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx']
            },
            
            # HTML
            'html': {
                'multi_line': [r'<!--[\s\S]*?-->'],
                'file_extensions': ['.html', '.htm', '.xhtml']
            },
            
            # CSS
            'css': {
                'multi_line': [r'/\*[\s\S]*?\*/'],
                'file_extensions': ['.css', '.scss', '.sass', '.less']
            },
            
            # SQL
            'sql': {
                'single_line': [r'--.*?$'],
                'multi_line': [r'/\*[\s\S]*?\*/'],
                'file_extensions': ['.sql']
            },
            
            # Shell/Bash
            'shell': {
                'single_line': [r'#.*?$'],
                'file_extensions': ['.sh', '.bash', '.zsh', '.fish']
            }
        }
    
    def _load_config(self) -> Dict:
        """Load comment stripping configuration from config file."""
        config_path = Path(__file__).parent.parent / "config" / "config.jsonc"
        
        # Default configuration
        default_config = {
            "enabled": False,
            "preserve_docstrings": True,
            "preserve_license_headers": True,
            "preserve_shebang": True,
            "min_line_length_after_strip": 3
        }
        
        try:
            if config_path.exists():
                full_config = load_jsonc(str(config_path))
                comment_config = full_config.get("comment_stripping", {})
                
                # Merge with defaults
                config = default_config.copy()
                config.update(comment_config)
                
                logger.debug(f"Loaded comment stripping configuration from {config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {config_path}, using defaults")
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}, using defaults")
            return default_config
    
    def is_enabled(self) -> bool:
        """Check if comment stripping is enabled."""
        return self.config.get("enabled", False)
    
    def _detect_language(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """Detect the programming language of the text."""
        # First try filename extension if provided
        if filename:
            filename_lower = filename.lower()
            for lang, patterns in self.comment_patterns.items():
                extensions = patterns.get('file_extensions', [])
                if any(filename_lower.endswith(ext) for ext in extensions):
                    return lang
        
        # Fallback to content-based detection
        if any(keyword in text for keyword in ['def ', 'import ', 'from ', '__init__', 'self.', 'print(']):
            return 'python'
        elif any(keyword in text for keyword in ['function ', 'var ', 'let ', 'const ', 'console.']):
            return 'javascript'
        elif any(keyword in text for keyword in ['#include', 'int main', 'std::', 'printf(']):
            return 'c_cpp'
        elif any(keyword in text for keyword in ['<html', '<body', '<div', '<script']):
            return 'html'
        elif any(keyword in text for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT']):
            return 'sql'
        elif text.startswith('#!') and any(shell in text[:100] for shell in ['/bin/bash', '/bin/sh', '/usr/bin/env']):
            return 'shell'
        
        return None
    
    def _strip_comments_for_language(self, text: str, language: str) -> str:
        """Strip comments for a specific language."""
        if language not in self.comment_patterns:
            logger.warning(f"Language '{language}' not supported for comment stripping")
            return text
        
        patterns = self.comment_patterns[language]
        result = text
        
        # Strip multi-line comments first
        if 'multi_line' in patterns:
            for pattern in patterns['multi_line']:
                result = re.sub(pattern, '', result, flags=re.MULTILINE | re.DOTALL)
        
        # Strip single-line comments
        if 'single_line' in patterns:
            for pattern in patterns['single_line']:
                result = self._strip_single_line_comments_safe(result, pattern)
        
        return result
    
    def _strip_single_line_comments_safe(self, text: str, comment_pattern: str) -> str:
        """Safely strip single-line comments while avoiding comments inside strings."""
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            # Simple heuristic: count quotes to determine if we're inside a string
            in_string = False
            quote_char = None
            escaped = False
            comment_start = -1
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if escaped:
                    escaped = False
                    i += 1
                    continue
                
                if char == '\\':
                    escaped = True
                    i += 1
                    continue
                
                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == '#' and comment_pattern.startswith(r'#'):
                        comment_start = i
                        break
                    elif char == '/' and i + 1 < len(line) and line[i + 1] == '/' and comment_pattern.startswith(r'//'):
                        comment_start = i
                        break
                else:
                    if char == quote_char:
                        in_string = False
                        quote_char = None
                
                i += 1
            
            # Remove comment if found outside of strings
            if comment_start >= 0 and not in_string:
                stripped_line = line[:comment_start].rstrip()
                min_length = self.config.get("min_line_length_after_strip", 3)
                if len(stripped_line) >= min_length or stripped_line.strip() == '':
                    result_lines.append(stripped_line)
                else:
                    result_lines.append('')  # Keep as empty line to preserve line numbers
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _calculate_token_savings(self, original_text: str, stripped_text: str) -> Dict:
        """Calculate token savings from comment stripping."""
        original_chars = len(original_text)
        stripped_chars = len(stripped_text)
        chars_saved = original_chars - stripped_chars
        
        # Try to use tiktoken for accurate token counting
        if TIKTOKEN_AVAILABLE:
            try:
                tokenizer = tiktoken.get_encoding("cl100k_base")
                
                original_tokens = len(tokenizer.encode(original_text))
                stripped_tokens = len(tokenizer.encode(stripped_text))
                actual_tokens_saved = original_tokens - stripped_tokens
                
                return {
                    "original_chars": original_chars,
                    "stripped_chars": stripped_chars,
                    "chars_saved": chars_saved,
                    "char_reduction_percent": (chars_saved / original_chars * 100) if original_chars > 0 else 0,
                    "original_tokens": original_tokens,
                    "stripped_tokens": stripped_tokens,
                    "tokens_saved": actual_tokens_saved,
                    "token_reduction_percent": (actual_tokens_saved / original_tokens * 100) if original_tokens > 0 else 0,
                    "method": "tiktoken"
                }
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}, falling back to estimation")
        
        # Fallback to estimation
        estimated_tokens_saved = chars_saved // 4
        return {
            "original_chars": original_chars,
            "stripped_chars": stripped_chars,
            "chars_saved": chars_saved,
            "char_reduction_percent": (chars_saved / original_chars * 100) if original_chars > 0 else 0,
            "estimated_tokens_saved": estimated_tokens_saved,
            "token_reduction_percent": (estimated_tokens_saved / (original_chars // 4) * 100) if original_chars > 0 else 0,
            "method": "estimation"
        }
    
    def strip_comments(self, text: str, filename: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Main entry point for comment stripping.
        
        Args:
            text: Source code text
            filename: Optional filename for language detection
            
        Returns:
            Tuple of (stripped_text, statistics_dict)
        """
        if not self.is_enabled():
            logger.debug("Comment stripping is disabled")
            return text, {"enabled": False}
        
        if not text or len(text.strip()) == 0:
            logger.debug("Empty or whitespace-only text, skipping comment stripping")
            return text, {"skipped": "empty_text"}
        
        # Detect language
        language = self._detect_language(text, filename)
        if not language:
            logger.debug("Could not detect programming language, skipping comment stripping")
            return text, {"skipped": "unknown_language"}
        
        logger.debug(f"Detected language: {language}")
        
        # Strip comments
        stripped_text = self._strip_comments_for_language(text, language)
        
        # Calculate savings
        savings = self._calculate_token_savings(text, stripped_text)
        
        # Add metadata
        statistics = {
            "enabled": True,
            "language": language,
            **savings
        }
        
        # Log results
        self._log_stripping_results(statistics)
        
        return stripped_text, statistics
    
    def _log_stripping_results(self, stats: Dict):
        """Log comment stripping results."""
        if not stats.get("enabled", False):
            return
        
        language = stats.get("language", "unknown")
        chars_saved = stats.get("chars_saved", 0)
        char_percent = stats.get("char_reduction_percent", 0)
        
        if stats.get("method") == "tiktoken":
            tokens_saved = stats.get("tokens_saved", 0)
            token_percent = stats.get("token_reduction_percent", 0)
            logger.info(f"📝 Comment Stripping Results ({language}):")
            logger.info(f"   Characters: {stats.get('original_chars', 0):,} → {stats.get('stripped_chars', 0):,} ({chars_saved:,} saved, {char_percent:.1f}%)")
            logger.info(f"   Tokens: {stats.get('original_tokens', 0):,} → {stats.get('stripped_tokens', 0):,} ({tokens_saved:,} saved, {token_percent:.1f}%)")
        else:
            estimated_tokens = stats.get("estimated_tokens_saved", 0)
            logger.info(f"📝 Comment Stripping Results ({language}):")
            logger.info(f"   Characters: {stats.get('original_chars', 0):,} → {stats.get('stripped_chars', 0):,} ({chars_saved:,} saved, {char_percent:.1f}%)")
            logger.info(f"   Estimated tokens saved: {estimated_tokens:,}") 