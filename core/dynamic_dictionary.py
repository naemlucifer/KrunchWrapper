#!/usr/bin/env python3
"""
Optimized Multi-threaded Dynamic Dictionary Analyzer for KrunchWrapper
Fast pattern recognition and compression with parallel processing.
"""

import json
import re
import os
import sys
import time
import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import threading

# Add tiktoken for token validation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available for token validation in dynamic dictionary")

from .rules import load_dict
from .model_context import get_effective_model, normalize_model_name
from .model_tokenizer_validator import get_model_tokenizer_validator
from .model_specific_symbol_selector import get_model_specific_symbol_selector, get_optimal_symbols_for_current_model
from .model_optimized_symbol_pool import get_model_optimized_symbol_pool
from .parameterized_pattern_detector import get_parameterized_pattern_detector
from .adaptive_multipass_compressor import get_adaptive_multipass_compressor
from .tool_identifier import contains_tool_calls
from .markdown_identifier import contains_markdown_content
from .persistent_token_cache import get_persistent_cache

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc

logger = logging.getLogger(__name__)



class FastPatternExtractor:
    """Optimized pattern extraction using efficient algorithms with parallel processing."""
    
    def __init__(self, min_length=3, min_frequency=2, num_threads=4, max_ngram=100):
        self.min_length = min_length
        self.min_frequency = min_frequency
        self.num_threads = num_threads
        self.max_ngram = max_ngram
        self._pattern_cache = {}
        
        # Pre-compiled regex patterns for performance
        self.token_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b|\b\w+\.\w+\b')
        self.func_call_pattern = re.compile(r'\w+\([^)]*\)')
        self.import_pattern = re.compile(r'(?:from|import)\s+[\w.]+')
        self.structure_pattern = re.compile(r'(?:if|for|while|def|class)\s+[^:]+:')
        
        # Code detection patterns
        self.code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', '()', ';']
        
    def preprocess_for_compression(self, text: str) -> str:
        """🎯 NEW: Preprocess text for better pattern detection."""
        # Normalize whitespace in specific patterns
        text = re.sub(r'(\w+)\s*=\s*', r'\1 = ', text)
        text = re.sub(r',\s+', ', ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        
        # Normalize method calls and spacing
        text = re.sub(r'\.(\w+)\s*\(', r'.\1(', text)
        text = re.sub(r'\)\s*:', '):', text)
        
        # Normalize string spacing but preserve quotes
        text = re.sub(r'"\s*([^"]*)\s*"', r'"\1"', text)
        text = re.sub(r"'\s*([^']*)\s*'", r"'\1'", text)
        
        return text
    
    def extract_all_patterns(self, text: str) -> List[Tuple[str, int]]:
        """Extract patterns using multiple strategies in parallel."""
        # 🎯 NEW: Preprocess text for better pattern detection
        processed_text = self.preprocess_for_compression(text)
        
        # For small texts, use single-threaded approach
        if len(processed_text) < 5000:
            patterns = self._extract_patterns_single_thread(processed_text)
        else:
            # For large texts, use parallel processing
            patterns = self._extract_patterns_parallel(processed_text)
        
        # Add pattern consolidation step
        consolidated_patterns = self._consolidate_similar_patterns(patterns)
        logger.debug(f"Pattern consolidation: {len(patterns)} -> {len(consolidated_patterns)} patterns")
        
        return consolidated_patterns
    
    def _extract_patterns_single_thread(self, text: str) -> List[Tuple[str, int]]:
        """Single-threaded pattern extraction for smaller texts."""
        patterns = Counter()
        
        # Fast token extraction
        tokens = self._extract_tokens_fast(text)
        patterns.update(tokens)
        
        # N-gram extraction for repeated substrings
        ngrams = self._extract_ngrams_fast(text)
        patterns.update(ngrams)
        
        # Code-specific patterns if detected
        if self._looks_like_code(text):
            code_patterns = self._extract_code_patterns(text)
            patterns.update(code_patterns)
        
        return self._filter_and_sort_patterns(patterns)
    
    def _extract_patterns_parallel(self, text: str) -> List[Tuple[str, int]]:
        """Parallel pattern extraction for large texts."""
        # Split text into chunks for parallel processing
        chunk_size = max(2000, len(text) // self.num_threads)
        chunks = []
        
        # Create overlapping chunks to catch patterns at boundaries
        overlap = 200
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Process chunks in parallel
        patterns = Counter()
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_chunk = {
                executor.submit(self._extract_patterns_from_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_patterns = future.result()
                    for pattern, count in chunk_patterns.items():
                        patterns[pattern] += count
                except Exception as e:
                    logger.warning(f"Pattern extraction failed for chunk: {e}")
        
        return self._filter_and_sort_patterns(patterns)
    
    def _extract_patterns_from_chunk(self, chunk: str) -> Counter:
        """Extract patterns from a single chunk."""
        patterns = Counter()
        
        # Extract different types of patterns
        tokens = self._extract_tokens_fast(chunk)
        patterns.update(tokens)
        
        ngrams = self._extract_ngrams_fast(chunk, max_ngram=30)  # Smaller for chunks
        patterns.update(ngrams)
        
        if self._looks_like_code(chunk):
            code_patterns = self._extract_code_patterns(chunk)
            patterns.update(code_patterns)
            
            # Extract parameterized patterns for code
            param_detector = get_parameterized_pattern_detector()
            param_patterns = param_detector.detect_parameterized_patterns(chunk)
            
            # Convert parameterized patterns to standard format and add them
            standard_param_patterns = param_detector.convert_to_standard_patterns(param_patterns)
            for pattern, count in standard_param_patterns:
                patterns[pattern] += count
        
        return patterns
    
    def _extract_tokens_fast(self, text: str) -> Counter:
        """Fast token extraction using compiled regex."""
        tokens = self.token_pattern.findall(text)
        return Counter(tokens)
    
    def _extract_ngrams_fast(self, text: str, max_ngram=None) -> Counter:
        """Extract repeated substrings using efficient sliding window with configurable settings."""
        ngrams = Counter()
        text_len = len(text)
        
        # Use configured max_ngram or instance default
        if max_ngram is None:
            max_ngram = self.max_ngram
        
        # Configurable upper limit for pattern length
        upper_limit = min(max_ngram, max(50, text_len // 20))  # At least 50 char patterns
        
        for length in range(self.min_length, upper_limit):
            seen = {}
            for i in range(text_len - length + 1):
                ngram = text[i:i + length]
                # Quick quality filter
                if self._is_meaningful_ngram(ngram):
                    if ngram in seen:
                        ngrams[ngram] += 1
                    else:
                        seen[ngram] = i
        
        # 🎯 NEW: Also try line-based patterns for code
        lines = text.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) >= self.min_length:
                ngrams[stripped] += 1
                
                # Check for similar lines in the next 50 lines
                for j in range(i + 1, min(i + 51, len(lines))):
                    if lines[j].strip() == stripped and len(stripped) >= self.min_length:
                        ngrams[stripped] += 1
        
        return ngrams
    
    def _is_meaningful_ngram(self, ngram: str) -> bool:
        """Quick check if ngram is worth considering."""
        # Must contain some alphabetic characters
        if not any(c.isalpha() for c in ngram):
            return False
        # Avoid mostly whitespace or punctuation
        if len(ngram.strip()) < self.min_length // 2:
            return False
        return True
    
    def _extract_code_patterns(self, text: str) -> Counter:
        """Extract code-specific patterns efficiently."""
        patterns = Counter()
        
        # Extract specific code patterns that commonly appear in repetitive code
        code_specific_patterns = self._extract_code_specific_patterns(text)
        patterns.update(code_specific_patterns)
        
        lines = text.split('\n')
        
        # Process lines in batches for efficiency
        batch_size = 100
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            batch_text = '\n'.join(batch)
            
            # Extract patterns using pre-compiled regex
            func_calls = self.func_call_pattern.findall(batch_text)
            patterns.update(func_calls)
            
            imports = self.import_pattern.findall(batch_text)
            patterns.update(imports)
            
            structures = self.structure_pattern.findall(batch_text)
            patterns.update(structures)
        
        return patterns
    
    def _extract_parameterized_patterns(self, text: str) -> Counter:
        """🎯 NEW: Extract parameterized patterns with variable parts."""
        patterns = Counter()
        
        # Method calls with arguments - group by method name
        method_call_regex = r'(\w+\.get)\(([^,]+),\s*([^)]+)\)'
        matches = list(re.finditer(method_call_regex, text))
        
        if len(matches) >= 2:
            # Group by method name
            method_groups = defaultdict(list)
            for match in matches:
                method = match.group(1)
                full_call = match.group(0)
                method_groups[method].append(full_call)
            
            # Create parameterized patterns
            for method, calls in method_groups.items():
                if len(calls) >= 2:
                    # Use the full pattern multiple times
                    for call in calls:
                        patterns[call] += 1
        
        # Config access patterns
        config_regex = r'(\w+)\.get\("([^"]+)",\s*([^)]+)\)'
        for match in re.finditer(config_regex, text):
            patterns[match.group(0)] += 1
        
        # Assignment patterns
        assignment_regex = r'(\w+)\s*=\s*(\w+)\.get\([^)]+\)'
        for match in re.finditer(assignment_regex, text):
            patterns[match.group(0)] += 1
        
        # Log message patterns with format strings  
        log_format_regex = r'log_message\(f"[^"]*\{[^}]+\}[^"]*"\)'
        for match in re.finditer(log_format_regex, text):
            patterns[match.group(0)] += 1
        
        # Timing breakdown patterns
        timing_regex = r'timing_breakdown\.get\([^)]+\)'
        for match in re.finditer(timing_regex, text):
            patterns[match.group(0)] += 1
        
        # Variable assignment patterns with common prefixes
        common_assignment_patterns = [
            r'\w+_chars_saved = [^=\n]+',
            r'\w+_tokens_saved = [^=\n]+',
            r'\w+_stats\.get\([^)]+\)',
            r'comment_\w+ = [^=\n]+',
            r'config_data\.get\([^)]+\)',
            r'logger\.\w+\(f"[^"]*"\)',
        ]
        
        for pattern_regex in common_assignment_patterns:
            for match in re.finditer(pattern_regex, text):
                if len(match.group()) < 100:  # Reasonable limit
                    patterns[match.group()] += 1
        
        # Format string patterns (f-strings with similar structures)
        format_patterns = [
            r'f"[^"]*→[^"]*: \{[^}]+\}s \(\{[^}]+\}%\)"',
            r'f"[^"]*: \{[^}]+\}s"',
            r'f"[^"]*\{[^}]+\}[^"]*characters"',
            r'f"[^"]*\{[^}]+\}[^"]*patterns"',
        ]
        
        for pattern_regex in format_patterns:
            for match in re.finditer(pattern_regex, text):
                if len(match.group()) < 120:
                    patterns[match.group()] += 1
        
        return patterns
    
    def _extract_code_specific_patterns(self, text: str) -> Counter:
        """Extract commonly repeated code fragments with aggressive detection."""
        patterns = Counter()
        
        # 🎯 NEW: Add parameterized pattern detection first
        param_patterns = self._extract_parameterized_patterns(text)
        patterns.update(param_patterns)
        
        # 1. Enhanced logging patterns (high frequency in analysis)
        logging_patterns = [
            r'logger\.info\([^)]*\)',
            r'logger\.debug\([^)]*\)',
            r'logger\.warning\([^)]*\)',
            r'logger\.error\([^)]*\)',
            r'logging\.getLogger\([^)]*\)',
            r'log\.info\([^)]*\)',
            r'log\.debug\([^)]*\)',
            r'log\.error\([^)]*\)',
        ]
        
        for pattern in logging_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 120:  # Reasonable limit
                    patterns[match.group()] += 1
        
        # 2. Common method calls and attribute access
        common_prefixes = [
            'self.', 'logger.', 'config.', 'analyzer.', 'data.', 'result.', 
            'item.', 'obj.', 'ctx.', 'req.', 'res.', 'app.', 'client.',
            'server.', 'db.', 'model.', 'parser.', 'handler.', 'manager.',
            'service.', 'utils.', 'helpers.', 'tools.', 'api.', 'cache.'
        ]
        
        for prefix in common_prefixes:
            # Method calls with the prefix
            pattern = rf'{re.escape(prefix)}\w+\([^)]*\)'
            for match in re.finditer(pattern, text):
                call = match.group()
                if len(call) < 100:
                    patterns[call] += 1
            
            # Simple attribute access
            pattern = rf'{re.escape(prefix)}\w+'
            for match in re.finditer(pattern, text):
                attr = match.group()
                if len(attr) < 50:
                    patterns[attr] += 1
        
        # 3. Function definitions and calls (comprehensive)
        function_patterns = [
            r'def \w+\(self[^)]*\):',
            r'def \w+\([^)]*\):',
            r'async def \w+\([^)]*\):',
            r'lambda [^:]+:',
            r'\w+\([^)]{1,80}\)',  # Function calls with reasonable argument length
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 120:
                    patterns[match.group()] += 1
        
        # 4. Import statements (very common in Python)
        import_patterns = [
            r'from [\w.]+ import [\w., ]+',
            r'import [\w., ]+',
            r'from \. import \w+',
            r'from \.[\w.]+ import \w+',
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 100:
                    patterns[match.group()] += 1
        
        # 5. Variable assignments and declarations
        assignment_patterns = [
            r'\w+ = [^=\n]{1,60}',  # Variable assignments
            r'\w+: [\w\[\]]+',      # Type annotations
            r'\w+: [\w\[\]]+ = [^=\n]{1,40}',  # Typed assignments
        ]
        
        for pattern in assignment_patterns:
            for match in re.finditer(pattern, text):
                assignment = match.group()
                if 10 < len(assignment) < 80:
                    patterns[assignment] += 1
        
        # 6. Control flow and error handling
        control_patterns = [
            r'if [^:]{1,50}:',
            r'elif [^:]{1,50}:',
            r'for \w+ in [^:]{1,50}:',
            r'while [^:]{1,50}:',
            r'try:',
            r'except \w+ as \w+:',
            r'except \w+:',
            r'finally:',
            r'with [^:]{1,50}:',
        ]
        
        for pattern in control_patterns:
            for match in re.finditer(pattern, text):
                patterns[match.group()] += 1
        
        # 7. String operations and formatting
        string_patterns = [
            r'f"[^"]*{[^}]+}[^"]*"',  # f-strings
            r'f\'[^\']*{[^}]+}[^\']*\'',
            r'"[^"]*"\.format\([^)]*\)',
            r'\'[^\']*\'\.format\([^)]*\)',
            r'\.join\([^)]+\)',
            r'\.split\([^)]*\)',
            r'\.strip\(\)',
            r'\.replace\([^)]+\)',
        ]
        
        for pattern in string_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 100:
                    patterns[match.group()] += 1
        
        # 8. Common data structure operations
        data_patterns = [
            r'\.get\([^)]+\)',
            r'\.items\(\)',
            r'\.keys\(\)',
            r'\.values\(\)',
            r'\.append\([^)]+\)',
            r'\.extend\([^)]+\)',
            r'\.update\([^)]+\)',
            r'len\([^)]+\)',
            r'enumerate\([^)]+\)',
            r'zip\([^)]+\)',
            r'range\([^)]+\)',
        ]
        
        for pattern in data_patterns:
            for match in re.finditer(pattern, text):
                patterns[match.group()] += 1
        
        # 9. Class and decorator patterns
        class_patterns = [
            r'class \w+\([^)]*\):',
            r'@\w+',
            r'@\w+\([^)]*\)',
            r'super\(\)\.\w+\([^)]*\)',
            r'self\.\w+ = [^=\n]{1,40}',
        ]
        
        for pattern in class_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 100:
                    patterns[match.group()] += 1
        
        # 10. Common expressions and operators
        expression_patterns = [
            r'is not None',
            r'is None',
            r'not in ',
            r' in ',
            r' and ',
            r' or ',
            r' == ',
            r' != ',
            r' >= ',
            r' <= ',
            r' > ',
            r' < ',
        ]
        
        for pattern in expression_patterns:
            count = text.count(pattern)
            if count >= 2:  # Must appear at least twice
                patterns[pattern] += count
        
        # 11. JSON and configuration patterns
        config_patterns = [
            r'config\[\'[^\']+\'\]',
            r'config\["[^"]+"\]',
            r'\.get\(\'[^\']+\', [^)]+\)',
            r'\.get\("[^"]+", [^)]+\)',
            r'json\.loads\([^)]+\)',
            r'json\.dumps\([^)]+\)',
        ]
        
        for pattern in config_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 80:
                    patterns[match.group()] += 1
        
        # 12. File and path operations
        file_patterns = [
            r'open\([^)]+\)',
            r'with open\([^)]+\) as \w+:',
            r'\.read\(\)',
            r'\.write\([^)]+\)',
            r'\.close\(\)',
            r'Path\([^)]+\)',
            r'os\.path\.\w+\([^)]+\)',
            r'\.exists\(\)',
            r'\.mkdir\([^)]*\)',
        ]
        
        for pattern in file_patterns:
            for match in re.finditer(pattern, text):
                if len(match.group()) < 100:
                    patterns[match.group()] += 1
        
        return patterns
    
    def _looks_like_code(self, text: str) -> bool:
        """Quick heuristic to detect if text is code."""
        # Check first 1000 characters for performance
        sample = text[:1000]
        indicator_count = sum(1 for ind in self.code_indicators if ind in sample)
        return indicator_count >= 3
    
    def _filter_and_sort_patterns(self, patterns: Counter) -> List[Tuple[str, int]]:
        """Filter patterns by frequency and length, then sort by value."""
        filtered = []
        
        for pattern, count in patterns.items():
            # Basic frequency and length check
            if count < self.min_frequency or len(pattern) < self.min_length:
                continue
            
            # Enhanced quality filters
            if not self._is_quality_pattern(pattern):
                continue
            
            filtered.append((pattern, count))
        
        # Sort by compression value (count * length)
        return sorted(filtered, key=lambda x: x[1] * len(x[0]), reverse=True)
    
    def _is_quality_pattern(self, pattern: str) -> bool:
        """Check if a pattern meets quality criteria for compression."""
        # Avoid very short patterns unless they're very frequent
        if len(pattern) < 4:
            return False
        
        # Avoid patterns that are mostly punctuation or whitespace
        alphanumeric_chars = sum(1 for c in pattern if c.isalnum())
        if alphanumeric_chars < len(pattern) * 0.4:  # Less than 40% alphanumeric
            return False
        
        # Avoid patterns that are just whitespace with minimal content
        stripped = pattern.strip()
        if len(stripped) < 3:
            return False
        
        # Avoid patterns that are just repeated characters
        unique_chars = len(set(pattern.lower()))
        if len(pattern) > 6 and unique_chars < 3:  # Long patterns with too few unique chars
            return False
        
        # Prefer patterns with some structure (word boundaries, identifiers, etc.)
        has_structure = any([
            '_' in pattern and pattern.replace('_', '').isalnum(),  # Identifier-like
            '.' in pattern and not pattern.startswith(' '),        # Method calls
            '(' in pattern,                                        # Function calls
            pattern.strip().isalnum() and len(pattern.strip()) >= 4, # Clean words
            any(word in pattern.lower() for word in ['log', 'msg', 'data', 'config', 'file', 'json', 'def', 'class']), # Common code terms
        ])
        
        if not has_structure:
            # Allow patterns without obvious structure if they're reasonably long and clean
            if len(pattern) < 8 or pattern != pattern.strip():
                return False
        
        return True
    
    def _consolidate_similar_patterns(self, patterns: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Consolidate similar patterns to avoid duplication and improve compression efficiency."""
        if not patterns:
            return patterns
        
        logger.debug(f"Consolidating {len(patterns)} patterns...")
        
        # Convert to dict for easier manipulation
        pattern_dict = {pattern: count for pattern, count in patterns}
        
        # Step 1: Normalize whitespace variations of the same core pattern
        normalized_groups = self._group_whitespace_variations(pattern_dict)
        
        # Step 2: Handle containment relationships (longer patterns containing shorter ones)
        containment_resolved = self._resolve_containment_conflicts(normalized_groups)
        
        # Step 3: Merge very similar patterns using edit distance
        similarity_merged = self._merge_similar_patterns(containment_resolved)
        
        # Convert back to list and sort by value
        consolidated = [(pattern, count) for pattern, count in similarity_merged.items()]
        consolidated.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
        
        # Log consolidation results
        if len(consolidated) != len(patterns):
            logger.debug(f"Consolidation examples:")
            for i, (pattern, count) in enumerate(consolidated[:5]):
                original_matches = [p for p, c in patterns if p == pattern or self._patterns_similar(p, pattern)]
                if len(original_matches) > 1:
                    logger.debug(f"  {i+1}. Consolidated '{pattern}' (count: {count}) from {len(original_matches)} variants")
        
        return consolidated
    
    def _group_whitespace_variations(self, pattern_dict: Dict[str, int]) -> Dict[str, int]:
        """Group patterns that differ only in leading/trailing whitespace."""
        groups = defaultdict(list)
        
        # Group patterns by their stripped version
        for pattern, count in pattern_dict.items():
            stripped = pattern.strip()
            if stripped:  # Only process non-empty stripped patterns
                groups[stripped].append((pattern, count))
        
        # For each group, pick the best representative
        consolidated = {}
        for stripped_pattern, variants in groups.items():
            if len(variants) == 1:
                # Single pattern, keep as-is
                pattern, count = variants[0]
                consolidated[pattern] = count
            else:
                # Multiple variants, consolidate them
                total_count = sum(count for _, count in variants)
                
                # Choose the best representative (prefer most common whitespace pattern)
                # or the longest one if counts are similar
                best_pattern = max(variants, key=lambda x: (x[1], len(x[0])))[0]
                
                consolidated[best_pattern] = total_count
                
                logger.debug(f"Whitespace consolidation: merged {len(variants)} variants of '{stripped_pattern}' -> '{best_pattern}' (total count: {total_count})")
        
        return consolidated
    
    def _resolve_containment_conflicts(self, pattern_dict: Dict[str, int]) -> Dict[str, int]:
        """Resolve conflicts where one pattern is contained within another."""
        patterns_by_length = sorted(pattern_dict.items(), key=lambda x: len(x[0]), reverse=True)
        resolved = {}
        
        for pattern, count in patterns_by_length:
            # Check if this pattern is contained in any longer pattern we've already processed
            is_contained = False
            containing_pattern = None
            
            for existing_pattern in resolved:
                if pattern != existing_pattern and pattern in existing_pattern:
                    # This pattern is contained in an existing longer pattern
                    is_contained = True
                    containing_pattern = existing_pattern
                    break
            
            if is_contained:
                # Decide whether to merge or keep separate
                containing_count = resolved[containing_pattern]
                
                # If the contained pattern has significantly more occurrences,
                # it might be worth keeping both
                if count > containing_count * 2:
                    # Keep both patterns
                    resolved[pattern] = count
                    logger.debug(f"Kept contained pattern '{pattern}' (count: {count}) despite being in '{containing_pattern}' (count: {containing_count})")
                else:
                    # Merge into the containing pattern
                    resolved[containing_pattern] += count
                    logger.debug(f"Merged contained pattern '{pattern}' into '{containing_pattern}' (total count: {resolved[containing_pattern]})")
            else:
                # Pattern is not contained, keep it
                resolved[pattern] = count
        
        return resolved
    
    def _merge_similar_patterns(self, pattern_dict: Dict[str, int]) -> Dict[str, int]:
        """Merge patterns that are very similar based on edit distance."""
        patterns = list(pattern_dict.items())
        merged = {}
        processed = set()
        
        for i, (pattern1, count1) in enumerate(patterns):
            if pattern1 in processed:
                continue
            
            # Find all similar patterns
            similar_group = [(pattern1, count1)]
            
            for j, (pattern2, count2) in enumerate(patterns[i+1:], i+1):
                if pattern2 in processed:
                    continue
                
                if self._patterns_similar(pattern1, pattern2):
                    similar_group.append((pattern2, count2))
                    processed.add(pattern2)
            
            # If we found similar patterns, merge them
            if len(similar_group) > 1:
                # Choose the best representative (longest or most frequent)
                best_pattern, best_count = max(similar_group, key=lambda x: (len(x[0]), x[1]))
                total_count = sum(count for _, count in similar_group)
                
                merged[best_pattern] = total_count
                processed.add(pattern1)
                
                similar_patterns = [p for p, c in similar_group if p != best_pattern]
                logger.debug(f"Similarity merge: '{best_pattern}' absorbed {len(similar_patterns)} similar patterns, total count: {total_count}")
            else:
                # No similar patterns found, keep as-is
                merged[pattern1] = count1
                processed.add(pattern1)
        
        return merged
    
    def _patterns_similar(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are similar enough to be merged."""
        # Skip if patterns are identical
        if pattern1 == pattern2:
            return False
        
        # Skip if length difference is too large
        len_diff = abs(len(pattern1) - len(pattern2))
        max_len = max(len(pattern1), len(pattern2))
        if len_diff > max_len * 0.3:  # More than 30% length difference
            return False
        
        # Calculate simple similarity based on common characters
        set1 = set(pattern1.lower())
        set2 = set(pattern2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return False
        
        jaccard_similarity = intersection / union
        
        # For short patterns, require higher similarity
        if max_len < 10:
            return jaccard_similarity > 0.8
        else:
            return jaccard_similarity > 0.7


class OptimizedSymbolAssigner:
    """Fast symbol assignment with caching and pre-computation."""
    
    # Class-level cache to track which model/tokenizer combinations have been initialized
    _initialization_logged = set()
    _init_lock = threading.Lock()
    
    def __init__(self, tokenizer=None, model_name=None):
        self.tokenizer = tokenizer
        self.model_name = model_name or get_effective_model() or "gpt-4"
        self._symbol_costs = {}
        self._token_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize persistent cache
        self._persistent_cache = get_persistent_cache()
        
        self._initialize_optimized_symbol_pool()
    
    def _initialize_optimized_symbol_pool(self):
        """Initialize optimized symbol pool using ModelOptimizedSymbolPool."""
        # Create a key for this initialization to prevent duplicate logging
        tokenizer_name = getattr(self.tokenizer, 'name', 'unknown') if self.tokenizer else 'none'
        init_key = f"{self.model_name}:{tokenizer_name}"
        
        # Check if we've already logged initialization for this model/tokenizer combination
        with self._init_lock:
            should_log = init_key not in self._initialization_logged
            if should_log:
                self._initialization_logged.add(init_key)
        
        logger.debug("Initializing optimized symbol pool...")
        
        # Get the optimized symbol pool
        symbol_pool = get_model_optimized_symbol_pool(self.tokenizer)
        
        # Get optimal symbols for the current model  
        optimal_symbol_data = symbol_pool.get_optimal_symbols(
            model_name=self.model_name,
            max_symbols=262,  # Match max_dictionary_size for consistency
            exclude_symbols=set(),
            content=None  # No content available during initialization
        )
        
        # Extract just the symbol strings and compute costs
        self.priority_symbols = []
        efficient_count = 0
        
        for symbol, token_count in optimal_symbol_data:
            self.priority_symbols.append(symbol)
            self._symbol_costs[symbol] = token_count
            if token_count == 1:
                efficient_count += 1
        
        # Only log detailed initialization info once per model/tokenizer combination
        if should_log:
            logger.info(f"Initialized optimized symbol pool for {self.model_name}:")
            logger.info(f"  Total symbols: {len(self.priority_symbols)}")
            logger.info(f"  Single-token symbols: {efficient_count} ({100*efficient_count/max(1,len(self.priority_symbols)):.1f}%)")
            logger.info(f"  Average tokens per symbol: {sum(self._symbol_costs.values())/max(1,len(self._symbol_costs)):.2f}")
            
            # Log sample of most efficient symbols
            single_token_symbols = [s for s in self.priority_symbols if self._symbol_costs.get(s) == 1]
            if single_token_symbols:
                logger.debug(f"Top efficient symbols: {single_token_symbols[:15]}")
            else:
                logger.warning("No single-token symbols found! This may cause negative compression.")
        else:
            logger.debug(f"Symbol pool already initialized for {self.model_name} (using cached initialization)")
            
    def update_model_context(self, model_name: str):
        """Update symbol pool when model context changes."""
        if model_name != self.model_name:
            logger.debug(f"Updating symbol pool from {self.model_name} to {model_name}")
            self.model_name = model_name
            self._initialize_optimized_symbol_pool()
    
    def get_content_aware_symbols(self, content: str, max_symbols: int = 100) -> List[Tuple[str, int]]:
        """Get symbols optimized for specific content to avoid conflicts."""
        symbol_pool = get_model_optimized_symbol_pool(self.tokenizer)
        
        return symbol_pool.get_optimal_symbols(
            model_name=self.model_name,
            max_symbols=max_symbols,
            exclude_symbols=set(),
            content=content
        )
            
    # Keep the old method for backward compatibility
    def _initialize_symbol_pool(self):
        """Deprecated - use _initialize_optimized_symbol_pool instead."""
        logger.warning("Using deprecated _initialize_symbol_pool method. Consider updating to _initialize_optimized_symbol_pool.")
        self._initialize_optimized_symbol_pool()
    
    def get_token_count(self, text: str) -> int:
        """Cached token counting for performance with persistent disk backing."""
        try:
            # Try persistent cache first
            cache_key = f"tokenizer:{getattr(self.tokenizer, 'name', 'unknown')}:{text}"
            cached_result = self._persistent_cache.get(cache_key, "token_count")
            if cached_result is not None:
                return cached_result
            
            # Compute token count
            if not self.tokenizer:
                result = max(1, len(text) // 4)  # Conservative character-based estimate
            else:
                try:
                    result = len(self.tokenizer.encode(text))
                except Exception:
                    result = max(1, len(text) // 4)
            
            # Cache the result
            self._persistent_cache.set(cache_key, result, "token_count")
            return result
        except Exception as e:
            logger.warning(f"Error in get_token_count for text '{text[:50]}...': {e}")
            # Fallback to simple character-based estimate without caching
            return max(1, len(text) // 4)
    
    def calculate_assignment_value(self, pattern: str, count: int, symbol: str) -> float:
        """Calculate compression value with comprehensive token boundary testing."""
        if not pattern or not symbol:
            return 0.0

        if not self.tokenizer:
            # Fallback to character-based estimation
            char_savings = (len(pattern) - len(symbol)) * count
            overhead = len(f"{pattern}={symbol}, ") 
            return max(0, char_savings - overhead) / 4  # Rough chars-to-tokens

        # Test pattern in multiple realistic contexts to account for token boundaries
        test_contexts = [
            f" {pattern} ",         # Surrounded by spaces
            f"\n{pattern}\n",       # On its own line
            f"({pattern})",         # In parentheses
            f".{pattern}",          # After dot
            f"{pattern}(",          # Before parenthesis
            f'"{pattern}"',         # In quotes
            f"'{pattern}'",         # In single quotes
            f"[{pattern}]",         # In brackets
            f"{pattern}:",          # With colon
            f"={pattern}",          # After equals
            f"{pattern}=",          # Before equals
        ]
        
        # SIMPLIFIED: Use direct tokenization for more predictable results
        try:
            # Use simple standalone tokenization (most predictable)
            pattern_tokens = len(self.tokenizer.encode(pattern))
            symbol_tokens = len(self.tokenizer.encode(symbol))
            raw_savings_per_occurrence = max(0, pattern_tokens - symbol_tokens)
            
            # ENHANCED: Add sanity check for unrealistic savings
            max_reasonable_savings = len(pattern) / 3  # Very generous estimate of tokens per char
            if raw_savings_per_occurrence > max_reasonable_savings:
                logger.debug(f"Capping unrealistic savings: {raw_savings_per_occurrence} -> {max_reasonable_savings:.1f} for pattern '{pattern[:30]}...'")
                savings_per_occurrence = max_reasonable_savings
            else:
                savings_per_occurrence = raw_savings_per_occurrence
            
            # If we get negative or zero savings, this substitution is not worth it
            if savings_per_occurrence <= 0:
                return -1.0  # Signal that this assignment is counterproductive
                
        except Exception:
            # Fallback to cached/standalone tokenization
            try:
                pattern_tokens = len(self.tokenizer.encode(pattern))
                symbol_tokens = len(self.tokenizer.encode(symbol))
                raw_savings = max(0, pattern_tokens - symbol_tokens)
                
                # Apply same sanity check
                max_reasonable_savings = len(pattern) / 3
                savings_per_occurrence = min(raw_savings, max_reasonable_savings)
            except Exception:
                pattern_tokens = self.get_token_count(pattern)
                symbol_tokens = self._symbol_costs.get(symbol, self.get_token_count(symbol))
                raw_savings = max(0, pattern_tokens - symbol_tokens)
                
                # Apply same sanity check
                max_reasonable_savings = len(pattern) / 3
                savings_per_occurrence = min(raw_savings, max_reasonable_savings)
        
        # Calculate total savings
        total_savings = savings_per_occurrence * count
        
        # Calculate overhead cost more accurately
        # The dictionary entry format in the decoder
        dictionary_entry = f"\n{pattern}={symbol}"
        try:
            overhead_cost = len(self.tokenizer.encode(dictionary_entry))
        except Exception:
            overhead_cost = max(2, len(dictionary_entry) // 4)
        
        # Net value = total savings - overhead cost
        net_value = total_savings - overhead_cost
        
        # 🎯 SUPER AGGRESSIVE: Be less conservative for high-frequency patterns
        if count >= 5 and net_value > -2:  # Allow slightly negative if very frequent
            net_value = max(0.1, net_value)  # Give it a small positive value
        
        # ENHANCED: Log details for debugging with more information
        if total_savings >= 2 or net_value < 0:  # Log negative values and significant savings
            pattern_preview = pattern[:30] + "..." if len(pattern) > 30 else pattern
            logger.debug(f"Pattern '{pattern_preview}' -> '{symbol}': "
                        f"saves {savings_per_occurrence} tokens/occurrence × {count} = {total_savings} total, "
                        f"overhead = {overhead_cost}, net = {net_value:.2f}")
            
            # ENHANCED: Log suspicious calculations
            if net_value > total_savings:
                logger.warning(f"Suspicious calculation: net_value ({net_value:.2f}) > total_savings ({total_savings})")
            if savings_per_occurrence > len(pattern) / 2:  # Unrealistic token savings
                logger.debug(f"High per-occurrence savings: {savings_per_occurrence} tokens for {len(pattern)}-char pattern")
        
        return net_value
    
    def assign_symbols_fast(self, patterns: List[Tuple[str, int]], max_assignments: int = 262, content: str = None, exclude_symbols: set = None) -> Dict[str, str]:
        """Fast symbol assignment using greedy algorithm with value optimization."""
        assignments = {}
        used_symbols = set()
        
        # CRITICAL FIX: Honor excluded symbols to prevent collisions within same request
        if exclude_symbols:
            used_symbols.update(exclude_symbols)
            logger.debug(f"Starting symbol assignment with {len(exclude_symbols)} excluded symbols to prevent collisions")
            logger.debug(f"Excluded symbols: {list(exclude_symbols)[:10]}{'...' if len(exclude_symbols) > 10 else ''}")
        
        logger.debug(f"Starting symbol assignment for {len(patterns)} patterns")
        
        # Use content-aware symbol selection if content is provided
        if content:
            logger.debug("Using content-aware symbol selection to avoid conflicts")
            content_aware_symbols = self.get_content_aware_symbols(content, max_symbols=262)
            available_symbol_pool = [symbol for symbol, _ in content_aware_symbols]
            logger.debug(f"Content-aware symbols: {len(available_symbol_pool)} symbols available")
            
            # Log what symbols we're using
            unicode_symbols = [s for s in available_symbol_pool if len(s) == 1 and ord(s) > 127]
            ascii_symbols = [s for s in available_symbol_pool if s not in unicode_symbols]
            logger.debug(f"  Unicode symbols: {len(unicode_symbols)} (e.g., {unicode_symbols[:10]})")
            if ascii_symbols:
                logger.debug(f"  ASCII symbols: {len(ascii_symbols)} (e.g., {ascii_symbols[:5]})")
        else:
            logger.debug("Using default symbol pool")
            available_symbol_pool = self.priority_symbols
        
        # Filter out excluded symbols from available pool
        if exclude_symbols:
            original_pool_size = len(available_symbol_pool)
            available_symbol_pool = [s for s in available_symbol_pool if s not in exclude_symbols]
            excluded_count = original_pool_size - len(available_symbol_pool)
            if excluded_count > 0:
                logger.debug(f"Filtered out {excluded_count} excluded symbols, {len(available_symbol_pool)} symbols remaining")
        
        # Sort patterns by potential value first (using a quick estimate)
        patterns_with_values = []
        rejected_patterns = []
        
        for pattern, count in patterns:
            # Quick value estimation using best symbol (ignoring availability for ranking)
            best_value = 0
            for symbol in available_symbol_pool[:5]:  # Test top 5 for ranking
                value = self.calculate_assignment_value(pattern, count, symbol)
                if value > best_value:
                    best_value = value
            
            if best_value > 0:
                patterns_with_values.append((pattern, count, best_value))
            else:
                rejected_patterns.append((pattern, count, best_value))
        
        logger.debug(f"Pattern pre-evaluation: {len(patterns_with_values)} accepted, {len(rejected_patterns)} rejected")
        
        # Sort by potential value (highest first)
        patterns_with_values.sort(key=lambda x: x[2], reverse=True)
        
        # Now assign symbols greedily, recalculating best available symbol for each pattern
        for pattern, count, estimated_value in patterns_with_values:
            if len(assignments) >= max_assignments:
                break
            
            # Find the best available symbol for this specific pattern
            best_value = 0
            best_symbol = None
            
            # Test available symbols to find the actual best choice
            available_symbols = [s for s in available_symbol_pool if s not in used_symbols]
            
            # Limit search to reasonable number for performance
            test_limit = min(20, len(available_symbols))
            for symbol in available_symbols[:test_limit]:
                value = self.calculate_assignment_value(pattern, count, symbol)
                if value > best_value:
                    best_value = value
                    best_symbol = symbol
            
            # Only assign if we found a worthwhile symbol
            if best_symbol and best_value > 0:
                # FIXED: Use symbol -> pattern format consistently throughout
                assignments[best_symbol] = pattern
                used_symbols.add(best_symbol)
                
                logger.debug(f"Assigned '{pattern}' -> '{best_symbol}' (count: {count}, value: {best_value:.1f})")
            else:
                logger.debug(f"No suitable symbol found for '{pattern}' (count: {count})")
                if len(available_symbols) == 0:
                    logger.warning(f"Ran out of available symbols! Used: {len(used_symbols)}, Total available: {len(available_symbol_pool)}")
                    break

        logger.debug(f"Final assignments: {len(assignments)} patterns assigned symbols")
        
        # Log symbol usage statistics
        if assignments:
            logger.debug(f"Symbols used: {len(used_symbols)}/{len(available_symbol_pool)}")
            logger.debug(f"Top 5 assignments:")
            
            # Sort assignments by value for display
            assignment_values = []
            # FIXED: assignments is now symbol -> pattern format
            for symbol, pattern in list(assignments.items())[:5]:
                count = next((c for p, c in patterns if p == pattern), 0)
                value = self.calculate_assignment_value(pattern, count, symbol)
                assignment_values.append((pattern, symbol, count, value))
            
            assignment_values.sort(key=lambda x: x[3], reverse=True)
            for i, (pattern, symbol, count, value) in enumerate(assignment_values):
                logger.debug(f"  {i+1}. '{pattern}' -> '{symbol}' (count: {count}, value: {value:.1f})")
        
        return assignments


class ParallelPatternAnalyzer:
    """Multi-threaded pattern analysis for large texts."""
    
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
    



class DynamicDictionaryAnalyzer:
    """
    Optimized multi-threaded dynamic dictionary analyzer with fast pattern recognition.
    Completely replaces the previous implementation with performance-focused approach.
    """
    
    # Class-level cache to track if analyzer initialization has been logged
    _analyzer_init_logged = set()
    _analyzer_init_lock = threading.Lock()
    
    def __init__(self, temp_dir: str = "temp"):
        """Initialize the optimized dynamic dictionary analyzer."""
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Get thread count from config
        self.num_threads = self.config.get("threads", 4)
        
        # Initialize components with threading support
        self.pattern_extractor = FastPatternExtractor(
            min_length=self.config.get("min_token_length", 3),
            min_frequency=self.config.get("min_frequency", 2),
            num_threads=self.num_threads,
            max_ngram=self.config.get("max_ngram_length", 100)
        )
        
        # Initialize tokenizer for symbol assignment
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Get current model for optimized symbol selection
        current_model = get_effective_model() or "gpt-4"
        self.symbol_assigner = OptimizedSymbolAssigner(self.tokenizer, current_model)
        self.parallel_analyzer = ParallelPatternAnalyzer(self.num_threads)
        
        # Only log analyzer initialization once per thread count/temp_dir combination
        init_key = f"{temp_dir}:{self.num_threads}"
        with self._analyzer_init_lock:
            should_log = init_key not in self._analyzer_init_logged
            if should_log:
                self._analyzer_init_logged.add(init_key)
                logger.info(f"Initialized optimized dynamic dictionary analyzer with {self.num_threads} threads")
            else:
                logger.debug(f"Reusing dynamic dictionary analyzer configuration ({self.num_threads} threads)")
    
    def _load_config(self) -> Dict:
        """Load configuration with dynamic dictionary and threading settings."""
        config_path = Path(__file__).parent.parent / "config" / "config.jsonc"
        
        # Default configuration with updated aggressive settings
        default_config = {
            "enabled": True,
            "min_token_length": 3,
            "min_frequency": 2,
            "max_dictionary_size": 262,  # Match config.jsonc setting
            "compression_threshold": 0.01,
            "min_prompt_length": 1000,
            "auto_detection_threshold": 0.20,
            "threads": 4,  # Default thread count
            "cleanup_max_age_hours": 24,
            "auto_cleanup": True
        }
        
        try:
            if config_path.exists():
                full_config = load_jsonc(str(config_path))
                
                # Extract dynamic dictionary config
                dynamic_config = full_config.get("dynamic_dictionary", {})
                
                # Extract thread count from compression config
                compression_config = full_config.get("compression", {})
                threads = compression_config.get("threads", 4)
                
                # Merge configs
                config = default_config.copy()
                config.update(dynamic_config)
                config["threads"] = threads  # Override with compression thread count
                
                logger.debug(f"Loaded config: {config['max_dictionary_size']} max entries, {threads} threads")
                return config
            else:
                logger.warning(f"Config file not found, using defaults with {default_config['threads']} threads")
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return default_config
    
    def is_enabled(self) -> bool:
        """Check if dynamic dictionary analysis is enabled."""
        return self.config.get("enabled", True)
    
    def should_analyze_prompt(self, prompt: str, skip_tool_detection: bool = False, cline_mode: bool = False) -> Tuple[bool, str]:
        """Determine if dynamic compression should be used based on prompt analysis."""
        if not self.is_enabled():
            return False, "Dynamic dictionary analysis is disabled"
        
        # Skip for tool calls and markdown to preserve structure (unless disabled for Cline compatibility)
        if not skip_tool_detection and not cline_mode and contains_tool_calls(prompt):
            return False, "Tool calls detected - preserving JSON structure"
        elif not skip_tool_detection and cline_mode and contains_tool_calls(prompt):
            # In Cline mode, we allow compression of tool call examples in system prompts
            from .tool_identifier import ToolCallIdentifier
            logger.debug("Cline mode: allowing compression despite tool call examples in system prompt")
        
        # Skip for markdown content UNLESS we're in cline_mode for system prompts
        if not cline_mode and contains_markdown_content(prompt):
            return False, "Markdown content detected - preserving formatting"
        elif cline_mode and contains_markdown_content(prompt):
            # In Cline mode, we allow compression of markdown content but log it
            from .markdown_identifier import MarkdownIdentifier
            logger.debug("Cline mode: allowing compression despite markdown content detection")
        
        # Length check
        min_length = self.config.get("min_prompt_length", 2000)
        if len(prompt) < min_length:
            return False, f"Prompt too short ({len(prompt)} < {min_length} characters)"
        
        # Quick repetition analysis
        words = prompt.split()
        if len(words) > 50:
            word_counts = Counter(words)
            repeated_words = sum(1 for count in word_counts.values() 
                               if count >= self.config.get("min_frequency", 2))
            repetition_ratio = repeated_words / len(word_counts)
            
            threshold = self.config.get("auto_detection_threshold", 0.35)
            if repetition_ratio > threshold:
                return True, f"High repetition detected ({repetition_ratio*100:.1f}%)"
        
        # Pattern detection for structured content
        pattern_indicators = [
            r'\bdef\s+\w+', r'\bclass\s+\w+', r'\bimport\s+\w+', 
            r'\w+\([^)]*\)', r'\{[^}]*\}', r'<[^>]+>'
        ]
        
        pattern_count = sum(len(re.findall(pattern, prompt)) for pattern in pattern_indicators)
        if pattern_count > 20:
            return True, f"Structured content detected ({pattern_count} patterns)"
        
        return True, "Meets criteria for analysis"
    
    def analyze_prompt(self, prompt: str, exclude_symbols: set = None) -> Dict:
        """
        Analyze prompt using optimized multi-threaded approach.
        Complete replacement of the previous methodology.
        
        Args:
            prompt: The text to analyze for compression opportunities
            exclude_symbols: Set of symbols to exclude from assignment (prevents collisions)
        """
        start_time = time.time()
        
        logger.debug(f"🚀 Starting optimized analysis of {len(prompt)} character prompt using {self.num_threads} threads")
        if exclude_symbols:
            logger.debug(f"🚫 Excluding {len(exclude_symbols)} symbols to prevent collisions: {list(exclude_symbols)[:10]}{'...' if len(exclude_symbols) > 10 else ''}")
        
        # Step 1: Fast parallel pattern extraction
        extraction_start = time.time()
        patterns = self.pattern_extractor.extract_all_patterns(prompt)
        extraction_time = time.time() - extraction_start
        
        logger.debug(f"⚡ Pattern extraction completed in {extraction_time:.2f}s - found {len(patterns)} patterns")
        
        # Debug: Show sample patterns found
        if patterns:
            sample_patterns = patterns[:10]
            logger.debug(f"Sample patterns found: {[(p, c) for p, c in sample_patterns]}")
            
            # Show patterns meeting frequency threshold
            freq_threshold = self.config.get("min_frequency", 2)
            qualifying_patterns = [(p, c) for p, c in patterns if c >= freq_threshold]
            logger.debug(f"Patterns meeting frequency threshold ({freq_threshold}+): {len(qualifying_patterns)}")
            
            if qualifying_patterns:
                logger.debug(f"Top qualifying patterns: {qualifying_patterns[:5]}")
        else:
            logger.warning("No patterns found during extraction!")
        
        # Step 2: Quick filtering and ranking
        max_patterns = self.config.get("max_dictionary_size", 262) * 2  # Extract more for optimization
        valuable_patterns = patterns[:max_patterns]
        
        # Step 3: Fast symbol assignment with value optimization and content-aware symbol selection
        assignment_start = time.time()
        assignments = self.symbol_assigner.assign_symbols_fast(
            valuable_patterns, 
            max_assignments=self.config.get("max_dictionary_size", 262),
            content=prompt,  # Pass content for conflict avoidance
            exclude_symbols=exclude_symbols  # CRITICAL FIX: Pass exclude_symbols to prevent collisions
        )
        assignment_time = time.time() - assignment_start
        
        logger.debug(f"🎯 Symbol assignment completed in {assignment_time:.2f}s - created {len(assignments)} assignments")
        
        # Debug: Show sample assignments
        if assignments:
            # FIXED: assignments is now symbol -> pattern format
            sample_assignments = [(pattern, symbol) for symbol, pattern in list(assignments.items())[:5]]
            logger.debug(f"Sample assignments: {sample_assignments}")
        else:
            logger.warning("No symbol assignments created! Checking value calculations...")
            
            # Debug why no assignments were made
            for pattern, count in valuable_patterns[:5]:
                test_symbol = 'α'  # Use first Greek letter for testing
                value = self.symbol_assigner.calculate_assignment_value(pattern, count, test_symbol)
                logger.debug(f"Pattern '{pattern}' (count={count}) -> value={value:.2f}")
        
        # Step 4: Calculate compression metrics efficiently
        metrics_start = time.time()
        compression_analysis = self._calculate_compression_metrics_fast(assignments, valuable_patterns, prompt)
        metrics_time = time.time() - metrics_start
        
        # Debug: Show compression analysis results
        compression_ratio = compression_analysis.get("compression_ratio", 0)
        threshold = self.config.get("compression_threshold", 0.01)
        logger.debug(f"Compression ratio: {compression_ratio:.4f} (threshold: {threshold:.4f})")
        
        if compression_ratio < threshold:
            logger.debug(f"Dynamic analysis shows minimal benefit ({compression_ratio*100:.2f}% < {threshold*100:.1f}%), returning uncompressed")
        else:
            logger.debug(f"Compression threshold met! {compression_ratio*100:.2f}% >= {threshold*100:.1f}%")
        
        total_time = time.time() - start_time
        
        logger.debug(f"📊 Optimized analysis completed in {total_time:.2f}s total")
        logger.debug(f"   Pattern extraction: {extraction_time:.2f}s")
        logger.debug(f"   Symbol assignment: {assignment_time:.2f}s") 
        logger.debug(f"   Metrics calculation: {metrics_time:.2f}s")
        logger.debug(f"   Created {len(assignments)} compression entries")
        
        return {
            "prompt_length": len(prompt),
            "total_patterns": len(patterns),
            "dictionary_entries": assignments,
            "compression_analysis": compression_analysis,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_method": "optimized_multithreaded",
            "processing_time": total_time,
            "thread_count": self.num_threads
        }
    
    def _calculate_compression_metrics_fast(self, assignments: Dict[str, str], 
                                          patterns: List[Tuple[str, int]], prompt: str) -> Dict:
        """Fast compression metrics calculation without redundant tokenization."""
        if not assignments:
            return {
                "total_char_savings": 0,
                "total_token_savings": 0,
                "compression_ratio": 0.0,
                "worthwhile_opportunities": [],
                "rejected_opportunities": []
            }
        
        # Create pattern lookup for counts
        pattern_counts = {pattern: count for pattern, count in patterns}
        
        # Calculate metrics for assigned patterns
        worthwhile = []
        total_char_savings = 0
        total_token_savings = 0  # This will be the net savings (already accounts for overhead)
        
        # ENHANCED: Debug logging for token savings calculation
        if self.tokenizer:
            original_tokens = self.symbol_assigner.get_token_count(prompt)
            logger.debug(f"Original prompt tokens: {original_tokens}, calculating individual pattern savings...")
        
        # FIXED: assignments is now symbol -> pattern format
        for symbol, pattern in assignments.items():
            count = pattern_counts.get(pattern, 0)
            if count == 0:
                continue
            
            # Calculate character savings
            char_savings = (len(pattern) - len(symbol)) * count
            total_char_savings += char_savings
            
            if self.tokenizer:
                # calculate_assignment_value already returns NET token savings (after overhead)
                net_token_savings = self.symbol_assigner.calculate_assignment_value(pattern, count, symbol)
                
                # ENHANCED: Cap individual pattern savings to reasonable bounds
                # No single pattern should save more than 1/10 of total tokens (very conservative)
                max_reasonable_per_pattern = original_tokens * 0.1
                if net_token_savings > max_reasonable_per_pattern:
                    logger.debug(f"Capping unrealistic pattern savings: {net_token_savings:.1f} -> {max_reasonable_per_pattern:.1f} for pattern '{pattern[:30]}...'")
                    net_token_savings = max_reasonable_per_pattern
                
                # ENHANCED: Debug high-value patterns
                if net_token_savings > 20:  # Log patterns with significant token savings
                    logger.debug(f"High-value pattern: '{pattern[:50]}...' -> '{symbol}' (count={count}, net_tokens={net_token_savings:.1f})")
                
                total_token_savings += max(0, net_token_savings)
                
                # Store individual token savings for reporting
                individual_token_savings = net_token_savings
            else:
                # Character-based estimation
                individual_token_savings = char_savings / 4  # Rough chars-to-tokens conversion
                total_token_savings += max(0, individual_token_savings)
            
            worthwhile.append({
                "token": pattern,
                "symbol": symbol,
                "count": count,
                "char_savings": char_savings,
                "token_savings": max(0, individual_token_savings),
                "length": len(pattern)
            })
        
        # ENHANCED: Debug total savings calculation with safeguards
        if self.tokenizer and total_token_savings > 0:
            logger.debug(f"Total calculated token savings: {total_token_savings:.1f} from {len(assignments)} patterns")
            
            # Additional safeguard: ensure total never exceeds 80% of original tokens
            max_reasonable_total = original_tokens * 0.8
            if total_token_savings > max_reasonable_total:
                logger.debug(f"Capping total savings from {total_token_savings:.1f} to {max_reasonable_total:.1f} (80% of {original_tokens} tokens)")
                total_token_savings = max_reasonable_total
        
        # Calculate compression ratios with realistic bounds
        char_ratio = total_char_savings / len(prompt) if len(prompt) > 0 else 0
        
        if self.tokenizer:
            original_tokens = self.symbol_assigner.get_token_count(prompt)
            
            # CRITICAL FIX: Token savings cannot exceed original tokens
            # This prevents impossible compression ratios > 100%
            realistic_token_savings = min(total_token_savings, original_tokens - 1)  # Must leave at least 1 token
            token_ratio = realistic_token_savings / original_tokens if original_tokens > 0 else 0
            
            # Log warning if we had to cap the savings (indicates calculation issue)
            if total_token_savings > original_tokens:
                logger.warning(f"Token savings calculation exceeded original tokens: {total_token_savings} > {original_tokens}, capped to {realistic_token_savings}")
                logger.debug(f"This suggests individual pattern values are being over-estimated or double-counted")
        else:
            token_ratio = char_ratio
        
        # Ensure compression ratio is reasonable (max 95% to be conservative)
        token_ratio = min(token_ratio, 0.95)  # Cap at 95% instead of 100% for realism
        char_ratio = min(char_ratio, 0.95)
        
        # Create rejected list (patterns that weren't assigned)
        # FIXED: assignments is symbol -> pattern, so get assigned patterns from values
        assigned_patterns = set(assignments.values())
        rejected = [
            {
                "token": pattern,
                "count": count,
                "length": len(pattern),
                "rejection_reason": "Not selected in optimization"
            }
            for pattern, count in patterns
            if pattern not in assigned_patterns
        ]
        
        return {
            "total_char_savings": total_char_savings,
            "total_token_savings": max(0, total_token_savings),
            "char_compression_ratio": char_ratio,
            "token_compression_ratio": token_ratio,
            "compression_ratio": token_ratio,  # Use token ratio as primary metric
            "worthwhile_opportunities": worthwhile,
            "rejected_opportunities": rejected[:20],  # Limit for performance
            "token_validation_used": self.tokenizer is not None
        }
    
    def create_temporary_dictionary(self, analysis_result: Dict, prefix: str = "dynamic") -> Optional[str]:
        """Create temporary dictionary file from analysis results."""
        dictionary_entries = analysis_result["dictionary_entries"]
        
        if not dictionary_entries:
            logger.debug("No compression opportunities found")
            return None
        
        # CRITICAL SAFETY: Ensure temp directory exists and is isolated
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create temporary dictionary file with unique timestamp and PID for complete isolation
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        pid = os.getpid()
        dict_filename = f"{prefix}_{timestamp}_{pid}.json"
        dict_path = self.temp_dir / dict_filename
        
        # CRITICAL: Create completely isolated copies of all data
        compression_analysis = analysis_result["compression_analysis"]
        
        # CRITICAL BUG FIX: Ensure correct symbol -> pattern format
        # The dictionary_entries should already be in symbol -> pattern format from assign_symbols_fast()
        # But let's validate and ensure it's correct
        validated_tokens = {}
        if dictionary_entries:
            for key, value in dictionary_entries.items():
                # Determine which is symbol and which is pattern based on length
                # Symbols should be short Unicode characters, patterns should be longer text
                if len(key) <= 3 and len(value) > 3:
                    # Correct: key is symbol, value is pattern
                    validated_tokens[key] = value
                elif len(value) <= 3 and len(key) > 3:
                    # Incorrect: key is pattern, value is symbol - need to flip
                    validated_tokens[value] = key
                    logger.warning(f"Fixed inverted dictionary entry: '{key[:50]}...' -> '{value}' corrected to '{value}' -> '{key[:50]}...'")
                else:
                    # Both are either short or long - use original order but log warning
                    validated_tokens[key] = value
                    logger.debug(f"Ambiguous dictionary entry: '{key}' -> '{value}' (lengths: {len(key)}, {len(value)})")
        
        json_data = {
            "metadata": {
                "created": analysis_result["analysis_timestamp"],
                "prompt_length": analysis_result["prompt_length"],
                "total_char_savings": compression_analysis.get("total_char_savings", 0),
                "total_token_savings": compression_analysis.get("total_token_savings", 0),
                "compression_ratio": compression_analysis["compression_ratio"],
                "processing_time": analysis_result.get("processing_time", 0),
                "thread_count": analysis_result.get("thread_count", self.num_threads),
                "analysis_method": "optimized_multithreaded",
                "type": "dynamic_dictionary",
                "pid": pid,  # Track which process created this
                "isolated": True  # Mark as isolated dictionary
            },
            # CRITICAL: Use validated tokens in correct symbol -> pattern format
            "tokens": validated_tokens
        }
        
        try:
            # CRITICAL: Use atomic write to prevent corruption
            temp_path = dict_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            # Atomic rename to final name
            temp_path.rename(dict_path)
            
            logger.debug(f"✅ Created isolated dictionary: {len(dictionary_entries)} entries")
            logger.debug(f"📁 Dictionary saved to: {dict_path}")
            logger.debug(f"🔒 Dictionary is isolated (PID: {pid})")
            
            return str(dict_path)
            
        except Exception as e:
            logger.error(f"Failed to create temporary dictionary: {e}")
            # Cleanup failed temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            return None
    
    def cleanup_old_dictionaries(self, max_age_hours: Optional[int] = None):
        """Clean up old temporary dictionary files with aggressive cleanup."""
        if not self.config.get("auto_cleanup", True):
            return
        
        if max_age_hours is None:
            max_age_hours = self.config.get("cleanup_max_age_hours", 24)
        
        current_time = time.time()
        cleanup_count = 0
        
        try:
            # CRITICAL: More aggressive cleanup - remove all temp files, not just old ones
            patterns_to_clean = ["dynamic_*.json", "dynamic_*.tmp", "*.tmp"]
            
            for pattern in patterns_to_clean:
                for dict_file in self.temp_dir.glob(pattern):
                    try:
                        file_age = current_time - dict_file.stat().st_mtime
                        # Clean up files older than specified age OR any .tmp files (immediate cleanup)
                        if file_age > (max_age_hours * 3600) or dict_file.suffix == '.tmp':
                            dict_file.unlink()
                            cleanup_count += 1
                            logger.debug(f"🧹 Cleaned up: {dict_file.name}")
                    except Exception as file_error:
                        logger.debug(f"Failed to cleanup {dict_file}: {file_error}")
                        
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        if cleanup_count > 0:
            logger.info(f"🧹 Cleaned up {cleanup_count} old/temporary dictionaries")
            
        # CRITICAL: Force cleanup of current process temp files on each run
        try:
            import os
            current_pid = os.getpid()
            for dict_file in self.temp_dir.glob("dynamic_*.json"):
                try:
                    # Check if this file was created by a different process
                    with open(dict_file, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        file_pid = data.get("metadata", {}).get("pid")
                        if file_pid and file_pid != current_pid:
                            # File from different process - check if process is still running
                            try:
                                os.kill(file_pid, 0)  # Check if process exists
                            except OSError:
                                # Process doesn't exist, safe to clean up
                                dict_file.unlink()
                                cleanup_count += 1
                                logger.debug(f"🧹 Cleaned up orphaned dictionary: {dict_file.name}")
                except Exception:
                    pass  # Skip files we can't process
        except Exception as e:
            logger.debug(f"PID-based cleanup failed: {e}")

    def compress_multipass(self, text: str, max_passes: int = 3) -> Tuple[str, Dict[str, str], Dict]:
        """
        Apply multi-pass compression - adaptive or traditional based on config.
        """
        logger.debug(f"Starting multi-pass compression (max_passes={max_passes})")
        
        # Check if adaptive multi-pass is enabled in config
        if self.config.get("multi_pass_adaptive", False):
            logger.debug("Using adaptive multi-pass compression")
            # Get the adaptive compressor
            adaptive_compressor = get_adaptive_multipass_compressor(self)
            
            # Use adaptive compression which handles all the intelligent logic
            compressed, all_substitutions, metrics = adaptive_compressor.compress_adaptive(text)
            
            return compressed, all_substitutions, metrics
        else:
            logger.debug("Using traditional multi-pass compression (adaptive disabled)")
            # Use traditional multi-pass logic (original implementation)
            compressed = text
            total_used = {}
            all_metrics = []
            
            for pass_num in range(max_passes):
                logger.info(f"Starting compression pass {pass_num + 1}")
                
                # Skip if text is too small - use same threshold as main compression
                min_length = self.config.get("min_prompt_length", 200)
                if len(compressed) < min_length:
                    logger.info(f"Text too small for pass {pass_num + 1} ({len(compressed)} chars < {min_length})")
                    break
                
                # Analyze remaining text
                analysis = self.analyze_prompt(compressed)
                
                # Stop if no more opportunities
                if not analysis.get("dictionary_entries"):
                    logger.info(f"No more compression opportunities found in pass {pass_num + 1}")
                    break
                
                # Apply compression from this pass
                temp_dict_file = self.create_temporary_dictionary(analysis, f"multipass_{pass_num + 1}")
                if not temp_dict_file:
                    logger.warning(f"Failed to create dictionary for pass {pass_num + 1}")
                    break
                
                # Load and apply the compression
                try:
                    from .rules import load_dict
                    temp_dict = load_dict(temp_dict_file)
                    pass_compressed = compressed
                    pass_used = {}
                    
                    # Apply each substitution - CORRECT FORMAT: symbol -> pattern
                    # CRITICAL BUG FIX: temp_dict is now in symbol -> pattern format
                    for symbol, pattern in temp_dict.items():
                        if pattern in pass_compressed:
                            occurrences = pass_compressed.count(pattern)
                            pass_compressed = pass_compressed.replace(pattern, symbol)
                            pass_used[symbol] = pattern  # CORRECT: symbol -> pattern for decompression
                            logger.debug(f"Pass {pass_num + 1}: Replaced '{pattern}' with '{symbol}' ({occurrences} times)")
                    
                    # Validate improvement
                    improvement_ratio = (len(compressed) - len(pass_compressed)) / len(compressed)
                    logger.info(f"Pass {pass_num + 1}: {improvement_ratio:.1%} improvement "
                               f"({len(compressed)} -> {len(pass_compressed)} chars)")
                    
                    if improvement_ratio < 0.02:  # Less than 2% improvement
                        logger.info(f"Minimal improvement in pass {pass_num + 1}, stopping")
                        os.unlink(temp_dict_file)  # Clean up
                        break
                    
                    # Accept the compression
                    compressed = pass_compressed
                    total_used.update(pass_used)
                    all_metrics.append(analysis.get("compression_analysis", {}))
                    
                    # Clean up temporary file
                    os.unlink(temp_dict_file)
                    
                except Exception as e:
                    logger.error(f"Error in pass {pass_num + 1}: {e}")
                    if temp_dict_file and os.path.exists(temp_dict_file):
                        os.unlink(temp_dict_file)
                    break
            
            # Calculate final overall metrics
            final_metrics = {
                "passes_completed": len(all_metrics),
                "total_original_length": len(text),
                "total_compressed_length": len(compressed),
                "total_compression_ratio": (len(text) - len(compressed)) / len(text) if text else 0,
                "total_patterns": len(total_used),
                "pass_metrics": all_metrics
            }
            
            logger.info(f"Traditional multi-pass compression complete: {final_metrics['total_compression_ratio']:.1%} "
                       f"compression in {final_metrics['passes_completed']} passes")
            
            return compressed, total_used, final_metrics


# Global cache for reusing analyzer instances to prevent duplicate initialization logging
_global_analyzer_cache = {}
_cache_lock = threading.Lock()

def get_dynamic_dictionary_analyzer(temp_dir: str = "temp") -> 'DynamicDictionaryAnalyzer':
    """
    Get a cached instance of DynamicDictionaryAnalyzer to prevent duplicate initialization logging.
    Creates one instance per temp_dir and reuses it across requests.
    """
    with _cache_lock:
        cache_key = temp_dir
        
        if cache_key not in _global_analyzer_cache:
            logger.debug(f"Creating new DynamicDictionaryAnalyzer instance for temp_dir: {temp_dir}")
            analyzer = DynamicDictionaryAnalyzer(temp_dir)
            _global_analyzer_cache[cache_key] = analyzer
        else:
            logger.debug(f"Reusing cached DynamicDictionaryAnalyzer instance for temp_dir: {temp_dir}")
            
        return _global_analyzer_cache[cache_key]

def clear_analyzer_cache():
    """Clear the global analyzer cache (useful for testing or configuration changes)."""
    global _global_analyzer_cache
    with _cache_lock:
        _global_analyzer_cache.clear()
        logger.debug("Cleared DynamicDictionaryAnalyzer cache")


