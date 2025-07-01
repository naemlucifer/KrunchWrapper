#!/usr/bin/env python3
"""
Model-Optimized Symbol Pool for Maximum Compression Efficiency

This module provides model-specific symbol pools optimized for tokenization efficiency,
prioritizing symbols that tokenize to single tokens for maximum compression benefit.
"""

import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from functools import lru_cache
from .persistent_token_cache import get_persistent_cache

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelOptimizedSymbolPool:
    """Generate model-specific optimal symbols that tokenize efficiently."""
    
    # Single-character Unicode symbols that typically tokenize to 1 token
    EFFICIENT_UNICODE = [
        # Greek letters (usually single tokens)
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
        'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
        'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
        
        # Mathematical symbols (often single tokens)
        '∂', '∇', '∆', '∑', '∏', '∫', '√', '∞', '≈', '≠', '≤', '≥',
        '∀', '∃', '∅', '∈', '∉', '∋', '∧', '∨', '∩', '∪', '⊂', '⊃',
        '⊆', '⊇', '⊕', '⊗', '⊥', '∥', '∝', '∴', '∵', '∶', '∷', '∸',
        
        # Subscript numbers (often single tokens)
        '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',
        '₊', '₋', '₌', '₍', '₎', 'ₐ', '₄', '₥', '₦', '₧',
        
        # Superscript numbers (often single tokens)
        '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',
        '⁺', '⁻', '⁼', '⁽', '⁾', 'ⁿ', 'ⁱ', '⁡', '⁢', '⁣',
        
        # Box drawing characters (efficient for structured content)
        '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '─', '│',
        '┏', '┓', '┗', '┛', '┣', '┫', '┳', '┻', '╋', '━', '┃',
        
        # Arrows (usually single tokens)
        '←', '→', '↑', '↓', '↔', '↕', '⇐', '⇒', '⇑', '⇓', '⇔', '⇕',
        '↖', '↗', '↘', '↙', '↩', '↪', '↺', '↻', '⟲', '⟳',
        
        # Geometric shapes (often single tokens)
        '○', '●', '□', '■', '△', '▲', '▽', '▼', '◇', '◆', '◊',
        '☆', '★', '☉', '☐', '☑', '☒', '◯', '◉', '◎', '⊙',
        '◀', '▶',  # Add missing left/right triangles
        
        # Currency and misc symbols (usually single tokens)  
        '€', '£', '¥', '¢', '₹', '₽', '₿', '©', '®', '™', '°', '±',
        '¬', '¦', '§', '¨', '¯', '¶', '·', '¸', '¿', '×', '÷',
        
        # Additional mathematical operators
        '∘', '∙', '∛', '∜', '∝', '∞', '∟', '∠', '∡', '∢', '∣', '∤',
        '∥', '∦', '∧', '∨', '∩', '∪', '∫', '∬', '∭', '∮', '∯', '∰',
        
        # Miscellaneous technical symbols
        '⌈', '⌉', '⌊', '⌋', '〈', '〉', '⟨', '⟩', '⟪', '⟫', '⟬', '⟭',
        '⌐', '⌑', '⌒', '⌓', '⌔', '⌕', '⌖', '⌗', '⌘', '⌙', '⌚', '⌛',
    ]
    
    # ASCII fallback symbols - ONLY use these if Unicode completely fails
    # Use extremely unique patterns with | prefix and other rare combinations
    ASCII_FALLBACK = [        
        # Pipe prefix (very uncommon as variable prefix)
        '|A', '|B', '|C', '|D', '|E', '|F', '|G', '|H', '|I', '|J',
        '|K', '|L', '|M', '|N', '|O', '|P', '|Q', '|R', '|S', '|T',
        
        # Double pipe prefix (extremely rare)
        '||A', '||B', '||C', '||D', '||E', '||F', '||G', '||H',
        
        # Tilde combinations (uncommon in most languages)
        '~A~', '~B~', '~C~', '~D~', '~E~', '~F~', '~G~', '~H~',
        
        # Triple symbols (very artificial)
        '^^^', '|||', '~~~', '===', '+++', '---', '***', '>>>', '<<<',
        
        # Bracket combinations (very unlikely in normal code)
        '[|A|]', '[|B|]', '[|C|]', '[|D|]', '[|E|]', '[|F|]',
        
        # Question mark combinations (rare as identifiers)
        '?A?', '?B?', '?C?', '?D?', '?E?', '?F?', '?G?', '?H?',
    ]
    
    def __init__(self, tokenizer=None):
        """Initialize with optional tokenizer for testing symbol efficiency."""
        self.tokenizer = tokenizer
        self.symbol_efficiency_cache = {}
        self.model_specific_rankings = {}
        
        # Initialize persistent cache
        self._persistent_cache = get_persistent_cache()
        
    def get_optimal_symbols(self, model_name: str, max_symbols: int = 100, 
                          exclude_symbols: Set[str] = None, content: str = None) -> List[Tuple[str, int]]:
        """
        Get optimal symbols for a specific model, sorted by tokenization efficiency.
        
        Args:
            model_name: Name of the model to optimize for
            max_symbols: Maximum number of symbols to return
            exclude_symbols: Set of symbols to exclude from selection
            content: Content being compressed (to avoid symbol conflicts)
            
        Returns:
            List of (symbol, token_count) tuples sorted by efficiency
        """
        if exclude_symbols is None:
            exclude_symbols = set()
            
        # Add symbols that appear in content to exclusion list
        if content:
            content_symbols = self._extract_symbols_from_content(content)
            exclude_symbols = exclude_symbols.union(content_symbols)
            if content_symbols:
                logger.debug(f"Excluding {len(content_symbols)} symbols found in content: {sorted(list(content_symbols))[:10]}...")
            
        # Check cache first
        cache_key = (model_name, max_symbols, frozenset(exclude_symbols))
        if cache_key in self.model_specific_rankings:
            return self.model_specific_rankings[cache_key]
        
        logger.debug(f"Computing optimal symbols for model: {model_name}")
        
        # Test symbol efficiency with tokenizer if available
        if self.tokenizer:
            symbol_scores = self._test_symbols_with_tokenizer(model_name, exclude_symbols)
        else:
            # Fallback to heuristic-based ranking
            symbol_scores = self._rank_symbols_heuristically(model_name, exclude_symbols)
        
        # Sort by efficiency (lowest token count first, then by symbol preference)
        symbol_scores.sort(key=lambda x: (x[1], self._get_symbol_preference(x[0])))
        
        # Take top symbols up to max_symbols
        optimal_symbols = symbol_scores[:max_symbols]
        
        # Cache result and log only on first computation (not cached)
        if cache_key not in self.model_specific_rankings:
            logger.info(f"Initialized optimized symbol pool for {model_name}:")
            if optimal_symbols:
                single_token_count = sum(1 for _, token_count in optimal_symbols if token_count == 1)
                unicode_count = sum(1 for symbol, _ in optimal_symbols if symbol in self.EFFICIENT_UNICODE)
                avg_tokens_per_symbol = sum(token_count for _, token_count in optimal_symbols) / len(optimal_symbols)
                logger.info(f"  Total symbols: {len(optimal_symbols)}")
                logger.info(f"  Single-token symbols: {single_token_count} ({100*single_token_count/len(optimal_symbols):.1f}%)")
                logger.info(f"  Average tokens per symbol: {avg_tokens_per_symbol:.2f}")
        
        self.model_specific_rankings[cache_key] = optimal_symbols
        
        return optimal_symbols
    
    def _extract_symbols_from_content(self, content: str) -> Set[str]:
        """Extract single-character symbols that appear in the content."""
        symbols_in_content = set()
        
        # Check for single character symbols from our pools
        all_candidate_symbols = set(self.EFFICIENT_UNICODE + self.ASCII_FALLBACK)
        
        for symbol in all_candidate_symbols:
            if symbol in content:
                symbols_in_content.add(symbol)
                
        return symbols_in_content
    
    def _test_symbols_with_tokenizer(self, model_name: str, 
                                   exclude_symbols: Set[str]) -> List[Tuple[str, int]]:
        """Test symbols with actual tokenizer to measure efficiency."""
        symbol_scores = []
        
        # Test Unicode symbols first (HIGHEST priority - no code conflicts)
        unicode_symbols = []
        for symbol in self.EFFICIENT_UNICODE:
            if symbol in exclude_symbols:
                continue
                
            try:
                token_count = len(self.tokenizer.encode(symbol))
                unicode_symbols.append((symbol, token_count))
            except Exception as e:
                logger.debug(f"Tokenization failed for '{symbol}': {e}")
                # Conservative fallback
                unicode_symbols.append((symbol, 2))
        
        # Add Unicode symbols to results first
        symbol_scores.extend(unicode_symbols)
        
        # Only add ASCII fallback if we really need more symbols and Unicode isn't sufficient
        # These are much safer ASCII symbols that are very unlikely to conflict with code
        if len(unicode_symbols) < 30:  # Only if we really need more symbols
            logger.warning(f"Limited Unicode symbols available ({len(unicode_symbols)}), adding ASCII fallback")
            for symbol in self.ASCII_FALLBACK:
                if symbol in exclude_symbols:
                    continue
                    
                try:
                    token_count = len(self.tokenizer.encode(symbol))
                    # Heavily penalize ASCII to strongly prefer Unicode
                    symbol_scores.append((symbol, token_count + 1.0))  # Much bigger penalty
                except Exception as e:
                    logger.debug(f"Tokenization failed for '{symbol}': {e}")
                    symbol_scores.append((symbol, 3.0))  # Even bigger penalty on failure
        
        # Log tokenization results for debugging
        unicode_count = len(unicode_symbols)
        single_token_unicode = sum(1 for _, c in unicode_symbols if c == 1)
        
        logger.debug(f"Symbol tokenization results:")
        logger.debug(f"  Unicode symbols: {unicode_count} ({single_token_unicode} single-token)")
        logger.debug(f"  Total symbols available: {len(symbol_scores)}")
        
        return symbol_scores
    
    def _rank_symbols_heuristically(self, model_name: str, 
                                  exclude_symbols: Set[str]) -> List[Tuple[str, int]]:
        """Rank symbols using heuristics when tokenizer is unavailable."""
        symbol_scores = []
        
        # ALWAYS prioritize Unicode symbols to avoid code conflicts
        # Unicode symbols don't interfere with programming syntax
        for symbol in self.EFFICIENT_UNICODE:
            if symbol not in exclude_symbols:
                # Heuristic: assume single-character Unicode symbols are 1 token
                token_count = 1 if len(symbol) == 1 else 2
                symbol_scores.append((symbol, token_count))
        
        # Only use ASCII fallback if we need more symbols
        # These are safer ASCII symbols that don't conflict with code
        for symbol in self.ASCII_FALLBACK:
            if symbol not in exclude_symbols:
                # Penalize ASCII symbols slightly to prefer Unicode
                token_count = 2 if len(symbol) <= 2 else 3
                symbol_scores.append((symbol, token_count))
        
        return symbol_scores
    
    def _get_symbol_preference(self, symbol: str) -> int:
        """Get preference ranking for symbol (lower is better)."""
        # Prefer shorter symbols
        if len(symbol) == 1:
            return 0
        elif len(symbol) == 2:
            return 1
        elif len(symbol) == 3:
            return 2
        else:
            return 3
    
    def test_symbol_in_contexts(self, symbol: str, test_contexts: Tuple[str, ...]) -> float:
        """Test how well a symbol tokenizes in various contexts with persistent caching."""
        # Create cache key from symbol and contexts
        contexts_str = "|".join(test_contexts)
        cache_key = f"symbol_test:{symbol}:{contexts_str}"
        
        # Try persistent cache first
        cached_result = self._persistent_cache.get(cache_key, "symbol_efficiency")
        if cached_result is not None:
            return cached_result
        
        if not self.tokenizer:
            result = 1.0  # Assume good if no tokenizer
        else:
            total_efficiency = 0.0
            valid_tests = 0
            
            for context in test_contexts:
                try:
                    # Test symbol in context
                    test_text = context.replace("{symbol}", symbol)
                    tokens = self.tokenizer.encode(test_text)
                    
                    # Test context without symbol
                    baseline_text = context.replace("{symbol}", "X")  # Single char replacement
                    baseline_tokens = self.tokenizer.encode(baseline_text)
                    
                    # Calculate efficiency (lower is better)
                    token_difference = len(tokens) - len(baseline_tokens)
                    efficiency = 1.0 / max(1, token_difference + 1)
                    
                    total_efficiency += efficiency
                    valid_tests += 1
                    
                except Exception:
                    continue
            
            result = total_efficiency / max(1, valid_tests)
        
        # Cache the result
        self._persistent_cache.set(cache_key, result, "symbol_efficiency")
        return result
    
    def get_context_optimized_symbols(self, text_sample: str, model_name: str,
                                    max_symbols: int = 50) -> List[Tuple[str, float]]:
        """Get symbols optimized for specific text context."""
        # Extract common contexts from the text
        contexts = self._extract_common_contexts(text_sample)
        
        # Convert to tuple for caching
        context_tuple = tuple(contexts[:10])  # Limit for performance
        
        # Get candidate symbols
        candidates = self.get_optimal_symbols(model_name, max_symbols * 2)
        
        # Test symbols in context
        symbol_efficiency = []
        for symbol, base_token_count in candidates:
            if base_token_count <= 2:  # Only test promising symbols
                efficiency = self.test_symbol_in_contexts(symbol, context_tuple)
                symbol_efficiency.append((symbol, efficiency))
        
        # Sort by efficiency and take top symbols
        symbol_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        return symbol_efficiency[:max_symbols]
    
    def _extract_common_contexts(self, text: str) -> List[str]:
        """Extract common contexts where symbols might be used."""
        contexts = []
        
        # Common code contexts
        if any(keyword in text for keyword in ['def ', 'class ', 'import ', 'function']):
            contexts.extend([
                " {symbol} ",
                ".{symbol}",
                "{symbol}(",
                "({symbol})",
                '"{symbol}"',
                "'{symbol}'",
                "={symbol}",
                "{symbol}=",
            ])
        
        # Common prose contexts
        if any(punct in text for punct in ['. ', ', ', '; ']):
            contexts.extend([
                " {symbol} ",
                ", {symbol},",
                ". {symbol}",
                "({symbol})",
            ])
        
        # Fallback contexts
        if not contexts:
            contexts = [
                " {symbol} ",
                "{symbol}",
                " {symbol}",
                "{symbol} ",
            ]
        
        return contexts
    
    def benchmark_symbol_efficiency(self, model_name: str, test_text: str = None) -> Dict:
        """Benchmark symbol efficiency for a specific model."""
        if not self.tokenizer:
            return {"error": "No tokenizer available for benchmarking"}
        
        # Use test text or create default
        if not test_text:
            test_text = """
            def process_data(self, data):
                logger.info(f"Processing {len(data)} items")
                for item in data:
                    result = self.analyze_item(item)
                    if result.is_valid():
                        self.store_result(result)
                return True
            """
        
        # Test symbol efficiency
        symbols_to_test = self.EFFICIENT_UNICODE[:20] + self.ASCII_FALLBACK[:20]
        results = {
            "model_name": model_name,
            "test_text_length": len(test_text),
            "symbols_tested": len(symbols_to_test),
            "single_token_symbols": [],
            "multi_token_symbols": [],
            "efficiency_stats": {}
        }
        
        for symbol in symbols_to_test:
            try:
                token_count = len(self.tokenizer.encode(symbol))
                
                if token_count == 1:
                    results["single_token_symbols"].append(symbol)
                else:
                    results["multi_token_symbols"].append((symbol, token_count))
                    
            except Exception as e:
                logger.debug(f"Benchmark failed for symbol '{symbol}': {e}")
        
        # Calculate statistics
        total_single = len(results["single_token_symbols"])
        total_multi = len(results["multi_token_symbols"])
        
        results["efficiency_stats"] = {
            "single_token_count": total_single,
            "multi_token_count": total_multi,
            "single_token_percentage": round(100 * total_single / (total_single + total_multi), 1),
            "average_multi_token_size": round(
                sum(count for _, count in results["multi_token_symbols"]) / max(1, total_multi), 2
            )
        }
        
        return results


# Global instance
_model_optimized_pool = None

def get_model_optimized_symbol_pool(tokenizer=None) -> ModelOptimizedSymbolPool:
    """Get singleton instance of ModelOptimizedSymbolPool."""
    global _model_optimized_pool
    if _model_optimized_pool is None:
        _model_optimized_pool = ModelOptimizedSymbolPool(tokenizer)
    elif tokenizer and not _model_optimized_pool.tokenizer:
        _model_optimized_pool.tokenizer = tokenizer
    return _model_optimized_pool 