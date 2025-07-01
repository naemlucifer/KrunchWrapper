#!/usr/bin/env python3
"""
Optimized Symbol Selection & Generation

This module provides high-performance symbol selection with batch processing,
precomputed rankings, and configurable multithreading for maximum efficiency.
"""

import json
import logging
import time
from typing import List, Tuple, Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import functools

try:
    from .model_specific_symbol_selector import get_model_specific_symbol_selector
    from .model_tokenizer_validator import get_model_tokenizer_validator
    from .dynamic_config_parser import load_config
except ImportError:
    # Handle imports when run as script
    from model_specific_symbol_selector import get_model_specific_symbol_selector
    from model_tokenizer_validator import get_model_tokenizer_validator
    from dynamic_config_parser import load_config

logger = logging.getLogger(__name__)

class FastSymbolSelector:
    """Optimized symbol selector with batch operations, caching, and configurable threading."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fast symbol selector.
        
        Args:
            config: Configuration dictionary (auto-loaded if None)
        """
        self.config = config or load_config()
        self.max_workers = self._get_thread_count()
        self.symbol_rankings = {}  # Precomputed rankings by model family
        self.validation_cache = {}  # Cache symbol validation results
        self.cache_hits = 0
        self.cache_misses = 0
        self._precompute_rankings()
        
        logger.info(f"FastSymbolSelector initialized with {self.max_workers} threads")
    
    def _get_thread_count(self) -> int:
        """Get thread count from config with sensible defaults."""
        try:
            threads = self.config.get("compression", {}).get("threads", 4)
            # Clamp to reasonable range for symbol validation
            return max(1, min(threads, 16))  # Max 16 threads for this task
        except Exception:
            return 4  # Safe default
    
    def _precompute_rankings(self):
        """Precompute symbol rankings for all model families."""
        start_time = time.time()
        
        try:
            # Load unified dictionary once
            dict_path = Path(__file__).parent.parent / "dictionaries" / "model_specific_symbols.json"
            if not dict_path.exists():
                logger.warning("Model-specific dictionary not found, using fallback rankings")
                self._create_fallback_rankings()
                return
            
            with open(dict_path, 'r', encoding='utf-8') as f:
                unified_dict = json.load(f)
            
            # Precompute rankings for each model family
            model_families = unified_dict.get('_metadata', {}).get('model_families', [])
            
            for family in model_families:
                family_rankings = []
                
                for symbol, data in unified_dict.items():
                    if symbol.startswith('_'):
                        continue
                    
                    token_counts = data.get('token_counts', {})
                    if family in token_counts:
                        token_count = token_counts[family]
                        family_rankings.append((symbol, token_count))
                
                # Sort by token count (ascending - fewer tokens = better)
                family_rankings.sort(key=lambda x: x[1])
                self.symbol_rankings[family] = family_rankings
                
                logger.debug(f"Precomputed {len(family_rankings)} symbols for {family}")
            
            elapsed = time.time() - start_time
            logger.info(f"Precomputed symbol rankings for {len(model_families)} model families in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to precompute symbol rankings: {e}")
            self._create_fallback_rankings()
    
    def _create_fallback_rankings(self):
        """Create fallback symbol rankings when main dictionary unavailable."""
        fallback_symbols = [
            # Greek letters (usually 1 token) - highest priority
            ('α', 1), ('β', 1), ('γ', 1), ('δ', 1), ('ε', 1), ('ζ', 1), ('η', 1), ('θ', 1),
            ('ι', 1), ('κ', 1), ('λ', 1), ('μ', 1), ('ν', 1), ('ξ', 1), ('ο', 1), ('π', 1),
            ('ρ', 1), ('σ', 1), ('τ', 1), ('υ', 1), ('φ', 1), ('χ', 1), ('ψ', 1), ('ω', 1),
            
            # Mathematical symbols (usually 1 token)
            ('∀', 1), ('∃', 1), ('∅', 1), ('∆', 1), ('∇', 1), ('∈', 1), ('∉', 1), ('∋', 1),
            ('∑', 1), ('∏', 1), ('∧', 1), ('∨', 1), ('∩', 1), ('∪', 1), ('∫', 1), ('∮', 1),
            
            # CJK punctuation marks (usually 1 token) - adds ~40 more single-token choices
            # These are rarely used in English code/logs but tokenize as single tokens
            ('、', 1), ('。', 1), ('《', 1), ('》', 1), ('〈', 1), ('〉', 1), ('『', 1), ('』', 1),
            ('「', 1), ('」', 1), ('（', 1), ('）', 1), ('【', 1), ('】', 1), ('〔', 1), ('〕', 1),
            ('…', 1), ('‥', 1), ('〜', 1), ('〝', 1), ('〞', 1), ('〰', 1), ('゠', 1), ('・', 1),
            ('※', 1), ('〒', 1), ('〓', 1), ('〔', 1), ('〕', 1), ('〖', 1), ('〗', 1), ('〘', 1),
            ('〙', 1), ('〚', 1), ('〛', 1), ('〜', 1), ('〝', 1), ('〞', 1), ('〟', 1), ('〠', 1),
            ('々', 1), ('〻', 1), ('〼', 1), ('〽', 1), ('〾', 1), ('〿', 1),
            
            # Currency symbols (usually 1 token)
            ('€', 1), ('£', 1), ('¥', 1), ('©', 1), ('®', 1), ('™', 1), ('°', 1), ('±', 1),
        ]
        
        # Use same rankings for all model families as fallback
        fallback_families = ['gpt-4', 'claude', 'llama', 'qwen', 'mistral', 'gemini']
        for family in fallback_families:
            self.symbol_rankings[family] = fallback_symbols.copy()
        
        logger.info(f"Created fallback rankings with {len(fallback_symbols)} symbols")
    
    def get_optimal_symbols_batch(self, 
                                model_name: str, 
                                count: int = 100,
                                exclude_symbols: Set[str] = None) -> List[Tuple[str, int]]:
        """Get optimal symbols in batch for maximum efficiency."""
        if exclude_symbols is None:
            exclude_symbols = set()
        
        # Get model family (cached)
        validator = get_model_tokenizer_validator()
        model_family = validator.detect_model_family(model_name)
        
        if not model_family or model_family not in self.symbol_rankings:
            logger.warning(f"No rankings for model family {model_family}, using fallback")
            model_family = 'gpt-4'  # Safe fallback
        
        # Get precomputed rankings
        rankings = self.symbol_rankings.get(model_family, [])
        
        # Filter out excluded symbols and take requested count
        optimal_symbols = []
        for symbol, token_count in rankings:
            if symbol not in exclude_symbols:
                optimal_symbols.append((symbol, token_count))
                if len(optimal_symbols) >= count:
                    break
        
        logger.debug(f"Selected {len(optimal_symbols)} optimal symbols for {model_family}")
        return optimal_symbols
    
    def validate_symbols_batch(self, 
                             symbol_token_pairs: List[Tuple[str, str]], 
                             model_name: str) -> Dict[Tuple[str, str], Dict]:
        """Validate multiple symbol-token pairs in batch using configurable threading."""
        
        if not symbol_token_pairs:
            return {}
        
        # Check cache first
        cached_results = {}
        uncached_pairs = []
        
        for pair in symbol_token_pairs:
            cache_key = (pair[0], pair[1], model_name)
            if cache_key in self.validation_cache:
                cached_results[pair] = self.validation_cache[cache_key]
                self.cache_hits += 1
            else:
                uncached_pairs.append(pair)
                self.cache_misses += 1
        
        if cached_results:
            logger.debug(f"Cache hit rate: {self.cache_hits}/{self.cache_hits + self.cache_misses} ({100*self.cache_hits/(self.cache_hits + self.cache_misses):.1f}%)")
        
        # Validate uncached pairs in parallel
        if uncached_pairs:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all validation tasks
                future_to_pair = {
                    executor.submit(self._validate_single_pair, symbol, token, model_name): (symbol, token)
                    for symbol, token in uncached_pairs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_pair, timeout=30):  # 30 second overall timeout
                    symbol, token = future_to_pair[future]
                    try:
                        result = future.result(timeout=2.0)  # 2 second timeout per validation
                        pair = (symbol, token)
                        cached_results[pair] = result
                        
                        # Cache the result (with size limit)
                        cache_key = (symbol, token, model_name)
                        if len(self.validation_cache) < 10000:  # Limit cache size
                            self.validation_cache[cache_key] = result
                            
                    except Exception as e:
                        logger.warning(f"Validation failed for {symbol}->{token}: {e}")
                        # Use conservative fallback
                        cached_results[(symbol, token)] = {
                            'valid': len(token) > len(symbol),
                            'token_savings': max(0, len(token) - len(symbol)),
                            'method': 'fallback_on_error',
                            'reason': f'Validation error: {str(e)}'
                        }
            
            elapsed = time.time() - start_time
            logger.debug(f"Batch validated {len(uncached_pairs)} pairs in {elapsed:.2f}s using {self.max_workers} threads")
        
        return cached_results
    
    def _validate_single_pair(self, symbol: str, token: str, model_name: str) -> Dict:
        """Validate a single symbol-token pair."""
        try:
            # Import here to avoid circular imports
            from .dynamic_dictionary import get_dynamic_dictionary_analyzer
            
            # Use cached analyzer instance for validation
            # This reuses the existing comprehensive validation logic
            analyzer = get_dynamic_dictionary_analyzer()
            return analyzer._validate_tokenization_efficiency(token, symbol, [])
        except Exception as e:
            logger.debug(f"Single pair validation failed: {e}")
            # Simple fallback validation
            return {
                'valid': len(token) > len(symbol) and len(symbol) == 1,
                'token_savings': max(0, len(token) - 1),
                'method': 'simple_fallback',
                'reason': f'Fallback validation: {len(token) - 1} char savings'
            }
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.validation_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1),
            'precomputed_model_families': len(self.symbol_rankings)
        }

class OptimizedDictionaryBuilder:
    """Build compression dictionaries with optimized symbol assignment and configurable threading."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize optimized dictionary builder.
        
        Args:
            config: Configuration dictionary (auto-loaded if None)
        """
        self.config = config or load_config()
        self.symbol_selector = FastSymbolSelector(self.config)
        self.max_workers = self.symbol_selector.max_workers
        
        logger.info(f"OptimizedDictionaryBuilder initialized with {self.max_workers} threads")
    
    def build_dictionary_fast(self, 
                            opportunities: List[Dict], 
                            model_name: str,
                            max_entries: int = 100) -> Dict[str, str]:
        """Build dictionary with optimized batch processing."""
        
        if not opportunities:
            return {}
        
        start_time = time.time()
        
        # Get optimal symbols in batch (get extra symbols as backup)
        symbol_count = min(len(opportunities) * 2, max_entries * 2)
        optimal_symbols = self.symbol_selector.get_optimal_symbols_batch(
            model_name, 
            count=symbol_count
        )
        
        if not optimal_symbols:
            logger.warning("No optimal symbols available for dictionary building")
            return {}
        
        # Create symbol-token pairs for validation
        symbol_token_pairs = []
        for i, opp in enumerate(opportunities[:max_entries]):
            if i < len(optimal_symbols):
                symbol, _ = optimal_symbols[i]
                token = opp['token']
                symbol_token_pairs.append((symbol, token))
        
        # Validate all pairs in batch
        validation_start = time.time()
        validation_results = self.symbol_selector.validate_symbols_batch(
            symbol_token_pairs, model_name
        )
        validation_time = time.time() - validation_start
        
        # Build final dictionary from valid pairs
        dictionary = {}
        token_savings_total = 0
        
        for (symbol, token), validation in validation_results.items():
            if validation.get('valid', False):
                dictionary[token] = symbol
                token_savings = validation.get('token_savings', 0)
                # Calculate total savings for this token across all occurrences
                token_count = next((opp['count'] for opp in opportunities if opp['token'] == token), 1)
                token_savings_total += token_savings * token_count
        
        elapsed = time.time() - start_time
        
        logger.info(f"Built optimized dictionary:")
        logger.info(f"  Entries: {len(dictionary)}/{len(opportunities)} opportunities")
        logger.info(f"  Total time: {elapsed:.2f}s (validation: {validation_time:.2f}s)")
        logger.info(f"  Estimated token savings: {token_savings_total}")
        logger.info(f"  Threading: {self.max_workers} workers")
        
        # Log cache performance
        cache_stats = self.symbol_selector.get_cache_stats()
        logger.debug(f"Cache performance: {cache_stats['hit_rate_percent']}% hit rate ({cache_stats['cache_hits']}/{cache_stats['cache_hits'] + cache_stats['cache_misses']})")
        
        return dictionary
    
    def build_dictionary_with_priorities(self,
                                       opportunities: List[Dict],
                                       model_name: str,
                                       max_entries: int = 100) -> Dict[str, str]:
        """Build dictionary with priority-based optimization for high-value tokens."""
        
        if not opportunities:
            return {}
        
        # Sort opportunities by priority (highest impact first)
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x.get('priority_score', 0) * x.get('count', 1), 
            reverse=True
        )
        
        # Use the fast builder with sorted opportunities
        return self.build_dictionary_fast(sorted_opportunities, model_name, max_entries)

# Global instances for singleton pattern
_fast_symbol_selector = None
_optimized_dictionary_builder = None

def get_fast_symbol_selector(config: Optional[Dict] = None) -> FastSymbolSelector:
    """Get singleton fast symbol selector."""
    global _fast_symbol_selector
    if _fast_symbol_selector is None:
        _fast_symbol_selector = FastSymbolSelector(config)
    return _fast_symbol_selector

def get_optimized_dictionary_builder(config: Optional[Dict] = None) -> OptimizedDictionaryBuilder:
    """Get singleton optimized dictionary builder."""
    global _optimized_dictionary_builder
    if _optimized_dictionary_builder is None:
        _optimized_dictionary_builder = OptimizedDictionaryBuilder(config)
    return _optimized_dictionary_builder

def benchmark_performance(opportunities: List[Dict], model_name: str = "gpt-4") -> Dict:
    """Benchmark the performance of optimized vs traditional symbol selection."""
    
    if not opportunities:
        return {"error": "No opportunities provided for benchmarking"}
    
    # Test optimized approach
    start_time = time.time()
    builder = get_optimized_dictionary_builder()
    optimized_dict = builder.build_dictionary_fast(opportunities, model_name)
    optimized_time = time.time() - start_time
    
    # Get cache stats
    cache_stats = builder.symbol_selector.get_cache_stats()
    
    return {
        "test_conditions": {
            "opportunities": len(opportunities),
            "model_name": model_name,
            "thread_count": builder.max_workers
        },
        "optimized_approach": {
            "time_seconds": round(optimized_time, 3),
            "dictionary_entries": len(optimized_dict),
            "cache_hit_rate": cache_stats['hit_rate_percent'],
            "threads_used": builder.max_workers
        },
        "performance_summary": {
            "total_time": round(optimized_time, 3),
            "entries_per_second": round(len(optimized_dict) / optimized_time, 1) if optimized_time > 0 else 0,
            "caching_enabled": cache_stats['cache_size'] > 0
        }
    }

if __name__ == "__main__":
    # Simple test/demo
    logging.basicConfig(level=logging.INFO)
    
    # Create test opportunities
    test_opportunities = [
        {"token": "function", "count": 10, "priority_score": 0.8},
        {"token": "variable", "count": 8, "priority_score": 0.7},
        {"token": "import", "count": 5, "priority_score": 0.6},
        {"token": "class", "count": 3, "priority_score": 0.5},
    ]
    
    # Run benchmark
    results = benchmark_performance(test_opportunities)
    print(f"Benchmark results: {json.dumps(results, indent=2)}") 