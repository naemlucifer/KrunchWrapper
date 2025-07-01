#!/usr/bin/env python3
"""
Model-Specific Symbol Selector

This module loads the unified model-specific dictionary and selects the most efficient
symbols for compression based on the current model architecture.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

try:
    from .model_context import get_effective_model, normalize_model_name
    from .model_tokenizer_validator import get_model_tokenizer_validator
except ImportError:
    # Handle imports when run as script
    from model_context import get_effective_model, normalize_model_name
    from model_tokenizer_validator import get_model_tokenizer_validator

logger = logging.getLogger(__name__)

class ModelSpecificSymbolSelector:
    """Selects optimal symbols for compression based on model architecture."""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the symbol selector.
        
        Args:
            dictionary_path: Path to the unified model-specific dictionary
        """
        if dictionary_path is None:
            dictionary_path = Path(__file__).parent.parent / "dictionaries" / "model_specific_symbols.json"
        
        self.dictionary_path = Path(dictionary_path)
        self.unified_dict = self._load_unified_dictionary()
        self.validator = get_model_tokenizer_validator()
        
    def _load_unified_dictionary(self) -> Dict:
        """Load the unified model-specific dictionary."""
        if not self.dictionary_path.exists():
            logger.warning(f"Model-specific dictionary not found at {self.dictionary_path}")
            logger.info("Run 'python utils/generate_model_specific_dictionary.py' to generate it")
            return {}
            
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                unified_dict = json.load(f)
            
            symbol_count = len([k for k in unified_dict.keys() if not k.startswith('_')])
            model_families = unified_dict.get('_metadata', {}).get('model_families', [])
            
            logger.info(f"Loaded unified dictionary with {symbol_count} symbols for {len(model_families)} model families")
            return unified_dict
            
        except Exception as e:
            logger.error(f"Failed to load unified dictionary from {self.dictionary_path}: {e}")
            return {}
    
    def get_symbols_for_model(self, model_name: str, max_symbols: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get symbols optimized for a specific model, sorted by token efficiency.
        
        Args:
            model_name: Name of the model to optimize for
            max_symbols: Maximum number of symbols to return (None for all)
            
        Returns:
            List of (symbol, token_count) tuples sorted by efficiency (lowest tokens first)
        """
        if not self.unified_dict:
            logger.warning("No unified dictionary loaded, falling back to priority symbols")
            return self._get_fallback_symbols(max_symbols)
        
        # Detect model family
        model_family = self.validator.detect_model_family(model_name)
        if not model_family:
            logger.warning(f"Unknown model family for {model_name}, using average token counts")
            return self._get_symbols_by_average(max_symbols)
        
        logger.info(f"Selecting symbols optimized for {model_family} (model: {model_name})")
        
        # Extract symbols with token counts for this model family
        symbols_with_counts = []
        
        for symbol, data in self.unified_dict.items():
            if symbol.startswith('_'):  # Skip metadata
                continue
                
            token_counts = data.get('token_counts', {})
            
            if model_family in token_counts:
                token_count = token_counts[model_family]
                symbols_with_counts.append((symbol, token_count))
            else:
                # Fallback to average if model family not available
                avg_tokens = data.get('average_tokens', 999)
                symbols_with_counts.append((symbol, int(avg_tokens)))
        
        # Sort by token count (ascending - fewer tokens = better)
        symbols_with_counts.sort(key=lambda x: x[1])
        
        # Apply limit if specified
        if max_symbols is not None:
            symbols_with_counts = symbols_with_counts[:max_symbols]
        
        logger.info(f"Selected {len(symbols_with_counts)} symbols for {model_family}")
        if symbols_with_counts:
            best_symbols = symbols_with_counts[:5]
            worst_symbols = symbols_with_counts[-5:]
            logger.debug(f"Best symbols: {best_symbols}")
            logger.debug(f"Worst symbols: {worst_symbols}")
        
        return symbols_with_counts
    
    def _get_symbols_by_average(self, max_symbols: Optional[int] = None) -> List[Tuple[str, int]]:
        """Get symbols sorted by average token count across all models."""
        symbols_with_counts = []
        
        for symbol, data in self.unified_dict.items():
            if symbol.startswith('_'):  # Skip metadata
                continue
                
            avg_tokens = data.get('average_tokens', 999)
            symbols_with_counts.append((symbol, int(avg_tokens)))
        
        # Sort by average token count
        symbols_with_counts.sort(key=lambda x: x[1])
        
        if max_symbols is not None:
            symbols_with_counts = symbols_with_counts[:max_symbols]
        
        logger.info(f"Selected {len(symbols_with_counts)} symbols by average token count")
        return symbols_with_counts
    
    def _get_fallback_symbols(self, max_symbols: Optional[int] = None) -> List[Tuple[str, int]]:
        """Fallback symbols when unified dictionary is not available."""
        fallback_symbols = [
            # Greek letters (usually 1 token)
            ('α', 1), ('β', 1), ('γ', 1), ('δ', 1), ('ε', 1), ('ζ', 1), ('η', 1), ('θ', 1),
            ('ι', 1), ('κ', 1), ('λ', 1), ('μ', 1), ('ν', 1), ('ξ', 1), ('ο', 1), ('π', 1),
            ('ρ', 1), ('σ', 1), ('τ', 1), ('υ', 1), ('φ', 1), ('χ', 1), ('ψ', 1), ('ω', 1),
            
            # Currency and mathematical symbols (usually 1 token)
            ('€', 1), ('£', 1), ('¥', 1), ('©', 1), ('®', 1), ('™', 1), ('°', 1), ('±', 1),
            ('²', 1), ('³', 1), ('¼', 1), ('½', 1), ('¾', 1), ('∞', 1), ('≈', 1), ('≠', 1),
            ('≤', 1), ('≥', 1), ('×', 1), ('÷', 1),
            
            # Arrows (usually 1 token)
            ('←', 1), ('→', 1), ('↑', 1), ('↓', 1), ('↔', 1), ('↕', 1),
            
            # Geometric shapes (usually 1-2 tokens)
            ('□', 1), ('■', 2), ('△', 1), ('▲', 2), ('○', 1), ('●', 2), ('◆', 2), ('◇', 1)
        ]
        
        if max_symbols is not None:
            fallback_symbols = fallback_symbols[:max_symbols]
        
        logger.warning(f"Using {len(fallback_symbols)} fallback symbols")
        return fallback_symbols
    
    def get_symbol_generator(self, model_name: str, max_symbols: Optional[int] = None):
        """
        Get a generator for symbols optimized for the specified model.
        
        Args:
            model_name: Name of the model to optimize for
            max_symbols: Maximum number of symbols to generate (None for all)
            
        Yields:
            str: Next optimal symbol for the model
        """
        symbols_with_counts = self.get_symbols_for_model(model_name, max_symbols)
        
        for symbol, token_count in symbols_with_counts:
            yield symbol
    
    def get_model_efficiency_stats(self, model_name: str) -> Dict:
        """Get efficiency statistics for a specific model."""
        if not self.unified_dict:
            return {"error": "No unified dictionary loaded"}
        
        model_family = self.validator.detect_model_family(model_name)
        if not model_family:
            return {"error": f"Unknown model family for {model_name}"}
        
        # Collect token counts for this model
        token_counts = []
        symbols_with_data = []
        
        for symbol, data in self.unified_dict.items():
            if symbol.startswith('_'):
                continue
                
            counts = data.get('token_counts', {})
            if model_family in counts:
                token_count = counts[model_family]
                token_counts.append(token_count)
                symbols_with_data.append((symbol, token_count, data))
        
        if not token_counts:
            return {"error": f"No token data for model family {model_family}"}
        
        # Calculate statistics
        stats = {
            "model_name": model_name,
            "model_family": model_family,
            "total_symbols": len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "average_tokens": round(sum(token_counts) / len(token_counts), 2),
            "single_token_symbols": len([c for c in token_counts if c == 1]),
            "multi_token_symbols": len([c for c in token_counts if c > 1])
        }
        
        # Find best and worst symbols
        sorted_symbols = sorted(symbols_with_data, key=lambda x: x[1])
        stats["best_symbols"] = [(s[0], s[1]) for s in sorted_symbols[:10]]
        stats["worst_symbols"] = [(s[0], s[1]) for s in sorted_symbols[-10:]]
        
        return stats
    
    def compare_model_efficiency(self, model1: str, model2: str) -> Dict:
        """Compare symbol efficiency between two models."""
        if not self.unified_dict:
            return {"error": "No unified dictionary loaded"}
        
        family1 = self.validator.detect_model_family(model1)
        family2 = self.validator.detect_model_family(model2)
        
        if not family1 or not family2:
            return {"error": f"Unknown model family for {model1 if not family1 else model2}"}
        
        # Compare symbols where both models have data
        comparison_data = []
        
        for symbol, data in self.unified_dict.items():
            if symbol.startswith('_'):
                continue
                
            counts = data.get('token_counts', {})
            if family1 in counts and family2 in counts:
                count1 = counts[family1]
                count2 = counts[family2]
                difference = count1 - count2  # Positive means model1 is worse
                comparison_data.append({
                    "symbol": symbol,
                    f"{family1}_tokens": count1,
                    f"{family2}_tokens": count2,
                    "difference": difference,
                    "better_for": family2 if difference > 0 else family1 if difference < 0 else "equal"
                })
        
        if not comparison_data:
            return {"error": f"No comparable data between {family1} and {family2}"}
        
        # Calculate summary statistics
        differences = [d["difference"] for d in comparison_data]
        model1_better = len([d for d in comparison_data if d["difference"] < 0])
        model2_better = len([d for d in comparison_data if d["difference"] > 0])
        equal = len([d for d in comparison_data if d["difference"] == 0])
        
        return {
            "model1": {"name": model1, "family": family1},
            "model2": {"name": model2, "family": family2},
            "total_compared_symbols": len(comparison_data),
            "model1_better_count": model1_better,
            "model2_better_count": model2_better,
            "equal_count": equal,
            "average_difference": round(sum(differences) / len(differences), 3),
            "max_difference": max(differences),
            "min_difference": min(differences),
            "details": sorted(comparison_data, key=lambda x: abs(x["difference"]), reverse=True)[:20]  # Top 20 differences
        }

# Global instance for easy access
_symbol_selector_instance = None

def get_model_specific_symbol_selector() -> ModelSpecificSymbolSelector:
    """Get the global model-specific symbol selector instance."""
    global _symbol_selector_instance
    if _symbol_selector_instance is None:
        _symbol_selector_instance = ModelSpecificSymbolSelector()
    return _symbol_selector_instance

def get_optimal_symbols_for_current_model(max_symbols: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Convenience function to get optimal symbols for the current model context.
    
    Args:
        max_symbols: Maximum number of symbols to return
        
    Returns:
        List of (symbol, token_count) tuples optimized for current model
    """
    model_name = get_effective_model()
    if not model_name:
        logger.warning("No model context available, using fallback symbols")
        selector = get_model_specific_symbol_selector()
        return selector._get_fallback_symbols(max_symbols)
    
    selector = get_model_specific_symbol_selector()
    return selector.get_symbols_for_model(model_name, max_symbols) 