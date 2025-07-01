"""
Enhanced model context and tokenizer caching for performance optimization.
"""

import functools
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import weakref
import logging
from .model_tokenizer_validator import ModelTokenizerValidator, get_model_tokenizer_validator
from .model_context import ModelContext, get_current_model, normalize_model_name

logger = logging.getLogger(__name__)

class OptimizedModelTokenizerValidator:
    """Enhanced validator with aggressive caching and lazy loading."""
    
    def __init__(self, 
                 base_validator: Optional[ModelTokenizerValidator] = None,
                 model_specific_cache: bool = True,
                 max_cache_size: int = 1000,
                 max_validation_time_samples: int = 100):
        """
        Initialize the optimized validator.
        
        Args:
            base_validator: Base validator to use for actual validation
            model_specific_cache: Whether to use model-specific caching (default: True)
            max_cache_size: Maximum number of validation results to cache (default: 1000)
            max_validation_time_samples: Maximum validation time samples to track (default: 100)
        """
        self.base_validator = base_validator or get_model_tokenizer_validator()
        self.model_specific_cache = model_specific_cache
        self.max_cache_size = max_cache_size
        self.tokenizer_cache = {}
        self.validation_cache = {}  # Cache validation results
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.RLock()
        
        # Weak reference cache for temporary objects
        self._weak_cache = weakref.WeakValueDictionary()
        
        # Performance tracking
        self._validation_times = []
        self._max_validation_time_samples = max_validation_time_samples
        
    @functools.lru_cache(maxsize=64)
    def detect_model_family_cached(self, model_name: str) -> Optional[str]:
        """Cached model family detection."""
        return self.base_validator.detect_model_family(model_name)
    
    def _create_cache_key(self, original_text: str, compressed_text: str, model_name: str) -> int:
        """Create a hash-based cache key from inputs."""
        if self.model_specific_cache:
            # Use normalized model name for model-specific caching
            normalized_model = normalize_model_name(model_name)
            return hash((original_text, compressed_text, normalized_model))
        else:
            # Model-agnostic caching - faster but less accurate
            return hash((original_text, compressed_text))
    
    def validate_token_efficiency_cached(self, 
                                       original_text: str, 
                                       compressed_text: str,
                                       model_name: str) -> Dict[str, Any]:
        """Cache validation results for identical inputs."""
        
        # Create cache key from inputs
        cache_key = self._create_cache_key(original_text, compressed_text, model_name)
        
        with self._lock:
            if cache_key in self.validation_cache:
                self.cache_hits += 1
                cache_mode = "model-specific" if self.model_specific_cache else "model-agnostic"
                logger.debug(f"Cache hit for model {model_name} (mode: {cache_mode})")
                return self.validation_cache[cache_key].copy()
            
            self.cache_misses += 1
            cache_mode = "model-specific" if self.model_specific_cache else "model-agnostic"
            logger.debug(f"Cache miss for model {model_name} (mode: {cache_mode})")
        
        # Perform validation with timing
        start_time = time.perf_counter()
        result = self.base_validator.validate_token_efficiency(original_text, compressed_text, model_name)
        validation_time = time.perf_counter() - start_time
        
        # Track performance
        self._track_validation_time(validation_time)
        
        # Add cache metadata to result
        result['cache_hit'] = False
        result['validation_time'] = validation_time
        
        with self._lock:
            # Cache result (limit cache size to prevent memory issues)
            if len(self.validation_cache) < self.max_cache_size:
                self.validation_cache[cache_key] = result.copy()
            else:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.validation_cache))
                del self.validation_cache[oldest_key]
                self.validation_cache[cache_key] = result.copy()
        
        return result
    
    def validate_batch(self, 
                      validations: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Perform batch validation for better performance."""
        results = []
        
        for original_text, compressed_text, model_name in validations:
            result = self.validate_token_efficiency_cached(original_text, compressed_text, model_name)
            results.append(result)
        
        return results
    
    def _track_validation_time(self, validation_time: float) -> None:
        """Track validation performance metrics."""
        with self._lock:
            self._validation_times.append(validation_time)
            if len(self._validation_times) > self._max_validation_time_samples:
                self._validation_times.pop(0)
    
    def get_tokenizer_cached(self, model_name: str) -> Optional[Any]:
        """Get tokenizer with enhanced caching."""
        with self._lock:
            if model_name in self.tokenizer_cache:
                return self.tokenizer_cache[model_name]
        
        # Use base validator to get tokenizer
        tokenizer = self.base_validator.get_tokenizer(model_name)
        
        if tokenizer:
            with self._lock:
                self.tokenizer_cache[model_name] = tokenizer
        
        return tokenizer
    
    def clear_cache(self, cache_type: str = "all") -> Dict[str, int]:
        """Clear specific or all caches."""
        cleared_counts = {}
        
        with self._lock:
            if cache_type in ["all", "validation"]:
                cleared_counts["validation"] = len(self.validation_cache)
                self.validation_cache.clear()
            
            if cache_type in ["all", "tokenizer"]:
                cleared_counts["tokenizer"] = len(self.tokenizer_cache)
                self.tokenizer_cache.clear()
            
            if cache_type in ["all", "family"]:
                cleared = self.detect_model_family_cached.cache_info().currsize
                self.detect_model_family_cached.cache_clear()
                cleared_counts["family"] = cleared
            
            # Reset performance counters
            if cache_type == "all":
                self.cache_hits = 0
                self.cache_misses = 0
                self._validation_times.clear()
        
        logger.info(f"Cleared caches: {cleared_counts}")
        return cleared_counts
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        # Calculate validation time statistics
        avg_validation_time = 0
        min_validation_time = 0
        max_validation_time = 0
        
        with self._lock:
            if self._validation_times:
                avg_validation_time = sum(self._validation_times) / len(self._validation_times)
                min_validation_time = min(self._validation_times)
                max_validation_time = max(self._validation_times)
        
        family_cache_info = self.detect_model_family_cached.cache_info()
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_tokenizers": len(self.tokenizer_cache),
            "cached_validations": len(self.validation_cache),
            "family_cache_hits": family_cache_info.hits,
            "family_cache_misses": family_cache_info.misses,
            "family_cache_size": family_cache_info.currsize,
            "avg_validation_time": avg_validation_time,
            "min_validation_time": min_validation_time,
            "max_validation_time": max_validation_time,
            "validation_samples": len(self._validation_times),
            # Configuration settings
            "model_specific_cache": self.model_specific_cache,
            "max_cache_size": self.max_cache_size,
            "max_validation_time_samples": self._max_validation_time_samples
        }

class FastModelContext:
    """Optimized model context with caching and batch operations."""
    
    def __init__(self):
        self._context_cache = {}
        self._access_times = {}  # Track access patterns
        self._lock = threading.RLock()
        
        # Performance tracking
        self._context_access_count = 0
        
    @functools.lru_cache(maxsize=32)
    def normalize_model_name_cached(self, model_name: str) -> str:
        """Cached model name normalization."""
        return normalize_model_name(model_name)
    
    def set_context_batch(self, contexts: Dict[str, str]) -> None:
        """Set multiple model contexts at once."""
        current_time = time.time()
        
        with self._lock:
            for key, value in contexts.items():
                self._context_cache[key] = value
                self._access_times[key] = current_time
    
    def get_context_cached(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get context with access tracking."""
        current_time = time.time()
        
        with self._lock:
            self._context_access_count += 1
            self._access_times[key] = current_time
            return self._context_cache.get(key, default)
    
    def get_effective_model_fast(self) -> Optional[str]:
        """Fast model retrieval with minimal overhead."""
        # Try context variable first (fastest)
        model = get_current_model()
        if model:
            return model
        
        # Fallback to cached global context
        return self.get_context_cached('global_model')
    
    def cleanup_stale_contexts(self, max_age_seconds: float = 3600) -> int:
        """Clean up contexts that haven't been accessed recently."""
        current_time = time.time()
        removed_count = 0
        
        with self._lock:
            stale_keys = [
                key for key, access_time in self._access_times.items()
                if current_time - access_time > max_age_seconds
            ]
            
            for key in stale_keys:
                if key in self._context_cache:
                    del self._context_cache[key]
                del self._access_times[key]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale context entries")
        
        return removed_count
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context performance statistics."""
        with self._lock:
            return {
                "cached_contexts": len(self._context_cache),
                "total_accesses": self._context_access_count,
                "tracked_access_times": len(self._access_times),
                "normalize_cache_info": self.normalize_model_name_cached.cache_info()._asdict()
            }

# Singleton instances with lazy initialization
_optimized_validator = None
_fast_context = None
_instances_lock = threading.Lock()

def get_optimized_validator(
    model_specific_cache: Optional[bool] = None,
    max_cache_size: Optional[int] = None,
    max_validation_time_samples: Optional[int] = None,
    force_new: bool = False
) -> OptimizedModelTokenizerValidator:
    """
    Get singleton optimized validator with optional configuration.
    
    Args:
        model_specific_cache: Whether to use model-specific caching (default: True)
        max_cache_size: Maximum number of validation results to cache (default: 1000)
        max_validation_time_samples: Maximum validation time samples (default: 100)
        force_new: Force creation of new instance (default: False)
    """
    global _optimized_validator
    
    # Create new instance if requested or if configuration differs
    if _optimized_validator is None or force_new:
        with _instances_lock:
            if _optimized_validator is None or force_new:
                kwargs = {}
                if model_specific_cache is not None:
                    kwargs['model_specific_cache'] = model_specific_cache
                if max_cache_size is not None:
                    kwargs['max_cache_size'] = max_cache_size
                if max_validation_time_samples is not None:
                    kwargs['max_validation_time_samples'] = max_validation_time_samples
                
                _optimized_validator = OptimizedModelTokenizerValidator(**kwargs)
                logger.info(f"Initialized OptimizedModelTokenizerValidator with config: {kwargs}")
    
    return _optimized_validator

def get_fast_context() -> FastModelContext:
    """Get singleton fast context."""
    global _fast_context
    if _fast_context is None:
        with _instances_lock:
            if _fast_context is None:
                _fast_context = FastModelContext()
                logger.info("Initialized FastModelContext")
    return _fast_context



# Context manager for performance monitoring
class PerformanceMonitor:
    """Context manager for monitoring validation performance."""
    
    def __init__(self, operation_name: str = "validation"):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            logger.debug(f"{self.operation_name} took {duration:.4f} seconds")

