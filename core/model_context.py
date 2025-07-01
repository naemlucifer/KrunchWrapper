"""
Model context tracking for KrunchWrapper compression system.
This module provides a way to track the current model being used for compression validation.
"""

import threading
import logging
from typing import Optional, Dict, Any
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable to track current model
_current_model: ContextVar[Optional[str]] = ContextVar('current_model', default=None)

class ModelContext:
    """Context manager for tracking the current model in compression operations."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.token = None
    
    def __enter__(self):
        self.token = _current_model.set(self.model_name)
        logger.debug(f"Set model context to: {self.model_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _current_model.reset(self.token)
            logger.debug(f"Reset model context from: {self.model_name}")

def get_current_model() -> Optional[str]:
    """Get the current model name from context."""
    return _current_model.get()

def set_current_model(model_name: str) -> None:
    """Set the current model name in context."""
    _current_model.set(model_name)
    logger.debug(f"Set current model to: {model_name}")

def extract_model_from_provider_format(model_id: str) -> str:
    """
    Extract clean model name from provider/model format used by cline.
    
    Examples:
        "anthropic/claude-3-5-sonnet-20241022" -> "claude-3-5-sonnet-20241022"
        "openai/gpt-4" -> "gpt-4"
        "qwen/qwen2.5-coder-32b-instruct" -> "qwen2.5-coder-32b-instruct"
    """
    if not model_id:
        return ""
    
    # Split by '/' and take the last part (model name)
    parts = model_id.split('/')
    if len(parts) > 1:
        return parts[-1]
    else:
        return model_id

def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for consistent detection.
    
    This function handles common variations in model naming.
    """
    if not model_name:
        return ""
    
    # Extract from provider format if needed
    clean_name = extract_model_from_provider_format(model_name)
    
    # Normalize to lowercase for consistent matching
    normalized = clean_name.lower()
    
    # Handle common variations
    normalized = normalized.replace("_", "-")
    normalized = normalized.replace(" ", "-")
    
    # Remove common suffixes that don't affect tokenization
    suffixes_to_remove = ["-instruct", "-chat", "-preview"]
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    
    return normalized

# Global model context for situations where context vars are not available
_global_model_context = None
_global_lock = threading.Lock()

def set_global_model_context(model_name: str) -> None:
    """Set the global model context as a fallback."""
    global _global_model_context
    with _global_lock:
        _global_model_context = model_name

def get_global_model_context() -> Optional[str]:
    """Get the global model context."""
    global _global_model_context
    with _global_lock:
        return _global_model_context

def get_effective_model() -> Optional[str]:
    """
    Get the effective model name, trying context first, then global fallback.
    """
    # Try context variable first
    model = get_current_model()
    if model:
        return model
    
    # Fall back to global context
    return get_global_model_context() 