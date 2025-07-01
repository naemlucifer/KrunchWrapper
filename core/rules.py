"""
Dictionary loading module - DEPRECATED

This module previously handled loading static compression dictionaries.
All dictionary-based compression has been removed in favor of dynamic compression.
This file is kept for compatibility but all functions now return empty results.
"""

import functools
from typing import Dict

@functools.lru_cache(maxsize=None)  
def load_dict(file_path: str) -> Dict[str, str]:
    """
    Load dictionary from temporary file created by dynamic analysis.
    
    Args:
        file_path: Path to temporary dictionary JSON file
        
    Returns:
        Dictionary mapping symbol -> pattern (for decompression)
    """
    import json
    import os
    
    # If it's an old-style lang parameter, return empty dict (backwards compatibility)
    if not file_path or not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract tokens dictionary from the JSON structure
            tokens = data.get("tokens", {})
            # Tokens are already in symbol -> pattern format from dynamic_dictionary.py
            return tokens
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load dictionary from {file_path}: {e}")
        return {} 