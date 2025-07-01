#!/usr/bin/env python3
"""
Dynamic Dictionary Configuration Manager
Manages configuration for the dynamic dictionary feature from config file.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.jsonc_parser import load_jsonc

logger = logging.getLogger('config_parser')


class DynamicConfigManager:
    """
    Manages dynamic dictionary configuration from the main config file.
    """
    
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "config" / "config.jsonc"
        self._config_cache = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load dynamic dictionary configuration from config file."""
        if self._config_cache is not None:
            return self._config_cache
        
        default_config = {
            "enabled": True,
            "min_token_length": 4,
            "min_frequency": 2,
            "max_dictionary_size": 262,  # Match config.jsonc setting
            "compression_threshold": 0.05,
            "enable_substring_analysis": True,
            "enable_phrase_analysis": True,
            "enable_pattern_analysis": True,
            "min_prompt_length": 500,
            "auto_detection_threshold": 0.3,
            "cleanup_max_age_hours": 24,
            "auto_cleanup": True
        }
        
        try:
            if self.config_path.exists():
                full_config = load_jsonc(str(self.config_path))
                dynamic_config = full_config.get("dynamic_dictionary", {})
                
                # Merge with defaults
                config = default_config.copy()
                config.update(dynamic_config)
                
                self._config_cache = config
                logger.debug(f"📋 Dynamic config loaded from {self.config_path.name}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                self._config_cache = default_config
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}, using defaults")
            self._config_cache = default_config
            return default_config
    
    def is_enabled(self) -> bool:
        """Check if dynamic dictionary analysis is enabled."""
        config = self.load_config()
        return config.get("enabled", True)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        config = self.load_config()
        return config.get(key, default)
    
    def reload_config(self):
        """Force reload of configuration from file."""
        self._config_cache = None
        return self.load_config()
    
    def get_configuration_summary(self) -> str:
        """Generate a summary of the current configuration."""
        config = self.load_config()
        return f"""
Dynamic Dictionary Configuration (from config/config.jsonc):

- Enabled: {config.get('enabled', True)}
- Minimum token length: {config.get('min_token_length', 4)} characters
- Minimum frequency: {config.get('min_frequency', 2)} occurrences  
- Maximum dictionary size: {config.get('max_dictionary_size', 262)} entries
- Compression threshold: {config.get('compression_threshold', 0.05)*100:.1f}%
- Minimum prompt length: {config.get('min_prompt_length', 500)} characters
- Auto detection threshold: {config.get('auto_detection_threshold', 0.3)*100:.1f}%
- Substring analysis: {'enabled' if config.get('enable_substring_analysis', True) else 'disabled'}
- Phrase analysis: {'enabled' if config.get('enable_phrase_analysis', True) else 'disabled'}
- Pattern analysis: {'enabled' if config.get('enable_pattern_analysis', True) else 'disabled'}
- Auto cleanup: {'enabled' if config.get('auto_cleanup', True) else 'disabled'}
- Cleanup max age: {config.get('cleanup_max_age_hours', 24)} hours

To modify these settings, edit the 'dynamic_dictionary' section in config/config.jsonc
""" 