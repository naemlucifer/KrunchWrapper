"""
Model-specific tokenizer mapping for accurate token validation.
This module provides tokenizer detection and loading for various model families.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Model family detection patterns (all lowercase for case-insensitive matching)
MODEL_PATTERNS = {
    # OpenAI models
    "gpt-4": ["gpt-4", "gpt4"],
    "gpt-3.5": ["gpt-3.5", "gpt-35", "turbo"],
    "gpt-3": ["davinci", "curie", "babbage", "ada"],
    
    # Anthropic models
    "claude": ["claude", "anthropic"],
    
    # Meta LLaMA models
    "llama": ["llama", "llama2", "llama-2", "llama3", "llama-3"],
    "codellama": ["codellama", "code-llama"],
    
    # Mistral models
    "mistral": ["mistral", "mixtral"],
    
    # Google models
    "gemini": ["gemini", "bard"],
    "palm": ["palm", "palm2"],
    
    # Qwen models
    "qwen": ["qwen", "qwen1.5", "qwen2", "qwen3", "qwen-"],
    
    # Other models
    "yi": ["yi-", "01-ai"],
    "deepseek": ["deepseek"],
    "phi": ["phi-", "microsoft/phi"],
    "falcon": ["falcon"],
    "starcoder": ["starcoder", "starcode"],
    "vicuna": ["vicuna"],
    "alpaca": ["alpaca"],
    "chatglm": ["chatglm", "glm-"],
}

# Tokenizer configurations for each model family
TOKENIZER_CONFIGS = {
    "gpt-4": {
        "type": "tiktoken",
        "encoding": "cl100k_base",
        "library": "tiktoken",
        "notes": "GPT-4 and GPT-3.5-turbo models"
    },
    "gpt-3.5": {
        "type": "tiktoken",
        "encoding": "cl100k_base",
        "library": "tiktoken",
        "notes": "Same as GPT-4"
    },
    "gpt-3": {
        "type": "tiktoken",
        "encoding": "p50k_base",
        "library": "tiktoken",
        "notes": "Legacy GPT-3 models"
    },
    "claude": {
        "type": "sentencepiece",
        "model_file": "claude_tokenizer.model",
        "library": "sentencepiece",
        "notes": "Claude uses a custom SentencePiece tokenizer"
    },
    "llama": {
        "type": "sentencepiece",
        "model_file": "tokenizer.model",
        "library": "sentencepiece",
        "vocab_size": 32000,
        "notes": "LLaMA/LLaMA2 tokenizer"
    },
    "llama3": {
        "type": "tiktoken",
        "encoding": "custom",
        "vocab_size": 128256,
        "library": "transformers",
        "notes": "LLaMA3 uses a custom tiktoken-style tokenizer"
    },
    "mistral": {
        "type": "sentencepiece",
        "model_file": "tokenizer.model",
        "library": "sentencepiece",
        "vocab_size": 32000,
        "notes": "Similar to LLaMA tokenizer"
    },
    "qwen": {
        "type": "tiktoken",
        "encoding": "qwen",
        "library": "transformers",
        "vocab_size": 151936,
        "special_tokens": {
            "<|endoftext|>": 151643,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        },
        "notes": "Qwen uses tiktoken with custom vocabulary"
    },
    "qwen2": {
        "type": "tiktoken", 
        "encoding": "qwen2",
        "library": "transformers",
        "vocab_size": 152064,
        "notes": "Qwen2 expanded vocabulary"
    },
    "qwen3": {
        "type": "tiktoken",
        "encoding": "qwen",
        "library": "transformers",
        "vocab_size": 151936,
        "notes": "Qwen3 uses same tokenizer as Qwen1"
    },
    "yi": {
        "type": "sentencepiece",
        "model_file": "tokenizer.model",
        "library": "sentencepiece",
        "vocab_size": 64000,
        "notes": "Yi models use expanded vocabulary"
    },
    "deepseek": {
        "type": "tiktoken",
        "encoding": "custom",
        "library": "transformers",
        "vocab_size": 100000,
        "notes": "DeepSeek custom tokenizer"
    },
    "gemini": {
        "type": "sentencepiece",
        "model_file": "spm_tokenizer.model",
        "library": "sentencepiece",
        "notes": "Google's Gemini tokenizer"
    },
    "chatglm": {
        "type": "sentencepiece",
        "model_file": "ice_text.model",
        "library": "sentencepiece",
        "vocab_size": 130344,
        "notes": "ChatGLM custom tokenizer"
    }
}

class ModelTokenizerValidator:
    """Validates token efficiency using model-specific tokenizers."""
    
    def __init__(self):
        self.tokenizer_cache = {}
        self.model_patterns = MODEL_PATTERNS.copy()  # Start with default patterns
        self._load_config()
        self._init_tokenizers()
    
    def _load_config(self):
        """Load configuration and merge custom model mappings."""
        try:
            from utils.jsonc_parser import load_jsonc
            config_path = Path(__file__).parents[1] / "config" / "config.jsonc"
            
            if config_path.exists():
                config = load_jsonc(str(config_path))
                model_config = config.get("model_tokenizer", {})
                
                # Load custom model mappings if present
                custom_mappings = model_config.get("custom_model_mappings", {})
                if custom_mappings:
                    logger.info(f"Loading {len(custom_mappings)} custom model mappings")
                    for family, patterns in custom_mappings.items():
                        if isinstance(patterns, list):
                            # Convert all patterns to lowercase for case-insensitive matching
                            lowercase_patterns = [pattern.lower() for pattern in patterns]
                            if family in self.model_patterns:
                                # Extend existing patterns
                                self.model_patterns[family].extend(lowercase_patterns)
                                logger.debug(f"Extended {family} patterns with: {lowercase_patterns}")
                            else:
                                # Create new family
                                self.model_patterns[family] = lowercase_patterns
                                logger.info(f"Added new model family '{family}' with patterns: {lowercase_patterns}")
                        else:
                            logger.warning(f"Custom mapping for '{family}' is not a list, skipping")
                            
                # Log configuration status
                if custom_mappings:
                    logger.info(f"Model tokenizer loaded with {len(self.model_patterns)} families (including {len(custom_mappings)} custom)")
                else:
                    logger.debug(f"Model tokenizer loaded with {len(self.model_patterns)} default families")
                    
        except Exception as e:
            logger.warning(f"Failed to load model tokenizer config: {e}")
            logger.info("Using default model patterns only")
    
    def _init_tokenizers(self):
        """Initialize available tokenizers based on installed libraries."""
        self.available_tokenizers = {
            "tiktoken": self._check_tiktoken(),
            "transformers": self._check_transformers(),
            "sentencepiece": self._check_sentencepiece()
        }
        
    def _check_tiktoken(self) -> bool:
        try:
            import tiktoken
            return True
        except ImportError:
            return False
            
    def _check_transformers(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False
            
    def _check_sentencepiece(self) -> bool:
        try:
            import sentencepiece
            return True
        except ImportError:
            return False
    
    def detect_model_family(self, model_name: str) -> Optional[str]:
        """Detect model family from model name/path (case-insensitive)."""
        if not model_name:
            return None
            
        model_lower = model_name.lower()
        
        # Special case for Qwen versions (prioritize more specific versions first)
        if "qwen3" in model_lower or "qwen-3" in model_lower:
            return "qwen3"
        elif "qwen2" in model_lower or "qwen-2" in model_lower:
            return "qwen2"
        elif "qwen" in model_lower:
            return "qwen"
        
        # Special case for LLaMA 3 - check for llama3 or llama-3 first
        if "llama3" in model_lower or "llama-3" in model_lower:
            return "llama3"
            
        # Check configured patterns (including custom ones)
        for family, patterns in self.model_patterns.items():
            for pattern in patterns:
                # Ensure pattern matching is case-insensitive
                pattern_lower = pattern.lower() if isinstance(pattern, str) else pattern
                if pattern_lower in model_lower:
                    logger.debug(f"Model '{model_name}' matched pattern '{pattern}' for family '{family}'")
                    return family
                    
        # If no match found, log for debugging
        logger.debug(f"No pattern match found for model: {model_name}")
        return None
    
    def get_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get appropriate tokenizer for the model."""
        # Check cache first
        if model_name in self.tokenizer_cache:
            return self.tokenizer_cache[model_name]
            
        family = self.detect_model_family(model_name)
        if not family or family not in TOKENIZER_CONFIGS:
            logger.warning(f"Unknown model family for: {model_name}")
            return None
            
        config = TOKENIZER_CONFIGS[family]
        tokenizer = None
        
        try:
            if config["type"] == "tiktoken":
                tokenizer = self._load_tiktoken_tokenizer(config, model_name)
            elif config["type"] == "sentencepiece":
                tokenizer = self._load_sentencepiece_tokenizer(config, model_name)
            elif config["type"] == "transformers":
                tokenizer = self._load_transformers_tokenizer(config, model_name)
                
            if tokenizer:
                self.tokenizer_cache[model_name] = tokenizer
                logger.info(f"Loaded {config['type']} tokenizer for {family} model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            
        return tokenizer
    
    def _load_tiktoken_tokenizer(self, config: Dict, model_name: str):
        """Load tiktoken-based tokenizer."""
        if not self.available_tokenizers["tiktoken"]:
            logger.warning("tiktoken not available")
            return None
            
        import tiktoken
        
        if config["encoding"] in ["cl100k_base", "p50k_base", "r50k_base"]:
            return tiktoken.get_encoding(config["encoding"])
        elif config["encoding"] == "qwen":
            # For Qwen models, try to use transformers if available
            if self.available_tokenizers["transformers"]:
                return self._load_qwen_tokenizer(model_name)
            else:
                # Fallback to a similar tiktoken encoding
                logger.warning("Using cl100k_base as fallback for Qwen model")
                return tiktoken.get_encoding("cl100k_base")
        else:
            # Custom encoding - try transformers
            return None
    
    def _load_qwen_tokenizer(self, model_name: str):
        """Load Qwen tokenizer using transformers."""
        if not self.available_tokenizers["transformers"]:
            return None
            
        try:
            from transformers import AutoTokenizer
            
            # Try to load from model name/path
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load Qwen tokenizer from transformers: {e}")
            # Fallback to tiktoken
            import tiktoken
            return tiktoken.get_encoding("cl100k_base")
    
    def _load_sentencepiece_tokenizer(self, config: Dict, model_name: str):
        """Load SentencePiece tokenizer."""
        if not self.available_tokenizers["sentencepiece"]:
            logger.warning("sentencepiece not available")
            return None
            
        import sentencepiece as spm
        
        # Try to find tokenizer model file
        model_path = Path(model_name).parent / config["model_file"]
        if model_path.exists():
            sp = spm.SentencePieceProcessor()
            sp.load(str(model_path))
            return sp
        else:
            logger.warning(f"SentencePiece model file not found: {model_path}")
            return None
    
    def _load_transformers_tokenizer(self, config: Dict, model_name: str):
        """Load tokenizer using transformers library."""
        if not self.available_tokenizers["transformers"]:
            logger.warning("transformers not available")
            return None
            
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with transformers: {e}")
            return None
    
    def validate_token_efficiency(self, 
                                original_text: str, 
                                compressed_text: str,
                                model_name: str) -> Dict[str, Any]:
        """Validate compression efficiency for specific model."""
        tokenizer = self.get_tokenizer(model_name)
        
        if not tokenizer:
            # Fallback to character-based estimation
            char_ratio = (len(original_text) - len(compressed_text)) / len(original_text) if len(original_text) > 0 else 0
            return {
                "valid": char_ratio > 0,
                "token_savings": int(char_ratio * len(original_text.split())),
                "original_tokens": len(original_text.split()),
                "compressed_tokens": len(compressed_text.split()),
                "method": "character_estimation",
                "model_family": self.detect_model_family(model_name)
            }
        
        # Get token counts based on tokenizer type
        if hasattr(tokenizer, 'encode'):
            # tiktoken or transformers style
            original_tokens = len(tokenizer.encode(original_text))
            compressed_tokens = len(tokenizer.encode(compressed_text))
        elif hasattr(tokenizer, 'encode_as_ids'):
            # sentencepiece style
            original_tokens = len(tokenizer.encode_as_ids(original_text))
            compressed_tokens = len(tokenizer.encode_as_ids(compressed_text))
        else:
            # Unknown tokenizer type
            return {
                "valid": True,
                "method": "unknown_tokenizer"
            }
        
        token_savings = original_tokens - compressed_tokens
        token_ratio = token_savings / original_tokens if original_tokens > 0 else 0
        
        return {
            "valid": token_savings > 0,
            "token_savings": token_savings,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_ratio": token_ratio,
            "method": "model_specific",
            "model_family": self.detect_model_family(model_name),
            "tokenizer_type": type(tokenizer).__name__
        }

# Global instance for easy access
_validator_instance = None

def get_model_tokenizer_validator() -> ModelTokenizerValidator:
    """Get the global model tokenizer validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ModelTokenizerValidator()
    return _validator_instance

def validate_with_model(original_text: str, compressed_text: str, model_name: str) -> float:
    """
    Wrapper function to use in compress.py
    
    Replace the validate_tokenization_efficiency function with:
    """
    validator = get_model_tokenizer_validator()
    result = validator.validate_token_efficiency(original_text, compressed_text, model_name)
    
    if result["valid"]:
        return result.get("token_ratio", 0.0)
    else:
        return -abs(result.get("token_ratio", 0.1))  # Negative for invalid 