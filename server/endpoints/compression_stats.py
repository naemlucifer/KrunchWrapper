"""Compression stats endpoint module."""
import time
from fastapi import HTTPException


async def get_compression_stats(config):
    """Get compression and system performance statistics."""
    try:
        result = {}
        
        # Conversation compression stats
        if config.conversation_compression_enabled:
            from core.conversation_compress import get_conversation_compression_stats
            stats = get_conversation_compression_stats()
            
            result["conversation_compression"] = {
                "enabled": True,
                "stats": stats,
                "config": {
                    "max_conversations": config.conversation_max_conversations,
                    "min_net_efficiency": config.conversation_min_net_efficiency,
                    "long_conversation_threshold": config.conversation_long_threshold,
                    "long_conversation_min_efficiency": config.conversation_long_min_efficiency
                }
            }
        else:
            result["conversation_compression"] = {
                "enabled": False,
                "message": "Conversation compression is disabled in configuration"
            }
        
        # Async logging performance stats
        try:
            from core.async_logger import get_optimized_logger
            async_logger = get_optimized_logger()
            async_stats = async_logger.get_stats()
            result["async_logging"] = {
                "enabled": True,
                "stats": async_stats,
                "note": "Configure performance via config/async_logging.jsonc"
            }
        except Exception as e:
            result["async_logging"] = {
                "enabled": False,
                "error": f"Failed to get async logging stats: {str(e)}"
            }
        
        # Basic system info
        result["system"] = {
            "compression_enabled": True,
            "min_characters": config.min_characters,
            "min_compression_ratio": config.min_compression_ratio
        }
        
        result["timestamp"] = time.time()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compression stats: {str(e)}")


async def reset_compression_state(config):
    """Reset conversation compression state (for testing/debugging)."""
    try:
        if config.conversation_compression_enabled:
            from core.conversation_compress import reset_conversation_compression
            reset_conversation_compression()
            
            return {
                "message": "Conversation compression state reset successfully",
                "timestamp": time.time()
            }
        else:
            return {
                "message": "Conversation compression is disabled",
                "timestamp": time.time()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset compression state: {str(e)}") 