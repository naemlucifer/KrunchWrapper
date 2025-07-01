import os
import argparse
import uvicorn
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import aiohttp
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field, validator
import pathlib
import sys
import time
import tiktoken

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jsonc_parser import load_jsonc

from core.compress import compress_with_dynamic_analysis, compress_with_selective_tool_call_analysis, decompress, decompress_multimodal_aware
from core.system_prompt import build_system_prompt
from core.system_prompt_interceptor import SystemPromptInterceptor
from core.cline_system_prompt_interceptor import ClineSystemPromptInterceptor
from core.webui_system_prompt_interceptor import get_webui_system_prompt_interceptor, detect_webui_request
from core.interface_engine import detect_and_process_compression, InterfaceEngine
from core.model_context import ModelContext, set_global_model_context, extract_model_from_provider_format
from core.optimized_model_validator import get_optimized_validator
from core.async_logger import setup_global_async_logging
from core.cline_server_handler import get_cline_handler, is_cline_request, should_disable_compression_for_cline

# Import all logging functions from the extracted module
from server.logging_utils import (
    log_message, log_config_message, flush_config_message_buffer,
    log_performance_metrics, log_verbose_content, log_passthrough_request,
    log_passthrough_response, log_proxy_request_content, log_proxy_response_content,
    log_completion_message
)

# Import request/response models and helper functions from extracted module
from server.models import (
    ChatMessage, ChatCompletionRequest,
    extract_message_content_for_compression, reconstruct_multimodal_messages
)

# Import configuration from extracted module
from server.config import ServerConfig

# Import streaming handlers from extracted module
from server.streaming import (
    create_streaming_timeout, create_aiohttp_session,
    handle_streaming_response_with_metrics, context_aware_decompress
)

# Import request utility functions from extracted module
from server.request_utils import (
    filter_timeout_parameters, apply_interface_specific_fixes, 
    _is_conversation_continuation
)

# Import response processing utilities from extracted module  
from server.response_utils import (
    decompress_response, fix_payload_encoding, clean_compression_artifacts,
    clean_messages_array
)

# Import endpoint functions from extracted modules
from server.endpoints.chat import chat_completion
from server.endpoints.models import list_models, get_model, get_server_props, legacy_models
from server.endpoints.compression_stats import get_compression_stats, reset_compression_state
from server.endpoints.completions import text_completion, legacy_completion, legacy_completions
from server.endpoints.embeddings import create_embeddings, legacy_embedding, legacy_embeddings
from server.endpoints.messages import anthropic_messages, AnthropicMessagesRequest

# Import proxy handler from extracted module
from server.proxy import proxy_request













# Helper function to filter timeout parameters
# (create_streaming_timeout and create_aiohttp_session moved to server.streaming)
# (filter_timeout_parameters moved to server.request_utils)

# ServerConfig class has been moved to server/config.py



config = ServerConfig()

# Log startup messages that shell script would have printed (only once, at main server startup)
from server.logging_utils import log_startup_messages
log_startup_messages()

app = FastAPI(title="KrunchWrapper Compression Proxy")





# _is_conversation_continuation moved to server.request_utils

@app.post("/v1/chat/completions")
async def chat_completion_endpoint(request: ChatCompletionRequest, http_request: Request):
    """Handle OpenAI-compatible chat completions with compression."""
    return await chat_completion(request, http_request, config)

@app.post("/v1/messages")
async def anthropic_messages_endpoint(request: AnthropicMessagesRequest, http_request: Request):
    """Handle Anthropic Messages API requests with compression."""
    return await anthropic_messages(request, http_request, config)

@app.get("/v1/models")
async def list_models_endpoint(request: Request):
    """List available models by forwarding to target LLM API."""
    return await list_models(request, config)

@app.get("/v1/models/{model_id}")
async def get_model_endpoint(model_id: str, request: Request):
    """Return information about a specific model."""
    return await get_model(model_id, request, config)

@app.get("/props")
async def get_server_props_endpoint(request: Request):
    """Get server properties for webui compatibility (llama.cpp format)."""
    return await get_server_props(request, config)

@app.get("/v1/compression/stats")
async def get_compression_stats_endpoint():
    """Get compression and system performance statistics."""
    return await get_compression_stats(config)

@app.post("/v1/compression/reset")
async def reset_compression_state_endpoint():
    """Reset conversation compression state (for testing/debugging)."""
    return await reset_compression_state(config)

@app.post("/v1/completions")
async def text_completion_endpoint(request: Request):
    """Handle legacy completions endpoint by proxying to target or adapting to chat completions."""
    return await text_completion(request, config)

@app.post("/v1/embeddings")
async def create_embeddings_endpoint(request: Request):
    """Handle embeddings request by proxying to target API."""
    return await create_embeddings(request, config)

@app.post("/completion")
async def legacy_completion_endpoint(request: Request):
    """Handle legacy completion endpoint (without /v1/ prefix and singular form)."""
    return await legacy_completion(request, config)

@app.post("/completions")
async def legacy_completions_endpoint(request: Request):
    """Handle legacy completions endpoint (without /v1/ prefix)."""
    return await legacy_completions(request, config)

@app.post("/embedding")
async def legacy_embedding_endpoint(request: Request):
    """Handle legacy embedding endpoint (without /v1/ prefix and singular form)."""
    return await legacy_embedding(request, config)

@app.post("/embeddings")
async def legacy_embeddings_endpoint(request: Request):
    """Handle legacy embeddings endpoint (without /v1/ prefix)."""
    return await legacy_embeddings(request, config)

@app.get("/models")
async def legacy_models_endpoint(request: Request):
    """Handle legacy models endpoint (without /v1/ prefix)."""
    return await legacy_models(request, config)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_request_endpoint(request: Request, path: str):
    """Proxy request endpoint that delegates to the extracted proxy handler."""
    return await proxy_request(request, path, config)



def main():
    parser = argparse.ArgumentParser(description="KrunchWrapper Compression Proxy Server")
    parser.add_argument("--port", type=int, default=config.port, help="Port to run the server on")
    parser.add_argument("--host", type=str, default=config.host, help="Host to run the server on")
    parser.add_argument("--target-url", type=str, default=None, help="Target LLM API URL (overrides target-host and target-port)")
    parser.add_argument("--target-host", type=str, default=None, help="Target LLM API host")
    parser.add_argument("--target-port", type=int, default=None, help="Target LLM API port")
    parser.add_argument("--min-compression-ratio", type=float, default=config.min_compression_ratio, 
                        help="Minimum compression ratio required to add decoder")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.port = args.port
    config.host = args.host
    
    # Handle target URL construction
    if args.target_url:
        config.target_url = args.target_url
    elif args.target_host or args.target_port:
        target_host = args.target_host or config.target_url.split("://")[1].split(":")[0].split("/")[0]
        target_port = args.target_port or int(config.target_url.split("://")[1].split(":")[1].split("/")[0])
        config.target_url = f"http://{target_host}:{target_port}/v1"
    
    config.min_compression_ratio = args.min_compression_ratio
    
    # ──────────────────────────────────────────────────────────────────────────────
    log_config_message("──────────────────────────────────────────────────────────────────────────────")
    log_config_message("🚀 SERVER STARTUP")
    log_config_message("──────────────────────────────────────────────────────────────────────────────")
    log_config_message(f"🔌 Listen address:               http://{config.host}:{config.port}")
    log_config_message(f"🎯 Target API:                   {config.target_url}")
    log_config_message(f"📊 Min compression ratio:        {config.min_compression_ratio}")
    log_config_message(f"🧠 Compression engine:           Dynamic analysis")
    
    # Enable global async logging using configuration file
    try:
        async_handler = setup_global_async_logging()
        if async_handler:
            # Log detailed async logging setup information
            import logging
            root_logger = logging.getLogger()
            file_handler_count = sum(1 for h in root_logger.handlers if hasattr(h, 'baseFilename'))
            stream_handler_count = sum(1 for h in root_logger.handlers if hasattr(h, 'stream'))
            total_handlers = len(root_logger.handlers)
            
            # ──────────────────────────────────────────────────────────────────────────────
            log_config_message("──────────────────────────────────────────────────────────────────────────────")
            log_config_message("⚡ ASYNC LOGGING SETUP")
            log_config_message("──────────────────────────────────────────────────────────────────────────────")
            log_config_message(f"📝 Status:                       Enabled (High-Performance Mode)")
            log_config_message(f"📈 Total destinations:           {total_handlers}")
            
            # Log details about where logs are going
            file_destinations = []
            console_destinations = []
            for i, handler in enumerate(root_logger.handlers):
                if hasattr(handler, 'baseFilename'):
                    file_destinations.append(handler.baseFilename)
                    log_config_message(f"📄 File destination {i+1}:           {handler.baseFilename}")
                elif hasattr(handler, 'stream'):
                    stream_name = getattr(handler.stream, 'name', '<stderr>')
                    console_destinations.append(stream_name)
                    log_config_message(f"📺 Console destination {i+1}:        {stream_name}")
            
            # Log async configuration details
            async_config = getattr(async_handler, 'async_handler', None)
            if async_config:
                batch_size = getattr(async_config, 'batch_size', 'unknown')
                worker_timeout = getattr(async_config, 'worker_timeout', 'unknown')
                max_queue_size = getattr(async_config, 'log_queue', None)
                queue_info = "unlimited" if (max_queue_size and max_queue_size.maxsize == 0) else f"{getattr(max_queue_size, 'maxsize', 'unknown')}"
                log_config_message(f"📊 Queue size:                   {queue_info}")
                log_config_message(f"⚡ Batch size:                   {batch_size}")
                log_config_message(f"⏱️  Worker timeout:               {worker_timeout}s")
            
            # Test that file logging is still working after async setup
            import logging
            test_logger = logging.getLogger('log.test')
            test_logger.info("🧪 Testing file logging after async setup - this should appear in log file")
            log_config_message("✅ File logging test:            Completed")
        else:
            # ──────────────────────────────────────────────────────────────────────────────
            log_config_message("──────────────────────────────────────────────────────────────────────────────")
            log_config_message("⚡ ASYNC LOGGING SETUP")
            log_config_message("──────────────────────────────────────────────────────────────────────────────")
            log_config_message("📝 Status:                       Disabled")
    except Exception as e:
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("⚡ ASYNC LOGGING SETUP")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"⚠️  Status:                       Failed ({e})")
        log_config_message("📝 Fallback:                     Standard logging")
    
    # Initialize persistent token cache and clean up expired entries
    try:
        from core.persistent_token_cache import get_persistent_cache
        cache = get_persistent_cache()
        cache.clear_expired()
        stats = cache.get_stats()
        
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🗄️  TOKEN CACHE")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"📁 Cache directory:              {stats['temp_dir']}")
        log_config_message(f"📊 Disk files:                   {stats['disk_files']}")
        log_config_message(f"🗂️  RAM entries:                  {stats['ram_entries']}")
        if stats['disk_files'] > 0:
            log_config_message(f"🔄 Lazy loading:                Enabled")
    except Exception as e:
        # ──────────────────────────────────────────────────────────────────────────────
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message("🗄️  TOKEN CACHE")
        log_config_message("──────────────────────────────────────────────────────────────────────────────")
        log_config_message(f"⚠️  Status:                       Failed ({e})")
        log_config_message("📝 Fallback:                     Standard operation without caching")
    
    # Configure uvicorn to use our logging setup
    # Get the existing file handler that we set up
    root_logger = logging.getLogger()
    file_handler = None
    for handler in root_logger.handlers:
        if hasattr(handler, 'baseFilename'):
            file_handler = handler
            break
    
    # Configure uvicorn logging to use our handlers
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)-16s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
                 "loggers": {
             "uvicorn": {
                 "handlers": ["default"],
                 "level": "INFO",
                 "propagate": False,  # Prevent duplicate logging
             },
             "uvicorn.error": {
                 "handlers": ["default"], 
                 "level": "INFO",
                 "propagate": False,  # Prevent duplicate logging
             },
             "uvicorn.access": {
                 "handlers": ["default"],
                 "level": "INFO", 
                 "propagate": False,  # Prevent duplicate logging
             },
         },
    }
    
    # If we have a file handler, add file logging for uvicorn too
    if file_handler:
        uvicorn_log_config["handlers"]["file"] = {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": file_handler.baseFilename,
            "mode": "a",
        }
        # Add file handler to all uvicorn loggers
        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            uvicorn_log_config["loggers"][logger_name]["handlers"].append("file")
    
    # ──────────────────────────────────────────────────────────────────────────────
    log_config_message("──────────────────────────────────────────────────────────────────────────────")
    log_config_message("🌟 UVICORN SERVER")
    log_config_message("──────────────────────────────────────────────────────────────────────────────")
    if file_handler:
        log_config_message(f"🔧 Configuration:                File + Console logging")
    else:
        log_config_message("🔧 Configuration:                Console logging only")
    log_config_message("⚡ Status:                       Starting...")
    log_config_message("=" * 80)
    log_config_message("✅ KrunchWrapper initialization complete - server ready!")
    log_config_message("=" * 80)
    
    # Run uvicorn with our logging configuration
    uvicorn.run(
        app, 
        host=config.host, 
        port=config.port,
        log_config=uvicorn_log_config,
        access_log=True
    )


if __name__ == "__main__":
    main()
