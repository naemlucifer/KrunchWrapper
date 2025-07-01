"""Embeddings endpoint module."""
import time
import json
from fastapi import Request, HTTPException
import aiohttp

from core.model_context import set_global_model_context, extract_model_from_provider_format
from server.logging_utils import log_message, log_performance_metrics, log_completion_message


async def create_embeddings(request: Request, config):
    """Handle embeddings requests by proxying to target or returning a placeholder."""
    # Start timing
    start_time = time.time()
    
    body = await request.body()
    
    # Set model context for tokenizer validation
    try:
        data = json.loads(body)
        model_name = data.get("model")
        if model_name:
            clean_model = extract_model_from_provider_format(model_name)
            set_global_model_context(clean_model)
            log_message(f"🔧 Set model context: {clean_model} (from {model_name})", "DEBUG")
    except Exception as e:
        log_message(f"⚠️  Failed to parse embeddings request for model context: {e}", "DEBUG")
    
    # Extract API key from request headers
    auth_header = request.headers.get("authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
    
    # Basic metrics for embeddings
    compression_ratio = 0.0
    compression_tokens_saved = 0
    dynamic_chars_used = 0
    
    preprocessing_end_time = time.time()
    preprocessing_time = preprocessing_end_time - start_time

    try:
        # Try proxying to target first
        llm_start_time = time.time()
        target_url = f"{config.target_url}/embeddings"
        async with aiohttp.ClientSession() as session:
            # Use client's API key if provided, otherwise use configured API key
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            async with session.post(target_url, headers=headers, data=body) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    
                    # Calculate and log metrics
                    end_time = time.time()
                    total_time = end_time - start_time
                    llm_time = end_time - llm_start_time
                    
                    # Extract usage from response (embeddings usually don't have completion tokens)
                    usage = response_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens)
                    
                    # Calculate rates
                    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
                    prompt_tokens_per_sec = prompt_tokens / preprocessing_time if preprocessing_time > 0 else 0
                    
                    # Log performance metrics
                    log_performance_metrics(
                        "embeddings",
                        total_time,
                        preprocessing_time,
                        llm_time,
                        compression_ratio,
                        compression_tokens_saved,
                        dynamic_chars_used,
                        total_tokens,
                        prompt_tokens,
                        0,
                        None,
                        llm_start_time
                    )
                    
                    # Log completion message with total compression stats
                    log_completion_message(
                        "embeddings",
                        compression_tokens_saved,
                        dynamic_chars_used,
                        {},  # No rule_union for embeddings
                        0,   # No original content size available
                        "generic",
                        None,  # Comment stats not available for embeddings endpoint
                        None   # Timing breakdown not available for embeddings endpoint
                    )
                    
                    return response_data
    except Exception as e:
        print(f"Error proxying to embeddings endpoint: {e}")
    
    # If target doesn't support embeddings, return a placeholder
    # This is just to prevent errors in frontends that expect this endpoint
    try:
        data = json.loads(body)
        input_text = data.get("input", "")
        
        # Convert input to list if it's a string
        if isinstance(input_text, str):
            inputs = [input_text]
        else:
            inputs = input_text
            
        # Generate placeholder embeddings (all zeros)
        # Most embedding models use 1536 dimensions
        embedding_size = 1536
        
        # Calculate and log metrics for placeholder
        end_time = time.time()
        total_time = end_time - start_time
        
        log_performance_metrics(
            "placeholder",
            total_time,
            preprocessing_time,
            0,
            compression_ratio,
            compression_tokens_saved,
            dynamic_chars_used,
            0,
            0,
            0,
            None,
            None
        )
        
        # Log completion message with total compression stats
        log_completion_message(
            "embeddings (placeholder)",
            compression_tokens_saved,
            dynamic_chars_used,
            {},  # No rule_union for placeholder
            0,   # No original content size available
            "generic",
            None,  # Comment stats not available for placeholder embeddings
            None   # Timing breakdown not available for placeholder embeddings
        )
        
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.0] * embedding_size,
                    "index": i
                }
                for i, _ in enumerate(inputs)
            ],
            "model": data.get("model", "placeholder-embedding-model"),
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        print(f"Error creating placeholder embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process embeddings request: {str(e)}")


async def legacy_embedding(request: Request, config):
    """Handle legacy embedding endpoint (without /v1/ prefix and singular form)."""
    print("Redirecting /embedding request to /v1/embeddings")
    return await create_embeddings(request, config)


async def legacy_embeddings(request: Request, config):
    """Handle legacy embeddings endpoint (without /v1/ prefix)."""
    print("Redirecting /embeddings request to /v1/embeddings")
    return await create_embeddings(request, config) 