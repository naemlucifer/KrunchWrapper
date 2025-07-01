"""Models endpoint module."""
import time
from fastapi import Request, HTTPException
import aiohttp

from server.streaming import create_aiohttp_session
from server.logging_utils import log_message


async def list_models(request: Request, config):
    """List available models by forwarding to target LLM API."""
    target_url = f"{config.target_url}/models"
    
    headers = {k: v for k, v in request.headers.items() 
              if k.lower() not in ["host", "content-length"]}
    
    # Extract API key from request headers if not present
    if "authorization" not in {k.lower() for k in headers} and config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    
    try:
        async with create_aiohttp_session() as session:
            async with session.get(target_url, headers=headers) as resp:
                response_data = await resp.json()
                return response_data
    except Exception as e:
        # Fallback response if target server doesn't support /models endpoint
        return {
            "object": "list",
            "data": [
                {
                    "id": "default-model",
                    "object": "model",
                    "owned_by": "krunchwrapper-proxy",
                    "created": 1677649963,  # Placeholder timestamp
                    "permission": [],
                    "root": "default-model",
                    "parent": None
                }
            ]
        }


async def get_model(model_id: str, request: Request, config):
    """Return information about a specific model."""
    # Extract API key from request headers
    auth_header = request.headers.get("authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
    
    target_url = f"{config.target_url}/models/{model_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Use client's API key if provided, otherwise use configured API key
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            async with session.get(target_url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        print(f"Error fetching model {model_id} from target API: {e}")
    
    # Default response if target doesn't support this endpoint
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "krunchwrapper"
    }


async def get_server_props(request: Request, config):
    """Get server properties for webui compatibility (llama.cpp format)."""
    try:
        # Extract API key from request headers
        auth_header = request.headers.get("authorization", "")
        api_key = ""
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        # Set up headers for target server requests
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        async with create_aiohttp_session() as session:
            # First try to forward the /props request directly (for llama.cpp servers)
            # Try both /props (root level) and /v1/props (OpenAI API style)
            for props_path in ["/props", "/v1/props"]:
                try:
                    # For root level endpoints like /props, use base URL without /v1
                    if props_path == "/props":
                        base_url = f"http://{config.target_host}:{config.target_port}"
                        target_url = f"{base_url}{props_path}"
                    else:
                        target_url = f"{config.target_url}{props_path}"
                    
                    async with session.get(target_url, headers=headers) as resp:
                        if resp.status == 200:
                            # Forward the target server's response directly
                            props = await resp.json()
                            log_message(f"📡 Forwarded server props directly from target server ({target_url})", "DEBUG")
                            return props
                        else:
                            log_message(f"⚠️ Target server {target_url} returned {resp.status}", "DEBUG")
                except Exception as e:
                    log_message(f"⚠️ Failed to get {props_path} from target server: {e}", "DEBUG")
            
            log_message(f"⚠️ Neither /props nor /v1/props worked, trying to build props from other endpoints", "DEBUG")
            
            # If /props doesn't work, try to gather information from other endpoints
            server_info = {}
            model_info = None
            
            # Try to get model information from /v1/models
            try:
                models_url = f"{config.target_url}/v1/models"
                async with session.get(models_url, headers=headers) as resp:
                    if resp.status == 200:
                        models_data = await resp.json()
                        if "data" in models_data and len(models_data["data"]) > 0:
                            # Use the first model as the primary model
                            model_info = models_data["data"][0]
                            server_info["model_path"] = model_info.get("id", "unknown-model")
                            log_message(f"📡 Got model info from /v1/models: {server_info['model_path']}", "DEBUG")
            except Exception as e:
                log_message(f"⚠️ Failed to get models from target server: {e}", "DEBUG")
            
            # Try to get server info from other common endpoints
            build_info = "Unknown Server"
            try:
                # Try some common server info endpoints
                for info_endpoint in ["/health", "/info", "/status", "/v1/info"]:
                    try:
                        info_url = f"{config.target_url}{info_endpoint}"
                        async with session.get(info_url, headers=headers, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.status == 200:
                                try:
                                    info_data = await resp.json()
                                    # Extract useful build/server information
                                    if "version" in info_data:
                                        build_info = f"Server Version: {info_data['version']}"
                                    elif "build" in info_data:
                                        build_info = str(info_data["build"])
                                    elif "server" in info_data:
                                        build_info = str(info_data["server"])
                                    log_message(f"📡 Got server info from {info_endpoint}: {build_info}", "DEBUG")
                                    break
                                except:
                                    # If JSON parsing fails, try to get text
                                    info_text = await resp.text()
                                    if info_text and len(info_text) < 200:
                                        build_info = f"Server Info: {info_text.strip()}"
                                        log_message(f"📡 Got server text info from {info_endpoint}", "DEBUG")
                                        break
                    except:
                        continue
            except Exception as e:
                log_message(f"⚠️ Could not get server build info: {e}", "DEBUG")
            
            # Build server properties from gathered information
            if server_info or model_info:
                # We got some real information from the target server
                props = {
                    "build_info": build_info,
                    "model_path": server_info.get("model_path", "unknown-model"),
                    "n_ctx": 32768,  # Conservative default, could try to detect this
                }
                
                # Try to detect context size from model name or other clues
                model_path = props["model_path"].lower()
                if "32k" in model_path or "32768" in model_path:
                    props["n_ctx"] = 32768
                elif "128k" in model_path or "131072" in model_path:
                    props["n_ctx"] = 131072
                elif "200k" in model_path:
                    props["n_ctx"] = 200000
                elif "1m" in model_path or "1000k" in model_path:
                    props["n_ctx"] = 1000000
                
                log_message(f"📡 Built server props from target server info: {props['model_path']}", "DEBUG")
                return props
        
        # Last resort fallback if we couldn't get any information
        log_message(f"⚠️ Could not get any information from target server, using minimal fallback", "WARNING")
        return {
            "build_info": f"Proxied Server (via KrunchWrapper) - Target: {config.target_host}:{config.target_port}",
            "model_path": "unknown-model",
            "n_ctx": 32768,
        }
        
    except Exception as e:
        log_message(f"❌ Error getting server props: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to get server props: {str(e)}")


async def legacy_models(request: Request, config):
    """Handle legacy models endpoint (without /v1/ prefix)."""
    print("Redirecting /models request to /v1/models")
    return await list_models(request, config) 