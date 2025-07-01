"""
Cline Server Handler Module

This module contains all Cline-specific logic for handling streaming responses
and client detection to fix the "Unexpected API Response" error.

Key Features:
- Cline request detection via user-agent and headers
- Streaming response validation and JSON structure preservation
- Client-specific compression disabling
- SSE format compliance for Cline compatibility
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
import aiohttp
from fastapi import Request
from fastapi.responses import StreamingResponse


class ClineServerHandler:
    """Handles Cline-specific server logic for streaming responses and client detection."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger('cline_handler')
        # Enable ultra-detailed logging for debugging (set to True when needed)
        self.ultra_debug = False
        
    def enable_ultra_debug(self, enabled: bool = True):
        """Enable/disable ultra-detailed logging for debugging streaming issues."""
        self.ultra_debug = enabled
        self.logger.info(f"🔧 [CLINE] Ultra-debug logging {'enabled' if enabled else 'disabled'}")
        
    def detect_cline_request(self, request: Request) -> bool:
        """
        Detect if a request is coming from Cline based on user-agent and headers.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            bool: True if request is from Cline, False otherwise
        """
        # CRITICAL FIX: Only skip detection if explicitly disabled, allow auto detection
        if self.config and hasattr(self.config, 'system_prompt') and isinstance(self.config.system_prompt, dict):
            use_cline = self.config.system_prompt.get('use_cline', False)
            if not use_cline:
                self.logger.debug("🔍 [CLINE] Cline mode disabled in config - not detecting as Cline")
                return False
        elif self.config and hasattr(self.config, 'interface_engine'):
            # Only skip detection if interface_engine is explicitly set to something other than "cline" or "auto"
            if self.config.interface_engine not in ["cline", "auto"]:
                self.logger.debug(f"🔍 [CLINE] Interface engine set to {self.config.interface_engine} (not cline/auto) - not detecting as Cline")
                return False
        
        # ENHANCED: Log all headers for debugging (temporarily more verbose)
        headers_dict = dict(request.headers)
        self.logger.info(f"🔍 [CLINE DEBUG] All request headers: {headers_dict}")
        
        # Check user-agent for Cline indicators
        user_agent = request.headers.get("user-agent", "").lower()
        self.logger.info(f"🔍 [CLINE DEBUG] User-Agent: '{user_agent}'")
        
        # Specific Cline detection patterns (more conservative)
        cline_patterns = [
            "cline",
            "claude-dev", 
            "claude_dev",
            "so/js"  # Real Cline user-agent pattern (so/js 4.83.0)
        ]
        
        self.logger.info(f"🔍 [CLINE DEBUG] Checking user-agent patterns: {cline_patterns}")
        for pattern in cline_patterns:
            if pattern in user_agent:
                self.logger.info(f"✅ [CLINE] DETECTED Cline request via user-agent pattern: '{pattern}' in '{user_agent}'")
                return True
            else:
                self.logger.debug(f"🔍 [CLINE DEBUG] Pattern '{pattern}' not found in user-agent")
                
        # Check for Cline-specific headers (case-insensitive)
        cline_headers = [
            "x-task-id",
            "x-cline-session", 
            "x-cline-version",
            "x-anthropic-version"
        ]
        
        self.logger.info(f"🔍 [CLINE DEBUG] Checking for Cline headers: {cline_headers}")
        available_headers = [h.lower() for h in request.headers.keys()]
        self.logger.info(f"🔍 [CLINE DEBUG] Available headers: {available_headers}")
        
        for header in cline_headers:
            if header in request.headers or header.lower() in available_headers:
                self.logger.info(f"✅ [CLINE] DETECTED Cline request via header: {header}")
                return True
            else:
                self.logger.debug(f"🔍 [CLINE DEBUG] Header '{header}' not found")
                
        # Check for VSCode/Cursor extension patterns (more specific)
        vscode_patterns = ["vscode-extension", "cursor-ai", "code-ai-assistant"]
        self.logger.info(f"🔍 [CLINE DEBUG] Checking VSCode patterns: {vscode_patterns}")
        for pattern in vscode_patterns:
            if pattern in user_agent:
                self.logger.info(f"✅ [CLINE] DETECTED VSCode/Cursor AI extension request via pattern: {pattern}")
                return True
            
        # REMOVED: Overly aggressive JSON detection
        # REMOVED: Fallback that treated everything as Cline
        
        # Default: Not a Cline request
        self.logger.info("❌ [CLINE] NOT detected as Cline request - no patterns matched")
        return False
        
    def should_disable_compression(self, request: Request) -> bool:
        """
        Determine if compression should be disabled for this client.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            bool: True if compression should be disabled
        """
        if not self.config:
            return False
            
        # Check if Cline compression is globally disabled
        if hasattr(self.config, 'disable_compression_for_cline') and self.config.disable_compression_for_cline:
            if self.detect_cline_request(request):
                self.logger.info("🚫 [CLINE] Compression disabled for Cline via global config")
                return True
                
        # Check disable_for_clients configuration
        if hasattr(self.config, 'disable_compression_clients'):
            user_agent = request.headers.get("user-agent", "").lower()
            for client in self.config.disable_compression_clients:
                if client.lower() in user_agent:
                    self.logger.info(f"🚫 [CLINE] Compression disabled for client: {client}")
                    return True
                    
        return False
        
    def validate_streaming_chunk(self, chunk_data: dict) -> bool:
        """
        Validate that a streaming chunk has the correct structure for Cline.
        
        Args:
            chunk_data: Parsed JSON data from streaming chunk
            
        Returns:
            bool: True if chunk is valid, False otherwise
        """
        try:
            # Ensure the data has the expected structure
            if "choices" in chunk_data and isinstance(chunk_data["choices"], list):
                for choice in chunk_data["choices"]:
                    if "delta" in choice:
                        # Ensure content is string or None, not corrupted
                        if "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content is not None and not isinstance(content, str):
                                self.logger.warning(f"⚠️ [CLINE] Invalid content type in delta: {type(content)}")
                                # Fix the content type
                                choice["delta"]["content"] = str(content)
                                
            return True
        except Exception as e:
            self.logger.error(f"⚠️ [CLINE] JSON validation error: {e}")
            return False
            
    def safe_decompress_streaming_chunk(self, original_content: str, rule_union: dict) -> str:
        """
        Safely decompress streaming chunk content only if compression symbols are present.
        
        Args:
            original_content: Original content from streaming chunk
            rule_union: Dictionary of compression rules
            
        Returns:
            str: Decompressed content or original if no symbols found
        """
        try:
            # Only decompress if compression symbols are actually present
            if rule_union and any(symbol in original_content for symbol in rule_union.keys()):
                from core.compress import decompress
                decompressed_content = decompress(original_content, rule_union)
                
                # Better decompression logging with more detail
                orig_preview = original_content[:200] + ("..." if len(original_content) > 200 else "")
                decomp_preview = decompressed_content[:200] + ("..." if len(decompressed_content) > 200 else "")
                self.logger.info(f"🔍 [CLINE DECOMPRESS] Found symbols, decompressing chunk")
                self.logger.info(f"🔍 [CLINE DECOMPRESS] Original:  '{orig_preview}'")
                self.logger.info(f"🔍 [CLINE DECOMPRESS] Decompressed: '{decomp_preview}'")
                return decompressed_content
            else:
                # No compression symbols present - pass through unchanged
                return original_content
                
        except Exception as e:
            self.logger.warning(f"⚠️ [CLINE] Decompression failed for chunk: {e}")
            # If decompression fails, return original content
            return original_content
            
    def process_streaming_chunk(self, line: str, rule_union: dict) -> str:
        """
        Process a single streaming chunk, handling decompression and validation.
        
        Args:
            line: Raw SSE line from the response
            rule_union: Dictionary of compression rules
            
        Returns:
            str: Processed line ready for output
        """
        try:
            # Handle empty lines (important for SSE)
            if not line.strip():
                return "\n"
                
            if line.startswith('data: '):
                data_str = line[6:].strip()
                
                # Handle [DONE] marker
                if data_str == "[DONE]":
                    self.logger.debug("🔍 [CLINE] Processing [DONE] marker")
                    return "data: [DONE]\n\n"
                
                # Handle empty data lines
                if not data_str:
                    return "data: \n\n"
                    
                # Parse JSON data
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as e:
                    preview = data_str[:300] + ("..." if len(data_str) > 300 else "")
                    self.logger.warning(f"⚠️ [CLINE] Invalid JSON in chunk: {preview}")
                    self.logger.warning(f"⚠️ [CLINE] JSON decode error: {e}")
                    # Return original line if JSON is invalid
                    return line + "\n\n"
                
                # Process choices with decompression
                for choice in data.get("choices", []):
                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                        original_content = choice["delta"]["content"]
                        
                        # Safe decompression
                        decompressed_content = self.safe_decompress_streaming_chunk(original_content, rule_union)
                        choice["delta"]["content"] = decompressed_content
                
                # CRITICAL: Ensure required fields are present for Cline compatibility
                if "id" not in data:
                    data["id"] = f"chatcmpl-{int(time.time())}"
                if "object" not in data:
                    data["object"] = "chat.completion.chunk"
                if "created" not in data:
                    data["created"] = int(time.time())
                        
                # Validate the structure
                if self.validate_streaming_chunk(data):
                    # Return properly formatted SSE line with double newline
                    formatted_line = f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                    return formatted_line
                else:
                    self.logger.warning("⚠️ [CLINE] Chunk validation failed, passing through original")
                    return line + "\n\n"
                    
            elif line.startswith('event: ') or line.startswith('id: ') or line.startswith('retry: '):
                # Pass through other SSE fields with proper formatting
                return line + "\n"
            else:
                # Pass through other lines with newline
                return line + "\n"
                
        except Exception as e:
            self.logger.error(f"⚠️ [CLINE] Unexpected error processing chunk: {e}")
            self.logger.error(f"⚠️ [CLINE] Problematic line: {line}")
            # Pass through original line with proper formatting
            return line + "\n\n"
    
    def _should_log_cline_stream(self) -> bool:
        """Check if Cline stream logging is enabled based on configuration."""
        return (hasattr(self.config, 'cline_stream_content_logging') and 
                getattr(self.config, 'cline_stream_content_logging', False))
    
    def _log_cline_stream_message(self, message: str, level: str = "DEBUG"):
        """Log a Cline stream message respecting the target configuration."""
        if not self._should_log_cline_stream():
            return
            
        target = getattr(self.config, 'cline_stream_logging_target', 'both')
        
        # Get the logger and log level
        log_func = getattr(self.logger, level.lower(), self.logger.debug)
        
        if target in ['both', 'terminal', 'file']:
            # Use standard logging which respects the logging configuration
            # The logging system will handle terminal vs file based on handlers
            log_func(message)
        # If target is 'disabled', we already returned early
            
    def _log_streaming_chunk(self, original_line: str, processed_line: str, rule_union: dict):
        """Enhanced logging for streaming chunks with better formatting and more detail."""
        try:
            # Extract content from SSE data lines for better readability
            if original_line.startswith('data: '):
                data_str = original_line[6:].strip()
                
                if data_str == "[DONE]":
                    self._log_cline_stream_message("🔍 [CLINE STREAM] Stream completed: [DONE]", "INFO")
                    return
                    
                if not data_str:
                    return  # Skip empty data lines
                
                try:
                    # Parse and extract meaningful content
                    data = json.loads(data_str)
                    
                    # Extract the actual text content being streamed
                    content_parts = []
                    for choice in data.get("choices", []):
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                content_parts.append(content)
                    
                    if content_parts:
                        combined_content = "".join(content_parts)
                        
                        # Show more content (up to 500 chars) and indicate if compressed
                        content_preview = combined_content[:500]
                        has_compression_symbols = rule_union and any(symbol in combined_content for symbol in rule_union.keys())
                        compression_indicator = " [COMPRESSED]" if has_compression_symbols else ""
                        truncated_indicator = "..." if len(combined_content) > 500 else ""
                        
                        # Log streaming content using the new targeted logging
                        self._log_cline_stream_message(f"🔍 [CLINE STREAM] Content{compression_indicator}: '{content_preview}{truncated_indicator}'")
                        
                        # Ultra-detailed logging for deep debugging
                        if self.ultra_debug:
                            self._log_cline_stream_message(f"🔍 [CLINE ULTRA-DEBUG] Full content: '{combined_content}'", "INFO")
                            self._log_cline_stream_message(f"🔍 [CLINE ULTRA-DEBUG] Original line: {original_line}", "INFO")
                            self._log_cline_stream_message(f"🔍 [CLINE ULTRA-DEBUG] Processed line: {processed_line}", "INFO")
                        
                        # If content was decompressed, show what happened
                        if has_compression_symbols:
                            symbols_found = [symbol for symbol in rule_union.keys() if symbol in combined_content]
                            self._log_cline_stream_message(f"🔍 [CLINE STREAM] Compression symbols found: {symbols_found}")
                    
                    # Log other interesting fields
                    if data.get("finish_reason"):
                        self._log_cline_stream_message(f"🔍 [CLINE STREAM] Finish reason: {data['finish_reason']}", "INFO")
                        
                except json.JSONDecodeError:
                    # If not valid JSON, just show raw content with more characters
                    preview = original_line[:500]
                    truncated = "..." if len(original_line) > 500 else ""
                    self._log_cline_stream_message(f"🔍 [CLINE STREAM] Raw data: {preview}{truncated}")
            else:
                # Non-data SSE lines (event, id, retry, etc.)
                self._log_cline_stream_message(f"🔍 [CLINE STREAM] SSE field: {original_line}")
                
        except Exception as e:
            # Fallback to simple logging if parsing fails
            preview = original_line[:200]
            truncated = "..." if len(original_line) > 200 else ""
            self.logger.warning(f"⚠️ [CLINE STREAM] Logging error ({e}): {preview}{truncated}")
            
    async def create_cline_compatible_stream(
        self, 
        target_url: str, 
        payload: dict, 
        rule_union: dict,
        request: Request
    ) -> StreamingResponse:
        """
        Create a Cline-compatible streaming response.
        
        Args:
            target_url: URL to forward the request to
            payload: Request payload
            rule_union: Compression rules dictionary
            request: Original request
            
        Returns:
            StreamingResponse: Cline-compatible streaming response
        """
        async def cline_stream_generator():
            try:
                # Set up proper headers for streaming
                headers = {k: v for k, v in request.headers.items() 
                          if k.lower() not in ["host", "content-length"]}
                headers["Content-Type"] = "application/json; charset=utf-8"
                
                # Add API key if configured
                if hasattr(self.config, 'api_key') and self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                    
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(target_url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            self.logger.error(f"🚨 [CLINE] Target API returned {resp.status}: {error_text}")
                            
                            # Yield error response in proper SSE format
                            error_response = {
                                "error": {
                                    "message": f"Target API error: {error_text}",
                                    "type": "api_error",
                                    "code": resp.status
                                }
                            }
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield f"data: [DONE]\n\n"
                            return
                            
                        # Ensure we're handling the response as a stream
                        if not resp.headers.get("content-type", "").startswith("text/event-stream"):
                            self.logger.warning("⚠️ [CLINE] Response is not SSE format, but treating as stream")
                            
                        # Process the SSE stream line by line
                        async for line in resp.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                processed_line = self.process_streaming_chunk(line, rule_union)
                                
                                # Enhanced logging for better debugging (only if enabled)
                                if self._should_log_cline_stream():
                                    self._log_streaming_chunk(line, processed_line, rule_union)
                                
                                yield processed_line
                                
            except aiohttp.ClientError as e:
                self.logger.error(f"🚨 [CLINE] Network error in streaming: {e}")
                error_response = {
                    "error": {
                        "message": f"Network error: {str(e)}",
                        "type": "connection_error",
                        "code": 500
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                self.logger.error(f"🚨 [CLINE] Unexpected streaming error: {e}")
                error_response = {
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "unknown_error",
                        "code": 500
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
                yield f"data: [DONE]\n\n"
                
        return StreamingResponse(
            cline_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*", 
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Expose-Headers": "*"
            }
        )
        
    def log_cline_request_details(self, request: Request, request_data: dict = None):
        """
        Log detailed information about Cline requests for debugging.
        
        Args:
            request: FastAPI Request object
            request_data: Parsed request data
        """
        if not self.detect_cline_request(request):
            return
            
        self.logger.info("🔍 [CLINE] Detected Cline request")
        self.logger.debug(f"🔍 [CLINE] Headers: {dict(request.headers)}")
        
        if request_data:
            self.logger.debug(f"🔍 [CLINE] Model: {request_data.get('model', 'N/A')}")
            self.logger.debug(f"🔍 [CLINE] Stream: {request_data.get('stream', False)}")
            self.logger.debug(f"🔍 [CLINE] Messages count: {len(request_data.get('messages', []))}")


# Global instance for use across the application
_cline_handler = None

def get_cline_handler(config=None):
    """Get or create the global Cline handler instance."""
    global _cline_handler
    if _cline_handler is None:
        _cline_handler = ClineServerHandler(config)
    return _cline_handler

def is_cline_request(request: Request) -> bool:
    """Convenience function to check if a request is from Cline."""
    handler = get_cline_handler()
    return handler.detect_cline_request(request)

def should_disable_compression_for_cline(request: Request, config=None) -> bool:
    """Convenience function to check if compression should be disabled for Cline."""
    handler = get_cline_handler(config)
    return handler.should_disable_compression(request) 