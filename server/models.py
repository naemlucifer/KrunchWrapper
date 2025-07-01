"""
Request/Response Models for KrunchWrapper Server

This module contains the Pydantic models and helper functions for handling
chat completion requests and multimodal message content processing.

Extracted from api/server.py to reduce file size and improve organization.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, validator

# Import compression utilities needed by helper functions
from core.compress import decompress_multimodal_aware


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], Dict[str, Any], Any]  # Accept multiple types for multimodal content
    name: Optional[str] = None
    _original_content: Optional[Any] = None  # Store original structure for multimodal reconstruction

    @validator('content')
    def content_not_none(cls, v, values):
        if v is None:
            return ""  # Convert None to empty string
        
        # Store the original content structure before processing
        values['_original_content'] = v
        
        # CRITICAL FIX: Handle complex content structures from webui, Cline, and other clients
        # This normalizes multimodal content arrays, dict structures, etc. to simple strings
        return extract_message_content_for_compression(v)
    
    @validator('role')
    def role_must_be_valid(cls, v):
        valid_roles = {'system', 'user', 'assistant', 'function'}
        if v not in valid_roles:
            # If role was compressed, try to map back common variations
            role_mappings = {
                'sys': 'system',
                'usr': 'user', 
                'user_message': 'user',
                'assistant_message': 'assistant',
                'ai': 'assistant',
                'human': 'user'
            }
            return role_mappings.get(v, 'user')  # Default to 'user' if unrecognized
        return v

    def get_original_content(self) -> Any:
        """Get the original multimodal content structure."""
        return self._original_content

    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content."""
        return (self._original_content is not None and 
                not isinstance(self._original_content, str))


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    filename: Optional[str] = None  # Optional language hint
    system: Optional[str] = None  # Claude-style system parameter
    system_instruction: Optional[str] = None  # Gemini-style system instruction
    
    # Additional sampling parameters that might be sent by clients
    # but should be filtered out if the target server doesn't support them
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    
    # Legacy parameters are handled in the double slash correction handler
    # They are converted to proper parameters before reaching this model
    
    class Config:
        extra = "allow"  # Allow additional fields to be passed through
        
    @validator('model', 'temperature', 'top_p', 'n', 'stream', 'max_tokens', 'presence_penalty', 'frequency_penalty', 'user', 'filename', 'system', 'system_instruction', pre=True)
    def handle_none_values(cls, v):
        """Convert None values to appropriate defaults to prevent validation errors."""
        if v is None:
            return None
        return v
    
    # Legacy parameter conversion is now handled in the request preprocessing
    # All legacy parameters are converted before reaching this Pydantic model


def extract_message_content_for_compression(content):
    """
    Extract text content from potentially complex message structures for compression.
    
    Supports various formats:
    - Simple strings: "Hello world"
    - Multimodal arrays: [{'type': 'text', 'text': 'Hello'}, {'type': 'attachment', 'name': 'file.txt'}]
    - Cline structures: [{'type': 'text', 'text': 'actual content'}] or {'text': 'content'}
    - WebUI attachments: [{'type': 'text', 'text': 'content'}, {'type': 'image_url', 'image_url': {...}}]
    
    This function normalizes these to simple strings for compression.
    """
    if isinstance(content, str):
        # Simple string content - return as-is
        return content
    elif isinstance(content, list):
        # List of content parts - extract text from each part
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                # Handle different content part formats
                part_type = part.get('type', '')
                
                if part_type == 'text':
                    # Standard text part: {"type": "text", "text": "content"}
                    text_parts.append(str(part.get('text', '')))
                elif part_type == 'attachment':
                    # Attachment part: {"type": "attachment", "name": "file.txt", "data": "content"}
                    # Include both filename and content for context
                    name = part.get('name', 'attachment')
                    data = part.get('data', '')
                    text_parts.append(f"[Attachment: {name}]\n{data}")
                elif part_type == 'image_url':
                    # Image part: {"type": "image_url", "image_url": {"url": "..."}}
                    # Just mention that there's an image, don't include the data
                    text_parts.append("[Image attachment]")
                elif part_type == 'image':
                    # Alternative image format
                    text_parts.append("[Image attachment]")
                else:
                    # Generic dict - look for text in common fields
                    text = (part.get('text') or 
                           part.get('content') or 
                           part.get('data') or 
                           str(part))
                    if text.strip():  # Only add non-empty text
                        text_parts.append(str(text))
            else:
                # Non-dict part - convert to string
                text_parts.append(str(part))
        
        return '\n'.join(filter(None, text_parts))  # Join with newlines and filter empty parts
    elif isinstance(content, dict):
        # Dictionary content - extract text from common fields
        text = (content.get('text') or 
               content.get('content') or 
               content.get('data') or 
               str(content))
        return str(text)
    else:
        # Other types - convert to string
        return str(content) if content is not None else ""


def reconstruct_multimodal_messages(messages: List[Dict[str, Any]], original_messages: List[ChatMessage], rule_union: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Reconstruct multimodal message structures after decompression.
    
    Args:
        messages: The messages with decompressed content
        original_messages: The original ChatMessage objects with preserved structure
        rule_union: The compression rules used
    
    Returns:
        Messages with reconstructed multimodal content
    """
    # Import logging here to avoid circular imports
    try:
        from server.logging_utils import log_message
    except ImportError:
        # Fallback logging function
        def log_message(msg, level="INFO", config=None):
            print(f"[{level}] {msg}")
    
    reconstructed_messages = []
    
    for i, (msg, original_msg) in enumerate(zip(messages, original_messages)):
        if hasattr(original_msg, 'is_multimodal') and original_msg.is_multimodal():
            # This message had multimodal content - reconstruct it
            compressed_content = msg.get("content", "")
            original_structure = original_msg.get_original_content()
            
            # Use multimodal-aware decompression
            reconstructed_content = decompress_multimodal_aware(
                compressed_content, 
                rule_union, 
                original_structure
            )
            
            # Create new message with reconstructed content
            reconstructed_msg = msg.copy()
            reconstructed_msg["content"] = reconstructed_content
            reconstructed_messages.append(reconstructed_msg)
            
            log_message(f"🔄 Reconstructed multimodal content for message {i}: {type(original_structure)} -> {type(reconstructed_content)}", "DEBUG", None)
        else:
            # Regular text message - keep as is
            reconstructed_messages.append(msg.copy())
    
    return reconstructed_messages 