"""Endpoint modules for the KrunchWrapper server."""

from .chat import chat_completion
from .models import list_models, get_model, get_server_props
from .compression_stats import get_compression_stats, reset_compression_state
from .completions import text_completion, legacy_completion, legacy_completions
from .embeddings import create_embeddings, legacy_embedding, legacy_embeddings
from .models import legacy_models

__all__ = [
    'chat_completion',
    'list_models', 'get_model', 'get_server_props',
    'get_compression_stats', 'reset_compression_state',
    'text_completion', 'legacy_completion', 'legacy_completions',
    'create_embeddings', 'legacy_embedding', 'legacy_embeddings',
    'legacy_models'
] 