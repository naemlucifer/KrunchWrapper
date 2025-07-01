"""
Conversation-Aware Compression for KrunchWrapper
Provides compression functions that maintain state and consistency across conversation turns.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

from .conversation_state import get_conversation_manager, ConversationState
from .compress import compress_with_dynamic_analysis, Packed

logger = logging.getLogger(__name__)

@dataclass
class ConversationCompressionResult:
    """Result of conversation-aware compression."""
    compressed_messages: List[Dict[str, Any]]
    compression_rules: Dict[str, str]  # symbol -> original_text (all symbols available for decompression)
    conversation_id: str
    turn_number: int
    compression_ratio: float
    tokens_saved: int
    system_prompt_overhead: int
    net_efficiency: float
    should_continue: bool
    metrics: Dict[str, Any]
    kv_cache_optimization_used: bool = False  # Flag to indicate KV cache optimization was used
    new_symbols_for_prompt: Dict[str, str] = None  # symbol -> original_text (only new symbols for stateful mode)
    stateful_mode: bool = False  # Flag to indicate if stateful mode was used

def compress_conversation_aware(messages: List[Dict[str, Any]], 
                              min_characters: int = 200,
                              force_new_analysis: bool = False,
                              session_id: str = None,
                              kv_cache_threshold: int = 20,
                              exclude_symbols: set = None,
                              stateful_mode: bool = False) -> ConversationCompressionResult:
    """
    Perform conversation-aware compression that maintains consistency across turns.
    
    Args:
        messages: List of chat messages for this request
        min_characters: Minimum characters needed to trigger compression
        force_new_analysis: Force new compression analysis (ignore existing state)
        session_id: Optional session ID for conversation isolation
        kv_cache_threshold: Threshold in characters below which KV cache optimization is used
        exclude_symbols: Set of symbols to exclude from compression
        stateful_mode: When True, optimize for persistent KV cache servers by only including new symbols in system prompt
        
    Returns:
        ConversationCompressionResult with compressed messages and state info
    """
    # ENHANCED ASYNC LOGGING: Track conversation compression entry
    conversation_entry_lines = [
        f"🗣️  [CONVERSATION COMPRESS] Starting conversation-aware compression",
        f"🗣️  [CONVERSATION COMPRESS] Parameters:",
        f"    - Messages count: {len(messages)}",
        f"    - Min characters: {min_characters}",
        f"    - Force new analysis: {force_new_analysis}",
        f"    - Session ID: {session_id}",
        f"    - KV cache threshold: {kv_cache_threshold}"
    ]
    
    # Use async logging for the complete entry message
    complete_entry_message = "\n".join(conversation_entry_lines)
    logger.info(complete_entry_message)
    
    # Also print to console for immediate visibility
    print(complete_entry_message)
    
    manager = get_conversation_manager()
    
    # CRITICAL DEBUG: Log what we're checking for
    conversation_id = manager._generate_conversation_id(messages, session_id)
    existing_state = conversation_id in manager.states
    
    # Build KV debug message for async logging
    kv_debug_lines = [
        f"🔍 [KV DEBUG] Checking conversation: {conversation_id}",
        f"🔍 [KV DEBUG] Existing state found: {existing_state}",
        f"🔍 [KV DEBUG] Total conversations in memory: {len(manager.states)}"
    ]
    
    if existing_state:
        kv_debug_lines.append(f"🔍 [KV DEBUG] Existing conversation turn: {manager.states[conversation_id].turn_number}")
    
    # Use async logging for KV debug info
    complete_kv_debug = "\n".join(kv_debug_lines)
    logger.info(complete_kv_debug)
    
    # Also print to console for immediate visibility
    print(complete_kv_debug)
    
    # Get or create conversation state and start timing for this turn
    conversation_state = manager.get_or_create_conversation_state(messages, session_id)
    conversation_state.start_turn()  # ENHANCED: Start timing tracking
    
    # Calculate original content metrics for cumulative tracking
    original_content_size = 0
    for msg in messages:
        if msg.get("role") in {"user", "assistant", "system"}:
            try:
                from api.server import extract_message_content_for_compression
                content = extract_message_content_for_compression(msg.get("content", ""))
                original_content_size += len(content)
            except ImportError:
                content = str(msg.get("content", ""))
                original_content_size += len(content)
    
    # FORCE KV CACHE CHECK FIRST - before any state updates
    if len(messages) > 1:  # Multi-message = potential continuation
        # Find last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
        
        if last_user_msg:
            # CRITICAL FIX: Extract actual text content from structured messages
            try:
                from api.server import extract_message_content_for_compression
                extracted_content = extract_message_content_for_compression(last_user_msg.get("content", ""))
                msg_size = len(extracted_content)
            except ImportError:
                # Fallback to raw content if extraction fails
                msg_size = len(str(last_user_msg.get("content", "")))
            
            # CRITICAL FIX: Also calculate total compressible content size
            total_compressible_size = 0
            for msg in messages:
                if msg.get("role") in {"user", "assistant", "system"}:
                    try:
                        from api.server import extract_message_content_for_compression
                        text_content = extract_message_content_for_compression(msg.get("content", ""))
                        total_compressible_size += len(text_content)
                    except ImportError:
                        total_compressible_size += len(str(msg.get("content", "")))
            
            # Build KV cache analysis message for async logging
            kv_analysis_lines = [
                f"🔍 [KV DEBUG] Last user message size: {msg_size} chars (extracted)",
                f"🔍 [KV DEBUG] Total compressible content: {total_compressible_size:,} chars",
                f"🔍 [KV DEBUG] KV cache threshold: {kv_cache_threshold} chars"
            ]
            
            # IMPROVED KV CACHE LOGIC: Only trigger if:
            # 1. Last message is small AND
            # 2. Total content is not massive (avoid wasting compression opportunities)
            should_use_kv_cache = (
                kv_cache_threshold > 0 and 
                msg_size < kv_cache_threshold and
                total_compressible_size < 20000  # Don't waste compression on huge requests
            )
            
            if should_use_kv_cache:
                kv_analysis_lines.append(f"🚀 [KV CACHE] FORCING KV cache optimization for {msg_size} char message")
            elif msg_size < kv_cache_threshold:
                kv_analysis_lines.append(f"⚠️ [KV CACHE] Short message ({msg_size} chars) but total content too large ({total_compressible_size:,} chars) - using full compression")
            
            # Use async logging for KV cache analysis
            complete_kv_analysis = "\n".join(kv_analysis_lines)
            logger.info(complete_kv_analysis)
            print(complete_kv_analysis)
            
            if should_use_kv_cache:
                
                # CRITICAL FIX: Apply existing rules AND comment stripping for KV cache optimization
                compressed_messages = []
                total_comment_chars_saved = 0
                total_comment_tokens_saved = 0
                comment_language = "unknown"
                all_used_symbols = set()  # Track symbols used across all messages
                
                for msg in messages:
                    if msg.get("role") in {"user", "assistant", "system"}:  # CRITICAL FIX: Include system messages!
                        original_content = msg.get("content", "")
                        if original_content:
                            # STEP 1: Extract text content properly from complex message structures (like multimodal)
                            try:
                                from api.server import extract_message_content_for_compression
                                text_content = extract_message_content_for_compression(original_content)
                            except ImportError:
                                # Fallback if import fails
                                text_content = str(original_content) if original_content is not None else ""
                            
                            # STEP 2: Apply comment stripping to the extracted text content
                            from core.comment_stripper import CommentStripper
                            comment_stripper = CommentStripper()
                            stripped_content, comment_stats = comment_stripper.strip_comments(text_content)
                            
                            # Collect comment stripping statistics
                            if comment_stats.get("enabled", False):
                                total_comment_chars_saved += comment_stats.get("chars_saved", 0)
                                total_comment_tokens_saved += comment_stats.get("tokens_saved", 0)
                                if comment_stats.get("language") != "unknown":
                                    comment_language = comment_stats.get("language", "unknown")
                            
                            # STEP 3: Apply existing compression rules to stripped content
                            if conversation_id in manager.states:
                                existing_rules = manager.states[conversation_id].compression_rules
                                compressed_content, used_symbols = _apply_existing_compression(stripped_content, existing_rules)
                                all_used_symbols.update(used_symbols)
                            else:
                                compressed_content, used_symbols = stripped_content, set()
                            
                            msg_copy = msg.copy()
                            msg_copy["content"] = compressed_content
                            compressed_messages.append(msg_copy)
                        else:
                            compressed_messages.append(msg.copy())
                    else:
                        compressed_messages.append(msg.copy())
                
                # Log comment stripping results for KV cache optimization
                if total_comment_chars_saved > 0:
                    logger.info(f"📝 [KV CACHE] Comment stripping saved {total_comment_chars_saved:,} chars, {total_comment_tokens_saved} tokens ({comment_language})")
                    logger.info(f"🚀 [KV CACHE] Applied comment stripping + {len(all_used_symbols)} existing compression symbols (no new analysis needed)")
                else:
                    logger.info(f"🚀 [KV CACHE] Applied {len(all_used_symbols)} existing compression symbols only (no comments found)")
                
                                    # ENHANCED: Update metrics for KV cache optimization
                    # CRITICAL FIX: Calculate real overhead and efficiency for stateful mode (KV cache subsequent turn)
                    used_compression_rules = {symbol: conversation_state.compression_rules[symbol] for symbol in all_used_symbols if symbol in conversation_state.compression_rules}
                    
                    if stateful_mode:
                        # In stateful mode, calculate total compression benefit but 0 new symbols overhead
                        if len(all_used_symbols) > 0:
                            # Calculate total compression happening (original vs compressed)
                            compressed_content_size = sum(len(msg.get("content", "")) for msg in compressed_messages if msg.get("role") in {"user", "assistant", "system"})
                            total_compression_ratio = (original_content_size - compressed_content_size) / original_content_size if original_content_size > 0 else 0
                        else:
                            total_compression_ratio = 0.0
                        
                        # No new symbols overhead in KV cache mode
                        real_system_prompt_overhead = 0
                        real_overhead_ratio = 0.0
                        real_net_efficiency = total_compression_ratio  # Full compression benefit, no new overhead
                    else:
                        # Original stateless logic
                        real_system_prompt_overhead = _estimate_system_prompt_overhead(used_compression_rules)
                        real_overhead_ratio = real_system_prompt_overhead / (original_content_size // 4) if original_content_size > 0 else 0
                        real_net_efficiency = 0.0 - real_overhead_ratio  # No new compression, but account for overhead
                
                kv_metrics = {
                    'compression_ratio': 0.0,
                    'overhead_ratio': real_overhead_ratio,  # FIXED: Use real overhead ratio
                    'chars_saved': 0,
                    'tokens_saved': 0,
                    'system_prompt_tokens': real_system_prompt_overhead,  # FIXED: Use real overhead
                    'original_chars': original_content_size,
                    'compressed_chars': original_content_size,  # No compression applied  
                    'messages_count': len(messages),
                    'kv_cache_used': True,
                    'symbols_used': all_used_symbols  # Track which symbols were actually used
                }
                # Update state with KV cache metrics
                manager.update_conversation_compression(conversation_id, {}, kv_metrics)
                
                # CRITICAL FIX: Always ensure we have conversation state before returning result
                # This prevents empty compression_rules when conversation state doesn't exist yet
                if conversation_id not in manager.states:
                    # Create initial conversation state so we can store/retrieve rules properly
                    conversation_state = manager.get_or_create_conversation_state(messages, session_id)
                    logger.debug(f"Created initial conversation state for KV cache optimization: {conversation_id}")
                
                # Now we're guaranteed to have a conversation state with proper rules
                current_compression_rules = manager.states[conversation_id].compression_rules
                # Only return the symbols that were actually used
                used_compression_rules = {symbol: current_compression_rules[symbol] for symbol in all_used_symbols if symbol in current_compression_rules}
                logger.debug(f"KV cache returning {len(used_compression_rules)} used compression rules (from {len(current_compression_rules)} total) for decompression")
                
                return ConversationCompressionResult(
                    compressed_messages=compressed_messages,
                    compression_rules=used_compression_rules,  # Only used symbols
                    conversation_id=conversation_id,
                    turn_number=manager.states[conversation_id].turn_number,
                    compression_ratio=0.0,
                    tokens_saved=0,
                    system_prompt_overhead=real_system_prompt_overhead,  # FIXED: Use real overhead
                    net_efficiency=real_net_efficiency,  # FIXED: Use real net efficiency
                    should_continue=True,
                    metrics=manager.get_conversation_metrics(conversation_id),
                    kv_cache_optimization_used=True,  # CRITICAL: Flag this was used
                    new_symbols_for_prompt=None if not stateful_mode else {},  # No new symbols in KV cache mode
                    stateful_mode=stateful_mode
                )
    
    # Generate content hash and conversation ID to check for existing state
    current_content_hash = manager._generate_content_hash(messages)
    
    # Check if conversation already exists and get previous hash
    previous_content_hash = ""
    if conversation_id in manager.states:
        previous_content_hash = manager.states[conversation_id].content_hash
    
    is_new_turn = current_content_hash != previous_content_hash
    
    # Get or create conversation state (this will update the hash)
    conversation_state = manager.get_or_create_conversation_state(messages, session_id)
    
    logger.debug(f"Processing conversation {conversation_id}, turn {conversation_state.turn_number + 1}, new turn: {is_new_turn}")
    
    # Check if we should continue using compression for this conversation
    # However, we still allow KV cache optimization even if full compression analysis is disabled
    # Get the configured minimum net efficiency from server config
    try:
        from api.server import config
        min_net_efficiency = config.conversation_min_net_efficiency
    except (ImportError, AttributeError):
        min_net_efficiency = 0.01  # Fallback default
    
    compression_disabled = not force_new_analysis and not manager.should_use_conversation_compression(conversation_id, min_net_efficiency)
    if compression_disabled:
        logger.info(f"Full compression analysis disabled for conversation {conversation_id} due to poor efficiency trend, but KV cache optimization still available")
    
    if conversation_state.turn_number == 0:
        # First turn: check total content size and compression eligibility
        # CRITICAL FIX: Handle complex content structures from Cline
        try:
            from api.server import extract_message_content_for_compression
            total_content_size = sum(len(extract_message_content_for_compression(msg.get("content", ""))) for msg in messages 
                                   if msg.get("role") in {"user", "assistant", "system"})
        except ImportError:
            # Fallback if import fails (shouldn't happen in normal operation)
            total_content_size = sum(len(str(msg.get("content", ""))) for msg in messages 
                                   if msg.get("role") in {"user", "assistant", "system"})
        
        if total_content_size < min_characters or compression_disabled:
            logger.debug(f"First turn: content size {total_content_size} below threshold {min_characters} or compression disabled, skipping compression")
            return _create_uncompressed_result(messages, conversation_state, stateful_mode)
    else:
        # Subsequent turns: check for KV cache optimization first (regardless of compression status)
        if not is_new_turn:
            # No new content - this shouldn't happen in normal flow, but handle gracefully
            logger.debug(f"No new content detected for conversation {conversation_id}, applying existing compression only")
        else:
            # Check if the most recent user message qualifies for KV cache optimization
            # Find the last user message (most likely to be the new short response)
            last_user_message = None
            last_user_index = -1
            
            # Search backwards from the end to find the most recent user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_message = messages[i]
                    last_user_index = i
                    break
            
            # Check if the most recent user message is short enough for KV cache optimization
            if last_user_message:
                # CRITICAL FIX: Extract actual text content from structured messages
                try:
                    from api.server import extract_message_content_for_compression
                    extracted_content = extract_message_content_for_compression(last_user_message.get("content", ""))
                    last_user_size = len(extracted_content)
                except ImportError:
                    # Fallback to raw content if extraction fails
                    last_user_size = len(str(last_user_message.get("content", "")))
                
                # CRITICAL FIX: Also calculate total compressible content size for subsequent turns
                total_compressible_size = 0
                for msg in messages:
                    if msg.get("role") in {"user", "assistant", "system"}:
                        try:
                            from api.server import extract_message_content_for_compression
                            text_content = extract_message_content_for_compression(msg.get("content", ""))
                            total_compressible_size += len(text_content)
                        except ImportError:
                            total_compressible_size += len(str(msg.get("content", "")))
                
                # IMPROVED KV CACHE LOGIC: Only trigger if both conditions met
                should_use_kv_cache = (
                    kv_cache_threshold > 0 and 
                    last_user_size < kv_cache_threshold and
                    total_compressible_size < 20000  # Don't waste compression on huge requests
                )
                
                # For very short user responses (like "cool", "ok", "thanks"), use KV cache optimization
                if should_use_kv_cache:
                    logger.info(f"Most recent user message short ({last_user_size} chars < {kv_cache_threshold}) and total content manageable ({total_compressible_size:,} chars), using KV cache optimization")
                    logger.info(f"  Message {last_user_index}: '{last_user_message.get('content', '')[:50]}...' ({last_user_size} chars extracted)")
                elif last_user_size < kv_cache_threshold:
                    logger.info(f"Most recent user message short ({last_user_size} chars < {kv_cache_threshold}) but total content too large ({total_compressible_size:,} chars), using full compression instead")
                    # Continue to full compression analysis
                else:
                    logger.debug(f"Most recent user message not short enough ({last_user_size} chars >= {kv_cache_threshold}), using full compression")
                    # Continue to full compression analysis
                
                if should_use_kv_cache:
                    
                    # CRITICAL FIX: Apply existing compression rules AND comment stripping but don't do new analysis  
                    compressed_messages = []
                    total_comment_chars_saved = 0
                    total_comment_tokens_saved = 0
                    comment_language = "unknown"
                    all_used_symbols = set()  # Track symbols used across all messages
                    
                    for msg in messages:
                        if msg.get("role") in {"user", "assistant", "system"}:  # CRITICAL FIX: Include system messages!
                            original_content = msg.get("content", "")
                            if original_content:
                                # STEP 1: Extract text content properly from complex message structures (like multimodal)
                                try:
                                    from api.server import extract_message_content_for_compression
                                    text_content = extract_message_content_for_compression(original_content)
                                except ImportError:
                                    # Fallback if import fails
                                    text_content = str(original_content) if original_content is not None else ""
                                
                                # STEP 2: Apply comment stripping to the extracted text content
                                from core.comment_stripper import CommentStripper
                                comment_stripper = CommentStripper()
                                stripped_content, comment_stats = comment_stripper.strip_comments(text_content)
                                
                                # Collect comment stripping statistics
                                if comment_stats.get("enabled", False):
                                    total_comment_chars_saved += comment_stats.get("chars_saved", 0)
                                    total_comment_tokens_saved += comment_stats.get("tokens_saved", 0)
                                    if comment_stats.get("language") != "unknown":
                                        comment_language = comment_stats.get("language", "unknown")
                                
                                # STEP 3: Apply existing compression rules to stripped content
                                compressed_content, used_symbols = _apply_existing_compression(stripped_content, conversation_state.compression_rules)
                                all_used_symbols.update(used_symbols)
                                compressed_msg = msg.copy()
                                compressed_msg["content"] = compressed_content
                                compressed_messages.append(compressed_msg)
                            else:
                                compressed_messages.append(msg.copy())
                        else:
                            compressed_messages.append(msg.copy())
                    
                    # Log comment stripping results for KV cache optimization
                    if total_comment_chars_saved > 0:
                        logger.info(f"📝 [KV CACHE TURN] Comment stripping saved {total_comment_chars_saved:,} chars, {total_comment_tokens_saved} tokens ({comment_language})")
                        logger.info(f"🚀 [KV CACHE TURN] Applied comment stripping + {len(all_used_symbols)} existing compression symbols (no new analysis)")
                    else:
                        logger.info(f"🚀 [KV CACHE TURN] Applied {len(all_used_symbols)} existing compression symbols only (no comments found)")
                    
                    # ENHANCED: Update metrics for KV cache optimization
                    # CRITICAL FIX: Calculate real overhead and efficiency for stateful mode
                    used_compression_rules = {symbol: conversation_state.compression_rules[symbol] for symbol in all_used_symbols if symbol in conversation_state.compression_rules}
                    
                    if stateful_mode:
                        # In stateful mode, calculate total compression benefit but 0 new symbols overhead
                        if len(all_used_symbols) > 0:
                            # Calculate total compression happening (original vs compressed)
                            total_compression_ratio = (original_content_size - original_content_size) / original_content_size if original_content_size > 0 else 0
                            # Actually recalculate based on compressed content
                            compressed_content_size = sum(len(msg.get("content", "")) for msg in compressed_messages if msg.get("role") in {"user", "assistant", "system"})
                            total_compression_ratio = (original_content_size - compressed_content_size) / original_content_size if original_content_size > 0 else 0
                        else:
                            total_compression_ratio = 0.0
                        
                        # No new symbols overhead in KV cache mode
                        real_system_prompt_overhead = 0
                        real_overhead_ratio = 0.0
                        real_net_efficiency = total_compression_ratio  # Full compression benefit, no new overhead
                    else:
                        # Original stateless logic
                        real_system_prompt_overhead = _estimate_system_prompt_overhead(used_compression_rules)
                        real_overhead_ratio = real_system_prompt_overhead / (original_content_size // 4) if original_content_size > 0 else 0
                        real_net_efficiency = 0.0 - real_overhead_ratio  # No new compression, but account for overhead
                    
                    kv_metrics = {
                        'compression_ratio': 0.0,
                        'overhead_ratio': real_overhead_ratio,  # FIXED: Use real overhead ratio
                        'chars_saved': 0,
                        'tokens_saved': 0,
                        'system_prompt_tokens': real_system_prompt_overhead,  # FIXED: Use real overhead
                        'original_chars': original_content_size,
                        'compressed_chars': original_content_size,  # No compression applied  
                        'messages_count': len(messages),
                        'kv_cache_used': True,
                        'symbols_used': all_used_symbols  # Track which symbols were actually used
                    }
                    # Update state with KV cache metrics
                    manager.update_conversation_compression(conversation_id, {}, kv_metrics)
                    
                    # Return result with existing rules but no new compression analysis
                    # Only return the symbols that were actually used
                    logger.debug(f"KV cache (subsequent turn) returning {len(used_compression_rules)} used compression rules (from {len(conversation_state.compression_rules)} total) for decompression")
                    
                    return ConversationCompressionResult(
                        compressed_messages=compressed_messages,
                        compression_rules=used_compression_rules,  # Only used symbols
                        conversation_id=conversation_id,
                        turn_number=conversation_state.turn_number,
                        compression_ratio=0.0,  # No new compression this turn
                        tokens_saved=0,  # No new tokens saved this turn
                        system_prompt_overhead=real_system_prompt_overhead,  # FIXED: Use real overhead
                        net_efficiency=real_net_efficiency,  # FIXED: Use real net efficiency
                        should_continue=True,  # Keep using existing compression
                        metrics=manager.get_conversation_metrics(conversation_id) or {},
                        kv_cache_optimization_used=True,  # KV cache optimization was used
                        new_symbols_for_prompt=None if not stateful_mode else {},  # No new symbols in KV cache mode
                        stateful_mode=stateful_mode
                    )
        
        # If we get here, KV cache optimization doesn't apply, so check if compression is disabled
        if compression_disabled:
            logger.info(f"Compression disabled for conversation {conversation_id} and KV cache optimization not applicable")
            return _create_uncompressed_result(messages, conversation_state, stateful_mode)
    
    # Proceed with full compression analysis
    # Perform compression on user/assistant messages
    compressed_messages = []
    new_compression_rules = {}
    total_original_size = 0
    total_compressed_size = 0
    all_used_symbols = set()  # Track all symbols used across all messages
    new_content_size = 0  # Track how much new content we're seeing this turn
    
    # CRITICAL FIX: Track symbols used within THIS REQUEST to prevent collisions between messages
    request_used_symbols = set()
    
    for msg in messages:
        if msg.get("role") in {"user", "assistant", "system"}:  # CRITICAL FIX: Include system messages!
            # CRITICAL FIX: Handle complex content structures from Cline
            try:
                from api.server import extract_message_content_for_compression
                original_content = extract_message_content_for_compression(msg.get("content", ""))
            except ImportError:
                original_content = str(msg.get("content", ""))
            
            if not original_content:
                compressed_messages.append(msg.copy())
                continue
                
            total_original_size += len(original_content)
            
            # Calculate new content size for smarter additive analysis
            # Consider content "new" if it's not been seen in this conversation before
            # For simplicity, consider assistant messages and recent user messages as "new"
            if msg.get("role") in {"user", "assistant"}:
                # For user messages, consider the last few as "new" content for this turn
                # For assistant messages, always consider as new
                new_content_size += len(original_content)
            
            # STEP 1: Always apply existing conversation compression rules first - this is the key to additive behavior
            compressed_content, used_symbols = _apply_existing_compression(original_content, conversation_state.compression_rules)
            all_used_symbols.update(used_symbols)
            
            # STEP 2: Enhanced logic for analyzing new symbols - consider both symbol count AND new content volume
            # This ensures we ALWAYS apply existing symbols but are intelligent about adding new ones
            should_analyze_for_new_symbols = (
                len(compressed_content) > min_characters * 2 and  # Content is still very large after existing compression
                (conversation_state.turn_number == 0 or  # First turn - establish initial symbols
                 len(conversation_state.compression_rules) < 262 or  # Match max_dictionary_size config
                 # ENHANCED: Use configured kv_cache_threshold for smart content analysis
                 (len(compressed_content) > kv_cache_threshold * 500 and len(conversation_state.compression_rules) < 350) or  # Large content based on config
                 # CRITICAL: If there's substantial new content this turn, analyze it using threshold
                 (new_content_size > kv_cache_threshold * 750 and len(conversation_state.compression_rules) < 500))  # Substantial new content based on config
            )
            
            if should_analyze_for_new_symbols:
                # Only for first turn or when we have room for more symbols
                # CRITICAL FIX: Create exclude_symbols from BOTH conversation symbols AND request symbols
                existing_symbols = set(conversation_state.compression_rules.keys())
                if exclude_symbols:
                    existing_symbols.update(exclude_symbols)
                
                # CRITICAL FIX: Also exclude symbols used by other messages in THIS REQUEST
                existing_symbols.update(request_used_symbols)
                
                # CRITICAL DEBUG: Track symbol exclusion
                logger.info(f"🔍 [SYMBOL EXCLUSION] Excluding {len(existing_symbols)} symbols from analysis: {list(existing_symbols)[:10]}...")
                logger.info(f"🔍 [SYMBOL EXCLUSION] Conversation has {len(conversation_state.compression_rules)} existing rules, request has {len(request_used_symbols)} used symbols")
                
                from core.compress import compress_with_dynamic_analysis
                result = compress_with_dynamic_analysis(
                    compressed_content,  # Apply to already-compressed content
                    skip_tool_detection=False, 
                    cline_mode=True,
                    exclude_symbols=existing_symbols  # CRITICAL FIX: Exclude BOTH conversation AND request symbols
                )
                
                # CRITICAL DEBUG: Track what new symbols were actually created
                logger.info(f"🔍 [NEW SYMBOLS] Dynamic analysis created {len(result.used)} potential symbols: {list(result.used.keys())[:10]}...")
                
                # CRITICAL FIX: Track symbols used by THIS MESSAGE for subsequent messages in the request
                message_symbols_used = set()
                
                # Merge new rules with conversation rules - but be selective to maintain quality
                symbols_filtered_out = 0
                symbols_added = 0
                for symbol, original_text in result.used.items():
                    # Only add if it's genuinely new, valuable, and doesn't conflict
                    if (original_text not in conversation_state.reverse_rules and 
                        len(original_text) >= 5 and  # Higher threshold for new symbols
                        symbol not in existing_symbols):  # Ensure no symbol collision
                        new_compression_rules[symbol] = original_text
                        all_used_symbols.add(symbol)  # Track that this new symbol was used
                        message_symbols_used.add(symbol)  # Track for this message
                        symbols_added += 1
                    else:
                        symbols_filtered_out += 1
                        # Debug why this symbol was filtered out
                        if original_text in conversation_state.reverse_rules:
                            logger.debug(f"🔍 [FILTER] Skipped {symbol}={original_text} - text already has symbol")
                        elif len(original_text) < 5:
                            logger.debug(f"🔍 [FILTER] Skipped {symbol}={original_text} - too short")
                        elif symbol in existing_symbols:
                            logger.debug(f"🔍 [FILTER] Skipped {symbol}={original_text} - symbol already used")
                
                # CRITICAL FIX: Add this message's symbols to the request exclusion list for subsequent messages
                request_used_symbols.update(message_symbols_used)
                
                # CRITICAL DEBUG: Track filtering results
                logger.info(f"🔍 [SYMBOL FILTERING] Added {symbols_added} symbols, filtered out {symbols_filtered_out} symbols")
                logger.info(f"🔍 [REQUEST COLLISION PREVENTION] Request now excludes {len(request_used_symbols)} symbols: {list(request_used_symbols)[:10]}...")
                
                compressed_content = result.text
                logger.info(f"🔍 [ADDITIVE] Added {len(new_compression_rules)} new symbols to conversation (turn {conversation_state.turn_number}, {len(all_used_symbols)} total used)")
                logger.info(f"🔍 [ADDITIVE] Analysis triggered: compressed_size={len(compressed_content)}, new_content_size={new_content_size}, existing_rules={len(conversation_state.compression_rules)}")
            else:
                # Use existing conversation symbols only - this is the common case for additive compression  
                logger.info(f"🔍 [ADDITIVE] Skipped new symbol analysis (turn {conversation_state.turn_number}): compressed_size={len(compressed_content)}, new_content_size={new_content_size}, existing_rules={len(conversation_state.compression_rules)}")
                logger.debug(f"Using {len(used_symbols)} existing conversation symbols only (turn {conversation_state.turn_number}, {len(conversation_state.compression_rules)} available)")
            
            total_compressed_size += len(compressed_content)
            
            # Create compressed message
            compressed_msg = msg.copy()
            compressed_msg["content"] = compressed_content
            compressed_messages.append(compressed_msg)
        else:
            # Keep other role messages unchanged (like function, tool, etc.)
            compressed_messages.append(msg.copy())
    
    # Calculate compression metrics
    compression_ratio = (total_original_size - total_compressed_size) / total_original_size if total_original_size > 0 else 0
    tokens_saved = (total_original_size - total_compressed_size) // 4  # Rough estimate
    
    # STATEFUL MODE: Calculate efficiency based on NEW symbols only for persistent KV cache servers
    if stateful_mode:
        # For stateful mode, only consider NEW symbols for overhead calculation
        # This is because persistent KV cache servers remember previous system prompts
        new_symbols_only = {symbol: new_compression_rules[symbol] for symbol in new_compression_rules.keys()}
        stateful_overhead = _estimate_system_prompt_overhead(new_symbols_only)
        stateful_overhead_ratio = stateful_overhead / (total_original_size // 4) if total_original_size > 0 else 0
        
        # CRITICAL FIX: Calculate total compression benefit including reused symbols
        # In stateful mode, we need to account for ALL compression happening this turn,
        # not just the compression from new symbols
        if len(all_used_symbols) > 0:
            # Calculate what the size would be WITHOUT any compression
            uncompressed_total_size = 0
            for msg in messages:
                if msg.get("role") in {"user", "assistant", "system"}:
                    try:
                        from api.server import extract_message_content_for_compression
                        original_content = extract_message_content_for_compression(msg.get("content", ""))
                        uncompressed_total_size += len(original_content)
                    except ImportError:
                        original_content = str(msg.get("content", ""))
                        uncompressed_total_size += len(original_content)
            
            # Total compression ratio including reused symbols
            total_compression_ratio = (uncompressed_total_size - total_compressed_size) / uncompressed_total_size if uncompressed_total_size > 0 else 0
        else:
            # No symbols used, so compression ratio is what we calculated
            total_compression_ratio = compression_ratio
        
        # Stateful efficiency = total compression benefit - new symbols overhead only
        stateful_net_efficiency = total_compression_ratio - stateful_overhead_ratio
        
        # Use stateful calculations for efficiency
        system_prompt_overhead = stateful_overhead
        overhead_ratio = stateful_overhead_ratio  
        net_efficiency = stateful_net_efficiency
        
        # Track both used and new symbols for return
        used_rules = {symbol: (conversation_state.compression_rules.get(symbol) or new_compression_rules.get(symbol)) 
                      for symbol in all_used_symbols}
        new_symbols_for_prompt = new_symbols_only
        
        logger.info(f"STATEFUL MODE: total compression {total_compression_ratio:.3f}, "
                   f"efficiency based on {len(new_symbols_only)} new symbols only "
                   f"(overhead: {stateful_overhead} tokens, net efficiency: {stateful_net_efficiency:.3f})")
    else:
        # STATELESS MODE: Calculate efficiency based on ALL symbols used (original behavior)
        used_rules = {symbol: (conversation_state.compression_rules.get(symbol) or new_compression_rules.get(symbol)) 
                      for symbol in all_used_symbols}
        system_prompt_overhead = _estimate_system_prompt_overhead(used_rules)
        overhead_ratio = system_prompt_overhead / (total_original_size // 4) if total_original_size > 0 else 0
        net_efficiency = compression_ratio - overhead_ratio
        new_symbols_for_prompt = None  # Not used in stateless mode
        
        logger.debug(f"STATELESS MODE: efficiency based on {len(used_rules)} total symbols used "
                    f"(overhead: {system_prompt_overhead} tokens, net efficiency: {net_efficiency:.3f})")
    
    # Update conversation state with new compression data
    compression_metrics = {
        'compression_ratio': compression_ratio,
        'overhead_ratio': overhead_ratio,
        'chars_saved': total_original_size - total_compressed_size,
        'tokens_saved': tokens_saved,
        'system_prompt_tokens': system_prompt_overhead,
        # ENHANCED: Additional metrics for cumulative tracking
        'original_chars': total_original_size,
        'compressed_chars': total_compressed_size, 
        'messages_count': len(messages),
        'kv_cache_used': False,  # This is full compression analysis
        'symbols_used': all_used_symbols  # Track which symbols were actually used this turn
    }
    
    # Update conversation state and get final rules (this adds new rules to the conversation dictionary)
    all_conversation_rules = manager.update_conversation_compression(
        conversation_id, 
        new_compression_rules,
        compression_metrics
    )
    
    # Get updated conversation metrics
    conversation_metrics = manager.get_conversation_metrics(conversation_id)
    should_continue = conversation_metrics['should_continue'] if conversation_metrics else True
    
    logger.info(f"Conversation {conversation_id} turn {conversation_state.turn_number}: "
               f"compression {compression_ratio:.3f}, overhead {overhead_ratio:.3f}, "
               f"net efficiency {net_efficiency:.3f}, used {len(all_used_symbols)}/{len(all_conversation_rules)} symbols")
    
    # ENHANCED: Log actual conversation-wide cumulative metrics if available
    if conversation_metrics and conversation_metrics.get('turn_number', 0) > 1:
        cumulative_ratio = conversation_metrics.get('cumulative_compression_ratio', 0)
        cumulative_efficiency = conversation_metrics.get('cumulative_net_efficiency', 0)
        cumulative_chars_saved = conversation_metrics.get('cumulative_chars_saved', 0)
        cumulative_original = conversation_metrics.get('cumulative_original_chars', 0)
        turn_number = conversation_metrics.get('turn_number', 0)
        
        logger.info("=" * 70)
        logger.info("📊 ACTUAL CONVERSATION-WIDE COMPRESSION SUMMARY")
        logger.info(f"📊 Cumulative compression across {turn_number} turns: {cumulative_ratio*100:.1f}% ({cumulative_chars_saved:,} chars from {cumulative_original:,})")
        logger.info(f"📊 Cumulative net efficiency: {cumulative_efficiency*100:.1f}%")
        logger.info(f"📊 Total compression rules: {len(all_conversation_rules)}")
        logger.info("=" * 70)
    
    return ConversationCompressionResult(
        compressed_messages=compressed_messages,
        compression_rules=used_rules,  # Only return symbols that were actually used in this request
        conversation_id=conversation_id,
        turn_number=conversation_state.turn_number,
        compression_ratio=compression_ratio,
        tokens_saved=tokens_saved,
        system_prompt_overhead=system_prompt_overhead,  # Overhead based on used symbols only
        net_efficiency=net_efficiency,
        should_continue=should_continue,
        metrics=conversation_metrics or {},
        kv_cache_optimization_used=False,  # Full compression analysis was performed
        new_symbols_for_prompt=new_symbols_for_prompt,  # New symbols for stateful mode
        stateful_mode=stateful_mode  # Flag to indicate stateful mode was used
    )

def _apply_existing_compression(content, compression_rules: Dict[str, str]) -> Tuple[str, Set[str]]:
    """
    Apply existing conversation compression rules to content.
    
    Returns:
        Tuple of (compressed_content, used_symbols_set)
        where used_symbols_set contains only the symbols that were actually found and applied
    """
    if not compression_rules:
        return content, set()
    
    # Track which symbols were actually used
    used_symbols = set()
    
    # CRITICAL FIX: Handle complex content structures from Cline (lists, dicts, etc.)
    try:
        from api.server import extract_message_content_for_compression
        
        # If content is complex (list/dict), extract text first
        if isinstance(content, (list, dict)):
            text_content = extract_message_content_for_compression(content)
        else:
            text_content = str(content) if content is not None else ""
        
        # Apply compression rules to the extracted text
        compressed_text = text_content
        
        # Sort by original text length (longest first) to avoid substring conflicts
        sorted_rules = sorted(compression_rules.items(), key=lambda x: len(x[1]), reverse=True)
        
        for symbol, original_text in sorted_rules:
            # Check if the original text is present before replacing
            if original_text in compressed_text:
                compressed_text = compressed_text.replace(original_text, symbol)
                used_symbols.add(symbol)
        
        # For complex content structures, we need to return the compressed text
        # The calling code will update the message content appropriately
        return compressed_text, used_symbols
        
    except ImportError:
        # Fallback if import fails
        if isinstance(content, str):
            compressed_content = content
            sorted_rules = sorted(compression_rules.items(), key=lambda x: len(x[1]), reverse=True)
            for symbol, original_text in sorted_rules:
                if original_text in compressed_content:
                    compressed_content = compressed_content.replace(original_text, symbol)
                    used_symbols.add(symbol)
            return compressed_content, used_symbols
        else:
            # Can't process non-string content without extract function
            return content, set()

def _estimate_system_prompt_overhead(compression_rules: Dict[str, str]) -> int:
    """
    Estimate the token overhead of the system prompt with compression rules.
    Uses actual format configuration to provide accurate overhead estimation.
    """
    if not compression_rules:
        return 0
    
    # Import here to avoid circular import
    from .system_prompt import _load_dictionary_format_config, _format_dictionary_minimal, _format_dictionary_verbose
    
    # Load current dictionary format configuration
    dict_config = _load_dictionary_format_config()
    style = dict_config.get("style", "minimal")
    threshold = dict_config.get("minimal_format_threshold", 3)
    
    # Determine which format will be used
    if style == "minimal" and len(compression_rules) >= threshold:
        # Use minimal format - much more compact
        format_content = _format_dictionary_minimal(compression_rules, "code", dict_config)
    else:
        # Use verbose format
        format_content = _format_dictionary_verbose(compression_rules, "code")
    
    # Calculate actual overhead based on character count
    # Use a more accurate token estimation (average is closer to 3.5 chars per token for mixed content)
    overhead_tokens = len(format_content) / 3.5
    
    # Add some overhead for API formatting and structure (~15% for metadata)
    return int(overhead_tokens * 1.15)

def _create_uncompressed_result(messages: List[Dict[str, Any]], 
                              conversation_state: ConversationState,
                              stateful_mode: bool = False) -> ConversationCompressionResult:
    """Create a result object for uncompressed messages while preserving historical metrics."""
    
    # CRITICAL FIX: When compression is disabled, we should still:
    # 1. Preserve existing compression rules (needed for decompressing responses)
    # 2. Preserve cumulative metrics (show historical performance)
    # 3. Continue counting turns (even if no new compression happens)
    
    # Get conversation manager to update turn count even when compression is disabled
    from .conversation_state import get_conversation_manager
    manager = get_conversation_manager()
    
    # Update turn count and metrics even when compression is disabled
    if conversation_state.conversation_id in manager.states:
        # Calculate current content size for tracking
        try:
            from api.server import extract_message_content_for_compression
            current_content_size = sum(len(extract_message_content_for_compression(msg.get("content", ""))) 
                                     for msg in messages if msg.get("role") in {"user", "assistant", "system"})
        except ImportError:
            current_content_size = sum(len(str(msg.get("content", ""))) 
                                     for msg in messages if msg.get("role") in {"user", "assistant", "system"})
        
        # Update metrics with no compression but track the turn
        disabled_metrics = {
            'compression_ratio': 0.0,
            'overhead_ratio': 0.0,
            'chars_saved': 0,
            'tokens_saved': 0,
            'system_prompt_tokens': 0,
            'original_chars': current_content_size,
            'compressed_chars': current_content_size,  # No compression applied
            'messages_count': len(messages),
            'kv_cache_used': False
        }
        
        # Update the conversation state to increment turn number and track metrics
        existing_rules = manager.update_conversation_compression(
            conversation_state.conversation_id, 
            {},  # No new compression rules
            disabled_metrics
        )
        
        # Get updated metrics that include cumulative totals
        conversation_metrics = manager.get_conversation_metrics(conversation_state.conversation_id) or {}
        
        logger.debug(f"Compression disabled but preserved {len(existing_rules)} rules and cumulative metrics for conversation {conversation_state.conversation_id}")
        
        return ConversationCompressionResult(
            compressed_messages=messages,
            compression_rules=existing_rules,  # FIXED: Preserve existing rules for decompression
            conversation_id=conversation_state.conversation_id,
            turn_number=conversation_state.turn_number,  # FIXED: Updated turn number
            compression_ratio=0.0,  # No compression this turn
            tokens_saved=0,  # No tokens saved this turn
            system_prompt_overhead=_estimate_system_prompt_overhead(existing_rules),
            net_efficiency=0.0,  # No efficiency this turn
            should_continue=False,  # Compression disabled
            metrics=conversation_metrics,  # FIXED: Preserve cumulative metrics
            kv_cache_optimization_used=False,
            new_symbols_for_prompt=None if not stateful_mode else {},  # No new symbols when compression disabled
            stateful_mode=stateful_mode
        )
    else:
        # Fallback for conversations without state (shouldn't happen in normal operation)
        return ConversationCompressionResult(
            compressed_messages=messages,
            compression_rules={},
            conversation_id=conversation_state.conversation_id,
            turn_number=conversation_state.turn_number,
            compression_ratio=0.0,
            tokens_saved=0,
            system_prompt_overhead=0,
            net_efficiency=0.0,
            should_continue=False,
            metrics={},
            kv_cache_optimization_used=False,
            new_symbols_for_prompt=None if not stateful_mode else {},
            stateful_mode=stateful_mode
        )

def get_conversation_compression_stats() -> Dict[str, Any]:
    """Get overall statistics for conversation compression system."""
    manager = get_conversation_manager()
    return manager.get_stats()

def reset_conversation_compression():
    """Reset all conversation compression state (for testing/debugging)."""
    from .conversation_state import reset_conversation_manager
    reset_conversation_manager()
    logger.info("Reset conversation compression state")

def should_use_conversation_compression(messages: List[Dict[str, Any]], session_id: str = None) -> Tuple[bool, str]:
    """
    Determine if conversation compression should be used for these messages.
    
    Returns:
        Tuple of (should_use, reason)
    """
    manager = get_conversation_manager()
    conversation_state = manager.get_or_create_conversation_state(messages, session_id)
    
    if conversation_state.turn_number == 0:
        return True, "first_turn"
    
    if not manager.should_use_conversation_compression(conversation_state.conversation_id):
        return False, "poor_efficiency_trend"
    
    # Check if conversation is getting too long
    if conversation_state.turn_number > 20:  # Arbitrary limit
        latest_efficiency = conversation_state.net_efficiency_trend[-1] if conversation_state.net_efficiency_trend else 0
        if latest_efficiency < 0.02:  # Less than 2% net benefit
            return False, "long_conversation_low_efficiency"
    
    return True, "continuing_efficient_compression" 