"""
Conversation State Manager for KrunchWrapper
Maintains compression dictionaries across conversation turns for optimal multi-turn compression.
"""

import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import json

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """State information for a conversation."""
    conversation_id: str
    turn_number: int = 0
    compression_rules: Dict[str, str] = field(default_factory=dict)  # symbol -> original_text
    reverse_rules: Dict[str, str] = field(default_factory=dict)  # original_text -> symbol
    token_usage_count: Dict[str, int] = field(default_factory=dict)  # token -> count
    total_chars_saved: int = 0
    total_tokens_saved: int = 0
    system_prompt_overhead: int = 0
    last_updated: float = field(default_factory=time.time)
    content_hash: str = ""  # Hash of all conversation content for consistency
    
    # Progressive compression metrics
    turn_compression_ratios: List[float] = field(default_factory=list)
    turn_overhead_ratios: List[float] = field(default_factory=list)
    net_efficiency_trend: List[float] = field(default_factory=list)
    
    # ENHANCED: Cumulative metrics for running tally
    cumulative_original_chars: int = 0      # Total characters processed across all turns
    cumulative_compressed_chars: int = 0    # Total compressed characters across all turns
    cumulative_messages_processed: int = 0  # Total messages processed across all turns
    turn_start_times: List[float] = field(default_factory=list)  # Track timing per turn
    turn_end_times: List[float] = field(default_factory=list)    # Track timing per turn
    kv_cache_optimizations_used: int = 0    # Count of KV cache optimizations
    
    # NEW: Symbol usage tracking for additive compression
    symbols_used_per_turn: List[Set[str]] = field(default_factory=list)  # Track which symbols were used each turn
    symbols_added_per_turn: List[Set[str]] = field(default_factory=list)  # Track which symbols were added each turn
    
    def add_turn_metrics(self, compression_ratio: float, overhead_ratio: float, 
                        original_chars: int = 0, compressed_chars: int = 0, 
                        messages_count: int = 0, kv_cache_used: bool = False,
                        symbols_used: Set[str] = None, symbols_added: Set[str] = None):
        """Add metrics for a turn and calculate efficiency trend."""
        self.turn_compression_ratios.append(compression_ratio)
        self.turn_overhead_ratios.append(overhead_ratio)
        
        # Net efficiency = compression benefit - overhead cost
        net_efficiency = compression_ratio - overhead_ratio
        self.net_efficiency_trend.append(net_efficiency)
        
        # Update cumulative metrics
        self.cumulative_original_chars += original_chars
        self.cumulative_compressed_chars += compressed_chars
        self.cumulative_messages_processed += messages_count
        
        if kv_cache_used:
            self.kv_cache_optimizations_used += 1
        
        # Track symbol usage for additive compression analysis
        if symbols_used is not None:
            self.symbols_used_per_turn.append(symbols_used.copy())
        else:
            self.symbols_used_per_turn.append(set())
            
        if symbols_added is not None:
            self.symbols_added_per_turn.append(symbols_added.copy())
        else:
            self.symbols_added_per_turn.append(set())
        
        # Track timing
        current_time = time.time()
        self.turn_end_times.append(current_time)
        
        # Update turn number
        self.turn_number += 1
        self.last_updated = current_time
    
    def start_turn(self):
        """Mark the start of a new turn for timing."""
        self.turn_start_times.append(time.time())
    
    def get_cumulative_compression_ratio(self) -> float:
        """Calculate overall compression ratio across all turns."""
        if self.cumulative_original_chars == 0:
            return 0.0
        return (self.cumulative_original_chars - self.cumulative_compressed_chars) / self.cumulative_original_chars
    
    def get_cumulative_net_efficiency(self) -> float:
        """Calculate overall net efficiency accounting for system prompt overhead."""
        if self.cumulative_original_chars == 0:
            return 0.0
        
        compression_ratio = self.get_cumulative_compression_ratio()
        # Convert overhead tokens to character equivalent for comparison
        overhead_chars = self.system_prompt_overhead * 4  # Rough tokens to chars conversion
        overhead_ratio = overhead_chars / self.cumulative_original_chars if self.cumulative_original_chars > 0 else 0
        
        return compression_ratio - overhead_ratio
    
    def get_total_processing_time(self) -> float:
        """Get total processing time across all turns."""
        if not self.turn_start_times or not self.turn_end_times:
            return 0.0
        
        total_time = 0.0
        for i in range(min(len(self.turn_start_times), len(self.turn_end_times))):
            total_time += self.turn_end_times[i] - self.turn_start_times[i]
        
        return total_time
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for this conversation including running tallies."""
        cumulative_compression_ratio = self.get_cumulative_compression_ratio()
        cumulative_net_efficiency = self.get_cumulative_net_efficiency()
        total_processing_time = self.get_total_processing_time()
        symbol_stats = self.get_symbol_usage_stats()
        
        return {
            # Basic info
            'conversation_id': self.conversation_id,
            'turn_number': self.turn_number,
            'total_rules': len(self.compression_rules),
            
            # Running tallies - the key new feature
            'cumulative_chars_saved': self.total_chars_saved,
            'cumulative_tokens_saved': self.total_tokens_saved,
            'cumulative_original_chars': self.cumulative_original_chars,
            'cumulative_compressed_chars': self.cumulative_compressed_chars,
            'cumulative_compression_ratio': cumulative_compression_ratio,
            'cumulative_net_efficiency': cumulative_net_efficiency,
            'cumulative_messages_processed': self.cumulative_messages_processed,
            
            # Symbol usage efficiency - NEW
            'symbol_usage_stats': symbol_stats,
            'avg_symbols_per_turn': symbol_stats.get('avg_symbols_per_turn', 0),
            'symbol_efficiency': symbol_stats.get('symbol_efficiency', 0),
            
            # Efficiency metrics
            'system_prompt_overhead_tokens': self.system_prompt_overhead,
            'net_tokens_saved': self.total_tokens_saved - self.system_prompt_overhead,
            'efficiency_trend': self.get_efficiency_trend(),
            'should_continue': self.should_continue_compression(),
            
            # Performance metrics
            'total_processing_time': total_processing_time,
            'avg_processing_time_per_turn': total_processing_time / self.turn_number if self.turn_number > 0 else 0,
            'kv_cache_optimizations_used': self.kv_cache_optimizations_used,
            
            # Current turn metrics
            'latest_compression_ratio': self.turn_compression_ratios[-1] if self.turn_compression_ratios else 0,
            'latest_net_efficiency': self.net_efficiency_trend[-1] if self.net_efficiency_trend else 0,
            'avg_compression_ratio': sum(self.turn_compression_ratios) / len(self.turn_compression_ratios) if self.turn_compression_ratios else 0,
            
            # Most used patterns
            'most_used_tokens': sorted(self.token_usage_count.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def get_efficiency_trend(self, last_n_turns: int = 3) -> str:
        """Get the efficiency trend for recent turns."""
        if len(self.net_efficiency_trend) < 2:
            return "insufficient_data"
        
        recent_turns = self.net_efficiency_trend[-last_n_turns:]
        if len(recent_turns) < 2:
            return "improving" if recent_turns[-1] > 0 else "declining"
        
        # Calculate trend
        if recent_turns[-1] > recent_turns[0]:
            return "improving"
        elif recent_turns[-1] < recent_turns[0]:
            return "declining"
        else:
            return "stable"
    
    def should_continue_compression(self, min_net_efficiency: float = 0.01) -> bool:
        """
        Determine if compression should continue based on current conditions.
        Each turn is evaluated independently - no permanent disabling.
        """
        if not self.net_efficiency_trend:
            return True  # First turn, always try
        
        # INDEPENDENT TURN ASSESSMENT: Each turn evaluated on its own merits
        
        recent_efficiency = self.net_efficiency_trend[-1]
        
        # If recent efficiency is acceptable, continue
        if recent_efficiency >= min_net_efficiency:
            return True
        
        # If we're in early conversation (≤ 5 turns), be forgiving - still learning patterns
        if len(self.net_efficiency_trend) <= 5:
            return True
        
        # For stateful mode: If we have existing symbols, compression can still be valuable
        # even with poor recent efficiency, because symbol reuse has zero overhead
        if len(self.compression_rules) > 0:
            # If we have a decent symbol library, continue (stateful mode benefit)
            return True
        
        # Check if we're seeing improvement trend (last 2 turns getting better)
        if len(self.net_efficiency_trend) >= 2:
            last_turn = self.net_efficiency_trend[-1]
            previous_turn = self.net_efficiency_trend[-2]
            if last_turn > previous_turn:  # Improving trend
                return True
        
        # Only disable if recent efficiency is very poor AND no mitigating factors
        return recent_efficiency >= min_net_efficiency * 0.5  # 50% of threshold as absolute floor
    
    def get_symbol_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about symbol usage across turns."""
        if not self.symbols_used_per_turn:
            return {'total_symbols': 0, 'avg_symbols_per_turn': 0, 'symbol_efficiency': 0}
        
        total_symbols_available = len(self.compression_rules)
        total_turns = len(self.symbols_used_per_turn)
        total_symbols_used = sum(len(used_set) for used_set in self.symbols_used_per_turn)
        avg_symbols_per_turn = total_symbols_used / total_turns if total_turns > 0 else 0
        
        # Calculate symbol efficiency: what percentage of available symbols are typically used?
        symbol_efficiency = avg_symbols_per_turn / total_symbols_available if total_symbols_available > 0 else 0
        
        # Track most and least used symbols
        symbol_usage_count = defaultdict(int)
        for used_set in self.symbols_used_per_turn:
            for symbol in used_set:
                symbol_usage_count[symbol] += 1
        
        most_used = sorted(symbol_usage_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_symbols_available': total_symbols_available,
            'total_symbols_added': len(set().union(*self.symbols_added_per_turn)) if self.symbols_added_per_turn else 0,
            'avg_symbols_per_turn': avg_symbols_per_turn,
            'symbol_efficiency': symbol_efficiency,
            'most_used_symbols': most_used,
            'turns_with_symbol_usage': total_turns
        }


class ConversationStateManager:
    """Manages conversation states across multiple conversations."""
    
    def __init__(self, max_conversations: int = 1000, cleanup_interval: int = 3600):
        self.states: Dict[str, ConversationState] = {}
        self.max_conversations = max_conversations
        self.cleanup_interval = cleanup_interval  # seconds
        self.last_cleanup = time.time()
        self.lock = threading.RLock()  # Re-entrant lock for nested calls
        
        # Symbol management for progressive compression
        self.available_symbols = self._load_available_symbols()
        self.global_symbol_usage: Dict[str, Set[str]] = defaultdict(set)  # symbol -> set of conversation_ids
        
    def _load_available_symbols(self) -> List[str]:
        """Load available Unicode symbols for compression."""
        try:
            # Try to load from the discovered symbols file
            import json
            from pathlib import Path
            
            symbols_file = Path(__file__).parent.parent / "utils" / "unicode_symbols_discovered.json"
            if symbols_file.exists():
                with open(symbols_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    symbols = data.get('symbols', [])
                    logger.debug(f"Loaded {len(symbols)} available symbols for progressive compression")
                    return symbols
        except Exception as e:
            logger.warning(f"Could not load Unicode symbols, using fallback: {e}")
        
        # Fallback to a basic set of symbols
        return ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ']
    
    def _generate_conversation_id(self, messages: List[Dict[str, Any]], session_id: str = None) -> str:
        """Generate a consistent conversation ID from message history and session context."""
        
        # If session_id provided, use it as primary identifier
        if session_id:
            # Clean session ID (remove special characters for safety)
            clean_session = "".join(c for c in session_id if c.isalnum() or c in "-_")[:32]
            
            # For session-based conversations, use first user message for context
            first_user_content = ""
            for msg in messages:
                if msg.get("role") == "user":
                    first_user_content = msg.get("content", "")[:100]
                    break
            
            # Create session-scoped conversation ID
            content_for_hash = f"{clean_session}:{first_user_content}".encode('utf-8')
            conversation_hash = hashlib.md5(content_for_hash).hexdigest()[:12]
            return f"sess_{clean_session}_{conversation_hash}"
        
        # Fallback: Legacy conversation identification (for backward compatibility)
        logger.debug("No session ID provided, using legacy conversation identification")
        
        # For single messages, generate unique ID to avoid false matches
        if len(messages) == 1:
            # Single message = likely new conversation, make it unique
            timestamp = int(time.time() * 1000)  # millisecond precision
            first_user_content = messages[0].get("content", "")[:50]
            content_for_hash = f"{timestamp}:{first_user_content}".encode('utf-8')
            conversation_hash = hashlib.md5(content_for_hash).hexdigest()[:12]
            return f"new_{conversation_hash}"
        
        # For multi-message conversations, use message sequence for ID
        # This creates unique IDs based on the actual conversation flow
        conversation_signature = []
        for i, msg in enumerate(messages[:3]):  # Use first 3 messages max
            role = msg.get("role", "")
            content = msg.get("content", "")[:50]  # First 50 chars
            conversation_signature.append(f"{role}:{content}")
        
        # Include message count to distinguish conversations of different lengths
        signature_text = f"{len(messages)}|" + "|".join(conversation_signature)
        content_for_hash = signature_text.encode('utf-8')
        conversation_hash = hashlib.md5(content_for_hash).hexdigest()[:12]
        return f"conv_{conversation_hash}"
    
    def _generate_content_hash(self, messages: List[Dict[str, Any]]) -> str:
        """Generate hash of all conversation content for consistency checking."""
        content_parts = []
        for msg in messages:
            if msg.get("role") in {"user", "assistant"}:
                content_parts.append(f"{msg.get('role')}:{msg.get('content', '')}")
        
        full_content = "\n".join(content_parts)
        return hashlib.sha256(full_content.encode('utf-8')).hexdigest()[:16]
    
    def get_or_create_conversation_state(self, messages: List[Dict[str, Any]], session_id: str = None) -> ConversationState:
        """Get existing conversation state or create new one."""
        with self.lock:
            conversation_id = self._generate_conversation_id(messages, session_id)
            content_hash = self._generate_content_hash(messages)
            
            if conversation_id in self.states:
                state = self.states[conversation_id]
                
                # Check if this is a new turn (content has grown)
                if content_hash != state.content_hash:
                    logger.debug(f"Detected new turn for conversation {conversation_id}")
                    state.content_hash = content_hash
                else:
                    logger.debug(f"Using existing state for conversation {conversation_id}")
                
                return state
            else:
                # Create new conversation state
                logger.debug(f"Creating new conversation state: {conversation_id}")
                if session_id:
                    logger.info(f"Started new session-based conversation: {conversation_id[:20]}... (session: {session_id[:10]}...)")
                else:
                    logger.debug(f"Started new legacy conversation: {conversation_id}")
                
                state = ConversationState(
                    conversation_id=conversation_id,
                    content_hash=content_hash
                )
                self.states[conversation_id] = state
                
                # Cleanup old conversations if needed
                self._maybe_cleanup()
                
                return state
    
    def update_conversation_compression(self, 
                                     conversation_id: str, 
                                     new_rules: Dict[str, str],
                                     compression_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Update conversation compression rules and return merged rules."""
        with self.lock:
            if conversation_id not in self.states:
                logger.warning(f"Conversation {conversation_id} not found for update")
                return new_rules
            
            state = self.states[conversation_id]
            
            # Track which symbols are being added this turn
            symbols_added_this_turn = set()
            
            # Merge rules progressively - existing rules take precedence for consistency
            merged_rules = state.compression_rules.copy()
            
            # CRITICAL FIX: Handle correct format - new_rules is symbol -> original_text
            symbols_to_assign = []
            for symbol, original_text in new_rules.items():  # FIXED: Correct format expectation
                if original_text not in state.reverse_rules:
                    # This is a new token to compress
                    if symbol not in merged_rules:
                        # Symbol is available, use it
                        merged_rules[symbol] = original_text
                        state.reverse_rules[original_text] = symbol
                        symbols_added_this_turn.add(symbol)
                        
                        # Track token usage
                        state.token_usage_count[original_text] = state.token_usage_count.get(original_text, 0) + 1
                        
                        logger.debug(f"✅ [ADDITIVE FIX] Added new compression rule: {symbol}={original_text}")
                    else:
                        # Symbol conflict, need to assign new symbol
                        logger.warning(f"🚨 [ADDITIVE FIX] Symbol conflict detected: {symbol} already used, reassigning text: {original_text}")
                        symbols_to_assign.append(original_text)
                else:
                    # Token already has a symbol, keep existing mapping for consistency
                    existing_symbol = state.reverse_rules[original_text]
                    logger.debug(f"🔄 [ADDITIVE FIX] Keeping existing mapping: {existing_symbol}={original_text}")
            
            # Assign new symbols for conflicting tokens
            if symbols_to_assign:
                new_symbols = self._assign_new_symbols(conversation_id, symbols_to_assign, merged_rules)
                merged_rules.update(new_symbols)
                for symbol, original_text in new_symbols.items():
                    state.reverse_rules[original_text] = symbol
                    state.token_usage_count[original_text] = state.token_usage_count.get(original_text, 0) + 1
                    symbols_added_this_turn.add(symbol)
            
            # Update state with debug tracking
            previous_symbol_count = len(state.compression_rules)
            state.compression_rules = merged_rules
            new_symbol_count = len(merged_rules)
            
            # CRITICAL DEBUG: Track symbol preservation
            logger.info(f"🔍 [ADDITIVE DEBUG] Symbol state change: {previous_symbol_count} → {new_symbol_count} total symbols")
            logger.info(f"🔍 [ADDITIVE DEBUG] Added {len(symbols_added_this_turn)} symbols this turn, reassigned {len(symbols_to_assign)} conflicts")
            
            # Extract symbol usage from compression metrics if available
            symbols_used_this_turn = compression_metrics.get('symbols_used', set())
            
            # ENHANCED: Add turn metrics with additional data for cumulative tracking including symbol usage
            compression_ratio = compression_metrics.get('compression_ratio', 0.0)
            overhead_ratio = compression_metrics.get('overhead_ratio', 0.0)
            original_chars = compression_metrics.get('original_chars', 0)
            compressed_chars = compression_metrics.get('compressed_chars', 0)
            messages_count = compression_metrics.get('messages_count', 0)
            kv_cache_used = compression_metrics.get('kv_cache_used', False)
            
            state.add_turn_metrics(
                compression_ratio=compression_ratio,
                overhead_ratio=overhead_ratio,
                original_chars=original_chars,
                compressed_chars=compressed_chars,
                messages_count=messages_count,
                kv_cache_used=kv_cache_used,
                symbols_used=symbols_used_this_turn,
                symbols_added=symbols_added_this_turn
            )
            
            # Update cumulative stats (legacy format for backward compatibility)
            state.total_chars_saved += compression_metrics.get('chars_saved', 0)
            state.total_tokens_saved += compression_metrics.get('tokens_saved', 0)
            state.system_prompt_overhead += compression_metrics.get('system_prompt_tokens', 0)
            
            symbol_stats = state.get_symbol_usage_stats()
            logger.info(f"Updated conversation {conversation_id}: turn {state.turn_number}, "
                       f"{len(merged_rules)} total rules, {len(symbols_used_this_turn)} used this turn, "
                       f"avg {symbol_stats.get('avg_symbols_per_turn', 0):.1f} symbols/turn, "
                       f"net efficiency: {state.net_efficiency_trend[-1]:.3f}")
            
            return merged_rules
    
    def _assign_new_symbols(self, 
                          conversation_id: str, 
                          tokens: List[str], 
                          existing_rules: Dict[str, str]) -> Dict[str, str]:
        """Assign new symbols for tokens, avoiding conflicts."""
        new_assignments = {}
        used_symbols = set(existing_rules.keys())
        
        # Find available symbols for this conversation
        available = []
        for symbol in self.available_symbols:
            if symbol not in used_symbols:
                # Check if symbol is used in other active conversations
                if conversation_id not in self.global_symbol_usage[symbol]:
                    available.append(symbol)
        
        # Assign symbols to tokens (prioritize by potential value)
        for i, token in enumerate(tokens[:len(available)]):
            symbol = available[i]
            new_assignments[symbol] = token
            self.global_symbol_usage[symbol].add(conversation_id)
            logger.debug(f"Assigned new symbol {symbol}={token} for conversation {conversation_id}")
        
        if len(tokens) > len(available):
            logger.warning(f"Not enough symbols available for conversation {conversation_id}. "
                         f"Needed {len(tokens)}, had {len(available)}")
        
        return new_assignments
    
    def get_conversation_metrics(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive metrics for a conversation."""
        with self.lock:
            if conversation_id not in self.states:
                return None
            
            state = self.states[conversation_id]
            
            # Use the new comprehensive metrics function
            return state.get_comprehensive_metrics()
    
    def should_use_conversation_compression(self, conversation_id: str, min_net_efficiency: float = 0.01) -> bool:
        """Determine if conversation-aware compression should be used."""
        with self.lock:
            if conversation_id not in self.states:
                return True  # First turn, always try
            
            state = self.states[conversation_id]
            # Use the configured minimum net efficiency from config
            return state.should_continue_compression(min_net_efficiency=min_net_efficiency)
    
    def _maybe_cleanup(self):
        """Clean up old conversation states if needed."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        if len(self.states) <= self.max_conversations:
            self.last_cleanup = current_time
            return
        
        # Remove oldest conversations
        sorted_states = sorted(
            self.states.items(),
            key=lambda x: x[1].last_updated
        )
        
        to_remove = len(self.states) - self.max_conversations + 100  # Remove extra to avoid frequent cleanups
        
        for conversation_id, state in sorted_states[:to_remove]:
            # Clean up global symbol usage
            for symbol in state.compression_rules.keys():
                self.global_symbol_usage[symbol].discard(conversation_id)
                if not self.global_symbol_usage[symbol]:
                    del self.global_symbol_usage[symbol]
            
            del self.states[conversation_id]
            logger.debug(f"Cleaned up old conversation state: {conversation_id}")
        
        self.last_cleanup = current_time
        logger.info(f"Cleaned up {to_remove} old conversation states")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall manager statistics."""
        with self.lock:
            total_turns = sum(state.turn_number for state in self.states.values())
            active_conversations = len(self.states)
            
            return {
                'active_conversations': active_conversations,
                'total_turns_processed': total_turns,
                'symbols_in_use': len(self.global_symbol_usage),
                'avg_turns_per_conversation': total_turns / active_conversations if active_conversations > 0 else 0,
                'memory_usage_kb': len(str(self.states)) / 1024  # Rough estimate
            }


# Global conversation state manager
_conversation_manager: Optional[ConversationStateManager] = None
_manager_lock = threading.Lock()

def get_conversation_manager() -> ConversationStateManager:
    """Get the global conversation state manager (singleton)."""
    global _conversation_manager
    
    if _conversation_manager is None:
        with _manager_lock:
            if _conversation_manager is None:  # Double-check pattern
                _conversation_manager = ConversationStateManager()
                logger.info("Initialized global conversation state manager")
    
    return _conversation_manager


def reset_conversation_manager():
    """Reset the global conversation manager (for testing)."""
    global _conversation_manager
    with _manager_lock:
        _conversation_manager = None 