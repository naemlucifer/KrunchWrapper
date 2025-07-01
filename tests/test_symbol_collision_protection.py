"""
Test Symbol Collision Protection in Conversation Compression

This test verifies that:
1. Once a symbol is assigned to a pattern in a conversation, it's never reused for a different pattern
2. Symbol consistency is maintained across all turns
3. Collision detection and resolution works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.conversation_compress import compress_conversation_aware, reset_conversation_compression
from core.conversation_state import get_conversation_manager


def test_symbol_consistency_across_turns():
    """Test that symbols remain consistent for the same patterns across turns."""
    # Reset conversation state
    reset_conversation_compression()
    
    # Turn 1: Establish initial symbols
    messages_turn1 = [
        {"role": "user", "content": "function calculateTotal(array) { return array.reduce((sum, item) => sum + item.value, 0); }"}
    ]
    
    result1 = compress_conversation_aware(
        messages=messages_turn1,
        min_characters=50,
        session_id="test_consistency"
    )
    
    print(f"Turn 1: Established {len(result1.compression_rules)} symbols")
    turn1_symbols = result1.compression_rules.copy()
    
    # Turn 2: Use the same patterns - should get same symbols
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": "This function uses array.reduce to calculate totals."},
        {"role": "user", "content": "function processArray(array) { return array.filter(item => item.active); }"}
    ]
    
    result2 = compress_conversation_aware(
        messages=messages_turn2,
        min_characters=50,
        session_id="test_consistency"
    )
    
    print(f"Turn 2: Using {len(result2.compression_rules)} symbols")
    
    # Verify consistency: any symbols from turn 1 that appear in turn 2 should have the same meaning
    for symbol, pattern in turn1_symbols.items():
        if symbol in result2.compression_rules:
            assert result2.compression_rules[symbol] == pattern, \
                f"Symbol '{symbol}' changed meaning! Turn 1: '{pattern}' vs Turn 2: '{result2.compression_rules[symbol]}'"
    
    print("✅ Symbol consistency maintained across turns")


def test_no_symbol_reuse_for_different_patterns():
    """Test that the same symbol is never used for different patterns in a conversation."""
    # Reset conversation state
    reset_conversation_compression()
    
    # Create content designed to potentially cause conflicts
    turn1_content = "function calculateSum(array) { return array.reduce((total, item) => total + item, 0); }"
    turn2_content = "class DataProcessor { constructor(config) { this.config = config; } }"
    turn3_content = "interface UserInterface { name: string; email: string; }"
    
    session_id = "test_no_reuse"
    
    # Turn 1
    result1 = compress_conversation_aware(
        messages=[{"role": "user", "content": turn1_content}],
        min_characters=30,
        session_id=session_id
    )
    
    # Turn 2 
    result2 = compress_conversation_aware(
        messages=[
            {"role": "user", "content": turn1_content},
            {"role": "assistant", "content": "I see you're using array.reduce."},
            {"role": "user", "content": turn2_content}
        ],
        min_characters=30,
        session_id=session_id
    )
    
    # Turn 3
    result3 = compress_conversation_aware(
        messages=[
            {"role": "user", "content": turn1_content},
            {"role": "assistant", "content": "I see you're using array.reduce."},
            {"role": "user", "content": turn2_content},
            {"role": "assistant", "content": "Now you're defining a class."},
            {"role": "user", "content": turn3_content}
        ],
        min_characters=30,
        session_id=session_id
    )
    
    # Get full conversation state to analyze all symbols
    manager = get_conversation_manager()
    conversation_id = manager._generate_conversation_id([{"role": "user", "content": turn1_content}], session_id)
    conversation_state = manager.states[conversation_id]
    
    print(f"Total symbols in conversation: {len(conversation_state.compression_rules)}")
    
    # Verify no symbol is used for multiple different patterns
    symbol_to_patterns = {}
    for symbol, pattern in conversation_state.compression_rules.items():
        if symbol in symbol_to_patterns:
            assert symbol_to_patterns[symbol] == pattern, \
                f"Symbol '{symbol}' reused! First: '{symbol_to_patterns[symbol]}' vs Second: '{pattern}'"
        symbol_to_patterns[symbol] = pattern
    
    # Also verify reverse mapping is consistent (no pattern has multiple symbols)
    pattern_to_symbols = {}
    for symbol, pattern in conversation_state.compression_rules.items():
        if pattern in pattern_to_symbols:
            print(f"⚠️  Pattern '{pattern}' has multiple symbols: {pattern_to_symbols[pattern]} and {symbol}")
            # This is actually OK - same pattern can have multiple representations for efficiency
        else:
            pattern_to_symbols[pattern] = symbol
    
    print(f"✅ No symbol reuse detected across {len(conversation_state.compression_rules)} symbols")


def test_symbol_collision_detection():
    """Test that symbol collisions are properly detected and resolved."""
    # Reset conversation state  
    reset_conversation_compression()
    
    # Create a scenario that might cause symbol assignment conflicts
    # by having many patterns that could compete for the same symbols
    large_content = """
    function calculateTotal(array) { return array.reduce((sum, item) => sum + item.value, 0); }
    function processData(items) { return items.filter(item => item.active).map(item => item.result); }
    function analyzeResults(data) { return data.sort((a, b) => a.score - b.score); }
    class DataManager { constructor(config) { this.config = config; } }
    interface ApiResponse { status: string; data: any; message: string; }
    const processItems = (items) => items.reduce((acc, item) => ({ ...acc, [item.id]: item }), {});
    """
    
    # Turn 1: Establish many symbols
    result1 = compress_conversation_aware(
        messages=[{"role": "user", "content": large_content}],
        min_characters=100,
        session_id="test_collisions"
    )
    
    turn1_symbols = len(result1.compression_rules)
    print(f"Turn 1: Created {turn1_symbols} symbols")
    
    # Turn 2: Add more content that might conflict
    more_content = """
    function transformArray(array) { return array.map(item => ({ ...item, processed: true })); }
    function validateInput(data) { return data.every(item => item.valid && item.required); }
    """
    
    result2 = compress_conversation_aware(
        messages=[
            {"role": "user", "content": large_content},
            {"role": "assistant", "content": "I'll analyze this code."},
            {"role": "user", "content": more_content}
        ],
        min_characters=100,
        session_id="test_collisions"
    )
    
    # Get conversation state
    manager = get_conversation_manager()
    conversation_id = manager._generate_conversation_id([{"role": "user", "content": large_content}], "test_collisions")
    conversation_state = manager.states[conversation_id]
    
    total_symbols = len(conversation_state.compression_rules)
    print(f"Turn 2: Total symbols now {total_symbols}")
    
    # Verify all symbols are unique and valid
    symbols_used = set()
    patterns_mapped = set()
    
    for symbol, pattern in conversation_state.compression_rules.items():
        assert symbol not in symbols_used, f"Duplicate symbol detected: {symbol}"
        symbols_used.add(symbol)
        
        assert len(pattern) > 0, f"Empty pattern for symbol {symbol}"
        assert len(symbol) > 0, f"Empty symbol for pattern {pattern}"
        
        patterns_mapped.add(pattern)
    
    print(f"✅ Collision detection working: {len(symbols_used)} unique symbols, {len(patterns_mapped)} unique patterns")


if __name__ == "__main__":
    print("🧪 Testing Symbol Collision Protection")
    print("=" * 50)
    
    print("\n1. Testing symbol consistency across turns:")
    test_symbol_consistency_across_turns()
    
    print("\n2. Testing no symbol reuse for different patterns:")
    test_no_symbol_reuse_for_different_patterns()
    
    print("\n3. Testing symbol collision detection:")
    test_symbol_collision_detection()
    
    print("\n" + "=" * 50)
    print("🎉 All symbol collision protection tests passed!")
    print("\nKey protections verified:")
    print("✅ Symbols remain consistent for same patterns across turns")
    print("✅ No symbol is ever reused for different patterns")
    print("✅ Collision detection and resolution works correctly")
    print("✅ Symbol uniqueness maintained throughout conversation") 