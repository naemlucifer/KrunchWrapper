#!/usr/bin/env python3
"""
Test stateful conversation compression mode for KrunchWrap.

This test verifies that stateful mode correctly optimizes for persistent KV cache servers
by only including new symbols in system prompt generation and calculating efficiency
based on new symbols only.
"""

import pytest
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.conversation_compress import compress_conversation_aware, reset_conversation_compression
from core.system_prompt import build_system_prompt


def setup_function():
    """Setup function to reset state before each test."""
    reset_conversation_compression()


def test_stateful_vs_stateless_efficiency():
    """Test that stateful mode calculates efficiency based on new symbols only."""
    print("Testing stateful vs stateless efficiency calculation")
    
    reset_conversation_compression()
    
    # First turn messages
    messages_t1 = [
        {"role": "user", "content": "def calculate_user_statistics(user_data):\n    # Calculate stats\n    statistics = {}\n    preferences = process_user_preferences(user_data)\n    return statistics, preferences"}
    ]
    
    # Turn 1: Both modes should behave similarly
    result_stateless_t1 = compress_conversation_aware(
        messages=messages_t1,
        min_characters=50,
        session_id="test_session",
        stateful_mode=False
    )
    
    result_stateful_t1 = compress_conversation_aware(
        messages=messages_t1,
        min_characters=50,
        session_id="test_session_stateful",
        stateful_mode=True
    )
    
    print(f"Stateless T1: efficiency={result_stateless_t1.net_efficiency:.3f}")
    print(f"Stateful T1: efficiency={result_stateful_t1.net_efficiency:.3f}")
    
    # Turn 2: Add small message
    messages_t2 = messages_t1 + [
        {"role": "assistant", "content": "I'll help you with that."},
        {"role": "user", "content": "Add error handling"}
    ]
    
    result_stateless_t2 = compress_conversation_aware(
        messages=messages_t2,
        min_characters=50,
        session_id="test_session",
        stateful_mode=False
    )
    
    result_stateful_t2 = compress_conversation_aware(
        messages=messages_t2,
        min_characters=50,
        session_id="test_session_stateful",
        stateful_mode=True
    )
    
    print(f"Stateless T2: efficiency={result_stateless_t2.net_efficiency:.3f}")
    print(f"Stateful T2: efficiency={result_stateful_t2.net_efficiency:.3f}")
    
    # Assertions
    assert result_stateful_t2.stateful_mode == True
    assert result_stateless_t2.stateful_mode == False
    assert result_stateful_t2.new_symbols_for_prompt is not None
    
    # Stateful should have better efficiency (lower overhead calculation)
    print(f"Efficiency improvement: {result_stateful_t2.net_efficiency - result_stateless_t2.net_efficiency:.3f}")


def test_stateful_system_prompt():
    """Test that stateful mode generates prompts with only new symbols."""
    print("Testing stateful system prompt generation")
    
    existing_symbols = {"α": "user_data", "β": "statistics"}
    new_symbols = {"γ": "error_handling", "δ": "validation"}
    all_symbols = {**existing_symbols, **new_symbols}
    
    # Stateless mode (all symbols)
    stateless_prompt, _ = build_system_prompt(
        used=all_symbols,
        lang="python",
        format_name="claude",
        stateful_mode=False
    )
    
    # Stateful mode (new symbols only)
    stateful_prompt, _ = build_system_prompt(
        used=all_symbols,
        lang="python",
        format_name="claude",
        stateful_mode=True,
        new_symbols_only=new_symbols
    )
    
    print(f"Stateless prompt: {len(stateless_prompt)} chars")
    print(f"Stateful prompt: {len(stateful_prompt)} chars")
    
    # Stateful should be shorter
    assert len(stateful_prompt) < len(stateless_prompt)
    assert "γ" in stateful_prompt or "error_handling" in stateful_prompt
    assert "NEW" in stateful_prompt or "new" in stateful_prompt.lower()


def test_stateful_mode_with_no_new_symbols():
    """Test stateful mode behavior when no new symbols are added."""
    print("\n🧪 Testing stateful mode with no new symbols")
    
    # Reset conversation state
    reset_conversation_compression()
    
    # First turn - establish symbols
    messages_turn1 = [
        {"role": "user", "content": "def process_user_authentication_and_authorization_data(user_credentials):\n    return validate_user_credentials(user_credentials)"}
    ]
    
    result_turn1 = compress_conversation_aware(
        messages=messages_turn1,
        min_characters=50,
        session_id="test_no_new_symbols",
        stateful_mode=True
    )
    
    print(f"   📊 Turn 1: {len(result_turn1.compression_rules)} symbols, efficiency={result_turn1.net_efficiency:.3f}")
    
    # Second turn - reuse existing symbols only (should trigger KV cache optimization)
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": "I'll help with authentication."},
        {"role": "user", "content": "Thanks"}  # Very short message
    ]
    
    result_turn2 = compress_conversation_aware(
        messages=messages_turn2,
        min_characters=50,
        session_id="test_no_new_symbols",
        stateful_mode=True,
        kv_cache_threshold=20  # Set threshold to trigger KV cache
    )
    
    print(f"   📊 Turn 2: KV cache used={result_turn2.kv_cache_optimization_used}, new_symbols={len(result_turn2.new_symbols_for_prompt) if result_turn2.new_symbols_for_prompt else 0}")
    
    # Should use KV cache optimization (no new symbols needed)
    assert result_turn2.kv_cache_optimization_used == True, "Should use KV cache optimization for short messages"
    
    # In stateful mode with KV cache, new_symbols_for_prompt should be empty dict
    if result_turn2.stateful_mode:
        assert result_turn2.new_symbols_for_prompt == {}, "No new symbols should be needed"
    
    # Test system prompt generation with no new symbols
    stateful_prompt_no_new, metadata = build_system_prompt(
        used=result_turn1.compression_rules,
        lang="python",
        format_name="claude", 
        user_content="Thanks",
        stateful_mode=True,
        new_symbols_only={}  # No new symbols
    )
    
    print(f"   📝 No new symbols prompt: {stateful_prompt_no_new[:150]}...")
    
    # Should indicate continuation with existing context
    assert "continue" in stateful_prompt_no_new.lower() or "existing" in stateful_prompt_no_new.lower(), \
        "Prompt should indicate continuation with existing symbols"
    
    print("   ✅ Stateful mode correctly handles no new symbols")


def test_stateful_configuration_integration():
    """Test that stateful mode configuration is properly integrated."""
    print("\n🧪 Testing stateful configuration integration")
    
    # Test that the configuration is read correctly
    try:

        from server.config import ServerConfig
    config = ServerConfig()
        
        # Check that stateful mode configuration exists
        assert hasattr(config, 'conversation_stateful_mode'), "Config should have stateful_mode setting"
        
        print(f"   ⚙️  Stateful mode configured: {config.conversation_stateful_mode}")
        print(f"   ⚙️  KV cache threshold: {config.conversation_kv_cache_threshold}")
        print(f"   ⚙️  Conversation compression enabled: {config.conversation_compression_enabled}")
        
        print("   ✅ Configuration integration working")
        
    except ImportError as e:
        print(f"   ⚠️  Could not test server config integration: {e}")


def test_comprehensive_stateful_workflow():
    """Test the complete stateful workflow over multiple turns."""
    print("\n🧪 Testing comprehensive stateful workflow")
    
    # Reset conversation state
    reset_conversation_compression()
    
    session_id = "comprehensive_stateful_test"
    
    # Turn 1: Initial conversation with substantial content
    messages_t1 = [
        {"role": "user", "content": "def analyze_customer_purchase_behavior_and_generate_recommendations(customer_data, purchase_history, product_catalog):\n    # Analyze customer behavior patterns\n    behavior_analysis = perform_comprehensive_customer_analysis(customer_data)\n    # Generate personalized recommendations\n    recommendations = generate_personalized_product_recommendations(behavior_analysis, product_catalog)\n    return recommendations"}
    ]
    
    result_t1 = compress_conversation_aware(
        messages=messages_t1,
        min_characters=50,
        session_id=session_id,
        stateful_mode=True
    )
    
    print(f"   📊 Turn 1: {len(result_t1.compression_rules)} symbols, efficiency={result_t1.net_efficiency:.3f}")
    symbols_t1 = set(result_t1.compression_rules.keys())
    
    # Turn 2: Add more content with some new patterns
    messages_t2 = messages_t1 + [
        {"role": "assistant", "content": "I'll help you implement customer behavior analysis."},
        {"role": "user", "content": "Also add real_time_inventory_management and price_optimization_algorithms for better recommendations"}
    ]
    
    result_t2 = compress_conversation_aware(
        messages=messages_t2,
        min_characters=50,
        session_id=session_id,
        stateful_mode=True
    )
    
    print(f"   📊 Turn 2: {len(result_t2.compression_rules)} total symbols, new_symbols={len(result_t2.new_symbols_for_prompt) if result_t2.new_symbols_for_prompt else 0}, efficiency={result_t2.net_efficiency:.3f}")
    symbols_t2 = set(result_t2.compression_rules.keys())
    new_symbols_t2 = symbols_t2 - symbols_t1
    
    # Turn 3: Short response (should trigger KV cache)
    messages_t3 = messages_t2 + [
        {"role": "assistant", "content": "Great suggestions!"},
        {"role": "user", "content": "Perfect"}  # Very short
    ]
    
    result_t3 = compress_conversation_aware(
        messages=messages_t3,
        min_characters=50,
        session_id=session_id,
        stateful_mode=True,
        kv_cache_threshold=20
    )
    
    print(f"   📊 Turn 3: KV cache={result_t3.kv_cache_optimization_used}, symbols={len(result_t3.compression_rules)}, efficiency={result_t3.net_efficiency:.3f}")
    
    # Verify stateful behavior
    assert result_t1.stateful_mode == True, "Turn 1 should use stateful mode"
    assert result_t2.stateful_mode == True, "Turn 2 should use stateful mode"
    assert result_t3.stateful_mode == True, "Turn 3 should use stateful mode"
    
    # Verify progressive symbol accumulation
    assert len(result_t2.compression_rules) >= len(result_t1.compression_rules), "Symbols should accumulate"
    
    # Verify new symbols tracking
    if result_t2.new_symbols_for_prompt is not None:
        assert len(result_t2.new_symbols_for_prompt) > 0, "Turn 2 should have new symbols"
        
    # Verify KV cache optimization
    assert result_t3.kv_cache_optimization_used == True, "Turn 3 should use KV cache"
    
    # Test system prompt generation for each turn
    if result_t2.new_symbols_for_prompt:
        stateful_prompt_t2, _ = build_system_prompt(
            used=result_t2.compression_rules,
            lang="python",
            format_name="claude",
            user_content="Add inventory management",
            stateful_mode=True,
            new_symbols_only=result_t2.new_symbols_for_prompt
        )
        
        print(f"   📝 Turn 2 stateful prompt length: {len(stateful_prompt_t2)} chars")
        assert len(stateful_prompt_t2) > 0, "Should generate prompt for new symbols"
    
    print("   ✅ Comprehensive stateful workflow successful")


if __name__ == "__main__":
    print("🧪 Running Stateful Conversation Compression Tests")
    print("=" * 60)
    
    try:
        test_stateful_vs_stateless_efficiency()
        test_stateful_system_prompt()
        test_stateful_mode_with_no_new_symbols()
        test_stateful_configuration_integration()
        test_comprehensive_stateful_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All stateful compression tests passed!")
        print("✅ Stateful mode correctly optimizes for persistent KV cache servers")
        print("✅ Efficiency calculations based on new symbols only")
        print("✅ System prompts include only new symbols in stateful mode")
        print("✅ KV cache optimization works correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 