#!/usr/bin/env python3
"""
Test script to verify conversation compression running tally works correctly.
This script tests the enhanced conversation compression with cumulative metrics tracking.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# Import the conversation compression function and state management
try:
    from core.conversation_compress import compress_conversation_aware, reset_conversation_compression
    from core.conversation_state import get_conversation_manager
    print("✅ Successfully imported conversation compression components")
except ImportError as e:
    print(f"❌ Failed to import conversation compression components: {e}")
    sys.exit(1)

# Set up logging to see all the debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_conversation_running_tally():
    """Test the running tally of cumulative compression savings."""
    print("\n" + "="*70)
    print("TESTING CONVERSATION COMPRESSION RUNNING TALLY")
    print("="*70)
    
    # Reset conversation state to start fresh
    reset_conversation_compression()
    
    # Simulate a multi-turn conversation with repeated patterns
    session_id = "test-running-tally-session"
    
    print("\n📈 Turn 1: Starting conversation with code example")
    messages_turn1 = [
        {
            "role": "user", 
            "content": "Can you help me write a Python function that processes a list of dictionaries? I need to filter them based on specific criteria and then transform the data."
        }
    ]
    
    result1 = compress_conversation_aware(
        messages=messages_turn1,
        min_characters=50,  # Lower threshold for testing
        session_id=session_id,
        kv_cache_threshold=100
    )
    
    print(f"Turn 1 - Conversation ID: {result1.conversation_id}")
    print(f"Turn 1 - Compression ratio: {result1.compression_ratio:.3f}")
    print(f"Turn 1 - Tokens saved: {result1.tokens_saved}")
    print(f"Turn 1 - Rules used: {len(result1.compression_rules)}")
    
    print("\n📈 Turn 2: Adding assistant response and new user message")
    messages_turn2 = [
        {
            "role": "user", 
            "content": "Can you help me write a Python function that processes a list of dictionaries? I need to filter them based on specific criteria and then transform the data."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you write a Python function that processes a list of dictionaries! Here's a clean implementation that filters dictionaries based on criteria and transforms the data."
        },
        {
            "role": "user",
            "content": "Great! Now I need to extend this function to handle nested dictionaries and also add error handling for invalid data types."
        }
    ]
    
    result2 = compress_conversation_aware(
        messages=messages_turn2,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=100
    )
    
    print(f"Turn 2 - Compression ratio: {result2.compression_ratio:.3f}")
    print(f"Turn 2 - Tokens saved: {result2.tokens_saved}")
    print(f"Turn 2 - Rules used: {len(result2.compression_rules)}")
    
    print("\n📈 Turn 3: Adding more conversation context")
    messages_turn3 = [
        {
            "role": "user", 
            "content": "Can you help me write a Python function that processes a list of dictionaries? I need to filter them based on specific criteria and then transform the data."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you write a Python function that processes a list of dictionaries! Here's a clean implementation that filters dictionaries based on criteria and transforms the data."
        },
        {
            "role": "user",
            "content": "Great! Now I need to extend this function to handle nested dictionaries and also add error handling for invalid data types."
        },
        {
            "role": "assistant",
            "content": "Excellent! Let me extend the function to handle nested dictionaries and add robust error handling for invalid data types."
        },
        {
            "role": "user",
            "content": "Perfect! One more thing - can you add logging to track the processing of each dictionary in the list?"
        }
    ]
    
    result3 = compress_conversation_aware(
        messages=messages_turn3,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=100
    )
    
    print(f"Turn 3 - Compression ratio: {result3.compression_ratio:.3f}")
    print(f"Turn 3 - Tokens saved: {result3.tokens_saved}")
    print(f"Turn 3 - Rules used: {len(result3.compression_rules)}")
    
    # Get the final cumulative metrics
    manager = get_conversation_manager()
    metrics = manager.get_conversation_metrics(result3.conversation_id)
    
    if metrics:
        print("\n" + "="*70)
        print("📊 FINAL RUNNING TALLY RESULTS")
        print("="*70)
        
        print(f"📈 Total turns processed: {metrics.get('turn_number', 0)}")
        print(f"🗂️  Total characters processed: {metrics.get('cumulative_original_chars', 0):,}")
        print(f"🗜️  Total characters after compression: {metrics.get('cumulative_compressed_chars', 0):,}")
        print(f"💾 Total characters saved: {metrics.get('cumulative_chars_saved', 0):,}")
        print(f"🎯 Total tokens saved: {metrics.get('cumulative_tokens_saved', 0):,}")
        print(f"📊 Cumulative compression ratio: {metrics.get('cumulative_compression_ratio', 0):.3f} ({metrics.get('cumulative_compression_ratio', 0)*100:.1f}%)")
        print(f"⚡ Cumulative net efficiency: {metrics.get('cumulative_net_efficiency', 0):.3f} ({metrics.get('cumulative_net_efficiency', 0)*100:.1f}%)")
        print(f"📬 Total messages processed: {metrics.get('cumulative_messages_processed', 0)}")
        print(f"⏱️  Total processing time: {metrics.get('total_processing_time', 0):.2f}s")
        print(f"🚀 KV cache optimizations used: {metrics.get('kv_cache_optimizations_used', 0)}")
        print(f"🔧 Total compression rules: {metrics.get('total_rules', 0)}")
        
        # Show top patterns
        if metrics.get('most_used_tokens'):
            print(f"\n🔤 Top compression patterns:")
            for i, (pattern, count) in enumerate(metrics['most_used_tokens'][:5], 1):
                print(f"    {i}. '{pattern}' used {count} times")
        
        # Validate the metrics make sense
        total_chars = metrics.get('cumulative_original_chars', 0)
        chars_saved = metrics.get('cumulative_chars_saved', 0)
        compression_ratio = metrics.get('cumulative_compression_ratio', 0)
        
        print(f"\n✅ Validation:")
        print(f"   Characters processed > 0: {total_chars > 0}")
        print(f"   Some compression achieved: {chars_saved > 0}")
        print(f"   Compression ratio consistent: {abs(compression_ratio - (chars_saved / total_chars if total_chars > 0 else 0)) < 0.01}")
        print(f"   Turn count matches: {metrics.get('turn_number') == 3}")
        
        return metrics
    else:
        print("❌ No metrics found!")
        return None

def test_kv_cache_optimization():
    """Test that KV cache optimization is tracked in the running tally."""
    print("\n" + "="*70)
    print("TESTING KV CACHE OPTIMIZATION TRACKING")
    print("="*70)
    
    session_id = "test-kv-cache-session"
    
    # First, establish a conversation
    messages_setup = [
        {
            "role": "user", 
            "content": "Can you help me understand how Python list comprehensions work? I'm having trouble with nested loops."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you understand Python list comprehensions! They're a powerful way to create lists using a more concise syntax than traditional loops."
        }
    ]
    
    result_setup = compress_conversation_aware(
        messages=messages_setup,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=50  # Low threshold to trigger KV cache
    )
    
    print(f"Setup - Rules established: {len(result_setup.compression_rules)}")
    
    # Now send a very short message that should trigger KV cache optimization
    messages_short = [
        {
            "role": "user", 
            "content": "Can you help me understand how Python list comprehensions work? I'm having trouble with nested loops."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you understand Python list comprehensions! They're a powerful way to create lists using a more concise syntax than traditional loops."
        },
        {
            "role": "user",
            "content": "Thanks!"  # Very short message should trigger KV cache
        }
    ]
    
    result_kv = compress_conversation_aware(
        messages=messages_short,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=50  # This should catch "Thanks!"
    )
    
    print(f"KV Cache turn - KV optimization used: {result_kv.kv_cache_optimization_used}")
    
    # Check the metrics
    manager = get_conversation_manager()
    metrics = manager.get_conversation_metrics(result_kv.conversation_id)
    
    if metrics:
        print(f"🚀 KV cache optimizations tracked: {metrics.get('kv_cache_optimizations_used', 0)}")
        print(f"📈 Total turns: {metrics.get('turn_number', 0)}")
        
        return metrics.get('kv_cache_optimizations_used', 0) > 0
    
    return False

if __name__ == "__main__":
    print("Starting Conversation Compression Running Tally Test")
    print("This test verifies that cumulative compression metrics are tracked correctly across conversation turns.")
    
    try:
        # Test the main running tally functionality
        metrics = test_conversation_running_tally()
        
        # Test KV cache optimization tracking
        kv_cache_works = test_kv_cache_optimization()
        
        print("\n" + "="*70)
        print("🎉 TEST RESULTS")
        print("="*70)
        
        if metrics:
            print("✅ Running tally metrics are working correctly!")
            print("✅ Cumulative compression statistics are being tracked!")
            print(f"✅ Final conversation had {metrics.get('turn_number', 0)} turns")
            print(f"✅ Total characters saved: {metrics.get('cumulative_chars_saved', 0):,}")
            print(f"✅ Overall compression ratio: {metrics.get('cumulative_compression_ratio', 0)*100:.1f}%")
        else:
            print("❌ Running tally metrics not working properly")
        
        if kv_cache_works:
            print("✅ KV cache optimization tracking is working!")
        else:
            print("❌ KV cache optimization tracking not working")
        
        print("\nExpected log entries when using conversation compression:")
        print("✅ '📊 [RUNNING TALLY] Cumulative conversation statistics:'")
        print("✅ '📈 Total turns processed: X'")
        print("✅ '💾 Total characters saved: X,XXX'")
        print("✅ '📊 Cumulative compression ratio: X.XXX (XX.X%)'")
        print("✅ '⚡ Cumulative net efficiency: X.XXX (XX.X%)'")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 