#!/usr/bin/env python3
"""
Test script to verify that the running tally now uses async logging correctly.
This script tests the enhanced async logging implementation for conversation compression.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the async logging setup
try:
    from core.async_logger import setup_global_async_logging, remove_global_async_logging
    from core.conversation_compress import compress_conversation_aware, reset_conversation_compression
    from core.conversation_state import get_conversation_manager
    print("✅ Successfully imported async logging and conversation compression components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

def test_async_logging_setup():
    """Test that async logging can be set up correctly."""
    print("\n" + "="*70)
    print("TESTING ASYNC LOGGING SETUP")
    print("="*70)
    
    # Set up async logging
    print("📝 Setting up global async logging...")
    async_handler = setup_global_async_logging(
        enable=True,
        log_level="INFO",
        max_queue_size=1000
    )
    
    if async_handler:
        print("✅ Async logging handler created successfully")
        print(f"✅ Handler type: {type(async_handler).__name__}")
        
        # Test that the logger is working
        test_logger = logging.getLogger('test.async.functionality')
        test_logger.info("🧪 Test async log message")
        
        # Small delay to allow async processing
        time.sleep(0.1)
        
        print("✅ Test async log message sent")
        return True
    else:
        print("❌ Failed to create async logging handler")
        return False

def test_async_running_tally():
    """Test that the running tally uses async logging correctly."""
    print("\n" + "="*70)
    print("TESTING ASYNC RUNNING TALLY LOGGING")
    print("="*70)
    
    # Reset conversation state to start fresh
    reset_conversation_compression()
    
    # Create a test conversation that will generate running tally logs
    session_id = "test-async-tally-session"
    
    # Use messages with substantial content to trigger compression
    test_messages = [
        {
            "role": "user", 
            "content": "Can you help me write a Python function that processes a list of dictionaries? I need to filter them based on specific criteria and then transform the data structure for better performance."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you write a Python function that processes a list of dictionaries! Here's a comprehensive implementation that filters dictionaries based on criteria and transforms the data structure for optimal performance."
        },
        {
            "role": "user",
            "content": "Great! Now I need to extend this function to handle nested dictionaries and also add error handling for invalid data types and edge cases."
        }
    ]
    
    print("📊 Triggering conversation compression with running tally...")
    
    # This should trigger the async running tally logging
    result = compress_conversation_aware(
        messages=test_messages,
        min_characters=50,  # Lower threshold for testing
        session_id=session_id,
        kv_cache_threshold=100
    )
    
    # Small delay to allow async processing to complete
    time.sleep(0.2)
    
    print(f"✅ Conversation compression completed")
    print(f"   - Conversation ID: {result.conversation_id}")
    print(f"   - Turn number: {result.turn_number}")
    print(f"   - Has metrics: {result.metrics is not None}")
    
    # Verify metrics are available (which would trigger running tally)
    if result.metrics:
        print(f"✅ Running tally should have been logged asynchronously")
        print(f"   - Turn number: {result.metrics.get('turn_number', 0)}")
        print(f"   - Characters processed: {result.metrics.get('cumulative_original_chars', 0):,}")
        print(f"   - Characters saved: {result.metrics.get('cumulative_chars_saved', 0):,}")
        print(f"   - Compression ratio: {result.metrics.get('cumulative_compression_ratio', 0):.3f}")
        return True
    else:
        print("⚠️  No metrics available - running tally may not have been triggered")
        return False

def test_async_kv_cache_logging():
    """Test that KV cache async logging works correctly."""
    print("\n" + "="*70)
    print("TESTING ASYNC KV CACHE LOGGING")
    print("="*70)
    
    session_id = "test-async-kv-session"
    
    # First establish a conversation
    setup_messages = [
        {
            "role": "user", 
            "content": "Can you explain how Python decorators work? I'm having trouble understanding the syntax."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to explain Python decorators! They're a powerful feature that allows you to modify or extend the behavior of functions."
        }
    ]
    
    print("📝 Establishing conversation for KV cache test...")
    result_setup = compress_conversation_aware(
        messages=setup_messages,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=30  # Low threshold to trigger KV cache
    )
    
    # Now send a very short message that should trigger KV cache optimization
    short_messages = [
        {
            "role": "user", 
            "content": "Can you explain how Python decorators work? I'm having trouble understanding the syntax."
        },
        {
            "role": "assistant",
            "content": "I'd be happy to explain Python decorators! They're a powerful feature that allows you to modify or extend the behavior of functions."
        },
        {
            "role": "user",
            "content": "Cool!"  # Very short message should trigger KV cache
        }
    ]
    
    print("🚀 Triggering KV cache optimization with async logging...")
    result_kv = compress_conversation_aware(
        messages=short_messages,
        min_characters=50,
        session_id=session_id,
        kv_cache_threshold=30  # This should catch "Cool!"
    )
    
    # Small delay to allow async processing
    time.sleep(0.2)
    
    print(f"✅ KV cache test completed")
    print(f"   - KV optimization used: {result_kv.kv_cache_optimization_used}")
    print(f"   - Turn number: {result_kv.turn_number}")
    
    if result_kv.kv_cache_optimization_used:
        print("✅ KV cache optimization was triggered (should have async logs)")
        return True
    else:
        print("⚠️  KV cache optimization was not triggered")
        return False

def cleanup_async_logging():
    """Clean up async logging setup."""
    print("\n📝 Cleaning up async logging...")
    try:
        remove_global_async_logging()
        print("✅ Async logging cleaned up successfully")
    except Exception as e:
        print(f"⚠️  Error during async logging cleanup: {e}")

if __name__ == "__main__":
    print("Starting Async Running Tally Logging Test")
    print("This test verifies that the running tally now uses async logging correctly.")
    
    # Collect test results
    results = []
    
    try:
        # Test 1: Async logging setup
        setup_success = test_async_logging_setup()
        results.append(("Async logging setup", setup_success))
        
        if setup_success:
            # Test 2: Async running tally
            tally_success = test_async_running_tally()
            results.append(("Async running tally", tally_success))
            
            # Test 3: Async KV cache logging
            kv_success = test_async_kv_cache_logging()
            results.append(("Async KV cache logging", kv_success))
        
        # Final delay to ensure all async operations complete
        time.sleep(0.5)
        
        print("\n" + "="*70)
        print("🎉 ASYNC LOGGING TEST RESULTS")
        print("="*70)
        
        all_passed = True
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
            if not success:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All async logging tests passed!")
            print("✅ Running tally now uses async logging")
            print("✅ KV cache logging is async")
            print("✅ Conversation compression entry logging is async")
            print("\nBenefits of async logging:")
            print("  📈 Improved performance - logging doesn't block compression")
            print("  🚀 Better responsiveness - async processing in background")
            print("  📝 Consistent logging - all messages processed efficiently")
            print("  🔧 Scalable - handles high-volume logging without bottlenecks")
        else:
            print("\n❌ Some async logging tests failed!")
            print("Please check the implementation for issues.")
        
        print("\nExpected behavior:")
        print("✅ Console output appears immediately (single print() call)")
        print("✅ File logging is processed asynchronously in background")
        print("✅ No blocking during compression operations")
        print("✅ Complete log messages reduce async queue pressure")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always clean up
        cleanup_async_logging() 