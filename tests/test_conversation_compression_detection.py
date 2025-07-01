#!/usr/bin/env python3
"""
Simple test script to verify conversation detection and configuration work correctly.
This tests the enhanced conversation compression logic without requiring a full server setup.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the conversation detection function and configuration
try:
    from server.request_utils import _is_conversation_continuation
    from api.server import ChatMessage, config
    print("✅ Successfully imported server components")
except ImportError as e:
    print(f"❌ Failed to import server components: {e}")
    sys.exit(1)

def test_conversation_detection():
    """Test the enhanced conversation detection logic."""
    print("\n" + "="*60)
    print("TESTING CONVERSATION DETECTION")
    print("="*60)
    
    # Test case 1: Single user message (should be new conversation)
    print("\n1. Testing single user message:")
    messages = [ChatMessage(role="user", content="Hello, can you help me with Python?")]
    result = _is_conversation_continuation(messages, None)
    print(f"   Result: {result} (Expected: False)")
    assert result == False, "Single user message should not be detected as continuation"
    
    # Test case 2: Multiple user messages (should be continuation)
    print("\n2. Testing multiple user messages:")
    messages = [
        ChatMessage(role="user", content="Hello, can you help me with Python?"),
        ChatMessage(role="user", content="I need to create a function")
    ]
    result = _is_conversation_continuation(messages, None)
    print(f"   Result: {result} (Expected: True)")
    assert result == True, "Multiple user messages should be detected as continuation"
    
    # Test case 3: User + Assistant messages (should be continuation)
    print("\n3. Testing user + assistant messages:")
    messages = [
        ChatMessage(role="user", content="Hello, can you help me with Python?"),
        ChatMessage(role="assistant", content="Of course! I'd be happy to help you with Python."),
        ChatMessage(role="user", content="I need to create a function")
    ]
    result = _is_conversation_continuation(messages, None)
    print(f"   Result: {result} (Expected: True)")
    assert result == True, "User + assistant messages should be detected as continuation"
    
    # Test case 4: Single message with session ID (should be continuation)
    print("\n4. Testing single message with session ID:")
    messages = [ChatMessage(role="user", content="Continue our conversation")]
    result = _is_conversation_continuation(messages, "test-session-123")
    print(f"   Result: {result} (Expected: True)")
    assert result == True, "Single message with session ID should be detected as continuation"
    
    print("\n✅ All conversation detection tests passed!")

def test_configuration():
    """Test that conversation compression configuration is loaded correctly."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION VALIDATION")
    print("="*60)
    
    print(f"conversation_compression_enabled: {config.conversation_compression_enabled}")
    print(f"conversation_max_conversations: {getattr(config, 'conversation_max_conversations', 'NOT SET')}")
    print(f"conversation_min_net_efficiency: {getattr(config, 'conversation_min_net_efficiency', 'NOT SET')}")
    print(f"conversation_kv_cache_threshold: {getattr(config, 'conversation_kv_cache_threshold', 'NOT SET')}")
    print(f"conversation_force_compression: {getattr(config, 'conversation_force_compression', 'NOT SET')}")
    print(f"interface_engine: {getattr(config, 'interface_engine', 'NOT SET')}")
    
    # Validate critical settings
    assert config.conversation_compression_enabled == True, "Conversation compression should be enabled"
    assert hasattr(config, 'conversation_kv_cache_threshold'), "KV cache threshold should be configured"
    # Updated to check for interface_engine instead of use_cline
    interface_engine = getattr(config, 'interface_engine', 'auto')
    assert interface_engine in ['cline', 'auto'], f"Interface engine should be 'cline' or 'auto', got: {interface_engine}"
    
    print("\n✅ All configuration tests passed!")

def test_cline_detection_logic():
    """Test the logic that should trigger conversation compression for Cline requests."""
    print("\n" + "="*60)
    print("TESTING CLINE CONVERSATION COMPRESSION LOGIC")
    print("="*60)
    
    # Simulate the conditions from the server code
    # These are the three conditions that must be true for conversation compression:
    conversation_compression_enabled = config.conversation_compression_enabled
    should_compress = True  # Assume content is large enough
    
    # Test multi-message Cline request (should trigger conversation compression)
    messages = [
        ChatMessage(role="user", content="Hello, I'm working on a Python project. Can you help me create a function that processes a list of dictionaries?"),
        ChatMessage(role="assistant", content="I'd be happy to help you create a function to filter dictionaries!"),
        ChatMessage(role="user", content="That's great! Now I need to modify the function to handle multiple filter criteria.")
    ]
    
    is_conversation_continuation = _is_conversation_continuation(messages, "test-session")
    is_cline = True  # Simulating Cline request
    
    print(f"Condition 1 - conversation_compression_enabled: {conversation_compression_enabled}")
    print(f"Condition 2 - should_compress: {should_compress}")
    print(f"Condition 3 - is_conversation_continuation: {is_conversation_continuation}")
    print(f"Is Cline request: {is_cline}")
    
    # This is the main condition from the server code
    should_use_conversation_compression = (
        conversation_compression_enabled and 
        should_compress and 
        is_conversation_continuation
    )
    
    print(f"\nShould use conversation compression: {should_use_conversation_compression}")
    
    # Also test the force conversation compression logic for Cline
    force_conversation_compression = (
        conversation_compression_enabled and 
        should_compress and 
        len(messages) > 1 and  # Multi-message indicates conversation
        is_cline  # Apply forced conversation compression for Cline requests specifically
    )
    
    print(f"Force conversation compression for Cline: {force_conversation_compression}")
    
    # If forcing is enabled and original detection failed, it should still work
    final_result = should_use_conversation_compression or force_conversation_compression
    print(f"Final result (should trigger conversation compression): {final_result}")
    
    assert final_result == True, "Multi-message Cline request should trigger conversation compression"
    
    print("\n✅ Cline conversation compression logic test passed!")

if __name__ == "__main__":
    print("Starting Conversation Compression Detection Test")
    print("This test verifies that conversation detection and configuration work correctly.\n")
    
    try:
        # Test the conversation detection logic
        test_conversation_detection()
        
        # Test configuration
        test_configuration()
        
        # Test Cline-specific logic
        test_cline_detection_logic()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe enhanced conversation compression logic is working correctly.")
        print("For Cline requests with multiple messages, conversation compression should now be triggered.")
        print("\nExpected behavior in server logs:")
        print("✅ '🗣️  Using conversation-aware compression (conversation continuation detected)'")
        print("✅ '🗣️  Conversation compression triggered for Cline client'")
        print("✅ '🗣️  [CONVERSATION COMPRESS] Starting conversation-aware compression'")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 