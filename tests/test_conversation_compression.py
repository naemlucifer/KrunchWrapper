#!/usr/bin/env python3
"""
Test script for conversation-aware compression.
Demonstrates how compression maintains consistency and improves efficiency across turns.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_compress import (
    compress_conversation_aware, 
    get_conversation_compression_stats,
    reset_conversation_compression
)

def test_multi_turn_conversation():
    """Test a realistic multi-turn conversation about Python programming."""
    
    print("🧪 Testing Multi-Turn Conversation Compression")
    print("=" * 60)
    
    # Reset state for clean test
    reset_conversation_compression()
    
    # Test both with and without session IDs
    test_session_id = "user123_chat456"
    
    # Simulate a conversation about Python programming
    conversation_turns = [
        # Turn 1: Initial question
        [
            {"role": "user", "content": "How do I create a function in Python that takes parameters and returns a value?"}
        ],
        
        # Turn 2: Follow-up with previous context
        [
            {"role": "user", "content": "How do I create a function in Python that takes parameters and returns a value?"},
            {"role": "assistant", "content": "To create a function in Python, use the 'def' keyword followed by the function name and parameters in parentheses. Here's the basic syntax:\n\n```python\ndef function_name(parameter1, parameter2):\n    # function body\n    result = parameter1 + parameter2\n    return result\n```\n\nThe 'return' statement is used to return a value from the function."},
            {"role": "user", "content": "What about default parameters? How do I create a function with optional parameters?"}
        ],
        
        # Turn 3: Building on the conversation
        [
            {"role": "user", "content": "How do I create a function in Python that takes parameters and returns a value?"},
            {"role": "assistant", "content": "To create a function in Python, use the 'def' keyword followed by the function name and parameters in parentheses. Here's the basic syntax:\n\n```python\ndef function_name(parameter1, parameter2):\n    # function body\n    result = parameter1 + parameter2\n    return result\n```\n\nThe 'return' statement is used to return a value from the function."},
            {"role": "user", "content": "What about default parameters? How do I create a function with optional parameters?"},
            {"role": "assistant", "content": "You can create functions with default parameters by assigning default values in the function definition:\n\n```python\ndef greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'\n\n# Usage:\nprint(greet('Alice'))  # Output: Hello, Alice!\nprint(greet('Bob', 'Hi'))  # Output: Hi, Bob!\n```\n\nParameters with default values must come after required parameters."},
            {"role": "user", "content": "Can you show me how to create a class with methods in Python?"}
        ],
        
        # Turn 4: Even longer conversation
        [
            {"role": "user", "content": "How do I create a function in Python that takes parameters and returns a value?"},
            {"role": "assistant", "content": "To create a function in Python, use the 'def' keyword followed by the function name and parameters in parentheses. Here's the basic syntax:\n\n```python\ndef function_name(parameter1, parameter2):\n    # function body\n    result = parameter1 + parameter2\n    return result\n```\n\nThe 'return' statement is used to return a value from the function."},
            {"role": "user", "content": "What about default parameters? How do I create a function with optional parameters?"},
            {"role": "assistant", "content": "You can create functions with default parameters by assigning default values in the function definition:\n\n```python\ndef greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'\n\n# Usage:\nprint(greet('Alice'))  # Output: Hello, Alice!\nprint(greet('Bob', 'Hi'))  # Output: Hi, Bob!\n```\n\nParameters with default values must come after required parameters."},
            {"role": "user", "content": "Can you show me how to create a class with methods in Python?"},
            {"role": "assistant", "content": "Certainly! Here's how to create a class with methods in Python:\n\n```python\nclass Calculator:\n    def __init__(self, name):\n        self.name = name\n    \n    def add(self, a, b):\n        return a + b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def get_info(self):\n        return f'Calculator: {self.name}'\n\n# Usage:\ncalc = Calculator('MyCalc')\nresult = calc.add(5, 3)\nprint(calc.get_info())\n```\n\nThe `__init__` method is the constructor that runs when creating an instance."},
            {"role": "user", "content": "How do I handle exceptions and errors in Python functions?"}
        ]
    ]
    
    results = []
    
    for turn_num, messages in enumerate(conversation_turns, 1):
        print(f"\n📝 Turn {turn_num}: Processing {len(messages)} messages")
        print(f"   Total content: {sum(len(msg['content']) for msg in messages)} chars")
        
        # Process with conversation-aware compression (with session ID)
        result = compress_conversation_aware(messages, min_characters=100, session_id=test_session_id)
        results.append(result)
        
        # Show compression results
        print(f"   Conversation ID: {result.conversation_id}")
        print(f"   Compression ratio: {result.compression_ratio:.3f}")
        print(f"   Net efficiency: {result.net_efficiency:.3f}")
        print(f"   Rules used: {len(result.compression_rules)}")
        print(f"   Should continue: {result.should_continue}")
        
        # Show some compression rules for first few turns
        if turn_num <= 2 and result.compression_rules:
            print(f"   Sample rules: {dict(list(result.compression_rules.items())[:5])}")
    
    print("\n📊 Conversation Analysis")
    print("=" * 40)
    
    # Show efficiency trends
    print("Turn | Compression | Net Efficiency | Rules | Continue")
    print("-----|-------------|----------------|-------|----------")
    for i, result in enumerate(results, 1):
        print(f"{i:4d} | {result.compression_ratio:10.3f} | {result.net_efficiency:13.3f} | {len(result.compression_rules):5d} | {'Yes' if result.should_continue else 'No'}")
    
    # Show final statistics
    final_result = results[-1]
    if final_result.metrics:
        print(f"\n🎯 Final Conversation Metrics:")
        metrics = final_result.metrics
        print(f"   Total turns: {metrics.get('turn_number', 0)}")
        print(f"   Total rules: {metrics.get('total_rules', 0)}")
        print(f"   Total tokens saved: {metrics.get('total_tokens_saved', 0)}")
        print(f"   System prompt overhead: {metrics.get('system_prompt_overhead', 0)}")
        print(f"   Net tokens saved: {metrics.get('net_tokens_saved', 0)}")
        print(f"   Efficiency trend: {metrics.get('efficiency_trend', 'unknown')}")
        print(f"   Average compression: {metrics.get('avg_compression_ratio', 0):.3f}")
        
        if metrics.get('most_used_tokens'):
            print(f"   Most used tokens:")
            for token, count in metrics['most_used_tokens'][:5]:
                print(f"     {token}: {count} times")
    
    return results

def test_efficiency_degradation():
    """Test what happens when compression becomes inefficient."""
    
    print("\n\n🔬 Testing Efficiency Degradation Detection")
    print("=" * 60)
    
    reset_conversation_compression()
    
    # Create a conversation with decreasing compression value
    messages = [
        {"role": "user", "content": "a b c d e f g h i j k l m n o p q r s t u v w x y z"},
        {"role": "assistant", "content": "Here are the letters of the alphabet in order."}
    ]
    
    for turn in range(10):
        print(f"\n📝 Turn {turn + 1}")
        
        # Add more random content each turn (harder to compress)
        random_content = " ".join([f"random_word_{i}_{turn}" for i in range(turn * 3)])
        messages.append({
            "role": "user", 
            "content": f"Tell me about {random_content} and how it relates to the alphabet."
        })
        
        result = compress_conversation_aware(messages, min_characters=50, session_id="efficiency_test_session")
        
        print(f"   Compression ratio: {result.compression_ratio:.3f}")
        print(f"   Net efficiency: {result.net_efficiency:.3f}")
        print(f"   Should continue: {result.should_continue}")
        
        if not result.should_continue:
            print(f"   ⚠️  Compression disabled due to poor efficiency!")
            break

def test_session_id_collision_prevention():
    """Test that session IDs prevent conversation collisions."""
    
    print("\n\n🔒 Testing Session ID Collision Prevention")
    print("=" * 60)
    
    reset_conversation_compression()
    
    # Same exact message content, different sessions
    identical_messages = [
        {"role": "user", "content": "How do I create a Python function?"}
    ]
    
    # User A with session ID
    result_a = compress_conversation_aware(
        identical_messages, 
        min_characters=10, 
        session_id="user_a_session_123"
    )
    
    # User B with different session ID (same message!)
    result_b = compress_conversation_aware(
        identical_messages, 
        min_characters=10, 
        session_id="user_b_session_456"
    )
    
    # User C without session ID (legacy mode)
    result_c = compress_conversation_aware(
        identical_messages, 
        min_characters=10, 
        session_id=None
    )
    
    print(f"User A conversation ID: {result_a.conversation_id}")
    print(f"User B conversation ID: {result_b.conversation_id}")
    print(f"User C conversation ID: {result_c.conversation_id}")
    
    # Verify they're all different
    all_ids = [result_a.conversation_id, result_b.conversation_id, result_c.conversation_id]
    unique_ids = set(all_ids)
    
    if len(unique_ids) == len(all_ids):
        print("✅ Success! All conversations have unique IDs despite identical content")
    else:
        print("❌ Failure! Conversation ID collision detected")
        return False
    
    # Test conversation continuation with session IDs
    print("\n🔄 Testing conversation continuation with session IDs...")
    
    # Continue User A's conversation
    continued_messages_a = [
        {"role": "user", "content": "How do I create a Python function?"},
        {"role": "assistant", "content": "To create a function, use the def keyword..."},
        {"role": "user", "content": "What about parameters?"}
    ]
    
    result_a_continued = compress_conversation_aware(
        continued_messages_a,
        min_characters=10,
        session_id="user_a_session_123"  # Same session ID
    )
    
    # Check if User A's conversation was properly continued
    if result_a_continued.conversation_id.startswith("sess_user_a_session_123"):
        print("✅ Session-based conversation continuation works correctly")
    else:
        print("❌ Session-based conversation continuation failed")
        return False
    
    return True

def test_conversation_stats():
    """Test the conversation statistics API."""
    
    print("\n\n📈 Testing Conversation Statistics")
    print("=" * 50)
    
    stats = get_conversation_compression_stats()
    
    print(f"Active conversations: {stats.get('active_conversations', 0)}")
    print(f"Total turns processed: {stats.get('total_turns_processed', 0)}")
    print(f"Symbols in use: {stats.get('symbols_in_use', 0)}")
    print(f"Average turns per conversation: {stats.get('avg_turns_per_conversation', 0):.1f}")
    print(f"Memory usage: {stats.get('memory_usage_kb', 0):.1f} KB")

def main():
    """Run all conversation compression tests."""
    
    print("🚀 Conversation-Aware Compression Test Suite")
    print("=" * 70)
    
    try:
        # Test normal multi-turn conversation
        test_multi_turn_conversation()
        
        # Test session ID collision prevention
        session_test_passed = test_session_id_collision_prevention()
        if not session_test_passed:
            print("\n❌ Session ID tests failed!")
            return 1
        
        # Test efficiency degradation detection
        test_efficiency_degradation()
        
        # Test statistics
        test_conversation_stats()
        
        print("\n\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 