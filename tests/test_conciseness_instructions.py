#!/usr/bin/env python3
"""
Test script for the configurable conciseness instructions feature.
This demonstrates how the system injects conciseness guidance into system prompts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.conciseness_instructions import ConcisenessInstructionsHandler
from core.system_prompt import build_system_prompt

def test_conciseness_handler():
    """Test the ConcisenessInstructionsHandler functionality"""
    print("=" * 60)
    print("Testing ConcisenessInstructionsHandler")
    print("=" * 60)
    
    handler = ConcisenessInstructionsHandler()
    
    print(f"Conciseness instructions enabled: {handler.is_enabled()}")
    
    if not handler.is_enabled():
        print("❌ Conciseness instructions are disabled in config. Enable them to test.")
        return
    
    # Test with different contexts
    test_cases = [
        {
            "user_content": "How do I debug this Python code that's not working?",
            "language": "python",
            "description": "Python debugging context"
        },
        {
            "user_content": "Please explain how to implement a REST API in Node.js",
            "language": "javascript", 
            "description": "JavaScript explanation context"
        },
        {
            "user_content": "Write a function to sort an array",
            "language": "python",
            "description": "Code implementation request"
        },
        {
            "user_content": "This is a short request",
            "language": "generic",
            "description": "Short generic request"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"User content: {test_case['user_content']}")
        print(f"Language: {test_case['language']}")
        
        # Test whether instructions should be injected
        should_inject = handler.should_inject_instructions(has_compression=True)
        print(f"Should inject instructions: {should_inject}")
        
        if should_inject:
            # Generate instructions
            instructions = handler.generate_instructions(
                user_content=test_case['user_content'],
                language=test_case['language']
            )
            
            print(f"Generated instructions:")
            if instructions:
                # Print each line with proper indentation
                for line in instructions.split('\n'):
                    print(f"  {line}")
            else:
                print("  (No instructions generated)")
                
        print()

def test_system_prompt_integration():
    """Test the integration with build_system_prompt"""
    print("=" * 60) 
    print("Testing System Prompt Integration")
    print("=" * 60)
    
    # Test with compression dictionary
    compression_dict = {
        "α": "function",
        "β": "return", 
        "γ": "variable"
    }
    
    user_content = "How do I debug this Python function that returns the wrong variable?"
    language = "python"
    format_name = "chatml"
    
    print(f"User content: {user_content}")
    print(f"Language: {language}")
    print(f"Compression dictionary: {compression_dict}")
    print(f"Format: {format_name}")
    
    # Build system prompt with conciseness instructions
    system_prompt, metadata = build_system_prompt(
        used=compression_dict,
        lang=language,
        format_name=format_name,
        user_content=user_content
    )
    
    print(f"\n--- Generated System Prompt ---")
    print(system_prompt)
    print(f"\n--- Metadata ---")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

def test_configuration_options():
    """Test different configuration options"""
    print("=" * 60)
    print("Testing Configuration Options") 
    print("=" * 60)
    
    handler = ConcisenessInstructionsHandler()
    
    if not handler.is_enabled():
        print("❌ Conciseness instructions are disabled. Cannot test configuration options.")
        return
    
    print(f"Injection position: {handler.get_injection_position()}")
    
    # Test with different injection scenarios
    scenarios = [
        {"has_compression": True, "description": "With compression"},
        {"has_compression": False, "description": "Without compression"}
    ]
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['description']} ---")
        should_inject = handler.should_inject_instructions(scenario['has_compression'])
        print(f"Should inject instructions: {should_inject}")

if __name__ == "__main__":
    print("🚀 Testing Configurable Conciseness Instructions Feature")
    print()
    
    try:
        test_conciseness_handler()
        test_system_prompt_integration()
        test_configuration_options()
        
        print("=" * 60)
        print("✅ All tests completed successfully!")
        print()
        print("💡 To customize the instructions:")
        print("   1. Edit config/conciseness-instructions.jsonc")
        print("   2. Modify config/config.jsonc conciseness_instructions section")
        print("   3. Restart the server to apply changes")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 