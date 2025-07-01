#!/usr/bin/env python3
"""
Simplified test script for debug category functionality in KrunchWrapper.
This tests only the core functionality without requiring full server initialization.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.logging_utils import _extract_debug_category_from_message

def test_debug_category_extraction():
    """Test the debug category extraction function."""
    print("\n🧪 Testing debug category extraction:")
    print("=" * 60)
    
    test_messages = [
        ("🔍 [ALL REQUESTS] Headers: {...}", "request_processing"),
        ("🔍 [CLINE] Detected Cline request", "cline_integration"),
        ("🔧 [SESSION] Generated session ID", "session_management"),
        ("🔍 [PROXY] Request content analysis", "compression_proxy"),
        ("🔍 [CONVERSATION DETECTION] Message analysis", "conversation_detection"),
        ("🔍 [KV DEBUG] Checking conversation", "kv_cache_optimization"),
        ("🗣️ [CONVERSATION COMPRESS] Starting compression", "conversation_compression"),
        ("🔍 [STREAMING DEBUG] Decompressed chunk", "streaming_responses"),
        ("🔍 [NON-STREAMING DEBUG] Found symbols", "response_decompression"),
        ("📊 Token calculation: 100 tokens saved", "token_calculations"),
        ("🔧 [FIX] Converted max_tokens", "system_fixes"),
        ("🚫 [TIMEOUT FILTER] Removed timeout parameter", "request_filtering"),
        ("🧹 [CLEANUP] Removing compression artifact", "cleanup_operations"),
        ("🚨 [ERROR] Target API returned 500", "error_handling"),
        ("📡 Forwarded server props", "server_communication"),
        ("🔧 Set model context: gpt-4", "model_context"),
        ("🚨 [PAYLOAD DEBUG] Processing request", "payload_debugging"),
        ("Generic debug message", "request_processing"),  # Default category
    ]
    
    all_passed = True
    for message, expected_category in test_messages:
        actual_category = _extract_debug_category_from_message(message)
        status = "✅" if actual_category == expected_category else "❌"
        print(f"{status} '{message[:40]}...' -> {actual_category}")
        if actual_category != expected_category:
            print(f"   Expected: {expected_category}, Got: {actual_category}")
            all_passed = False
    
    return all_passed

def test_debug_category_filtering_logic():
    """Test the debug category filtering logic without full config."""
    print("\n🧪 Testing debug category filtering logic:")
    print("=" * 60)
    
    # Simulate a simple config object
    class MockConfig:
        def __init__(self):
            self.debug_categories = {
                "request_processing": True,
                "cline_integration": False,  # Disabled
                "session_management": True,
                "compression_proxy": False,  # Disabled
                "streaming_responses": True,
                "error_handling": True,
            }
        
        def is_debug_category_enabled(self, category: str) -> bool:
            if not self.debug_categories:
                return True
            return self.debug_categories.get(category, True)
    
    config = MockConfig()
    
    # Test messages and expected results
    test_cases = [
        ("🔍 [CLINE] This should be filtered", False),  # cline_integration disabled
        ("🔍 [PROXY] This should be filtered", False),   # compression_proxy disabled  
        ("🔍 [ALL REQUESTS] This should pass", True),    # request_processing enabled
        ("🔍 [STREAMING DEBUG] This should pass", True), # streaming_responses enabled
        ("🚨 [ERROR] This should pass", True),           # error_handling enabled
        ("Generic debug message", True),                 # request_processing (default) enabled
    ]
    
    all_passed = True
    print("Testing category filtering logic:")
    for message, should_pass in test_cases:
        extracted_category = _extract_debug_category_from_message(message)
        is_enabled = config.is_debug_category_enabled(extracted_category)
        
        status = "✅" if is_enabled == should_pass else "❌"
        print(f"{status} '{message[:40]}...'")
        print(f"    Category: {extracted_category}, Enabled: {is_enabled}")
        
        if is_enabled != should_pass:
            print(f"    ❌ Expected enabled={should_pass}, got enabled={is_enabled}")
            all_passed = False
    
    return all_passed

def test_configuration_structure():
    """Test that the configuration structure is correct."""
    print("\n🧪 Testing configuration structure:")
    print("=" * 60)
    
    # Test loading the configuration file
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
        from jsonc_parser import load_jsonc
        
        config_path = Path(__file__).parent.parent / "config" / "server.jsonc"
        if config_path.exists():
            config_data = load_jsonc(str(config_path))
            debug_categories = config_data.get("logging", {}).get("debug_categories", {})
            
            print(f"✅ Configuration file loaded successfully")
            print(f"✅ Debug categories found: {len(debug_categories)}")
            
            expected_categories = [
                "request_processing", "cline_integration", "session_management",
                "compression_core", "compression_proxy", "conversation_detection",
                "kv_cache_optimization", "conversation_compression", "symbol_management",
                "streaming_responses", "response_decompression", "token_calculations",
                "system_fixes", "request_filtering", "cleanup_operations", "error_handling",
                "server_communication", "model_context", "payload_debugging", "test_utilities"
            ]
            
            for category in expected_categories:
                if category in debug_categories:
                    enabled = debug_categories[category]
                    status_icon = "✅" if enabled else "🔧"
                    print(f"  {status_icon} {category}: {enabled}")
                else:
                    print(f"  ❌ Missing category: {category}")
            
            return True
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing KrunchWrapper Debug Categories (Simplified)")
    print("=" * 60)
    
    all_tests_passed = True
    
    try:
        if not test_debug_category_extraction():
            all_tests_passed = False
            
        if not test_debug_category_filtering_logic():
            all_tests_passed = False
            
        if not test_configuration_structure():
            all_tests_passed = False
        
        if all_tests_passed:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed - check output above")
            
        print("\n📝 Usage Instructions:")
        print("1. Set log_level to 'DEBUG' in config/server.jsonc")
        print("2. Enable/disable specific categories in the debug_categories section")
        print("3. Start the server to see filtered debug output")
        print("\n🔧 Example: To see only Cline-related debug messages:")
        print('   - Set all categories to false except "cline_integration": true')
        
        return 0 if all_tests_passed else 1
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 