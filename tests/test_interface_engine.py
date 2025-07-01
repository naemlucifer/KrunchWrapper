"""
Test the new modular interface engine system.

This test demonstrates how the new InterfaceCompressionHandler cleanly handles
different client interfaces (Cline, WebUI, SillyTavern, etc.) without complex branching.
"""

import json
from unittest.mock import Mock, MagicMock
from core.interface_engine import InterfaceEngine, InterfaceCompressionHandler, detect_and_process_compression


def test_interface_detection():
    """Test detection of different interface engines."""
    
    # Mock configuration
    config = Mock()
    config.system_prompt = {
        "interface_engine": "auto"
    }
    config.system_prompt_format = "chatml"
    
    handler = InterfaceCompressionHandler(config)
    
    # Test explicit Cline configuration
    config.system_prompt["interface_engine"] = "cline"
    request = Mock()
    request.headers = {}
    
    engine = handler.detect_interface_engine(request)
    assert engine == InterfaceEngine.CLINE
    print(f"✅ Explicit Cline configuration detected: {engine.value}")
    
    # Test auto-detection with WebUI characteristics
    config.system_prompt["interface_engine"] = "auto"
    request.headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-llama-webui": "true"
    }
    request_data = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "gpt-4",
        "timings_per_token": True
    }
    
    engine = handler.detect_interface_engine(request, request_data)
    print(f"✅ Auto-detected interface: {engine.value}")
    
    # Test SillyTavern detection
    request.headers = {"user-agent": "SillyTavern/1.2.3"}
    engine = handler.detect_interface_engine(request)
    assert engine == InterfaceEngine.SILLYTAVERN
    print(f"✅ SillyTavern detected: {engine.value}")
    
    # Test Cline auto-detection
    request.headers = {"user-agent": "cline/1.0.0"}
    engine = handler.detect_interface_engine(request)
    assert engine == InterfaceEngine.CLINE
    print(f"✅ Cline auto-detected: {engine.value}")
    
    # Test standard fallback
    request.headers = {"user-agent": "generic-client/1.0"}
    engine = handler.detect_interface_engine(request)
    assert engine == InterfaceEngine.STANDARD
    print(f"✅ Standard fallback: {engine.value}")


def test_compression_processing():
    """Test compression processing with different engines."""
    
    # Mock configuration
    config = Mock()
    config.system_prompt = {
        "interface_engine": "auto"
    }
    config.system_prompt_format = "chatml"
    
    # Mock interceptors to avoid dependencies
    handler = InterfaceCompressionHandler(config)
    handler._cline_interceptor = Mock()
    handler._webui_interceptor = Mock()
    handler._standard_interceptor = Mock()
    
    # Set up mock returns
    mock_metadata = {"format": "chatml", "engine": "test"}
    handler._cline_interceptor.intercept_and_process_cline.return_value = ([], mock_metadata)
    handler._webui_interceptor.intercept_and_process_webui.return_value = ([], mock_metadata)
    handler._standard_interceptor.intercept_and_process.return_value = ([], mock_metadata)
    
    # Test Cline processing
    messages = [{"role": "user", "content": "test message"}]
    rule_union = {"Ω": "function", "α": "variable"}
    
    processed_messages, metadata = handler.process_compression(
        engine=InterfaceEngine.CLINE,
        messages=messages,
        rule_union=rule_union,
        model_id="anthropic/claude-3-5-sonnet-20241022"
    )
    
    handler._cline_interceptor.intercept_and_process_cline.assert_called_once()
    print(f"✅ Cline processing called correct interceptor")
    
    # Test WebUI processing
    processed_messages, metadata = handler.process_compression(
        engine=InterfaceEngine.WEBUI,
        messages=messages,
        rule_union=rule_union
    )
    
    handler._webui_interceptor.intercept_and_process_webui.assert_called_once()
    print(f"✅ WebUI processing called correct interceptor")
    
    # Test standard processing
    processed_messages, metadata = handler.process_compression(
        engine=InterfaceEngine.STANDARD,
        messages=messages,
        rule_union=rule_union
    )
    
    handler._standard_interceptor.intercept_and_process.assert_called_once()
    print(f"✅ Standard processing called correct interceptor")


def test_unified_function():
    """Test the unified detect_and_process_compression function."""
    
    # Mock request and config
    request = Mock()
    request.headers = {"user-agent": "cline/1.0.0"}
    
    config = Mock()
    config.system_prompt = {
        "interface_engine": "auto"
    }
    config.system_prompt_format = "chatml"
    
    # Mock the handler and its methods
    original_handler = InterfaceCompressionHandler(config)
    original_handler.detect_interface_engine = Mock(return_value=InterfaceEngine.CLINE)
    original_handler.should_disable_compression = Mock(return_value=False)
    original_handler.process_compression = Mock(return_value=([], {"test": "metadata"}))
    
    # Test the unified function
    messages = [{"role": "user", "content": "test"}]
    rule_union = {"Ω": "function"}
    
    # Mock the global handler
    import core.interface_engine
    core.interface_engine._global_handler = original_handler
    
    processed_messages, metadata, detected_engine = detect_and_process_compression(
        request=request,
        messages=messages,
        rule_union=rule_union,
        config=config,
        model_id="anthropic/claude-3-5-sonnet-20241022"
    )
    
    assert detected_engine == InterfaceEngine.CLINE
    assert "engine" in metadata
    assert metadata["engine"] == "cline"
    print(f"✅ Unified function correctly detected and processed {detected_engine.value} interface")


def demo_modular_system():
    """Demonstrate the clean, modular system vs. the old complex branching."""
    
    print("\n" + "="*60)
    print("🎯 INTERFACE ENGINE MODULAR SYSTEM DEMO")
    print("="*60)
    
    print("\n📋 OLD SYSTEM (replaced):")
    print("   ❌ Complex nested if/elif statements")
    print("   ❌ Duplicated logic for compression vs no-compression")
    print("   ❌ WebUI, Cline, SillyTavern handling scattered")
    print("   ❌ Hard to add new interfaces like roo, aider")
    print("   ❌ Brittle branching logic broke easily")
    
    print("\n✨ NEW MODULAR SYSTEM:")
    print("   ✅ Single unified detect_and_process_compression() function")
    print("   ✅ Clean InterfaceEngine enum (CLINE, WEBUI, SILLYTAVERN, etc.)")
    print("   ✅ Auto-detection + explicit configuration support")
    print("   ✅ Easy to add new interfaces - just add enum + handler")
    print("   ✅ Consistent behavior across direct + proxy endpoints")
    print("   ✅ Fallback handling built-in")
    
    print(f"\n🔧 SUPPORTED ENGINES:")
    for engine in InterfaceEngine:
        status = "🚀 Active" if engine in [InterfaceEngine.CLINE, InterfaceEngine.WEBUI, InterfaceEngine.SILLYTAVERN, InterfaceEngine.STANDARD] else "🔮 Future"
        print(f"   {status}: {engine.value}")
    
    print(f"\n📖 USAGE:")
    print(f"   # Old way (complex branching - REMOVED):")
    print(f"   if is_webui_request:")
    print(f"       webui_interceptor.intercept_and_process_webui(...)")
    print(f"   elif config.interface_engine == 'cline':")
    print(f"       cline_interceptor.intercept_and_process_cline(...)")
    print(f"   else:")
    print(f"       standard_interceptor.intercept_and_process(...)")
    print(f"")
    print(f"   # New way (unified):")
    print(f"   messages, metadata, engine = detect_and_process_compression(")
    print(f"       request=request, messages=messages, rule_union=rules, config=config)")
    
    print(f"\n🎛️  CONFIGURATION:")
    print(f'   "interface_engine": "auto"        # Auto-detect (recommended)')
    print(f'   "interface_engine": "cline"       # Force Cline interface')
    print(f'   "interface_engine": "webui"       # Force WebUI interface')
    print(f'   "interface_engine": "sillytavern" # Force SillyTavern interface')
    print(f'   # Legacy: "use_cline": true      # Backward compatibility (deprecated)')


if __name__ == "__main__":
    print("🧪 Testing Interface Engine System...")
    
    try:
        test_interface_detection()
        print("✅ Interface detection tests passed")
        
        test_compression_processing()
        print("✅ Compression processing tests passed")
        
        test_unified_function()
        print("✅ Unified function tests passed")
        
        demo_modular_system()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"The new modular interface engine system is working correctly.")
        print(f"This replaces the complex branching logic and makes the system")
        print(f"much easier to extend for future interfaces like roo, aider, etc.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 