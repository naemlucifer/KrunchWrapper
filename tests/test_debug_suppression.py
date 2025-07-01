#!/usr/bin/env python3
"""
Test script to demonstrate selective debug suppression functionality.
This shows how to suppress debug logging from specific modules while keeping DEBUG level active.
"""

import logging
import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.jsonc_parser import load_jsonc

def test_selective_debug_suppression():
    """Test the selective debug suppression functionality."""
    
    print("🧪 Testing selective debug suppression functionality")
    print("=" * 60)
    
    # Configure logging to DEBUG level
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)-20s - %(levelname)s - %(message)s'
    )
    
    # Load server config to get suppress_debug_modules setting
    server_config_path = Path(__file__).parent.parent / "config" / "server.jsonc"
    config = {}
    if server_config_path.exists():
        config = load_jsonc(str(server_config_path))
    
    # Apply environment variable override (same logic as server.py)
    suppress_debug_env = os.environ.get("KRUNCHWRAPPER_SUPPRESS_DEBUG_MODULES", "")
    if suppress_debug_env:
        suppress_debug_modules = [module.strip() for module in suppress_debug_env.split(",") if module.strip()]
        print(f"🌍 Environment override detected: {suppress_debug_env}")
    else:
        suppress_debug_modules = config.get("suppress_debug_modules", [])
    print(f"📋 Config suppress_debug_modules: {suppress_debug_modules}")
    
    # Test modules
    test_modules = [
        "core.dynamic_dictionary",
        "core.compress", 
        "core.other_module"
    ]
    
    print(f"\n🎯 Testing with modules: {test_modules}")
    print(f"🤫 Suppressed modules: {suppress_debug_modules}")
    
    # Apply selective suppression (same logic as in server.py)
    if suppress_debug_modules:
        print(f"\n🔧 Applying selective debug suppression to {len(suppress_debug_modules)} modules:")
        for module_name in suppress_debug_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            print(f"   📵 {module_name}: DEBUG → INFO")
    
    print(f"\n📝 Testing logging from each module:")
    print("-" * 40)
    
    # Test logging from each module
    for module_name in test_modules:
        logger = logging.getLogger(module_name)
        
        # Show current effective level
        effective_level = logger.getEffectiveLevel()
        level_name = logging.getLevelName(effective_level)
        is_suppressed = module_name in suppress_debug_modules
        
        print(f"\n🔍 Module: {module_name}")
        print(f"   Level: {level_name} (suppressed: {is_suppressed})")
        
        # Test different log levels
        logger.debug(f"DEBUG message from {module_name}")
        logger.info(f"INFO message from {module_name}")
        logger.warning(f"WARNING message from {module_name}")
    
    print(f"\n✅ Test completed!")
    print("=" * 60)
    print("📖 How to use:")
    print("1. Edit config/server.jsonc")
    print("2. Uncomment the desired modules in 'suppress_debug_modules'")
    print("3. Or set KRUNCHWRAPPER_SUPPRESS_DEBUG_MODULES=core.dynamic_dictionary,core.compress")
    print("4. Start the server with DEBUG level logging")

if __name__ == "__main__":
    test_selective_debug_suppression() 