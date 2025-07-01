#!/usr/bin/env python3
"""
Test to verify that the symbol selection now uses Unicode symbols
instead of conflicting programming symbols.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.dynamic_dictionary import DynamicDictionaryAnalyzer

def create_test_code():
    """Create test code with programming symbols that could conflict."""
    return """
def process_data(data_items):
    # Comments with # symbols
    for item in data_items:
        value = item % 100  # Modulo operator
        if value & 0xFF:    # Bitwise AND
            result = value / 2  # Division
            @decorator
            def func(x):
                return x * value
            print(f"Result: {result}")
        else:
            print("Zero value")
    return True

class DataProcessor:
    def __init__(self):
        self.config = {
            "enabled": True,
            "path": "/tmp/data"
        }
    
    def validate(self, data):
        return data is not None
""" * 10  # Repeat to create patterns

def test_symbol_selection():
    """Test that symbol selection avoids programming symbols."""
    print("🧪 Testing Symbol Selection for Conflict Avoidance")
    print("=" * 60)
    
    test_code = create_test_code()
    print(f"Test code length: {len(test_code):,} characters")
    
    # Check what symbols are present in the code
    programming_symbols = set('#%&/@\\^`')
    symbols_in_code = programming_symbols.intersection(set(test_code))
    print(f"Programming symbols found in code: {sorted(symbols_in_code)}")
    
    # Create analyzer
    analyzer = DynamicDictionaryAnalyzer()
    
    # Test content-aware symbol selection
    print(f"\n🔍 Testing content-aware symbol selection...")
    content_aware_symbols = analyzer.symbol_assigner.get_content_aware_symbols(
        content=test_code, 
        max_symbols=20
    )
    
    print(f"Content-aware symbols selected: {len(content_aware_symbols)}")
    
    # Check what types of symbols were selected
    selected_symbols = [symbol for symbol, _ in content_aware_symbols]
    unicode_symbols = [s for s in selected_symbols if len(s) == 1 and ord(s) > 127]
    ascii_symbols = [s for s in selected_symbols if s not in unicode_symbols]
    conflicting_symbols = [s for s in selected_symbols if s in symbols_in_code]
    
    print(f"\n📊 Symbol Analysis:")
    print(f"  Unicode symbols: {len(unicode_symbols)}")
    print(f"  ASCII symbols: {len(ascii_symbols)}")
    print(f"  Conflicting symbols: {len(conflicting_symbols)}")
    
    print(f"\n✨ Selected Unicode symbols: {unicode_symbols[:10]}")
    if ascii_symbols:
        print(f"⚠️  Selected ASCII symbols: {ascii_symbols[:5]}")
    if conflicting_symbols:
        print(f"❌ Conflicting symbols: {conflicting_symbols}")
    else:
        print(f"✅ No conflicting symbols selected!")
    
    # Test full analysis
    print(f"\n🚀 Testing full analysis with conflict avoidance...")
    analysis_result = analyzer.analyze_prompt(test_code)
    
    dictionary_entries = analysis_result.get("dictionary_entries", {})
    print(f"Dictionary entries created: {len(dictionary_entries)}")
    
    if dictionary_entries:
        # Check symbols used in final dictionary
        used_symbols = list(dictionary_entries.keys())
        used_unicode = [s for s in used_symbols if len(s) == 1 and ord(s) > 127]
        used_conflicting = [s for s in used_symbols if s in symbols_in_code]
        
        print(f"\n📚 Final Dictionary Analysis:")
        print(f"  Unicode symbols used: {len(used_unicode)} ({used_unicode[:10]})")
        print(f"  Conflicting symbols used: {len(used_conflicting)} ({used_conflicting})")
        
        # Show sample assignments
        print(f"\n🔗 Sample assignments:")
        for i, (symbol, pattern) in enumerate(list(dictionary_entries.items())[:5]):
            pattern_preview = pattern[:50] + "..." if len(pattern) > 50 else pattern
            print(f"  {i+1}. '{symbol}' -> '{pattern_preview}'")
        
        if not used_conflicting:
            print(f"\n🎉 SUCCESS: No conflicting symbols in final dictionary!")
            return True
        else:
            print(f"\n❌ FAILURE: Found conflicting symbols in dictionary: {used_conflicting}")
            return False
    else:
        print(f"\n⚠️  No dictionary entries created")
        return False

if __name__ == "__main__":
    success = test_symbol_selection()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ Symbol conflict avoidance is working correctly!")
        print("The system now uses Unicode symbols that don't conflict with code.")
    else:
        print("❌ Symbol conflict avoidance needs further adjustment.")
    print("=" * 60) 