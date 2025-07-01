#!/usr/bin/env python
"""Test the fallback compression functionality."""
import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compress import compress

def test_fallback():
    """Test fallback compression with unknown language."""
    
    # Test with some generic code-like text
    test_text = """
    function calculateValue(input) {
        if (input > 0) {
            return input * 2;
        } else {
            return 0;
        }
    }
    
    def process_data(data):
        for item in data:
            if item is not None:
                print(item)
        return True
    
    package main
    
    import "fmt"
    
    func main() {
        fmt.Println("Hello World")
    }
    """
    
    print("🧪 Testing Fallback Compression")
    print("=" * 40)
    print(f"Test text length: {len(test_text)} characters")
    
    # Test with no language specified (should trigger fallback)
    print(f"\n🔄 Testing with no language specified...")
    result = compress(test_text, None)
    
    compression_ratio = (1 - len(result.text) / len(test_text)) * 100
    
    print(f"✅ Compression complete!")
    print(f"   Original size: {len(test_text)} characters")
    print(f"   Compressed size: {len(result.text)} characters")
    print(f"   Compression ratio: {compression_ratio:.1f}%")
    print(f"   Language detected: {result.language}")
    print(f"   Fallback used: {result.fallback_used}")
    print(f"   Tokens used: {len(result.used)}")
    
    if result.used:
        print(f"\n🎯 Sample replacements:")
        for i, (symbol, token) in enumerate(list(result.used.items())[:10]):
            print(f"      '{token}' → '{symbol}'")
    
    # Test with invalid language (should also trigger fallback)
    print(f"\n🔄 Testing with invalid language 'xyz'...")
    result2 = compress(test_text, 'xyz')
    
    compression_ratio2 = (1 - len(result2.text) / len(test_text)) * 100
    
    print(f"✅ Compression complete!")
    print(f"   Compression ratio: {compression_ratio2:.1f}%")
    print(f"   Language detected: {result2.language}")
    print(f"   Fallback used: {result2.fallback_used}")
    print(f"   Tokens used: {len(result2.used)}")

if __name__ == "__main__":
    test_fallback() 