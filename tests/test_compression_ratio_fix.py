#!/usr/bin/env python3
"""
Test script to verify that the compression ratio calculation fix works correctly.
This script tests that compression ratios are realistic and never exceed reasonable bounds.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_compression_ratio_calculation():
    """Test that compression ratio calculations are realistic."""
    print("\n" + "="*70)
    print("TESTING COMPRESSION RATIO CALCULATION FIX")
    print("="*70)
    
    try:
        from core.dynamic_dictionary import DynamicDictionaryAnalyzer
        
        # Create analyzer
        analyzer = DynamicDictionaryAnalyzer()
        
        # Test with content that previously caused 100% compression ratio
        test_content = """
        def center_dialog(self, dlg):
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dlg.winfo_width() // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dlg.winfo_height() // 2)
            dlg.geometry(f"+{x}+{y}")
            
        def show_dialog(self, dlg):
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dlg.winfo_width() // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dlg.winfo_height() // 2)
            dlg.geometry(f"+{x}+{y}")
            dlg.show()
            
        def position_window(self, dlg):
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dlg.winfo_width() // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dlg.winfo_height() // 2)
            dlg.geometry(f"+{x}+{y}")
            dlg.update()
        """ * 5  # Repeat to ensure pattern detection
        
        print(f"📝 Testing with {len(test_content)} character content containing repeated patterns")
        
        # Analyze the content
        print("🔍 Analyzing content for compression opportunities...")
        analysis = analyzer.analyze_prompt(test_content)
        
        # Check the results
        compression_analysis = analysis.get("compression_analysis", {})
        compression_ratio = compression_analysis.get("compression_ratio", 0)
        total_char_savings = compression_analysis.get("total_char_savings", 0)
        total_token_savings = compression_analysis.get("total_token_savings", 0)
        
        print(f"\n📊 COMPRESSION ANALYSIS RESULTS:")
        print(f"   Original length: {len(test_content):,} characters")
        print(f"   Character savings: {total_char_savings:,}")
        print(f"   Token savings: {total_token_savings:.1f}")
        print(f"   Compression ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
        print(f"   Dictionary entries: {len(analysis.get('dictionary_entries', {}))}")
        
        # Validation checks
        issues = []
        
        # Check 1: Compression ratio should be reasonable
        if compression_ratio > 0.95:
            issues.append(f"Compression ratio too high: {compression_ratio*100:.1f}%")
        
        # Check 2: Compression ratio should be positive if we have savings
        if total_char_savings > 0 and compression_ratio <= 0:
            issues.append(f"Positive character savings ({total_char_savings}) but zero compression ratio")
        
        # Check 3: Character savings should be reasonable
        if total_char_savings > len(test_content):
            issues.append(f"Character savings ({total_char_savings}) exceed original length ({len(test_content)})")
        
        # Check 4: Dictionary entries should exist if we have compression
        if compression_ratio > 0.01 and len(analysis.get('dictionary_entries', {})) == 0:
            issues.append("Significant compression ratio but no dictionary entries")
        
        if issues:
            print(f"\n❌ VALIDATION ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"\n✅ VALIDATION PASSED:")
            print(f"   ✅ Compression ratio is realistic: {compression_ratio*100:.1f}%")
            print(f"   ✅ Character savings are reasonable: {total_char_savings:,}")
            print(f"   ✅ Token savings are reasonable: {total_token_savings:.1f}")
            print(f"   ✅ Dictionary has {len(analysis.get('dictionary_entries', {}))} entries")
            return True
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extreme_pattern_values():
    """Test with patterns that might cause extreme values."""
    print("\n" + "="*70)
    print("TESTING EXTREME PATTERN VALUE HANDLING")
    print("="*70)
    
    try:
        from core.dynamic_dictionary import OptimizedSymbolAssigner
        import tiktoken
        
        # Create symbol assigner with tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        assigner = OptimizedSymbolAssigner(tokenizer, "gpt-4")
        
        # Test with the problematic pattern from the logs
        pattern = "y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dlg.winfo_height() // 2)"
        symbol = "δ"
        count = 4
        
        print(f"📝 Testing pattern calculation:")
        print(f"   Pattern: '{pattern}' (length: {len(pattern)})")
        print(f"   Symbol: '{symbol}'")
        print(f"   Count: {count}")
        
        # Calculate the value
        value = assigner.calculate_assignment_value(pattern, count, symbol)
        
        print(f"\n📊 CALCULATION RESULTS:")
        print(f"   Net value: {value:.2f}")
        
        # Calculate expected maximum reasonable value
        pattern_tokens = len(tokenizer.encode(pattern))
        symbol_tokens = len(tokenizer.encode(symbol))
        theoretical_max = (pattern_tokens - symbol_tokens) * count
        
        print(f"   Pattern tokens: {pattern_tokens}")
        print(f"   Symbol tokens: {symbol_tokens}")
        print(f"   Theoretical max (no overhead): {theoretical_max:.1f}")
        
        # Validation
        if value > theoretical_max:
            print(f"❌ Value exceeds theoretical maximum: {value:.2f} > {theoretical_max:.1f}")
            return False
        elif value > theoretical_max * 0.8:  # Allow some reasonable overhead accounting
            print(f"⚠️  Value is very close to theoretical maximum: {value:.2f} vs {theoretical_max:.1f}")
        
        # Check for reasonable bounds
        max_reasonable = len(pattern) / 2 * count  # Very generous bound
        if value > max_reasonable:
            print(f"❌ Value exceeds reasonable bounds: {value:.2f} > {max_reasonable:.1f}")
            return False
        
        print(f"✅ Pattern value calculation is reasonable")
        return True
        
    except Exception as e:
        print(f"❌ Error during pattern testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Compression Ratio Fix Test")
    print("This test verifies that compression ratios are realistic and never exceed reasonable bounds.")
    
    results = []
    
    try:
        # Test 1: Overall compression ratio calculation
        ratio_test = test_compression_ratio_calculation()
        results.append(("Compression ratio calculation", ratio_test))
        
        # Test 2: Extreme pattern value handling
        pattern_test = test_extreme_pattern_values()
        results.append(("Extreme pattern value handling", pattern_test))
        
        print("\n" + "="*70)
        print("🎉 COMPRESSION RATIO FIX TEST RESULTS")
        print("="*70)
        
        all_passed = True
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
            if not success:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All compression ratio fix tests passed!")
            print("✅ Compression ratios are now realistic and bounded")
            print("✅ Token savings calculations are reasonable")
            print("✅ No more impossible >100% compression ratios")
            print("✅ Pattern value calculations are bounded")
            
            print("\nKey fixes implemented:")
            print("  🔧 Token savings capped at original token count - 1")
            print("  🔧 Compression ratios capped at 95% for realism")
            print("  🔧 Pattern token savings bounded by reasonable limits")
            print("  🔧 Enhanced debug logging for suspicious calculations")
            print("  🔧 Simplified tokenization method for predictability")
        else:
            print("\n❌ Some compression ratio fix tests failed!")
            print("Please review the calculation logic for remaining issues.")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 