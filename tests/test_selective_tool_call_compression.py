#!/usr/bin/env python3
"""
Test script for selective tool call compression.

This demonstrates the difference between:
1. Old approach: Skip compression entirely for content with tool calls
2. New approach: Selectively compress content within tool calls while preserving structure
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compress import compress_with_dynamic_analysis, compress_with_selective_tool_call_analysis, decompress
from core.tool_identifier import contains_tool_calls

def create_test_content_with_tool_call():
    """Create test content that contains a tool call with large file content."""
    # Simulate a large file content that would benefit from compression
    large_file_content = """
def calculate_fibonacci_sequence(n):
    '''Calculate the Fibonacci sequence up to n numbers.
    
    This function uses dynamic programming to efficiently calculate
    the Fibonacci sequence. The Fibonacci sequence is a series of numbers
    where each number is the sum of the two preceding ones.
    
    Args:
        n (int): Number of Fibonacci numbers to calculate
        
    Returns:
        list: List containing the Fibonacci sequence
        
    Example:
        >>> calculate_fibonacci_sequence(10)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    '''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    # Initialize the sequence with the first two numbers
    fibonacci_sequence = [0, 1]
    
    # Calculate the rest of the sequence using dynamic programming
    for i in range(2, n):
        next_number = fibonacci_sequence[i-1] + fibonacci_sequence[i-2]
        fibonacci_sequence.append(next_number)
        
        # Optional: Add some logging for demonstration
        if i % 5 == 0:
            print(f"Calculated Fibonacci number {i}: {next_number}")
            
    return fibonacci_sequence

def main():
    '''Main function to demonstrate the Fibonacci calculation.'''
    print("Welcome to the Fibonacci Calculator!")
    
    try:
        # Get user input for the number of Fibonacci numbers to calculate
        num_count = int(input("Enter the number of Fibonacci numbers to calculate: "))
        
        if num_count < 0:
            print("Please enter a non-negative integer.")
            return
            
        # Calculate the Fibonacci sequence
        result = calculate_fibonacci_sequence(num_count)
        
        # Display the results
        print(f"\\nFibonacci sequence with {num_count} numbers:")
        for i, num in enumerate(result):
            print(f"F({i}) = {num}")
            
        # Calculate some statistics
        if len(result) > 0:
            print(f"\\nStatistics:")
            print(f"Largest number: {max(result)}")
            print(f"Sum of all numbers: {sum(result)}")
            print(f"Average: {sum(result) / len(result):.2f}")
            
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
""" * 3  # Triple the content to make it larger
    
    # Create a tool call JSON with this large content
    tool_call = {
        "tool": "read_file",
        "path": "fibonacci_calculator.py",
        "content": large_file_content.strip(),
        "operationIsLocatedInWorkspace": True
    }
    
    # Embed the tool call in a larger context
    full_content = f"""
I need to analyze this Python file for optimization opportunities.

Here is the file content from the read operation:

{json.dumps(tool_call, indent=2)}

Please review this code and suggest optimizations for:
1. Performance improvements
2. Code readability
3. Memory usage
4. Error handling

Focus particularly on the Fibonacci calculation algorithm and see if there are any mathematical optimizations we can apply.
"""
    
    return full_content

def test_compression_approaches():
    """Test both compression approaches and compare results."""
    print("🧪 Testing Selective Tool Call Compression\n")
    print("=" * 60)
    
    # Create test content
    content = create_test_content_with_tool_call()
    original_size = len(content)
    
    print(f"📏 Original content size: {original_size:,} characters")
    print(f"🔍 Contains tool calls: {contains_tool_calls(content)}")
    print()
    
    # Test 1: Standard compression (skips tool calls entirely)
    print("🔄 Test 1: Standard Compression (skips tool calls)")
    print("-" * 50)
    
    try:
        packed_standard = compress_with_dynamic_analysis(content, skip_tool_detection=False)
        standard_size = len(packed_standard.text)
        standard_ratio = (original_size - standard_size) / original_size
        
        print(f"   Compressed size: {standard_size:,} characters")
        print(f"   Compression ratio: {standard_ratio:.1%}")
        print(f"   Rules generated: {len(packed_standard.used)}")
        print(f"   Result: {'COMPRESSION APPLIED' if len(packed_standard.used) > 0 else 'NO COMPRESSION (tool calls detected)'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        standard_size = original_size
        standard_ratio = 0
        packed_standard = None
    
    print()
    
    # Test 2: Selective tool call compression
    print("🔧 Test 2: Selective Tool Call Compression")
    print("-" * 50)
    
    try:
        packed_selective = compress_with_selective_tool_call_analysis(content, skip_tool_detection=False)
        selective_size = len(packed_selective.text)
        selective_ratio = (original_size - selective_size) / original_size
        
        print(f"   Compressed size: {selective_size:,} characters")
        print(f"   Compression ratio: {selective_ratio:.1%}")
        print(f"   Rules generated: {len(packed_selective.used)}")
        print(f"   Result: SELECTIVE COMPRESSION APPLIED")
        
        # Show which rules are for tool call content vs standard content
        tool_call_rules = [k for k in packed_selective.used.keys() if '_' in k and any(
            field in k.split('_')[0].lower() for field in 
            ['content', 'file_content', 'text', 'data', 'body', 'source', 'code']
        )]
        standard_rules = [k for k in packed_selective.used.keys() if k not in tool_call_rules]
        
        print(f"   Tool call content rules: {len(tool_call_rules)}")
        print(f"   Standard compression rules: {len(standard_rules)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        selective_size = original_size
        selective_ratio = 0
        packed_selective = None
    
    print()
    
    # Test 3: Verify decompression works correctly
    print("🔍 Test 3: Decompression Verification")
    print("-" * 50)
    
    if packed_selective and len(packed_selective.used) > 0:
        try:
            decompressed = decompress(packed_selective.text, packed_selective.used)
            is_identical = decompressed == content
            
            print(f"   Decompression successful: {is_identical}")
            if is_identical:
                print("   ✅ Original content perfectly restored")
            else:
                print("   ❌ Decompression mismatch!")
                print(f"   Original length: {len(content)}")
                print(f"   Decompressed length: {len(decompressed)}")
                
                # Find first difference
                for i, (a, b) in enumerate(zip(content, decompressed)):
                    if a != b:
                        print(f"   First difference at position {i}: '{a}' vs '{b}'")
                        break
                        
        except Exception as e:
            print(f"   ❌ Decompression error: {e}")
    else:
        print("   ⏭️ Skipped (no compression applied)")
    
    print()
    
    # Summary
    print("📊 Summary")
    print("-" * 50)
    print(f"Original size:          {original_size:,} chars")
    print(f"Standard compression:   {standard_size:,} chars ({standard_ratio:.1%} reduction)")
    print(f"Selective compression:  {selective_size:,} chars ({selective_ratio:.1%} reduction)")
    
    if selective_ratio > standard_ratio:
        improvement = selective_ratio - standard_ratio
        print(f"\n🎉 Selective compression achieved {improvement:.1%} better compression!")
        print("    This preserves tool call structure while compressing content.")
    elif selective_ratio == standard_ratio == 0:
        print(f"\n🤔 Neither approach achieved compression (content may be too small or not repetitive enough)")
    else:
        print(f"\n📝 Standard compression performed better (or both achieved same result)")
    
    print()
    
    # Show example of what happens to tool calls
    if packed_selective and contains_tool_calls(packed_selective.text):
        print("🔍 Tool Call Structure Preservation")
        print("-" * 50)
        print("Tool calls are still present and parseable in the compressed content.")
        print("Only the large content fields within tool calls have been compressed.")
        print()
        
        # Try to extract and show a tool call
        try:
            from core.tool_identifier import get_tool_calls
            tool_calls = get_tool_calls(packed_selective.text)
            if tool_calls:
                print(f"Found {len(tool_calls)} tool call(s) in compressed content:")
                for i, tc in enumerate(tool_calls[:1]):  # Show first one
                    print(f"  Tool {i+1}: {tc.get('tool', 'unknown')}")
                    print(f"  Path: {tc.get('path', 'N/A')}")
                    content_field = tc.get('content', '')
                    if content_field:
                        preview = content_field[:100] + '...' if len(content_field) > 100 else content_field
                        print(f"  Content: {preview}")
                        print(f"  Content compressed: {'🔧' if len(content_field) < 1000 else '📄'}")  # Rough heuristic
        except Exception as e:
            print(f"Error analyzing tool calls: {e}")

if __name__ == "__main__":
    test_compression_approaches() 