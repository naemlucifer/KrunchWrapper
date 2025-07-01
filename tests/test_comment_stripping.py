#!/usr/bin/env python3
"""
Test script for comment stripping functionality.
"""

import sys
import os
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.comment_stripper import CommentStripper
from core.compress import compress_with_dynamic_analysis

def test_python_comment_stripping():
    """Test comment stripping for Python code."""
    print("🐍 Testing Python Comment Stripping")
    print("=" * 40)
    
    python_code = '''# This is a Python script
import os
import sys  # Standard imports

def hello_world():
    """This is a docstring that should be preserved."""
    # This is a comment inside function
    name = "World"  # Another comment
    print(f"Hello, {name}!")
    
    # Multi-line comment
    # that spans multiple lines
    return name

# Main execution
if __name__ == "__main__":
    hello_world()  # Call the function
'''
    
    stripper = CommentStripper()
    stripped_code, stats = stripper.strip_comments(python_code)
    
    print("Original code:")
    print(python_code)
    print("\nStripped code:")
    print(stripped_code)
    print(f"\nStatistics: {stats}")

def test_javascript_comment_stripping():
    """Test comment stripping for JavaScript code."""
    print("\n🟨 Testing JavaScript Comment Stripping")
    print("=" * 40)
    
    js_code = '''// JavaScript example
function greetUser(name) {
    /* Multi-line comment
       explaining the function */
    const greeting = "Hello"; // Inline comment
    
    // Another single line comment
    console.log(`${greeting}, ${name}!`);
    
    /* Another multi-line
       comment block */
    return greeting;
}

// Main execution
greetUser("World"); // Function call
'''
    
    stripper = CommentStripper()
    stripped_code, stats = stripper.strip_comments(js_code)
    
    print("Original code:")
    print(js_code)
    print("\nStripped code:")
    print(stripped_code)
    print(f"\nStatistics: {stats}")

def test_c_cpp_comment_stripping():
    """Test comment stripping for C/C++ code."""
    print("\n⚙️ Testing C/C++ Comment Stripping")
    print("=" * 40)
    
    cpp_code = '''// C++ example
#include <iostream>
#include <string>  // For string handling

/* Main function
   Entry point of the program */
int main() {
    std::string name = "World";  // Variable declaration
    
    // Print greeting
    std::cout << "Hello, " << name << "!" << std::endl;
    
    /* Return success code
       Standard practice */
    return 0;  // Success
}
'''
    
    stripper = CommentStripper()
    stripped_code, stats = stripper.strip_comments(cpp_code)
    
    print("Original code:")
    print(cpp_code)
    print("\nStripped code:")
    print(stripped_code)
    print(f"\nStatistics: {stats}")

def test_html_comment_stripping():
    """Test comment stripping for HTML."""
    print("\n🌐 Testing HTML Comment Stripping")
    print("=" * 40)
    
    html_code = '''<!DOCTYPE html>
<!-- This is an HTML document -->
<html>
<head>
    <title>Test Page</title>
    <!-- Meta tags and CSS would go here -->
</head>
<body>
    <!-- Main content area -->
    <h1>Hello World</h1>
    <p>This is a test page.</p>
    
    <!-- Footer section -->
    <footer>
        <p>&copy; 2024 Test</p>  
    </footer>
</body>
</html>
<!-- End of document -->
'''
    
    stripper = CommentStripper()
    stripped_code, stats = stripper.strip_comments(html_code)
    
    print("Original code:")
    print(html_code)
    print("\nStripped code:")
    print(stripped_code)
    print(f"\nStatistics: {stats}")

def test_integration_with_compression():
    """Test comment stripping integration with compression pipeline."""
    print("\n🔧 Testing Integration with Compression Pipeline")
    print("=" * 50)
    
    # Test with comment stripping disabled (default)
    python_code = '''# This is a test Python script
import os
import sys

def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    # Base cases
    if n <= 1:
        return n
    
    # Recursive case
    # F(n) = F(n-1) + F(n-2)
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Test the function
for i in range(10):  # Loop through first 10 numbers
    result = calculate_fibonacci(i)  # Calculate Fibonacci
    print(f"F({i}) = {result}")  # Print result
'''
    
    print("Testing compression without comment stripping:")
    packed_without = compress_with_dynamic_analysis(python_code)
    print(f"Original length: {len(python_code)} chars")
    print(f"Compressed length: {len(packed_without.text)} chars")
    print(f"Compression ratio: {((len(python_code) - len(packed_without.text)) / len(python_code) * 100):.1f}%")
    
    # Note: To test with comment stripping enabled, the config would need to be modified
    print("\nNote: To test with comment stripping enabled, set 'enabled': true in config/config.jsonc")

def test_edge_cases():
    """Test edge cases for comment stripping."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    # Test comments inside strings (should not be stripped)
    code_with_string_comments = '''
# Real comment
print("This is not a # comment in string")
print('Also not a // comment')
name = "John"  # This is a real comment
print(f"Hello {name}")  # Another real comment
'''
    
    stripper = CommentStripper()
    stripped, stats = stripper.strip_comments(code_with_string_comments)
    
    print("Code with comments in strings:")
    print(code_with_string_comments)
    print("\nAfter stripping:")
    print(stripped)
    
    # Test empty input
    empty_result, empty_stats = stripper.strip_comments("")
    print("\nEmpty input test:")
    print(f"Result: '{empty_result}'")
    print(f"Stats: {empty_stats}")

def main():
    """Run all comment stripping tests."""
    print("🧪 Comment Stripping Test Suite")
    print("=" * 50)
    
    # Test individual language comment strippers
    test_python_comment_stripping()
    test_javascript_comment_stripping() 
    test_c_cpp_comment_stripping()
    test_html_comment_stripping()
    
    # Test integration with compression
    test_integration_with_compression()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n✅ All tests completed!")
    print("\nTo enable comment stripping:")
    print("1. Edit config/config.jsonc")
    print("2. Set 'comment_stripping' -> 'enabled' to true")
    print("3. Configure language-specific settings as needed")

if __name__ == "__main__":
    main() 