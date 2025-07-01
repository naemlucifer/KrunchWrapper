# KrunchWrapper Tests Documentation

This document provides an overview of the test files in the KrunchWrapper project and recommends a testing workflow.

## Test Files Overview

| File | Description |
|------|-------------|
| `test_compression.py` | Basic compression tests for Python and JavaScript code samples. Tests the core compression and decompression functionality with simple code snippets. |
| `test_roundtrip.py` | Simple test to verify that compression followed by decompression returns the original text. |
| `test_fallback.py` | Tests the fallback compression mechanism when no language is specified or an invalid language is provided. |
| `test_server.py` | Tests the KrunchWrapper server's chat completion endpoint with a sample Python code explanation request. |
| `test_enhanced_compression.py` | Tests compression with the enhanced token dictionary on large Python files from PyTorch. Measures compression ratios and validates roundtrip accuracy. |
| `test_real_file_compression.py` | Comprehensive test on real Python files, including token analysis, compression statistics, and detailed breakdown of compression efficiency. |

## Recommended Testing Workflow

1. **Unit Testing**
   - Start with `test_roundtrip.py` to verify basic compression/decompression functionality
   - Run `test_compression.py` to check language-specific compression
   - Run `test_fallback.py` to ensure the system handles unknown languages gracefully

2. **Integration Testing**
   - Run `test_real_file_compression.py` to test compression on actual code files
   - Run `test_enhanced_compression.py` to validate the enhanced token dictionary performance

3. **Server Testing**
   - Ensure the server is running
   - Run `test_server.py` to verify the API endpoints are working correctly

## Running Tests

Individual tests can be run directly:

```bash
# Run a specific test
python tests/test_compression.py

# Run all tests
for test in tests/test_*.py; do python "$test"; done
```

## Test Environment

For consistent results, tests should be run in a controlled environment:

1. Use a clean Python environment with all dependencies installed
2. Ensure all language dictionaries are up-to-date
3. For server tests, make sure the server is running on the expected port

## Extending Tests

When adding new features:

1. Create a new test file following the naming convention `test_feature_name.py`
2. Include both basic functionality tests and edge cases
3. Ensure all tests validate roundtrip accuracy (original text = decompressed text)
4. Add performance metrics where appropriate (compression ratio, processing time) 