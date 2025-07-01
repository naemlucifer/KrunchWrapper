# KrunchWrapper Dictionary Workflows

This document outlines the step-by-step workflows for creating new compression dictionaries and testing them within the KrunchWrapper project.

## Workflow 1: Standing Up a New Dictionary

### Step 1: Discover Available Unicode Symbols
```bash
python scripts/discover_unicode_symbols.py
```
This script systematically scans Unicode blocks to find suitable symbols for compression that are:
- Single characters
- Printable and visible
- Compatible with programming syntax
- Understandable by LLMs

The discovered symbols are saved to `generated_symbols.py` for use in dictionary creation.

### Step 2: Build the Initial Dictionary
Choose the appropriate script based on your language needs:

**For Python-specific dictionary:**
```bash
python scripts/build_python_dict.py
```
This creates a Python compression dictionary with Unicode symbols assigned to Python keywords, built-ins, exceptions, and standard library modules.

**For comprehensive multi-language dictionaries:**
```bash
python scripts/build_comprehensive_dicts.py
```
This builds dictionaries for multiple languages (Python, JavaScript, Go, Rust) with language-specific tokens.

**For enhanced dictionaries with common words:**
```bash
python scripts/build_enhanced_dicts.py
```
This creates dictionaries with a larger symbol pool and support for common English words.

### Step 3: Analyze Codebases for Compression Opportunities

**For single file analysis:**
```bash
python scripts/analyze_single_file.py --file [path_to_file]
```
This analyzes a specific file for compression opportunities.

**For large codebase analysis:**
```bash
python scripts/analyze_large_python_codebase.py --directory [path_to_codebase]
```
This performs comprehensive analysis of an entire codebase to find common patterns.

**For token frequency analysis:**
```bash
python scripts/analyze_codebase_tokens.py --directory [path_to_codebase]
```
This discovers frequently used tokens across a codebase.

### Step 4: Enhance the Dictionary with Discovered Opportunities

**Add opportunities from single file:**
```bash
python scripts/add_single_file_opportunities.py --dict [path_to_dict] --words [path_to_words.txt]
```

**Add opportunities from codebase analysis:**
```bash
python scripts/add_opportunities_to_dict.py --dict [path_to_dict] --opportunities [path_to_opportunities_file]
```

**Automatically expand dictionaries:**
```bash
python scripts/auto_expand_dictionaries.py --directory [path_to_codebase]
```

**Extract tokens from documentation (optional):**
```bash
python scripts/extract_from_devdocs.py --output [output_file]
```
This extracts common programming tokens from DevDocs documentation.

### Step 5: Clean and Validate the Dictionary
```bash
python scripts/clean_dictionary.py --dict [path_to_dict]
```
This removes problematic Unicode artifacts and specialized tokens.

```bash
python scripts/check_dict_conflicts.py
```
This validates that there are no conflicts between dictionary entries.

## Workflow 2: Testing a Dictionary

### Step 1: Debug Compression Process
```bash
python scripts/debug_compression.py --file [sample_file] --dict [path_to_dict]
```
This provides detailed information about what's happening during compression on a sample file.

### Step 2: Analyze Already Compressed Files
```bash
python scripts/analyze_compressed_opportunities.py --file [compressed_file]
```
This identifies patterns in compressed text and suggests new tokens for dictionaries.

### Step 3: Validate Dictionary Integrity
```bash
python scripts/check_dict_conflicts.py
```
This checks for duplicate short tokens, zero-gain mappings, and cross-file conflicts.

### Step 4: Iterative Improvement
After testing, you may want to:
1. Return to the enhancement step to add more valuable tokens
2. Clean the dictionary again to remove any problematic entries
3. Re-test with sample files to measure compression improvements

## Best Practices

1. **Start Small**: Begin with a core set of language-specific tokens before expanding
2. **Prioritize by Frequency**: Focus on adding tokens that appear frequently in real code
3. **Avoid Conflicts**: Ensure no two long tokens resolve to the same short macro
4. **Test Thoroughly**: Use real codebases to validate compression effectiveness
5. **Iterate**: Dictionary building is an iterative process; continue to refine based on analysis results 