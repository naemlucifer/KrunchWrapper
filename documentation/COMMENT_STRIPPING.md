# Comment Stripping

The KrunchWrapper comment stripping feature automatically removes comments from code before compression, providing additional token savings while preserving code functionality.

## Overview

Comment stripping is a pre-compression step that removes comments from detected code, which can provide significant token savings. This feature:

- **Runs before main compression**: Comments are stripped before dynamic dictionary compression
- **Language-aware**: Supports multiple programming languages with appropriate comment syntax
- **Safe**: Preserves comments inside strings and important comments like license headers
- **Configurable**: Can be enabled/disabled globally or per language
- **Detailed logging**: Shows token savings from comment removal

## Supported Languages

| Language | Single-line Comments | Multi-line Comments | File Extensions |
|----------|---------------------|---------------------|-----------------|
| Python | `#` | `"""` `'''` | `.py`, `.pyw` |
| JavaScript/TypeScript | `//` | `/* */` | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs` |
| C/C++ | `//` | `/* */` | `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx` |
| HTML | - | `<!-- -->` | `.html`, `.htm`, `.xhtml` |
| CSS | - | `/* */` | `.css`, `.scss`, `.sass`, `.less` |
| SQL | `--` | `/* */` | `.sql` |
| Shell/Bash | `#` | - | `.sh`, `.bash`, `.zsh`, `.fish` |

## Configuration

Comment stripping is configured in `config/config.jsonc` under the `comment_stripping` section:

```jsonc
"comment_stripping": {
    /* Enable comment stripping functionality */
    "enabled": false,
    
    /* Preserve important comments like license headers */
    "preserve_license_headers": true,
    
    /* Preserve shebang lines (#! at start of files) */
    "preserve_shebang": true,
    
    /* Preserve docstrings in languages that support them */
    "preserve_docstrings": true,
    
    /* Minimum line length to keep after stripping comments */
    "min_line_length_after_strip": 3,
    
    /* Language-specific settings */
    "languages": {
        "python": true,
        "javascript": true,
        "c_cpp": true,
        "html": true,
        "css": true,
        "sql": true,
        "shell": true
    }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable comment stripping |
| `preserve_license_headers` | boolean | `true` | Keep copyright/license comments |
| `preserve_shebang` | boolean | `true` | Keep shebang lines (#!) |
| `preserve_docstrings` | boolean | `true` | Keep Python docstrings |
| `min_line_length_after_strip` | integer | `3` | Minimum line length to preserve |
| `languages.{lang}` | boolean | `true` | Enable per language |

## How It Works

### 1. Language Detection

The system detects the programming language using:

1. **File extension** (if filename provided)
2. **Content analysis** (keywords, syntax patterns)

### 2. Comment Removal

Comments are removed using language-specific patterns:

- **Single-line comments**: Removed from end of lines
- **Multi-line comments**: Completely removed
- **String safety**: Comments inside strings are preserved
- **Important comments**: License headers and shebangs are preserved

### 3. Token Calculation

Token savings are calculated using:

- **Tiktoken**: Accurate token counting when available
- **Estimation**: Fallback using ~4 characters per token

## Example Results

### Python Code

**Before:**
```python
# This is a Python script
import os
import sys  # Standard imports

def hello_world():
    """This docstring is preserved."""
    # This comment is removed
    name = "World"  # Another comment
    print(f"Hello, {name}!")
    return name
```

**After:**
```python

import os
import sys

def hello_world():
    """This docstring is preserved."""
    
    name = "World"
    print(f"Hello, {name}!")
    return name
```

**Savings:** 56.3% token reduction (58 tokens saved)

### JavaScript Code

**Before:**
```javascript
// JavaScript example
function greetUser(name) {
    /* Multi-line comment
       explaining the function */
    const greeting = "Hello"; // Inline comment
    console.log(`${greeting}, ${name}!`);
    return greeting;
}
```

**After:**
```javascript

function greetUser(name) {
    
    const greeting = "Hello";
    console.log(`${greeting}, ${name}!`);
    return greeting;
}
```

**Savings:** 50.6% token reduction (41 tokens saved)

## Integration with Compression

Comment stripping integrates seamlessly with the compression pipeline:

1. **Pre-processing**: Comments are stripped before dynamic analysis
2. **Fallback handling**: If compression is disabled, comment-stripped code is still returned
3. **Error handling**: On compression errors, comment-stripped code is preserved
4. **Logging**: Both comment stripping and compression results are logged

## Safety Features

### String Protection

Comments inside strings are never removed:

```python
# This comment will be removed
print("This # is not a comment")  # This comment will be removed
```

### Important Comment Preservation

License headers and copyright notices are automatically preserved:

```python
# Copyright 2024 Example Corp
# Licensed under MIT License
```

### Shebang Preservation

Script execution shebangs are always preserved:

```bash
#!/bin/bash
# This comment is removed
echo "Hello World"
```

## Performance Impact

Comment stripping adds minimal overhead:

- **Language detection**: Very fast pattern matching
- **Comment removal**: Efficient regex-based processing
- **Token counting**: Optional tiktoken integration
- **Memory usage**: Minimal additional memory

## Logging Output

When enabled, comment stripping provides detailed logging:

```
📝 Comment Stripping Results (python):
   Characters: 412 → 166 (246 saved, 59.7%)
   Tokens: 103 → 45 (58 saved, 56.3%)
```

## Testing

Test the comment stripping functionality:

```bash
# Run the test suite
python tests/test_comment_stripping.py
```

The test suite covers:
- Multiple programming languages
- Edge cases (strings, empty input)
- Integration with compression pipeline
- Token savings calculation

## Best Practices

### When to Enable

Comment stripping is most beneficial for:

- **Code-heavy prompts** with many comments
- **Educational content** with extensive commenting
- **Documentation examples** with inline explanations
- **Large codebases** where every token counts

### When to Disable

Consider disabling for:

- **Short prompts** where overhead outweighs benefits
- **Documentation** where comments are essential
- **Code reviews** where comments need to be preserved
- **Debugging** where comments provide context

### Language-Specific Considerations

- **Python**: Docstrings are preserved by default
- **JavaScript**: JSDoc comments are removed (configure if needed)
- **C/C++**: Doxygen comments are removed
- **HTML**: All comments removed (often safe)

## Troubleshooting

### Common Issues

**Language not detected:**
- Check file extension or content patterns
- Add explicit language indicators

**Comments not removed:**
- Verify language is enabled in config
- Check for comments inside strings

**Important comments removed:**
- Enable preservation options
- Check license header patterns

### Debug Logging

Enable debug logging to see detailed processing:

```jsonc
"logging": {
    "log_level": "DEBUG",
    "verbose_logging": true
}
```

## Future Enhancements

Potential future improvements:

- **Custom preservation patterns**
- **Language-specific docstring handling**
- **Context-aware comment removal**
- **Integration with language servers**
- **Custom comment syntax support** 