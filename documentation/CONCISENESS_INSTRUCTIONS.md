# Configurable Conciseness Instructions

The KrunchWrapper system now supports configurable conciseness instructions that can be injected into system prompts to encourage more focused, brief responses from language models.

## Overview

This feature allows you to:
- Add configurable instructions to system prompts that encourage concise responses
- Customize instructions based on context (programming language, user intent, etc.)
- Control when and how these instructions are applied
- Format instructions in different styles and positions

## Configuration

### Main Configuration (`config/config.jsonc`)

Add this section to your main configuration:

```jsonc
"conciseness_instructions": {
    /* Enable injection of conciseness instructions into system prompts */
    "enabled": true,
    
    /* Configuration file for conciseness instructions */
    "instructions_file": "conciseness-instructions.jsonc",
    
    /* Position where to inject instructions relative to compression decoder
       Options: "before", "after", "separate_section" */
    "injection_position": "after",
    
    /* Whether to apply instructions only when compression is active */
    "only_with_compression": false
}
```

### Instructions Configuration (`config/conciseness-instructions.jsonc`)

This file contains the actual instructions and their formatting rules:

```jsonc
{
    /* Core instructions - always included when enabled */
    "core_instructions": [
        "Be concise and direct in your responses",
        "Respond as briefly and precisely as possible", 
        "Provide only essential information unless specifically asked for details",
        "Limit explanations unless explicitly requested"
    ],
    
    /* Code-specific instructions - included when programming language is detected */
    "code_instructions": [
        "Use minimal but descriptive function and variable names",
        "Minimize comments unless they clarify complex logic", 
        "Prefer concise code patterns over verbose alternatives",
        "Avoid unnecessary intermediate variables"
    ],
    
    /* ... additional configuration options ... */
}
```

## Usage Examples

### Basic Usage

When enabled, the system automatically injects appropriate conciseness instructions based on:
- Whether compression is active
- The detected programming language
- Keywords in the user's request
- Custom instruction sets you've enabled

### Example System Prompts

**Without Conciseness Instructions:**
```
You will read python code in a compressed DSL. Apply these symbol substitutions when understanding and responding: α=function, β=return, γ=variable. This reduces token usage.
```

**With Conciseness Instructions:**
```
You will read python code in a compressed DSL. Apply these symbol substitutions when understanding and responding: α=function, β=return, γ=variable. This reduces token usage.

📝 CONCISENESS GUIDELINES:
• Be concise and direct in your responses
• Respond as briefly and precisely as possible
• Provide only essential information unless specifically asked for details
• Limit explanations unless explicitly requested
• Use minimal but descriptive function and variable names
• Minimize comments unless they clarify complex logic
• Identify the specific problem first
• Provide the corrected code immediately after diagnosis
• Explain the fix in one concise sentence
```

## Configuration Options

### Injection Position

Control where instructions appear relative to compression decoder:
- `"before"`: Instructions appear before compression decoder
- `"after"`: Instructions appear after compression decoder (default)
- `"separate_section"`: Instructions appear in a distinct section with separators

### Conditional Instructions

Instructions can be triggered based on keywords in user content:

```jsonc
"conditional_instructions": {
    "debugging_context": {
        "enabled": true,
        "trigger_keywords": ["debug", "error", "fix", "broken", "not working"],
        "instructions": [
            "Identify the specific problem first",
            "Provide the corrected code immediately after diagnosis"
        ]
    }
}
```

### Custom Instruction Sets

Enable special modes for specific use cases:

```jsonc
"custom_instruction_sets": {
    "ultra_brief": {
        "enabled": false,
        "instructions": [
            "Respond in the fewest words possible",
            "Use sentence fragments if they convey the meaning"
        ]
    }
}
```

### Formatting Options

Customize how instructions are presented:

```jsonc
"formatting": {
    "prefix": "💡 RESPONSE GUIDANCE:",
    "instruction_separator": "\n• ",
    "use_section_wrapper": true,
    "section_wrapper": {
        "start": "\n\n📝 CONCISENESS GUIDELINES:\n",
        "end": "\n"
    }
}
```

## Advanced Features

### Context-Aware Selection

The system automatically selects relevant instructions based on:
- **Programming Language**: Adds code-specific instructions when language is detected
- **User Intent**: Analyzes keywords to determine if user is debugging, asking for explanations, etc.
- **Content Length**: Different instructions for short vs. long responses

### Instruction Limits

Control how many instructions are included:

```jsonc
"selection_criteria": {
    "max_instructions": 6,
    "prioritize_by_order": true,
    "always_include_core": true
}
```

### Conditional Application

Instructions can be applied only under certain conditions:
- Only when compression is active (`only_with_compression: true`)
- Only for specific programming languages
- Only when certain keywords are detected

## Testing

Run the test script to verify your configuration:

```bash
python tests/test_conciseness_instructions.py
```

This will show:
- Whether the feature is enabled
- What instructions are generated for different contexts
- How the system prompt looks with instructions included

## Benefits

### Token Efficiency
- Reduces verbose responses, saving tokens
- Encourages focused, relevant answers
- Particularly effective with code generation

### Consistency
- Standardizes response style across interactions
- Ensures consistent brevity preferences
- Maintains quality while reducing length

### Flexibility
- Completely configurable per use case
- Can be enabled/disabled without code changes
- Context-aware instruction selection

## Best Practices

1. **Start Conservative**: Begin with core instructions only, then add context-specific ones
2. **Test Thoroughly**: Use the test script to verify instruction generation
3. **Monitor Results**: Track whether responses become more concise as expected
4. **Customize Gradually**: Add conditional instructions as you identify patterns
5. **Balance Quality**: Ensure conciseness doesn't sacrifice accuracy or helpfulness

## Troubleshooting

### Instructions Not Appearing
- Check that `enabled: true` in main config
- Verify the instructions file path is correct
- Ensure the instructions file has valid JSONC syntax

### Too Many/Few Instructions
- Adjust `max_instructions` in selection criteria
- Review conditional instruction trigger keywords
- Check if `only_with_compression` is limiting application

### Formatting Issues
- Verify section wrapper configuration
- Check instruction separator settings
- Test different injection positions

## Integration with Other Features

This feature works seamlessly with:
- **Compression System**: Can be applied with or without compression
- **Language Detection**: Automatically includes language-specific instructions
- **System Prompt Interceptor**: Properly merges with existing system prompts
- **Multiple Formats**: Works with Claude, ChatGPT, Gemini, and other formats 