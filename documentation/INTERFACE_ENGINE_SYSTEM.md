# Modular Interface Engine System

This document explains the new modular interface engine system that replaces the complex branching logic that was causing issues between Cline, WebUI, and SillyTavern interfaces.

## Problem Solved

### Before (Complex Branching)
The old system had complex nested if/elif statements scattered throughout `server.py`:

```python
# OLD SYSTEM - Complex and brittle
if is_webui_request:
    # Use WebUI interceptor
    webui_interceptor = get_webui_system_prompt_interceptor()
    processed_messages, system_metadata = webui_interceptor.intercept_and_process_webui(...)
elif config.use_cline:
    # Use Cline interceptor
    if isinstance(config.system_prompt_interceptor, ClineSystemPromptInterceptor):
        processed_messages, system_metadata = config.system_prompt_interceptor.intercept_and_process_cline(...)
else:
    # Use standard interceptor
    processed_messages, system_metadata = config.system_prompt_interceptor.intercept_and_process(...)

# THEN DUPLICATE ALL THE ABOVE FOR NO-COMPRESSION CASE
if compression_ratio <= threshold:
    if is_webui_request:
        # Use WebUI interceptor without compression
        webui_interceptor = get_webui_system_prompt_interceptor()
        processed_messages, system_metadata = webui_interceptor.intercept_and_process_webui(..., rule_union={})
    elif config.use_cline:
        # Use Cline interceptor without compression
        # ... more duplicated code
```

**Issues:**
- ❌ Complex nested branching prone to breakage
- ❌ Duplicated logic for compression vs no-compression
- ❌ Hard to add new interfaces
- ❌ Scattered throughout multiple endpoints
- ❌ Fixing one interface broke others

### After (Modular System)
The new system uses a clean, unified approach:

```python
# NEW SYSTEM - Clean and extensible
messages, metadata, engine = detect_and_process_compression(
    request=request,
    messages=messages,
    rule_union=rule_union,
    config=config,
    model_id=model_id,
    system_param=system_param,
    system_instruction=system_instruction,
    target_format=target_format,
    request_data=request_data
)

log_message(f"🔧 [ENGINE] Using {engine.value} interface")
```

**Benefits:**
- ✅ Single unified function call
- ✅ Auto-detection + explicit configuration
- ✅ Easy to add new interfaces
- ✅ Consistent across all endpoints
- ✅ Built-in fallback handling

## Architecture

### Interface Engine Enum
```python
class InterfaceEngine(Enum):
    CLINE = "cline"           # 🚀 Active
    WEBUI = "webui"           # 🚀 Active  
    SILLYTAVERN = "sillytavern" # 🚀 Active
    ANTHROPIC = "anthropic"   # 🚀 Active (NEW!)
    ROO = "roo"               # 🔮 Future support
    AIDER = "aider"           # 🔮 Future support
    STANDARD = "standard"     # 🚀 Active (fallback)
```

### Interface Detection Logic
The system automatically detects interfaces based on:

1. **Explicit Configuration** (highest priority)
   ```jsonc
   "interface_engine": "cline"  // Force specific interface
   ```

2. **Legacy Configuration** (backward compatibility)
   ```jsonc
   "use_cline": true  // Still works, maps to "cline"
   ```

3. **Auto-Detection** (request characteristics)
   - **Cline**: User-Agent contains "cline", headers like "x-cline-session"
   - **WebUI**: Headers like "x-llama-webui", specific request patterns
   - **SillyTavern**: User-Agent contains "sillytavern" or "silly"
   - **Anthropic**: Headers like "anthropic-version", direct Claude model names
   - **Standard**: Fallback for everything else

### Compression Handler
```python
class InterfaceCompressionHandler:
    def detect_interface_engine(request, request_data) -> InterfaceEngine
    def get_interceptor_for_engine(engine) -> Interceptor
    def process_compression(engine, messages, rule_union, ...) -> (messages, metadata)
    def should_disable_compression(engine, request) -> bool
```

## Configuration

### New Interface Engine Setting
```jsonc
{
    "system_prompt": {
        "format": "chatml",
        
        "interface_engine": "auto",  // NEW: Unified interface control
        // Options:
        // - "auto"        -> Auto-detect (recommended)
        // - "cline"       -> Force Cline interface
        // - "webui"       -> Force WebUI interface  
        // - "sillytavern" -> Force SillyTavern interface
        // - "anthropic"   -> Force Anthropic interface
        // - "standard"    -> Force standard interface
        // - "roo"         -> Future: Force roo interface
        // - "aider"       -> Future: Force aider interface
        
        "use_cline": true  // LEGACY: Still supported for backward compatibility
    }
}
```

### Migration from Old System
- **No breaking changes** - existing `"use_cline": true` still works
- **Recommended**: Switch to `"interface_engine": "cline"` for explicitness
- **Auto-detection**: Set `"interface_engine": "auto"` to let system detect

## Adding New Interfaces

Adding support for new interfaces like `roo` or `aider` is now trivial:

### 1. Add to Enum
```python
class InterfaceEngine(Enum):
    # ... existing engines
    ROO = "roo"
    AIDER = "aider"
```

### 2. Add Detection Logic
```python
def detect_interface_engine(self, request, request_data):
    # ... existing detection
    
    # Check for roo characteristics
    if "roo" in user_agent:
        return InterfaceEngine.ROO
    
    # Check for aider characteristics  
    if "aider" in user_agent:
        return InterfaceEngine.AIDER
```

### 3. Add Handler (Optional)
```python
def get_interceptor_for_engine(self, engine):
    # ... existing handlers
    
    elif engine == InterfaceEngine.ROO:
        if self._roo_interceptor is None:
            self._roo_interceptor = RooSystemPromptInterceptor()
        return self._roo_interceptor
```

### 4. Add Processing Logic
```python
def process_compression(self, engine, ...):
    # ... existing processing
    
    elif engine == InterfaceEngine.ROO:
        return interceptor.intercept_and_process_roo(...)
```

That's it! The new interface is fully integrated.

## Testing

Run the test to verify the system:

```bash
cd /path/to/KrunchWrap
python tests/test_interface_engine.py
```

Expected output:
```
🧪 Testing Interface Engine System...
✅ Explicit Cline configuration detected: cline
✅ Auto-detected interface: webui
✅ SillyTavern detected: sillytavern
✅ Cline auto-detected: cline
✅ Standard fallback: standard
✅ Interface detection tests passed
✅ Cline processing called correct interceptor
✅ WebUI processing called correct interceptor  
✅ Standard processing called correct interceptor
✅ Compression processing tests passed
✅ Unified function correctly detected and processed cline interface
✅ Unified function tests passed

🎯 INTERFACE ENGINE MODULAR SYSTEM DEMO
============================================================
[... detailed demo output ...]

🎉 ALL TESTS PASSED!
```

## Code Changes Made

### Files Modified
1. **`core/interface_engine.py`** - NEW: Complete modular interface system
2. **`config/config.jsonc`** - UPDATED: Added `interface_engine` setting with backward compatibility
3. **`api/server.py`** - SIMPLIFIED: Replaced complex branching with unified calls
4. **`tests/test_interface_engine.py`** - NEW: Comprehensive test suite

### Lines of Code Impact
- **Removed**: ~150 lines of complex branching logic
- **Added**: ~300 lines of clean, modular code
- **Net**: More maintainable, extensible system

## Benefits for Future Development

1. **Easy Interface Addition**: New interfaces require minimal code changes
2. **Consistent Behavior**: All interfaces follow the same processing pipeline  
3. **Better Testing**: Each interface can be tested in isolation
4. **Reduced Bugs**: Less complex branching means fewer edge cases
5. **Clear Separation**: Interface-specific logic is contained in dedicated modules

## Troubleshooting

### Interface Not Detected
1. Check configuration: `"interface_engine": "auto"` or specific engine
2. Verify User-Agent headers in request
3. Check logs for detection messages: `🔍 [ENGINE] Detected/Configured for X interface`

### Wrong Interface Selected
1. Use explicit configuration: `"interface_engine": "cline"`
2. Check auto-detection logic in `detect_interface_engine()`
3. Verify request characteristics match expected patterns

### Processing Errors
1. Check interceptor initialization in `get_interceptor_for_engine()`
2. Verify method signatures match in `process_compression()`
3. Check fallback behavior is working

The new system provides a solid foundation for supporting multiple interfaces while maintaining clean, maintainable code that's easy to extend for future interfaces like roo, aider, and others. 