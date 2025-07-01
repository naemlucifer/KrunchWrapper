# KrunchWrapper WebUI

This WebUI is derived from the [llama.cpp](https://github.com/ggml-org/llama.cpp) project and has been adapted for use with KrunchWrapper.

## Attribution

- **Original Source**: [llama.cpp WebUI](https://github.com/ggml-org/llama.cpp)
- **Original License**: MIT License
- **Original Copyright**: Copyright (c) 2023 Georgi Gerganov
- **Modifications**: Adapted for KrunchWrapper compression proxy integration

## License

This WebUI code is licensed under the MIT License. See the [LICENSE](./LICENSE) file for the full license text.

## Modifications Made

This WebUI has been modified from the original llama.cpp version to:

- Connect to KrunchWrapper's compression proxy API
- Dynamically read configuration from `../config/server.jsonc`
- Support configurable API and WebUI ports
- Integrate with KrunchWrapper's start scripts

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Configuration

The WebUI automatically reads configuration from `../config/server.jsonc` to determine:
- API server port (for proxy configuration)
- WebUI development server port

## Original Project

For the original llama.cpp project and WebUI, please visit:
https://github.com/ggml-org/llama.cpp