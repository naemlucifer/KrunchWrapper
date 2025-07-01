# Cline Integration for KrunchWrapper

This document explains how to use KrunchWrapper with Cline, a VS Code extension for interacting with various LLM APIs.

## Overview

Cline supports multiple LLM providers including OpenAI, Anthropic, Google (Gemini), and many others through a unified API. KrunchWrapper now includes a specialized system prompt interceptor for Cline that can properly handle the provider-specific formatting required by different LLM backends.

## How It Works

The `ClineSystemPromptInterceptor` extends the standard `SystemPromptInterceptor` with Cline-specific functionality:

1. It detects the provider from the model ID (format: `provider/model`)
2. It applies the appropriate system prompt format based on the detected provider
3. It handles provider-specific message formats and system prompt placement

## Configuration

To enable Cline integration in KrunchWrapper, update your `config/config.jsonc` file:

```jsonc
{
    "system_prompt": {
        "format": "claude",  // Default format when not using Cline
        "use_cline": true    // Enable Cline-specific handling
    }
}
```

**Note**: This is the single source of truth for Cline integration. The `use_cline` setting was previously duplicated in `config/server.jsonc` but has been consolidated to avoid confusion.

## Supported Providers

The Cline interceptor supports all providers available in Cline:

| Provider | Model ID Format | System Prompt Format |
|----------|----------------|---------------------|
| Anthropic | `anthropic/claude-*` | Claude Messages API |
| OpenAI | `openai/*` | ChatGPT/ChatML |
| OpenAI Native | `openai-native/*` | ChatGPT/ChatML |
| Google/Gemini | `google/*` or `gemini/*` | Gemini API |
| DeepSeek | `deepseek/*` | ChatML (R1 format for Reasoner models) |
| Qwen | `qwen/*` | Qwen format |
| Gemma | `gemma/*` | Gemma format |
| Mistral | `mistral/*` | ChatGPT/ChatML |
| X-AI/XAI (Grok) | `x-ai/*` or `xai/*` | ChatGPT/ChatML |
| Ollama | `ollama/*` | ChatGPT/ChatML |
| AWS Bedrock | `bedrock/*` | Claude format for Claude models, ChatGPT/ChatML for others |
| Google Vertex AI | `vertex/*` | Gemini format for Gemini models, ChatGPT/ChatML for others |
| Fireworks | `fireworks/*` | ChatGPT/ChatML |
| Together AI | `together/*` | ChatGPT/ChatML |
| LM Studio | `lmstudio/*` | ChatGPT/ChatML |
| LiteLLM | `litellm/*` | ChatGPT/ChatML |
| OpenRouter | `openrouter/*` | ChatGPT/ChatML |
| Nebius | `nebius/*` | ChatGPT/ChatML |
| Doubao | `doubao/*` | ChatGPT/ChatML |
| AskSage | `asksage/*` | ChatGPT/ChatML |
| SambaNova | `sambanova/*` | ChatGPT/ChatML |
| Cerebras | `cerebras/*` | ChatGPT/ChatML |
| SAP AI Core | `sapaicore/*` | ChatGPT/ChatML |
| Requesty | `requesty/*` | ChatGPT/ChatML |
| VS Code LM | `vscode-lm/*` | ChatGPT/ChatML |

For any other providers, the interceptor defaults to ChatGPT/ChatML format.

## Using with Cline

To use KrunchWrapper with Cline:

1. Start the KrunchWrapper proxy server with Cline integration enabled
2. Configure Cline to use the KrunchWrapper proxy URL as its API endpoint
3. Use Cline normally - KrunchWrapper will automatically handle compression and system prompt formatting

### Example Configuration in Cline

In VS Code, open your Cline settings and set:

```json
{
    "cline.apiProvider": "openai",
    "cline.openAiBaseUrl": "http://localhost:5001/v1",
    "cline.openAiApiKey": "dummy-key"  // KrunchWrapper proxy doesn't validate keys
}
```

## How It Processes Requests

When a request comes through the KrunchWrapper proxy with Cline integration enabled:

1. The proxy detects if the request is a Cline request by checking the model ID format
2. It compresses the messages if they exceed the minimum character threshold
3. It determines the appropriate system prompt format based on the provider
4. It processes and merges system prompts with KrunchWrapper compression instructions
5. It formats the request according to the provider's requirements
6. It forwards the processed request to the target LLM API

## Debugging

If you encounter issues with Cline integration:

1. Check the KrunchWrapper server logs for errors
2. Verify that `use_cline` is set to `true` in your config
3. Make sure the model ID follows the expected format (`provider/model`)
4. Check that the target LLM API URL is correctly configured

## Advanced Usage

### Custom Provider Mappings

If you need to add custom provider mappings, you can modify the `_determine_target_format` method in `core/cline_system_prompt_interceptor.py`.

### Handling Special Cases

Some models require special handling (like DeepSeek Reasoner models). The interceptor includes logic to handle these cases, and you can extend it for additional special cases if needed. 