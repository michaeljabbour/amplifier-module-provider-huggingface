# amplifier-module-provider-huggingface

HuggingFace Inference API provider module for [Amplifier](https://github.com/microsoft/amplifier).

## Overview

Provides access to HuggingFace models via the Inference API (Serverless and Inference Endpoints).
Uses the OpenAI-compatible chat completions format through `huggingface_hub`.

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | secret | `$HF_TOKEN` | HuggingFace API token |
| `base_url` | text | `https://router.huggingface.co/v1` | API base URL (custom for Inference Endpoints) |
| `default_model` | text | `meta-llama/Llama-3.3-70B-Instruct` | Default model to use |

## Supported Models

Any model on HuggingFace Hub that supports chat completion, including:

- `meta-llama/Llama-3.3-70B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-72B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`

## Setup

```bash
export HF_TOKEN="hf_..."
```

## License

MIT
