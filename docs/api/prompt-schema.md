# ComfyUI Prompt API Schema

This document describes the JSON format used by the `/prompt` endpoint in ComfyUI.

## Overview

A ComfyUI prompt is a JSON object where:
- **Keys** are unique node IDs (arbitrary strings, typically numeric)
- **Values** are node objects containing `class_type`, `inputs`, and optional `_meta`

## Format Specification

### Node Object

```json
{
  "class_type": "KSampler",
  "inputs": {
    "seed": 123456,
    "steps": 20,
    "model": ["4", 0]
  },
  "_meta": {
    "title": "My KSampler"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class_type` | string | ✅ | Must match a registered node class in `NODE_CLASS_MAPPINGS` |
| `inputs` | object | ✅ | Input parameters for the node |
| `_meta` | object | ❌ | UI metadata (display title, position, etc.) |

### Input Values

Each input can be one of two types:

#### Direct Value
Any JSON-serializable value:
```json
{"seed": 123456, "steps": 20, "cfg": 7.0, "width": 512}
```

#### Node Link (Output Reference)
A 2-element array `[source_node_id, output_index]`:
```json
{"model": ["4", 0], "positive": ["6", 0]}
```

### Validation Rules

Based on `execution.py:validate_prompt()`:

1. Every node **must** have a `class_type` field
2. `class_type` **must** reference a registered node (existing in `NODE_CLASS_MAPPINGS`)
3. The prompt **must** contain at least one `OUTPUT_NODE`
4. All linked inputs must reference valid node IDs and output indices
5. Required inputs for each node class must be provided

### Common Error Types

| Type | Description |
|------|-------------|
| `missing_node_type` | Node has no `class_type` or class not found |
| `prompt_no_outputs` | No OUTPUT_NODE found in the prompt |

## Validation

Use the JSON Schema to validate prompts:

```bash
# Using ajv-cli
npx ajv validate -s schemas/prompt.json -d my_prompt.json

# Using Python
pip install jsonschema
python -c "
import json, jsonschema
with open('schemas/prompt.json') as f:
    schema = json.load(f)
with open('my_prompt.json') as f:
    prompt = json.load(f)
jsonschema.validate(prompt, schema)
print('Valid!')
"
```

## Minimal Example

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "model.safetensors"}
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "hello world", "clip": ["1", 1]}
  },
  "3": {
    "class_type": "EmptyLatentImage",
    "inputs": {"width": 512, "height": 512, "batch_size": 1}
  },
  "4": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 0, "steps": 20, "cfg": 7,
      "sampler_name": "euler", "scheduler": "normal",
      "denoise": 1, "model": ["1", 0],
      "positive": ["2", 0], "negative": ["2", 0],
      "latent_image": ["3", 0]
    }
  },
  "5": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["4", 0], "vae": ["1", 2]}
  },
  "6": {
    "class_type": "SaveImage",
    "inputs": {"filename_prefix": "test", "images": ["5", 0]}
  }
}
```

## Workflow vs Prompt Format

The **prompt format** (this schema) is the compact API format used by `/prompt`.

The **workflow format** is the UI serialization format that includes additional layout data, group information, and node positions. Workflows are converted to prompts before execution.

## Future Enhancements

This schema documents the current static format. Future versions may include:
- Per-node schemas with required/optional input definitions
- Dynamic schema generation based on installed custom nodes
- LLM-friendly structured output definitions
