"""OpenAPI spec and Swagger UI for the wrapper API.

The wrapper exposes its own docs so consumers never have to learn the
internal ComfyUI endpoints. Served at /api/wrapper/openapi.json and
/api/wrapper/docs.
"""

WRAPPER_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "ComfyUI Wrapper API",
        "description": (
            "Simplified generation API on top of ComfyUI. Submit a prompt (plus any input image, "
            "video or audio the workflow takes) and the wrapper handles the rest: it builds the "
            "prompt graph, downloads any missing model files, queues the job on the ComfyUI "
            "execution server and streams back the finished artifact. Every endpoint is "
            "synchronous and returns the file itself; the X-Wrapper-Models / X-Wrapper-Settings "
            "response headers report which checkpoints and resolved parameters produced it. "
            "VRAM and RAM are released automatically as soon as a job finishes; POST "
            "/api/wrapper/free releases them on demand.\n\n"
            "The MiniMax H3 video endpoints are capped to what the host can actually finish: "
            "clip length and canvas*frames are checked before the job is queued and an "
            "over-budget request gets a 400 rather than OOM-killing the server. The form "
            "defaults are the largest values measured to complete on this machine."),
        "version": "0.1.0",
    },
    "servers": [{"url": "/"}],
    "paths": {
        "/api/wrapper/{workflow}/generate": {
            "post": {
                "summary": "Run a workflow synchronously and download the result",
                "description": "Uploads the input image, ensures the workflow's model files are present (downloading them if missing), runs the workflow to completion and returns the final artifact (image, video, audio, 3D asset, ...) directly as the file download. Model downloads or the fp4 conversion can take a while on first use.",
                "operationId": "generate",
                "parameters": [
                    {"name": "workflow", "in": "path", "required": True, "schema": {"type": "string", "enum": ["flux2klein9b"]}, "description": "Workflow to run; see GET /api/wrapper/workflows."},
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["prompt", "image"],
                                "properties": {
                                    "prompt": {"type": "string", "description": "The edit instruction, e.g. 'make it snow'."},
                                    "image": {"type": "string", "format": "binary", "description": "Input image (png/jpg/webp)."},
                                    "negative_prompt": {"type": "string", "default": "", "description": "Things to avoid in the output."},
                                    "seed": {"type": "integer", "minimum": 0, "description": "Random seed; a random one is used when omitted."},
                                    "steps": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 20},
                                    "cfg": {"type": "number", "minimum": 0, "maximum": 100, "default": 5.0, "description": "CFG scale used by the guider."},
                                    "megapixels": {"type": "number", "minimum": 0.01, "maximum": 16.0, "default": 1.0, "description": "Input image is scaled to roughly this many megapixels before editing."},
                                    "timeout": {"type": "integer", "minimum": 0, "maximum": 86400, "default": 0, "description": "Seconds to wait before returning 504. 0 (default) waits until the workflow finishes, however long that takes; the timeout response includes the job_id to poll via /jobs/{job_id}."},
                                    "free_vram": {"type": "boolean", "default": True, "description": "Release VRAM and RAM as soon as this job finishes (set to false to keep models loaded between jobs, e.g. batch runs)."},
                                    "quantization": {"type": "string", "enum": ["fp8", "fp4"], "default": "fp8", "description": "Weight precision. fp8 uses the shipped fp8 checkpoint (loaded as bf16 on MPS). fp4 converts the checkpoint to NVFP4 (~half the size) for CUDA/CPU; on MPS it automatically falls back to fp8."},
                                    "vram": {"type": "string", "enum": ["auto", "low", "normal", "high"], "default": "auto", "description": "VRAM management for this job: auto (default) lets ComfyUI's dynamic VRAM decide, streaming model weights to/from RAM so a model larger than the GPU never OOMs. low/normal/high force the legacy vram state for this job (advisory while dynamic VRAM is active)."},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "The final output image (image/png), shown inline in the docs.",
                        "headers": {
                            "X-Wrapper-Models": {"schema": {"type": "string"}, "description": "The checkpoint files this run actually loaded. The setup may substitute a variant already on disk for the requested quantization, so this is the authoritative record of what produced the output."},
                            "X-Wrapper-Settings": {"schema": {"type": "string"}, "description": "The resolved width/height/duration/steps/scheduler after clamping and rounding."},
                            "X-Wrapper-Note": {"schema": {"type": "string"}, "description": "Advisory note from the setup (model substitution, canvas downscale, ...). Absent when there is nothing to report."},
                            "X-Wrapper-Job-Id": {"schema": {"type": "string", "format": "uuid"}, "description": "Id of the job that produced this file; usable with /api/wrapper/jobs/{job_id}."},
                            "X-Wrapper-Prompt-File": {"schema": {"type": "string"}, "description": "Output-directory path of the sidecar JSON saved next to the artifact, recording the prompt that generated it: the free-form input, the prompt the LLM wrote from it (or the raw_prompt used verbatim), the model, the H3 mode and the settings. The same prompt is also written to the server log. Present only on workflows whose prompts are generated."},
                        },
                        "content": {
                            "image/png": {"schema": {"type": "string", "format": "binary"}},
                            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
                        },
                    },
                    "400": {
                        "description": "Invalid request: unknown workflow, missing prompt/image, bad parameter value, or a required model could not be downloaded (e.g. gated model without HF_TOKEN).",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    },
                    "500": {
                        "description": "The workflow ran but failed (execution error) or produced no output file.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    },
                    "502": {
                        "description": "The prompt-rewriting model could not be reached, timed out, or returned nothing usable. Retry, or send 'raw_prompt' to skip the rewrite.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    },
                    "504": {
                        "description": "The workflow did not finish within the timeout; poll /api/wrapper/jobs/{job_id} with the returned id.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TimeoutResponse"}}},
                    },
                },
            },
        },
        "/api/wrapper/workflows": {
            "get": {
                "summary": "List the available workflows",
                "operationId": "listWorkflows",
                "responses": {
                    "200": {
                        "description": "Workflow names usable in /api/wrapper/{workflow}/generate.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/WorkflowList"}}},
                    },
                },
            },
        },
        "/api/wrapper/free": {
            "post": {
                "summary": "Free VRAM and RAM used by model execution",
                "description": "Unloads every loaded model (releasing GPU VRAM and RAM) and empties the torch caches. If a job is currently running or queued the free happens automatically right after that job finishes (unloading mid-job would crash it); when idle it happens immediately.",
                "operationId": "free",
                "responses": {
                    "200": {
                        "description": "Free scheduled/completed.",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/FreeResponse"}}},
                    },
                },
            },
        },
        "/api/wrapper/jobs/{job_id}": {
            "get": {
                "summary": "Get the status, progress and result of a generation job",
                "operationId": "getJob",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                ],
                "responses": {
                    "200": {"description": "Job details.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}},
                    "404": {"description": "Unknown job id.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            },
            "delete": {
                "summary": "Delete everything a finished job left on this worker",
                "description": (
                    "Worker cleanup, for once the caller has stored the result (or after the job failed or was "
                    "cancelled). Deletes the job's output files (every file its history entry lists, the prompt "
                    "sidecar JSON, and the job's own output folder output/wrapper/{job_id}/), the input files the "
                    "wrapper saved for it (its upload folder input/wrapper/{job_id}/: conditioning images, "
                    "audio, reference files) and its history entry. Idempotent: an unknown or already deleted job "
                    "answers 200 with zeros. Nothing outside ComfyUI's output/input/temp directories is ever "
                    "deleted: paths are resolved to their real paths, and one that resolves outside (a traversal "
                    "or a symlink pointing out) is refused and counted in `refused` instead. A job that is still "
                    "pending or running, or whose generate request is still open, is refused with 409."),
                "operationId": "deleteJob",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                ],
                "responses": {
                    "200": {"description": "What was deleted.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CleanupResponse"}}}},
                    "400": {"description": "job_id is not a canonical UUID.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    "409": {"description": "The job is still pending or running (or its generate request is still open); wait or cancel it, then retry.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    "500": {"description": "Some files could not be deleted (e.g. permissions). The history entry is kept so a retry can finish the job; the body also carries `deleted` and `refused` so far.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            },
        },
        "/api/wrapper/jobs/{job_id}/image": {
            "get": {
                "summary": "Get the generated image of a completed job",
                "description": "Redirects to the ComfyUI /view endpoint serving the image bytes.",
                "operationId": "getJobImage",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}},
                ],
                "responses": {
                    "302": {"description": "Redirect to the image."},
                    "404": {"description": "Unknown job id.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                    "409": {"description": "Job has not completed yet.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            },
        },
    },
    "components": {
        "schemas": {
            "WorkflowList": {
                "type": "object",
                "properties": {
                    "workflows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "title": {"type": "string"},
                                "requires_image": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "PromptPreview": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The rewritten prompt, ready to send back as 'raw_prompt'."},
                    "model": {"type": "string", "description": "The model that wrote it (after 'auto' resolution)."},
                    "used_image": {"type": "boolean", "description": "Whether the model actually saw a context image. False when none was attached, or when a text-only model was pinned and the wrapper's own context image was skipped."},
                    "note": {"type": "string", "nullable": True, "description": "Set when something about the request was adjusted rather than refused, e.g. a context image dropped because the pinned model cannot read one."},
                    "mode": {"type": "string", "enum": ["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"], "description": "H3 input mode inferred from which assets were attached."},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "frames": {"type": "integer", "description": "Frame count the prompt's timing was written against."},
                    "duration": {"type": "number", "description": "Seconds at 24 fps for that frame count."},
                },
            },
            "TimeoutResponse": {
                "type": "object",
                "properties": {
                    "error": {"type": "object", "properties": {"type": {"type": "string"}, "message": {"type": "string"}, "details": {"type": "string"}}},
                    "job_id": {"type": "string", "format": "uuid", "description": "Poll /api/wrapper/jobs/{job_id} with this id."},
                },
            },
            "CleanupResponse": {
                "type": "object",
                "properties": {
                    "deleted": {
                        "type": "object",
                        "properties": {
                            "outputs": {"type": "integer", "minimum": 0, "description": "Output (and temp) files deleted, prompt sidecar included."},
                            "inputs": {"type": "integer", "minimum": 0, "description": "Files deleted from the job's upload folder."},
                            "history": {"type": "boolean", "description": "Whether a history entry existed and was deleted."},
                        },
                    },
                    "refused": {"type": "integer", "minimum": 0, "description": "Paths left alone because they resolve outside ComfyUI's output/input/temp directories (or outside the job's own upload folder, for input files). Normally 0; each one is logged."},
                },
            },
            "FreeResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok"]},
                    "deferred": {"type": "boolean", "description": "True when a job is running/queued and the free happens after it finishes; false means memory was released immediately."},
                },
            },
            "Job": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "cancelled"]},
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "subfolder": {"type": "string"},
                                "type": {"type": "string"},
                                "url": {"type": "string", "description": "Relative URL to fetch the image."},
                            },
                        },
                        "description": "Generated images (only present once the job completed).",
                    },
                    "execution_error": {"type": "object", "nullable": True, "description": "Details when the job failed."},
                    "progress": {"$ref": "#/components/schemas/JobProgress"},
                },
            },
            "JobProgress": {
                "type": "object",
                "description": (
                    "How far the job has got. Present while it is pending or in progress, and as 100 once it "
                    "completed; absent for failed or cancelled jobs (older servers never send it). Each node of the "
                    "prompt weighs its reported step total (the sampler's steps dominate) or 1, finished nodes count "
                    "fully, and step counts the prompt asks for but no node has reported yet are reserved up front. "
                    "Never decreases for a job and stays below 100 until the job completed."
                ),
                "properties": {
                    "fraction": {"type": "number", "minimum": 0, "maximum": 1},
                    "percent": {"type": "integer", "minimum": 0, "maximum": 100},
                    "node": {"type": "string", "nullable": True, "description": "Id of the node running now."},
                    "node_class": {"type": "string", "nullable": True, "description": "Its class, e.g. SamplerCustomAdvanced."},
                    "step": {
                        "type": "object",
                        "nullable": True,
                        "description": "The running node's own steps, when it reports them.",
                        "properties": {"value": {"type": "integer"}, "max": {"type": "integer"}},
                    },
                    "queue_position": {"type": "integer", "nullable": True, "description": "1-based place among pending jobs (1 runs next); null once running."},
                },
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "object", "properties": {"type": {"type": "string"}, "message": {"type": "string"}, "details": {"type": "string"}}},
                    "missing": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"folder": {"type": "string"}, "filename": {"type": "string"}, "error": {"type": "string"}}},
                        "description": "Models that could not be downloaded.",
                    },
                },
            },
        },
    },
}

OUTPUT_MEDIA_TYPES = {"video": "video/mp4", "audio": "audio/wav"}


def _expanded_operation(template_operation, workflow, task, operation_suffix):
    """One concrete generate operation for a workflow/task: unique operationId,
    per-task form fields, required uploads, response media type."""
    import copy

    operation = copy.deepcopy(template_operation)
    operation["parameters"] = [p for p in operation.get("parameters", [])
                               if p.get("name") != "workflow"]
    operation["operationId"] = operation["operationId"] + "".join(
        part.capitalize() for part in operation_suffix.replace("-", "_").split("_"))
    schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    form = task.get("form")
    if form:
        universal = ("timeout", "free_vram", "quantization", "vram")
        schema["properties"] = {k: v for k, v in schema["properties"].items()
                                if k in form or k in universal}
        # Deep-copied: a workflow's extra_form_properties dict is shared by all
        # of its tasks, and the quantization enum below is patched per task.
        schema["properties"].update(copy.deepcopy(task.get("extra_form_properties") or {}))
        for up_name, up_spec in (task.get("uploads") or {}).items():
            if up_name not in schema["properties"]:
                schema["properties"][up_name] = {
                    "type": "string", "format": "binary",
                    "description": f"Upload ({up_spec['ext']}; up to {up_spec['max']})."}
        # With a rewriter, either field satisfies the request, so neither can be
        # marked required on its own; the handler enforces "one of".
        # Likewise an image the task accepts in more than one field (image or
        # ref_images) is "one of", so only a single field can be required.
        image_fields = task.get("image_fields", ["image"])
        schema["required"] = ([] if task.get("prompt_rewrite") else ["prompt"]) + (
            image_fields if task.get("requires_image") and len(image_fields) == 1 else [])
        if task.get("quantization_options"):
            schema["properties"]["quantization"]["enum"] = task["quantization_options"]
    if task.get("prompt_rewrite") and "prompt" in schema["properties"]:
        schema["properties"]["prompt"] = dict(
            schema["properties"]["prompt"],
            description="Plain description of the video you want, in your own words -- e.g. "
                        "\"two wrestlers in a gym, one chokeslams the other onto a crash mat, "
                        "heavy metal soundtrack\". It is rewritten into a full MiniMax H3 prompt "
                        "(alignment instruction, timed shots, overall_soundscape, "
                        "non_diegetic_music) by a Xiaomi MiMo model following MiniMax's own "
                        "H3 prompt-writing guide, using this request's real mode, duration, "
                        "canvas and reference labels. Requires MIMO_API_KEY on the server. "
                        "Send 'raw_prompt' instead if you already have an H3 prompt and want to "
                        "skip the rewrite; exactly one of the two is required. Preview the "
                        "rewrite without generating a video via the /prompt endpoint.")
    if workflow.get("example_prompt"):
        prompt_prop = schema["properties"]["prompt"]
        prompt_prop["description"] = (
            f"{prompt_prop['description']} This workflow natively understands "
            f"structured JSON prompts; example:\n\n```json\n{workflow['example_prompt']}\n```"
        )
    if task.get("example_prompt"):
        # The example is written in the workflow's native prompt format, so on a
        # rewriting task it belongs on raw_prompt -- 'prompt' takes plain intent.
        target = "raw_prompt" if task.get("prompt_rewrite") and "raw_prompt" in schema["properties"] \
            else "prompt"
        schema["properties"][target] = dict(
            schema["properties"][target],
            description=f"{schema['properties'][target]['description']} Example:"
                        f"\n\n{task['example_prompt']}")
    media_type = OUTPUT_MEDIA_TYPES.get(workflow.get("output_type"), "image/png")
    operation["responses"]["200"]["content"] = {
        media_type: {"schema": {"type": "string", "format": "binary"}},
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    }
    return operation


def _prompt_preview_operation(workflow_name, task_name, task):
    """The /prompt operation: rewrite only, no generation."""
    extra = task.get("extra_form_properties") or {}
    properties = {
        "prompt": {"type": "string", "description": "Plain description of the video you want, in your own words. Rewritten into the workflow's native prompt format and returned as JSON."},
        "llm_image": extra.get("llm_image", {"type": "string", "format": "binary"}),
        "llm_model": extra.get("llm_model", {"type": "string"}),
        "width": extra.get("width", {"type": "integer"}),
        "height": extra.get("height", {"type": "integer"}),
        "duration": extra.get("duration", {"type": "number"}),
    }
    for upload_name, upload_spec in (task.get("uploads") or {}).items():
        if upload_name not in properties:
            properties[upload_name] = {
                "type": "string", "format": "binary",
                "description": f"Optional ({upload_spec['ext']}). Only its presence and count matter "
                               "here: they tell the rewriter which mode and reference labels the real "
                               "generation will use."}
    task_note = (
        "The task is inferred from what you attach: reference assets mean ref2va, a first/last "
        "frame means image-to-video, and nothing attached means text-to-video -- so a prompt on "
        "its own is a complete request. Use /{task}/prompt to pin one explicitly."
        if task_name is None else
        f"Fixed to the {task_name} task; the H3 input mode still follows which frames you attach.")
    return {
        "summary": (f"Turn a free-form prompt into a {workflow_name} prompt"
                    if task_name is None else
                    f"Preview the rewritten prompt for {workflow_name}/{task_name}"),
        "description": "Runs only the LLM prompt rewrite and returns the result as JSON -- no GPU "
                       "work, no model downloads, no video. Iterating on a prompt through "
                       "/generate costs a full render, so use this to get the wording right, then "
                       "send the result back to /generate as 'raw_prompt' to render it verbatim. "
                       f"{task_note} Whichever image you attach also gives the model visual "
                       "context. Accepts multipart, form-urlencoded or JSON. Requires "
                       "MIMO_API_KEY.\n\n"
                       "Expect 10-20 s: this is one call to a large model, not a lookup. Give your "
                       "client at least 60 s -- a short timeout aborts mid-flight and is easy to "
                       "misread as the endpoint not existing. The server's own cap on the model "
                       "call is MIMO_TIMEOUT (120 s by default).",
        "operationId": "previewPrompt" + "".join(
            part.capitalize()
            for part in f"{workflow_name}_{task_name or ''}".replace("-", "_").split("_")),
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": {
                "type": "object", "required": ["prompt"], "properties": properties}}},
        },
        "responses": {
            "200": {
                "description": "The rewritten prompt and the parameters it was written against.",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PromptPreview"}}},
            },
            "400": {"description": "Missing prompt, invalid parameter, over-budget canvas/length, or the rewriter is not configured (MIMO_API_KEY unset).", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
            "404": {"description": "This workflow task does not support prompt rewriting.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
            "502": {"description": "The prompt-rewriting model could not be reached, timed out, or returned nothing usable.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        },
    }


def spec_with_workflows(workflows):
    """Return a copy of the spec with one concrete /api/wrapper/{name}/generate
    path per registered workflow (plus /api/wrapper/{name}/{task} paths for
    task workflows), so the docs show the real endpoints."""
    import copy

    spec = copy.deepcopy(WRAPPER_OPENAPI_SPEC)
    paths = spec["paths"]
    template = paths.pop("/api/wrapper/{workflow}/generate")
    template_operation = template["post"]
    for name, workflow in workflows.items():
        rewriting = [(t_name, t) for t_name, t in (workflow.get("tasks") or {}).items()
                     if t.get("prompt_rewrite")]
        for task_name, task in rewriting:
            paths[f"/api/wrapper/{name}/{task_name}/prompt"] = {
                "post": _prompt_preview_operation(name, task_name, task)}
        if rewriting:
            # Task-less alias: the task is inferred from the attached assets,
            # so turning a free-form idea into a prompt needs nothing but text.
            merged = dict(rewriting[0][1])
            merged["uploads"] = {k: v for _, t in rewriting
                                 for k, v in (t.get("uploads") or {}).items()}
            paths[f"/api/wrapper/{name}/prompt"] = {
                "post": _prompt_preview_operation(name, None, merged)}
        tasks = workflow.get("tasks")
        if tasks:
            for task_name, task in tasks.items():
                operation = _expanded_operation(template_operation, workflow, task,
                                                f"{name}_{task_name}")
                paths[f"/api/wrapper/{name}/{task_name}/generate"] = {"post": operation}
        else:
            operation = _expanded_operation(template_operation, workflow, workflow, name)
            paths[f"/api/wrapper/{name}/generate"] = {"post": operation}
    return spec


WRAPPER_SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI Wrapper API Docs</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
        .topbar { display: none; }
        .swagger-ui .info { margin: 20px 0; }
        .swagger-ui .info .title { font-size: 28px; }
    </style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
<script>
window.onload = function() {
  SwaggerUIBundle({
    url: "/api/wrapper/openapi.json",
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    layout: "StandaloneLayout",
    defaultModelsExpandDepth: -1,
    docExpansion: "list",
    filter: true,
    tryItOutEnabled: true,
  })
}
</script>
</body>
</html>"""
