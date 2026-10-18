# Automated Testing

## Running tests locally

Additional requirements for running tests:
```
pip install pytest
pip install websocket-client==1.6.1
opencv-python==4.6.0.66
scikit-image==0.21.0
```
Run inference tests:
```
pytest tests/inference
```

## Quality regression test
Compares images in 2 directories to ensure they are the same

1) Run an inference test to save a directory of "ground truth" images
```
    pytest tests/inference --output_dir tests/inference/baseline
```
2) Make code edits

3) Run inference and quality comparison tests
```
pytest
```

## Custom node pack checks

Holds the custom-node packs the stable-release bar names -- KJNodes,
VideoHelperSuite, ComfyUI-Manager -- against this build: did each import, does
it register the nodes it is known for, are its node schemas well formed. The
pins live in `tests/custom_nodes/manifest.json`; CI runs this as
`.github/workflows/test-custom-nodes.yml`.

Locally, against a server that has the pinned packs installed:
```
python tests/custom_nodes/install_pinned_packs.py
for r in custom_nodes/*/requirements.txt; do pip install -r "$r"; done
python main.py --cpu 2>&1 | tee console_output.log
```
then, in another shell:
```
COMFYUI_URL=http://127.0.0.1:8188 COMFYUI_STARTUP_LOG=console_output.log \
    pytest tests/custom_nodes
```
