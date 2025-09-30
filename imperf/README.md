# imperf (seed)

A tiny, composable pipeline to sort a set of images by visual/semantic traits and render a contact sheet + teaser video.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r imperf/requirements.txt

# run extraction + sequence + render
python imperf/cli/vf_run.py \
  --images "${IMAGES_DIR}" \
  --extract_cfg imperf/pipelines/extract.yaml \
  --sequence_cfg imperf/pipelines/sequence_color.yaml
```

Outputs land in `imperf/out/`.

If `torch` fails to install with CUDA on your machine, try CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
