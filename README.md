# imbatch2soundvisu

An image-first playground for finding unexpected sequences inside loosely
connected collections.  Configure a folder full of images, extract a handful of
interpretable traits, then line everything up to see which orderings tease out
surprising coherence.

## Quick start

1. Install the lightweight dependencies:

   ```bash
   pip install Pillow
   ```

2. Edit [`imbatch2soundvisu/config.py`](imbatch2soundvisu/config.py) and set
   `DATA_ROOT` to the directory that contains the images you want to explore.
   You only have to do this once—afterwards you can keep tweaking the rest of
   the code.

3. Adjust the experiment description in
   [`imbatch2soundvisu/session.py`](imbatch2soundvisu/session.py).  Each
   `SequencePlan` describes one visual narrative you want to inspect.

4. Run the session:

   ```bash
   python -m imbatch2soundvisu
   ```

   The script prints a feature overview, logs each generated sequence together
   with the driving metric, and renders animated GIFs into
   [`outputs/`](outputs) so you can immediately watch every ordering come to
   life.

## How can we sort one-subject image sets?

Start with traits that are easy to compute yet highly perceptual:

* **Luminance sweeps** — dark-to-bright progressions feel like lights turning on
  in slow motion and work well for portraits or studio material.
* **Dominant hue** — rotating through the color wheel creates an immediate sense
  of order.  It is especially effective when the subject is consistently lit.
* **Colorfulness and saturation** — transition from muted, desaturated takes to
  loud, high-energy frames to emphasize mood changes.
* **Subject scale** — approximate the framing tightness and move from wide
  establishing shots toward macro details.
* **Edge density** — low-frequency shapes followed by detailed textures read as
  a zoom into complexity.
* **Aspect ratio / orientation** — landscape-to-portrait flips give rhythm to
  grids or split-screen presentations.

Once the basics feel good, layer in heavier cues: skin detection or face
landmark distances (for portraits), saliency maps for emphasis shifts, or depth
estimation to travel from flat compositions into deep scenes.

## What works well with accessible computer vision tooling?

Non-commercial, widely available models handle several traits gracefully:

* **Color descriptors** using Pillow, OpenCV, or scikit-image are stable and
  fast.
* **Edge and texture statistics** (Canny, Sobel, Laplacian filters) describe
  complexity without training data.
* **Pose estimation** via MediaPipe or OpenPifPaf can align human limbs or body
  orientation when you want choreographed motion.
* **Face embeddings** (FaceNet derivatives, ArcFace) cluster expressions and
  gaze direction surprisingly well on varied lighting.
* **Image embeddings** from CLIP or OpenCLIP deliver semantic similarity for
  more chaotic rolls once you move beyond single subjects.

Combine these primitives with handcrafted heuristics (cropping ratios, dominant
lines) to craft sequences that feel deliberate yet computationally inexpensive.

## Datasets worth exploring

* **Oxford Flowers 102** — consistent subject matter with strong color
  variation; ideal for hue and saturation experiments.
* **CelebA-HQ** — aligned portrait imagery for expression, gaze, and crop
  studies.
* **DeepFashion In-Shop Clothes** — repeat subjects with varying garments; great
  for texture and colorfulness sweeps.
* **Google Landmarks v2 (clean subset)** — architectural repetition with wide
  framing variance.
* **ImageNet (single synset slices)** — pick a category like "school bus" to
  probe orientation and background cues.
* **LAION-Aesthetics filtered subsets** — large, messy, and stylistically
  diverse once you're ready to move beyond curated material.

## Architecture sketch

```
imbatch2soundvisu/
├── config.py        # Set DATA_ROOT and tweak supported extensions here.
├── datasets.py      # Discovers image files and creates lightweight records.
├── features.py      # Scalar descriptors (luminance, hue, subject scale, …).
├── pipeline.py      # Orchestrates feature extraction, sequencing, and rendering.
├── session.py       # Your sandbox: describe sequences, stats, and logging.
├── sorting.py       # Helpers for ordering and formatting sequences.
├── visuals.py       # Compiles ordered frames into animated GIFs.
└── __main__.py      # Allows `python -m imbatch2soundvisu` to run the session.
```

Everything is ordinary Python.  Modify `session.py` to try new feature mixes or
swap in additional processing steps.  Because configuration lives in code you
can version your experiments, fork paths, and script more elaborate behaviours
without wrestling with command-line plumbing.

## Next steps

* Emit contact sheets or storyboards alongside the GIFs for frame-accurate
  planning.
* Capture thumbnails for quick visual inspection alongside the printed values.
* Cache feature computations to keep large experiments responsive.
* Add opt-in hooks for CLIP embeddings, segmentation masks, or motion-aware
  sequencing once you start feeding in camera rolls and video frame dumps.
