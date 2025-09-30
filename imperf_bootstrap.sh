#!/usr/bin/env bash
# bootstrap_imperf.sh — creates a minimal imperf project with a first pipeline
# Usage: bash bootstrap_imperf.sh /path/to/your/images
# After it finishes, activate venv and run the demo command shown at the end.

set -euo pipefail
IMAGES_DIR=${1:-"/ABSOLUTE/PATH/TO/IMAGES"}

# --- project skeleton ---
mkdir -p imperf/{vf_core,cli,pipelines,db,out}

############################################
# requirements
############################################
cat > imperf/requirements.txt << 'REQ'
pillow
opencv-python
numpy
pandas
pyarrow
scikit-image
tqdm
pyyaml
faiss-cpu
imageio
imageio-ffmpeg
# CLIP stack (you can pin to your CUDA version or use cpu)
open-clip-torch
torch
REQ

############################################
# README
############################################
cat > imperf/README.md << 'MD'
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
MD

############################################
# pipeline configs
############################################
cat > imperf/pipelines/extract.yaml << 'YAML'
ingest:
  glob: "**/*.{jpg,jpeg,png,JPG,JPEG,PNG}"
normalize:
  long_edge: 512
  letterbox: true
features:
  - name: hue_stats
  - name: sharpness_laplacian
  - name: edge_orientation_hist
  - name: saliency_ratio
  - name: clip_embedding
store:
  parquet_path: "imperf/db/features.parquet"
  image_table_path: "imperf/db/images.parquet"
  index_path: "imperf/db/clip.faiss"
YAML

cat > imperf/pipelines/sequence_color.yaml << 'YAML'
select:
  filter: "has_clip == True"
order:
  mode: "composed"
  traits:
    - key: hue_mean
      weight: 1.0
      circular: true
    - key: saliency_ratio
      weight: 0.3
    - key: clip_pc1
      weight: 0.2
smoothing:
  window: 3
  method: "swap_local_min"
render:
  take_every_n: 1         # keep 1 to use all; raise if you want fewer frames
  contact_sheet:
    cols: 10
    thumb: 256
    path: "imperf/out/contact_sheet.jpg"
  video:
    fps: 24
    xfade_frames: 5       # ~150ms at 30fps; with fps=24 this is ~208ms
    path: "imperf/out/sequence.mp4"
YAML

############################################
# vf_core: __init__
############################################
cat > imperf/vf_core/__init__.py << 'PY'
__all__ = []
PY

############################################
# ingest.py
############################################
cat > imperf/vf_core/ingest.py << 'PY'
from pathlib import Path
from typing import List

def discover_images(root: Path, patterns: List[str]):
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    # unique, stable order
    return sorted(set(files))
PY

############################################
# normalize.py
############################################
cat > imperf/vf_core/normalize.py << 'PY'
from typing import Tuple
from PIL import Image, ImageOps

# long-edge resize with optional letterbox

def to_canvas(img: Image.Image, long_edge: int = 512, letterbox: bool = True) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
    im2 = img.convert('RGB').resize((new_w, new_h), Image.LANCZOS)
    if letterbox:
        canvas = Image.new('RGB', (long_edge, long_edge), (0,0,0))
        canvas.paste(im2, ((long_edge - new_w)//2, (long_edge - new_h)//2))
        return canvas
    return im2
PY

############################################
# features_low.py
############################################
cat > imperf/vf_core/features_low.py << 'PY'
import cv2
import numpy as np
from typing import Dict

# ---- hue stats ----

def hue_stats(img_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].astype(np.float32) * (360.0/180.0)
    s = (hsv[...,1].astype(np.float32))/255.0
    v = (hsv[...,2].astype(np.float32))/255.0
    # circular mean for hue (simple approximation)
    h_rad = np.deg2rad(h)
    mean_angle = np.arctan2(np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad)))
    hue_mean = (np.rad2deg(mean_angle) + 360.0) % 360.0
    return {
        'hue_mean': float(hue_mean),
        'hue_std': float(np.std(h)),
        'sat_mean': float(np.mean(s)),
        'val_mean': float(np.mean(v)),
    }

# ---- sharpness (variance of Laplacian) ----

def sharpness_laplacian(img_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())

# ---- edge orientation histogram ----

def edge_orientation_hist(img_bgr: np.ndarray, bins: int = 8) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang, bins=bins, range=(0,360), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return {f'edge_bin_{i}': float(hist[i]) for i in range(bins)}

# ---- spectral residual saliency (fast) ----

def saliency_ratio(img_bgr: np.ndarray, thresh: float = 0.7) -> float:
    # convert to gray float
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # FFT
    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    # average filter in frequency domain
    kernel = cv2.boxFilter(log_amp, ddepth=-1, ksize=(3,3))
    spectral_residual = log_amp - kernel
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j*phase)))**2
    saliency = cv2.GaussianBlur(saliency, (9,9), 2.5)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    # ratio of salient pixels
    return float((saliency > thresh).mean())
PY

############################################
# features_sem.py
############################################
cat > imperf/vf_core/features_sem.py << 'PY'
from typing import Dict
import numpy as np
from PIL import Image
import torch
import open_clip

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
_model = _model.to(_device).eval()

@torch.no_grad()
def clip_embedding(pil_img: Image.Image) -> np.ndarray:
    x = preprocess(pil_img).unsqueeze(0).to(_device)
    feat = _model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype('float32')[0]
PY

############################################
# store.py
############################################
cat > imperf/vf_core/store.py << 'PY'
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss

@dataclass
class Stores:
    parquet_path: Path
    image_table_path: Path
    index_path: Path

class FeatureStore:
    def __init__(self, stores: Stores):
        self.stores = stores
        self.stores.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.stores.image_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.stores.index_path.parent.mkdir(parents=True, exist_ok=True)

    def append_features(self, df: pd.DataFrame):
        table = pa.Table.from_pandas(df)
        if self.stores.parquet_path.exists():
            pq.write_to_dataset(table, root_path=str(self.stores.parquet_path))
        else:
            pq.write_table(table, self.stores.parquet_path)

    def write_images(self, df: pd.DataFrame):
        pq.write_table(pa.Table.from_pandas(df), self.stores.image_table_path)

    def build_faiss(self, vecs: np.ndarray):
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        faiss.write_index(index, str(self.stores.index_path))

    def read_all(self) -> pd.DataFrame:
        if self.stores.parquet_path.is_dir():
            return pq.ParquetDataset(str(self.stores.parquet_path)).read_pandas().to_pandas()
        return pd.read_parquet(self.stores.parquet_path)
PY

############################################
# sequence.py
############################################
cat > imperf/vf_core/sequence.py << 'PY'
import numpy as np
import pandas as pd
from typing import List, Dict

# project CLIP to PC1 for semantic drift

def add_clip_pcs(df: pd.DataFrame, col: str = 'clip_embedding', k: int = 2) -> pd.DataFrame:
    mats = np.vstack(df[col].to_list())
    # PCA via SVD
    U, S, Vt = np.linalg.svd(mats - mats.mean(0, keepdims=True), full_matrices=False)
    pcs = U[:, :k] * S[:k]
    for i in range(k):
        df[f'clip_pc{i+1}'] = pcs[:, i]
    return df

# distance between consecutive items in a weighted feature space

def neighbor_cost(a: Dict, b: Dict, traits):
    tot = 0.0
    for t in traits:
        key = t['key']
        w = t.get('weight', 1.0)
        circ = t.get('circular', False)
        va, vb = a[key], b[key]
        if circ:
            # minimal circular distance on [0,360)
            d = abs(((va - vb + 180) % 360) - 180)
        else:
            d = abs(va - vb)
        tot += w * d
    return tot

# simple local smoothing: try swapping within a sliding window if it reduces total cost

def smooth_order(df: pd.DataFrame, order: List[int], traits, window: int = 3, iters: int = 2) -> List[int]:
    idx = order[:]
    n = len(idx)
    for _ in range(iters):
        for i in range(n - window):
            best = idx[i:i+window]
            best_cost = path_cost(df, best, traits)
            # try pairwise swap neighbors
            for j in range(i, i+window-1):
                cand = idx[i:i+window]
                cand[j], cand[j+1] = cand[j+1], cand[j]
                c = path_cost(df, cand, traits)
                if c < best_cost:
                    best, best_cost = cand, c
            idx[i:i+window] = best
    return idx


def path_cost(df: pd.DataFrame, subidx: List[int], traits) -> float:
    c = 0.0
    for a, b in zip(subidx, subidx[1:]):
        c += neighbor_cost(df.loc[a], df.loc[b], traits)
    return c


def composed_sort(df: pd.DataFrame, traits) -> List[int]:
    # lexicographic on first key, then key2, etc., with circular handling on first only
    primary = traits[0]
    circ = primary.get('circular', False)
    key = primary['key']
    vals = df[key].to_numpy()
    if circ:
        # rotate so mean is near center to avoid wrap jump
        shift = (np.mean(vals) + 360.0) % 360.0
        vals = (vals - shift) % 360.0
    df2 = df.copy()
    df2['__primary__'] = vals
    keys = ['__primary__'] + [t['key'] for t in traits[1:]]
    order = df2.sort_values(keys, ascending=True).index.to_list()
    return order
PY

############################################
# render.py
############################################
cat > imperf/vf_core/render.py << 'PY'
from pathlib import Path
from typing import List
from PIL import Image
import imageio.v2 as imageio


def contact_sheet(paths: List[Path], cols: int, thumb: int, out_path: Path):
    thumbs = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        im.thumbnail((thumb, thumb), Image.LANCZOS)
        thumbs.append(im)
    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumb
    h = rows * thumb
    sheet = Image.new('RGB', (w, h), (0, 0, 0))
    for i, im in enumerate(thumbs):
        x = (i % cols) * thumb
        y = (i // cols) * thumb
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def sequence_video(paths: List[Path], fps: int, xfade_frames: int, out_path: Path):
    # simple frame writer with linear crossfades generated in-Python
    frames = []
    def load(p):
        return Image.open(p).convert('RGB')
    base = load(paths[0])
    base = base.resize((1280, int(1280*base.height/base.width)))
    frames.append(base)
    for nxt_path in paths[1:]:
        nxt = load(nxt_path)
        nxt = nxt.resize((base.width, base.height))
        # hold base for a moment
        hold = max(1, fps//2)
        frames.extend([base]*hold)
        # crossfade
        for t in range(1, max(1, xfade_frames)+1):
            a = 1.0 - t/float(xfade_frames+1)
            b = 1.0 - a
            blend = Image.blend(base, nxt, b)
            frames.append(blend)
        base = nxt
    # final hold
    frames.extend([base]* (fps//2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, [f for f in frames], fps=fps)
PY

############################################
# cli/vf_run.py — glue for v1
############################################
cat > imperf/cli/vf_run.py << 'PY'
import argparse
import json
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

from imperf.vf_core.ingest import discover_images
from imperf.vf_core.normalize import to_canvas
from imperf.vf_core import features_low as F0
from imperf.vf_core.features_sem import clip_embedding
from imperf.vf_core.store import FeatureStore, Stores
from imperf.vf_core.sequence import composed_sort, smooth_order, add_clip_pcs
from imperf.vf_core.render import contact_sheet, sequence_video


def pil_to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Root folder with images')
    ap.add_argument('--extract_cfg', required=True)
    ap.add_argument('--sequence_cfg', required=True)
    args = ap.parse_args()

    # load cfgs
    ext = yaml.safe_load(open(args.extract_cfg))
    seq = yaml.safe_load(open(args.sequence_cfg))

    root = Path(args.images).expanduser().resolve()
    pats = ext['ingest']['glob'].split()
    if len(pats)==1 and '**' in pats[0]:
        # allow brace expansion style in one string
        pats = pats[0].split(',') if ',' in pats[0] else [pats[0]]
    files = discover_images(root, patterns=[p.strip() for p in pats])
    if not files:
        raise SystemExit('No images found. Check your path/glob.')
    print(f"Found {len(files)} images")

    stores = Stores(
        parquet_path=Path(ext['store']['parquet_path']),
        image_table_path=Path(ext['store']['image_table_path']),
        index_path=Path(ext['store']['index_path'])
    )
    fs = FeatureStore(stores)

    rows = []
    img_rows = []
    for i, p in enumerate(tqdm(files)):
        pil = Image.open(p)
        orig_w, orig_h = pil.size
        norm = to_canvas(pil, long_edge=ext['normalize']['long_edge'], letterbox=ext['normalize']['letterbox'])
        bgr = pil_to_bgr(norm)
        feats = {}
        feats.update(F0.hue_stats(bgr))
        feats['sharpness_laplacian'] = F0.sharpness_laplacian(bgr)
        feats.update(F0.edge_orientation_hist(bgr))
        feats['saliency_ratio'] = F0.saliency_ratio(bgr)
        # semantic
        try:
            emb = clip_embedding(norm)
            feats['clip_embedding'] = emb
            feats['has_clip'] = True
        except Exception as e:
            feats['clip_embedding'] = np.zeros(512, dtype='float32')
            feats['has_clip'] = False
        feats['id'] = i
        feats['uri'] = str(p.resolve())
        rows.append(feats)
        img_rows.append({'id': i, 'uri': str(p.resolve()), 'orig_w': orig_w, 'orig_h': orig_h})

    # pack into DataFrame
    df = pd.DataFrame(rows)
    # add clip PCs if available
    if df['has_clip'].any():
        df = add_clip_pcs(df, col='clip_embedding', k=2)
    fs.write_images(pd.DataFrame(img_rows))
    fs.append_features(df.drop(columns=[], errors='ignore'))

    # selection
    sel_expr = seq['select'].get('filter', None)
    df_sel = df.query(sel_expr) if sel_expr else df

    # order
    traits = seq['order']['traits']
    order = composed_sort(df_sel, traits)
    order = smooth_order(df_sel, order, traits, window=seq['smoothing']['window'])

    # output paths (subsample if configured)
    take_n = int(seq['render'].get('take_every_n', 1))
    ordered_paths = [Path(df_sel.loc[i, 'uri']) for i in order][::take_n]

    # contact sheet
    cs = seq['render']['contact_sheet']
    contact_sheet(ordered_paths, cols=cs['cols'], thumb=cs['thumb'], out_path=Path(cs['path']))

    # video
    vid = seq['render']['video']
    sequence_video(ordered_paths, fps=vid['fps'], xfade_frames=vid['xfade_frames'], out_path=Path(vid['path']))

    print("\nDone. Outputs:\n -", cs['path'], "\n -", vid['path'])

if __name__ == '__main__':
    main()
PY

############################################
# finishing message
############################################
echo "\n✅ imperf scaffold created. Next steps:\n1) python3 -m venv .venv && source .venv/bin/activate\n2) pip install -r imperf/requirements.txt\n3) python imperf/cli/vf_run.py --images \"${IMAGES_DIR}\" --extract_cfg imperf/pipelines/extract.yaml --sequence_cfg imperf/pipelines/sequence_color.yaml\n\nOutputs will be in imperf/out/."
