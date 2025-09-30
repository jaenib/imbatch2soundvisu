import argparse
import json
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import imperf

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
