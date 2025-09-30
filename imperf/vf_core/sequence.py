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
