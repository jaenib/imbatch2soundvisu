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
