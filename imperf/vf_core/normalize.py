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
