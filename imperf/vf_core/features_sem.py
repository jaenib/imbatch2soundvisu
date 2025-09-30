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
