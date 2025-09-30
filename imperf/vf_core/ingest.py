from pathlib import Path
from typing import List

def discover_images(root: Path, patterns: List[str]):
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    # unique, stable order
    return sorted(set(files))
