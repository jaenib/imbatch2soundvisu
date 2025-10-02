"""Rendering helpers for turning ordered image sequences into visuals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from .datasets import ImageRecord
from .sorting import SequencePlan


@dataclass(slots=True)
class RenderOptions:
    """Control how sequence animations are written to disk."""

    output_dir: Path
    image_size: tuple[int, int] | None = (720, 720)
    frame_duration_ms: int = 220
    max_frames: int | None = None


def slugify(value: str) -> str:
    """Return a filesystem-friendly version of ``value``."""

    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "sequence"


def render_sequence(
    records: Sequence[ImageRecord], plan: SequencePlan, options: RenderOptions
) -> Path | None:
    """Write ``records`` to an animated GIF and return the resulting path."""

    if not records:
        return None

    filename = plan.output_name or slugify(plan.title)
    output_path = options.output_dir / f"{filename}.gif"
    frames = _prepare_frames(records, options.image_size, options.max_frames)
    if not frames:
        return None
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        format="GIF",
        loop=0,
        duration=max(int(options.frame_duration_ms), 1),
    )
    return output_path


def _prepare_frames(
    records: Iterable[ImageRecord],
    image_size: tuple[int, int] | None,
    max_frames: int | None,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index, record in enumerate(records):
        if max_frames is not None and index >= max_frames:
            break
        with Image.open(record.path) as image:
            frame = image.convert("RGB")
            if image_size is not None:
                frame = frame.resize(image_size, Image.LANCZOS)
            frames.append(frame.copy())
    return frames
