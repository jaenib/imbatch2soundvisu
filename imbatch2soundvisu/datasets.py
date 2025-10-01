"""Utilities for discovering and loading image files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .config import IMAGE_EXTENSIONS


@dataclass(slots=True)
class ImageRecord:
    """Container describing an image and the features extracted from it."""

    path: Path
    features: dict[str, float] = field(default_factory=dict)


def discover_images(
    root: Path,
    *,
    recursive: bool = True,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Return the list of image paths under ``root``.

    ``extensions`` defaults to :data:`~imbatch2soundvisu.config.IMAGE_EXTENSIONS`.
    The return value is sorted to provide deterministic ordering across runs.
    """

    normalized_exts = tuple(e.lower() for e in (extensions or IMAGE_EXTENSIONS))
    iterator: Iterable[Path]
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")

    files = [
        path
        for path in iterator
        if path.suffix.lower() in normalized_exts and path.is_file()
    ]
    files.sort()
    return files


def iter_image_records(paths: Sequence[Path]) -> Iterator[ImageRecord]:
    """Yield :class:`ImageRecord` entries for ``paths``."""

    for path in paths:
        yield ImageRecord(path=path)
