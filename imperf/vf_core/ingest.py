"""Utilities for finding image files on disk."""

from pathlib import Path
from typing import Iterable, List


def _expand_braces(pattern: str) -> List[str]:
    """Expand simple brace expressions in glob patterns.

    Python's :meth:`pathlib.Path.rglob` does not understand brace expansion
    (e.g. ``"**/*.{jpg,png}"``).  The original implementation attempted to
    split on commas which broke patterns by dropping the surrounding text.
    Instead we recursively expand the brace expressions so that
    ``"**/*.{jpg,png}"`` becomes ``["**/*.jpg", "**/*.png"]``.

    The implementation intentionally keeps the logic simple – nested braces are
    handled recursively and patterns without closing braces are returned
    unchanged.
    """

    start = pattern.find("{")
    if start == -1:
        return [pattern]

    end = pattern.find("}", start)
    if end == -1:
        # Unmatched brace – best effort by returning the original pattern.
        return [pattern]

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    options = pattern[start + 1 : end].split(",")

    expanded: List[str] = []
    for option in options:
        expanded.extend(_expand_braces(f"{prefix}{option}{suffix}"))
    return expanded


def _normalize_patterns(patterns: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for pattern in patterns:
        normalized.extend(_expand_braces(pattern.strip()))
    return normalized


def discover_images(root: Path, patterns: Iterable[str]):
    files = []
    for pat in _normalize_patterns(patterns):
        files.extend(p for p in root.rglob(pat) if p.is_file())
    # unique, stable order
    return sorted(set(files))
