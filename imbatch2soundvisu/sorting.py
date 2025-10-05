"""Helpers for sequencing images by computed traits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .datasets import ImageRecord


@dataclass(slots=True)
class SequencePlan:
    """Describe how a sequence should be generated."""

    title: str
    feature: str
    reverse: bool = False
    render: bool = True
    output_name: str | None = None


def sort_by_feature(
    records: Sequence[ImageRecord],
    feature: str,
    *,
    reverse: bool = False,
) -> list[ImageRecord]:
    """Return a new list ordered by ``feature``."""

    def key(record: ImageRecord) -> float:
        return record.features.get(feature, float("nan"))

    return sorted(records, key=key, reverse=reverse)


def summarize_sequence(records: Iterable[ImageRecord], feature: str) -> list[str]:
    """Return printable lines describing the ordered records."""

    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        value = record.features.get(feature, float("nan"))
        lines.append(f"{index:>3} | {value:8.3f} | {record.path}")
    if not lines:
        lines.append("    (no images matched the current configuration)")
    return lines
