"""Composable session pipeline for experimentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .datasets import ImageRecord, discover_images, iter_image_records
from .features import DEFAULT_FEATURES, FeatureExtractor
from .sorting import SequencePlan, sort_by_feature

SessionHook = Callable[[Sequence[ImageRecord], SequencePlan], None]


@dataclass(slots=True)
class SessionConfig:
    """Describe what the current experiment should do."""

    dataset_root: Path
    recursive: bool = True
    features: dict[str, Callable] = field(default_factory=lambda: dict(DEFAULT_FEATURES))
    sequences: list[SequencePlan] = field(default_factory=list)
    on_sequence: SessionHook | None = None


def run_session(config: SessionConfig) -> list[list[ImageRecord]]:
    """Execute ``config`` and return the generated sequences."""

    if not config.dataset_root.exists():
        raise FileNotFoundError(
            f"Configured dataset directory does not exist: {config.dataset_root}"
        )

    image_paths = discover_images(config.dataset_root, recursive=config.recursive)
    records = iter_image_records(image_paths)
    extractor = FeatureExtractor(config.features)
    enriched_records = extractor.process(records)

    sequences: list[list[ImageRecord]] = []
    for plan in config.sequences:
        ordered = sort_by_feature(enriched_records, plan.feature, reverse=plan.reverse)
        sequences.append(ordered)
        if config.on_sequence is not None:
            config.on_sequence(ordered, plan)
    return sequences
