"""Composable session pipeline for experimentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from .datasets import ImageRecord, discover_images, iter_image_records
from .features import DEFAULT_FEATURES, FeatureExtractor
from .sorting import SequencePlan, sort_by_feature
from .visuals import RenderOptions, render_sequence

SessionHook = Callable[[Sequence[ImageRecord], SequencePlan], None]
RenderHook = Callable[[SequencePlan, Path | None], None]
FeatureSummaryHook = Callable[[dict[str, "FeatureStats"]], None]


@dataclass(slots=True)
class FeatureStats:
    """Aggregate metrics for a computed feature."""

    count: int
    minimum: float
    maximum: float
    mean: float


@dataclass(slots=True)
class SessionConfig:
    """Describe what the current experiment should do."""

    dataset_root: Path
    recursive: bool = True
    features: dict[str, Callable] = field(default_factory=lambda: dict(DEFAULT_FEATURES))
    sequences: list[SequencePlan] = field(default_factory=list)
    on_sequence: SessionHook | None = None
    on_render: RenderHook | None = None
    on_feature_summary: FeatureSummaryHook | None = None
    render: RenderOptions | None = None


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

    if config.on_feature_summary is not None:
        config.on_feature_summary(
            summarize_features(enriched_records, tuple(config.features.keys()))
        )

    sequences: list[list[ImageRecord]] = []
    render_options = config.render
    if render_options is not None:
        render_options.output_dir.mkdir(parents=True, exist_ok=True)
    for plan in config.sequences:
        ordered = sort_by_feature(enriched_records, plan.feature, reverse=plan.reverse)
        sequences.append(ordered)
        if config.on_sequence is not None:
            config.on_sequence(ordered, plan)
        render_path: Path | None = None
        if render_options is not None and plan.render:
            render_path = render_sequence(ordered, plan, render_options)
        if config.on_render is not None:
            config.on_render(plan, render_path)
    return sequences


def summarize_features(
    records: Sequence[ImageRecord], feature_names: Sequence[str]
) -> dict[str, FeatureStats]:
    """Return min/max/mean aggregates for ``feature_names``."""

    summary: dict[str, FeatureStats] = {}
    for feature in feature_names:
        values = [
            value
            for record in records
            if (value := record.features.get(feature)) is not None
        ]
        if not values:
            continue
        summary[feature] = FeatureStats(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=mean(values),
        )
    return summary
