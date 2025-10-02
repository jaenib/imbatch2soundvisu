"""Edit this file to steer your current experiment."""

from __future__ import annotations

from pathlib import Path

from .config import get_data_root
from .pipeline import FeatureStats, SessionConfig, run_session
from .sorting import SequencePlan, summarize_sequence
from .visuals import RenderOptions


# Configure which features you want to compute and how to order them.
SEQUENCES = [
    SequencePlan(title="Luminance (dark to bright)", feature="luminance"),
    SequencePlan(title="Luminance (bright to dark)", feature="luminance", reverse=True),
    SequencePlan(title="Subject scale (loose to tight)", feature="subject_scale"),
    SequencePlan(title="Dominant hue (0° to 360°)", feature="hue"),
    SequencePlan(title="Edge density (minimal to detailed)", feature="edge_density"),
]

OUTPUT_DIR = Path("outputs")
FRAME_DURATION_MS = 220
TARGET_SIZE = (720, 720)
MAX_FRAMES = 240


def create_config(dataset_root: Path | None = None) -> SessionConfig:
    root = dataset_root or get_data_root()

    def on_feature_summary(summary: dict[str, FeatureStats]) -> None:
        if not summary:
            print("No features could be summarized. Check your dataset configuration.")
            return
        print()
        print("Feature statistics across the dataset:")
        for name, stats in summary.items():
            span = stats.maximum - stats.minimum
            print(
                f"  {name:>15} | count={stats.count:3d} | "
                f"min={stats.minimum:8.3f} | max={stats.maximum:8.3f} | span={span:8.3f} | "
                f"mean={stats.mean:8.3f}"
            )

    def on_sequence(records, plan: SequencePlan) -> None:
        feature_name = plan.feature
        print()
        print(f"Sequence based on {plan.title} [{feature_name}] ({len(records)} image(s)):")
        for line in summarize_sequence(records, feature_name):
            print(f"  {line}")

    def on_render(plan: SequencePlan, output_path) -> None:
        if output_path is None:
            print(f"  Skipped rendering for {plan.title} (no frames available).")
            return
        print(f"  Saved animation to {output_path}")

    return SessionConfig(
        dataset_root=root,
        sequences=SEQUENCES,
        on_sequence=on_sequence,
        on_render=on_render,
        on_feature_summary=on_feature_summary,
        render=RenderOptions(
            output_dir=OUTPUT_DIR,
            image_size=TARGET_SIZE,
            frame_duration_ms=FRAME_DURATION_MS,
            max_frames=MAX_FRAMES,
        ),
    )


def run() -> None:
    """Execute the configured experiment."""

    config = create_config()
    try:
        sequences = run_session(config)
    except FileNotFoundError as error:
        print(error)
        print("Edit imbatch2soundvisu/config.py and point DATA_ROOT at a valid folder.")
        return

    if not sequences:
        print("No sequences were generated. Edit imbatch2soundvisu/session.py to add some.")


if __name__ == "__main__":
    run()
