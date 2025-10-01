"""Edit this file to steer your current experiment."""

from __future__ import annotations

from pathlib import Path

from .config import get_data_root
from .pipeline import SessionConfig, run_session
from .sorting import SequencePlan, summarize_sequence


# Configure which features you want to compute and how to order them.
SEQUENCES = [
    SequencePlan(title="Luminance (dark to bright)", feature="luminance"),
    SequencePlan(title="Luminance (bright to dark)", feature="luminance", reverse=True),
    SequencePlan(title="Subject scale (loose to tight)", feature="subject_scale"),
    SequencePlan(title="Dominant hue (0° to 360°)", feature="hue"),
    SequencePlan(title="Edge density (minimal to detailed)", feature="edge_density"),
]


def create_config(dataset_root: Path | None = None) -> SessionConfig:
    root = dataset_root or get_data_root()

    def on_sequence(records, plan: SequencePlan) -> None:
        feature_name = plan.feature
        print()
        print(f"Sequence based on {plan.title} [{feature_name}] ({len(records)} image(s)):")
        for line in summarize_sequence(records, feature_name):
            print(f"  {line}")

    return SessionConfig(
        dataset_root=root,
        sequences=SEQUENCES,
        on_sequence=on_sequence,
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
