"""Executable entry point for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import ENV_VAR_NAME, get_data_root

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff")


def _iter_audio_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
            yield path


def _format_report(root: Path, files: list[Path]) -> str:
    lines = [
        "imbatch2soundvisu",
        "=================",
        f"Configured dataset: {root}",
    ]

    if files:
        lines.append("")
        lines.append(f"Discovered {len(files)} audio file(s):")
        for file in files:
            lines.append(f"  - {file.relative_to(root)}")
    else:
        lines.extend(
            [
                "",
                "No audio files were found. Ensure that the directory exists",
                "and that it contains files with one of the supported extensions",
                f"({', '.join(AUDIO_EXTENSIONS)}).",
            ]
        )

    lines.extend(
        [
            "",
            "Tip: temporarily override the configured path by setting",
            f"the {ENV_VAR_NAME} environment variable.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Entry point used by ``python -m imbatch2soundvisu``."""

    root = get_data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"Configured dataset directory does not exist: {root}. "
            "Edit DATA_ROOT in imbatch2soundvisu/config.py to point to a valid "
            "location."
        )

    files = list(_iter_audio_files(root))
    print(_format_report(root, files))


if __name__ == "__main__":  # pragma: no cover - convenience for scripts
    main()
