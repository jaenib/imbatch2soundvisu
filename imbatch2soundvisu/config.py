"""Central configuration knobs for the project.

Edit :data:`DATA_ROOT` once so the toolkit knows where to look for your image
material.  The value can be absolute or relative to this file.  An environment
variable override remains available for quick experiments with alternative
collections without touching the code.
"""

from __future__ import annotations

import os
from pathlib import Path
from imbatch2soundvisu.user_secrets import user_path

#: Directory that contains the image material you want to experiment with.
#: Update this value right after cloning the repository so everything else in
#: the toolkit can discover your files.
if str(user_path).startswith("~"):
    DATA_ROOT: Path = Path(user_secrets.user_path).expanduser().resolve()
else:
    DATA_ROOT: Path = Path(user_path).resolve()

#: Optional environment variable that overrides :data:`DATA_ROOT` when set.
ENV_VAR_NAME = "IMBATCH2SOUNDVISU_DATA_ROOT"

#: File extensions that are treated as images.  Feel free to extend the tuple if
#: you work with additional formats.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
)


def get_data_root() -> Path:
    """Return the configured dataset path.

    The function prefers an environment variable so that you can temporarily
    point to a different dataset without editing the code.  When the environment
    variable is not set, the function falls back to :data:`DATA_ROOT` defined in
    this module.
    """

    override = os.getenv(ENV_VAR_NAME)
    if override:
        return Path(override).expanduser().resolve()
    return DATA_ROOT.expanduser().resolve()
