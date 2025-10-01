"""Project configuration.

Edit :data:`DATA_ROOT` to point at your local dataset directory. The path is
resolved relative to the current file, so you can use either absolute or
relative values. Environment variables can still override the configuration when
needed without having to modify the code again.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Directory that contains your audio dataset. Update this once after cloning
#: the repository. The default intentionally points at a clearly invalid path so
#: you do not accidentally run the project without configuring it first.
DATA_ROOT: Path = Path("/path/to/your/audio/dataset")

#: Optional environment variable that overrides :data:`DATA_ROOT` when set.
ENV_VAR_NAME = "IMBATCH2SOUNDVISU_DATA_ROOT"


def get_data_root() -> Path:
    """Return the configured dataset path.

    The function prefers an environment variable so that you can temporarily
    point to a different dataset without editing the code. When the environment
    variable is not set, the function falls back to :data:`DATA_ROOT` defined in
    this module.
    """

    override = os.getenv(ENV_VAR_NAME)
    if override:
        return Path(override).expanduser().resolve()
    return DATA_ROOT.expanduser().resolve()
