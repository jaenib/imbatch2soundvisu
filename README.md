# imbatch2soundvisu

This repository has been reset to a clean starting point. The new layout is
centered around a small Python module that discovers the audio files inside a
configured dataset directory. Update the dataset path once in
[`imbatch2soundvisu/config.py`](imbatch2soundvisu/config.py), then run the module
to verify that everything is wired correctly.

## Quick start

1. Edit [`imbatch2soundvisu/config.py`](imbatch2soundvisu/config.py) and set
   `DATA_ROOT` to the directory that contains your audio files.
2. Run the module:

   ```bash
   python -m imbatch2soundvisu
   ```

   The script prints a short report listing the discovered audio files.

## Development notes

* Configuration lives entirely in Python so you only have to update the path
  once.
* The code is intentionally lightweight to make it easy to extend with your own
  processing steps.
