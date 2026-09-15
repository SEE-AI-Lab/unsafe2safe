# Legacy utilities

These scripts reproduce older experiments and are not the primary Unsafe2Safe
implementation. The active paper path is the small set of modules in
`stage2/`: `data.py`, `model.py`, `attention.py`, and `external.py`.

The legacy batch editor is exposed through
`src/unsafe2safe/scripts/run_unsafe2safe.sh` for compatibility with existing
checkpoints. Its input column names are command-line arguments, so it works
with the manifest schema already used by a local dataset.
