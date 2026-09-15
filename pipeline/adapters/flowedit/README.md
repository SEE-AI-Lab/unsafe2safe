# Unsafe2Safe adapter for FlowEdit

This directory contains the FlowEdit integration: CSV condition mapping, image
loading and saving, configuration, and a batch launcher.
The FlowEdit sampler and model assets remain in a separately installed
upstream checkout.

## Upstream setup

The audited upstream source is [fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit)
at commit `2620a9364f0f9c21368c36266e6200ff250273ec`.

```bash
git clone https://github.com/fallenshock/FlowEdit.git /path/to/FlowEdit
git -C /path/to/FlowEdit checkout 2620a9364f0f9c21368c36266e6200ff250273ec
python -m pip install torch diffusers==0.30.0 transformers accelerate sentencepiece protobuf pandas pillow pyyaml tqdm
export FLOWEDIT_ROOT=/path/to/FlowEdit
```

The upstream README reports CUDA 12.4 and diffusers 0.30.0 as its tested
environment and notes that newer diffusers versions may be incompatible. Model
downloads use the normal Hugging Face environment configuration; credentials
are never passed through this repository or stored in its files.

## Data and run settings

Run from the Unsafe2Safe repository root:

```bash
bash pipeline/scripts/run_flowedit_unsafe2safe.sh \
  --input-csv /path/to/metadata.csv \
  --image-root /path/to/coco \
  --output-dir /path/to/outputs \
  --source-column SOURCE_COLUMN \
  --condition TARGET_COLUMN
```

The example config is [`config.example.yaml`](config.example.yaml). It reproduces the
paper-era FlowEdit SD3 route: 50 steps, `n_avg=1`, source guidance 3.5,
target guidance 13.5, `n_min=0`, `n_max=33`, seed 42, and a maximum resolution
of 1536 pixels. The source and target text columns are command-line inputs.
Pass multiple target columns to run them all:

```bash
... --source-column SOURCE_COLUMN --condition TARGET_A TARGET_B
```

Or run every other string-valued CSV column as a target condition:

```bash
... --source-column SOURCE_COLUMN --all-conditions
```

When multiple conditions are selected, outputs are placed in separate
`output-dir/COLUMN/` directories. Rows whose `file` value begins with `val`
are excluded by default, matching the paper's train-only generation convention.

The CSV must contain the configured file, source, and target columns. File
values are relative paths; `image-root/file` is read and the generated image is
written to `output-dir/file` for one condition or `output-dir/COLUMN/file` for
multiple conditions. Existing outputs are skipped unless
`--overwrite` is supplied.

## External dependency

`FlowEdit_utils.py`, upstream configs, demo assets, datasets, checkpoints, and
the historical scripts are intentionally not copied into this repository.
Follow the upstream [license](https://github.com/fallenshock/FlowEdit/blob/main/LICENSE)
when obtaining that dependency.

Full GPU/model execution was not run in this workspace because it requires the
external model dependencies, Hugging Face access, and a CUDA-capable runtime.
