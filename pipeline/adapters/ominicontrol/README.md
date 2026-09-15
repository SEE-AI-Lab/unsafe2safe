# Unsafe2Safe adapter for OminiControl

This directory contains only the Unsafe2Safe-specific adapter. The OminiControl
implementation is an external dependency and is not copied into this repository.
That keeps the upstream code, license, and history separate from the contribution
needed to reproduce the Unsafe2Safe OminiControl experiment.

## Install OminiControl

Clone OminiControl beside this repository (or use any other local checkout):

```bash
git clone https://github.com/Yuanshi9815/OminiControl.git ../OminiControl
cd ../OminiControl
git checkout 65d929e
python -m pip install -r train/requirements.txt
cd -
```

The adapter is compatible with the upstream OminiControl training API. The
revision above is the vanilla checkout used during development. For exact
paper-era provenance, the older OminiControl commit `54913bc` is also supported;
select one revision and record it with the experiment outputs.

Install this repository's requirements in the same environment as OminiControl,
then expose both roots when launching training:

```bash
python -m pip install -r requirements.txt
export OMINICONTROL_ROOT=/absolute/path/to/OminiControl
export PYTHONPATH="$PWD:$OMINICONTROL_ROOT:${PYTHONPATH:-}"
```

No OminiControl source files need to be copied or modified.

## Data expected by the adapter

The default config expects the filtered MS-COCO edited-pair data used by the
paper. Paths are relative to the repository root when using
`pipeline/adapters/ominicontrol/config.example.yaml` unless absolute paths are
supplied. The pair CSV must contain:

* `file`: relative path shared by the source and edited target images;
* `caption`: the edit instruction used as the text condition.

`image_root/file` is the unsafe source condition and
`target_root/file` is the safe training target. If `clip_score_path` is provided,
the adapter retains rows with `clip_edit / clip_orig > 0.7`, matching the paper's
filtering rule. It uses the same deterministic 75/25 train/validation split and
zero positional offset as the local OminiControl experiment.

Copy the example config and update its data paths before training. Do not commit private
datasets, model tokens, generated images, or machine-specific checkpoint paths.

## Train

From the Unsafe2Safe repository root:

```bash
export OMINICONTROL_ROOT=/absolute/path/to/OminiControl
bash pipeline/scripts/train_ominicontrol_unsafe2safe.sh
```

The checked-in recipe uses FLUX.1-dev, subject conditioning, batch size 4, and
12,000 training steps. OminiControl's upstream trainer writes checkpoints and a
copy of the resolved config below `train.save_path`.

## Generate outputs

After training, run the project-specific generation wrapper against an upstream
OminiControl checkout and a saved adapter:

```bash
export PYTHONPATH="$PWD:$OMINICONTROL_ROOT:${PYTHONPATH:-}"
python -m pipeline.adapters.ominicontrol.generate_unsafe2safe \
  --input-csv data/MSCOCO_Qwen_4B_detailed_face.csv \
  --image-root data/coco \
  --output-dir outputs/ominicontrol \
  --checkpoint runs/ominicontrol/<run>/ckpt/<step> \
  --base-model black-forest-labs/FLUX.1-schnell \
  --split-column split --split val
```

The generation defaults follow the existing local inference recipe (512x512,
8 steps, and a fixed seed), while all paths and model choices are explicit CLI
arguments. The paper specifies the OminiControl training setup more precisely
than the inference defaults, so record any changed inference settings with the
results.

## Provenance

The adapter is derived from the OminiControl subject-training interface, but the
Unsafe2Safe contribution is the dataset mapping: unsafe image condition, safe
target, edit-instruction text, and zero positional offset. OminiControl remains
the external upstream project and is governed by its own Apache-2.0 license.
