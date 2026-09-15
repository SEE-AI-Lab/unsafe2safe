# Pipeline code

This directory contains the paper-owned dataset, editing, evaluation, and Safe Attention code.

The diffusion entry points expect a vanilla InstructPix2Pix checkout outside this repository. Set `INSTRUCT_PIX2PIX_ROOT` to that checkout, or use the training launcher below. The base diffusion repository is intentionally not vendored or modified here.

For the current local reproduction, the base is
[`timothybrooks/instruct-pix2pix`](https://github.com/timothybrooks/instruct-pix2pix)
at commit `0dffd1e`. Pin a clean checkout when reproducing results:

```bash
git clone https://github.com/timothybrooks/instruct-pix2pix.git /path/to/instruct-pix2pix
git -C /path/to/instruct-pix2pix checkout 0dffd1e
```

`instruct_pix2pix.py` is the import boundary for the external checkout. `safe_attention.py` supplies the project-specific public-caption/edit-instruction attention adapter and patches the external UNet in memory at model-construction time.

`unsafe2safe_model.py` provides the project-specific training wrapper on top of the external diffusion base. It does not require edits inside the external checkout.

The canonical Stage 2 CSV contains `file`, `c1`, and `caption`. Historical exports sometimes call `c1` `priv_caption`; in the project metadata convention, `c1` is the public semantic caption. The loader returns this as `caption_public` together with `caption_edit`.

The dataset and metrics helpers can be used independently with local image and metadata paths.

To keep edited training pairs with enough semantic overlap, first compute the original and edited CLIP scores, then run:

```bash
python pipeline/dataset_creation/filter_dataset.py scores.csv filtered_scores.csv --threshold 0.7
```

The default threshold follows the paper's MS-COCO filtering step.

Train the project wrapper against a vanilla checkout:

```bash
./pipeline/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  pipeline/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

Update the dataset and checkpoint paths in `pipeline/configs/train_unsafe2safe.yaml` before launching. The launcher adds this repository and the external checkout to `PYTHONPATH`, then imports the project adapter from `pipeline/`.

Collect VLM anonymization scores from generated caption JSON files:

```bash
python pipeline/metrics/vlm_score.py outputs/scores outputs/vlm_scores.json
```

`pipeline/metrics/face_similarity.py` also exposes `nearest_face_similarity`, which follows the paper's nearest-counterpart FaceSim definition.

`pipeline/metrics/privacy_scores.py` provides the paper's token-set TextSim and normalized Race Entropy formula helpers.

`pipeline/metrics/image_similarity.py` provides the paper's SSIM and VGG-16 LPIPS pair scores.

`pipeline/metrics/utility_scores.py` provides the downstream top-1 classification accuracy helper.

`pipeline/metrics/caption_scores.py` provides the BLEU-4 and CIDEr captioning utility scores.

## OminiControl reproduction

`pipeline/ominicontrol/` contains only the Unsafe2Safe dataset adapter and
launch wrappers. OminiControl is intentionally an external dependency: clone
and install it separately, set `OMINICONTROL_ROOT`, and follow
[`pipeline/ominicontrol/README.md`](ominicontrol/README.md). No OminiControl
source files are copied or modified in this repository.
