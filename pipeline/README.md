# Pipeline

This directory contains the project-owned dataset, editing, training adapters, evaluation, and prompt-demo code. External model repositories and checkpoints are not copied into this repository.

## InstructPix2Pix integration

The InstructPix2Pix path expects a clean external checkout. The current local reproduction used [`timothybrooks/instruct-pix2pix`](https://github.com/timothybrooks/instruct-pix2pix) at commit `0dffd1e`.

```bash
git clone https://github.com/timothybrooks/instruct-pix2pix.git /path/to/instruct-pix2pix
git -C /path/to/instruct-pix2pix checkout 0dffd1e
```

`instruct_pix2pix.py` defines the external import boundary. `unsafe2safe_model.py` and `safe_attention.py` provide the project-specific model integration without modifying the external checkout.

Train with the released configuration:

```bash
./pipeline/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  pipeline/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

The training CSV should contain `file`, `c1`, and `caption`. The loader treats `c1` as the public semantic caption and `caption` as the edit instruction. Older exports may call `c1` `priv_caption`.

Run the legacy batch editor from the repository root:

```bash
./pipeline/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT
```

That editor expects the external diffusion checkout at `stable_diffusion/`, a compatible config, and a checkpoint. Keep all three outside version control when possible.

## Dataset filtering

Filter edited pairs by normalized CLIP similarity:

```bash
python pipeline/dataset_creation/filter_dataset.py \
  scores.csv filtered_scores.csv \
  --threshold 0.7
```

The input CSV must contain `clip_orig` and `clip_edit`. A row is kept when `clip_edit / clip_orig` is greater than the threshold.

## OminiControl adapter

`ominicontrol/` contains only the Unsafe2Safe-specific dataset adapter and launch wrappers. Install OminiControl separately, set `OMINICONTROL_ROOT`, and follow [`ominicontrol/README.md`](ominicontrol/README.md). The upstream OminiControl and FLUX source remain external.

## Evaluation helpers

The reusable modules under `metrics/` provide:

- CLIP and directional CLIP similarity.
- SSIM and LPIPS image similarity.
- Nearest-counterpart FaceSim.
- Token-set TextSim and normalized Race Entropy.
- VLM anonymization score collection.
- BLEU-4 and CIDEr captioning scores.
- Downstream top-1 classification accuracy.

For example, collect VLM scores from generated caption JSON files:

```bash
python pipeline/metrics/vlm_score.py outputs/scores outputs/vlm_scores.json
```

The prompt demo is optional and requires its own `datasets`, `gradio`, and `openai` installation:

```bash
python pipeline/prompt_app.py --openai-api-key "$OPENAI_API_KEY" --openai-model MODEL_NAME
```
