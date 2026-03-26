# Unsafe2Safe: Controllable Image Anonymization for Downstream Utility

Official repository for the paper **Unsafe2Safe: Controllable Image Anonymization for Downstream Utility**.

[![Paper](PLACEHOLDER_PAPER_URL)](PLACEHOLDER_PAPER_URL)
[![Project Page](PLACEHOLDER_PROJECT_PAGE_URL)](PLACEHOLDER_PROJECT_PAGE_URL)
[![Demo](PLACEHOLDER_DEMO_URL)](PLACEHOLDER_DEMO_URL)
[![Dataset](PLACEHOLDER_DATASET_URL)](PLACEHOLDER_DATASET_URL)
[![Model Weights](PLACEHOLDER_WEIGHTS_URL)](PLACEHOLDER_WEIGHTS_URL)


> [!IMPORTANT]
> **Update (CVPR 2026):** Unsafe2Safe was **accepted at CVPR 2026**.

> [!TODO]
> **Current TODOs**
> - [ ] Replace all `PLACEHOLDER_*` links with final project resources
> - [ ] Release training/inference code for the clean refactored structure
> - [ ] Release pretrained checkpoints
> - [ ] Release anonymized dataset metadata and download instructions
> - [ ] Add full reproduction guide (exact configs + commands)
> - [ ] Add final camera-ready citation and author list


## TL;DR
Unsafe2Safe is a two-stage pipeline for privacy-preserving image anonymization with downstream utility:

1. **Stage 1: Privacy-aware text prior generation**
- Detect privacy risk
- Produce private/public captions
- Generate edit instructions

2. **Stage 2: Safe image generation**
- Edit unsafe images using diffusion-based editors
- Preserve non-sensitive semantics and scene structure
- Improve privacy while maintaining utility

---

## Repository Structure (tentative)

```text
unsafe2safe/
  README.md
  LICENSE
  pyproject.toml
  requirements.txt

  configs/
    stage1/
      privacy_inspection.yaml
      captioning.yaml
      instruction_gen.yaml
    stage2/
      editor_train.yaml
      editor_infer.yaml
    eval/
      privacy.yaml
      quality.yaml
      utility.yaml

  src/
    unsafe2safe/
      stage1/
        inspector.py
        captioner.py
        instruction_generator.py
        prompts/
      stage2/
        editors/
          instructpix2pix_editor.py
          flowedit_editor.py
        trainer.py
        infer.py
      data/
        datasets.py
        transforms.py
        io.py
      evaluation/
        quality.py
        cheating.py
        privacy.py
        utility.py
      utils/
        logging.py
        seed.py
        paths.py

  scripts/
    stage1_generate_priors.sh
    stage2_train_editor.sh
    stage2_infer_safe_images.sh
    evaluate_all.sh

  data/
    README.md
    PLACEHOLDER_DATA_LAYOUT.md

  checkpoints/
    .gitkeep

  outputs/
    .gitkeep

  docs/
    method.md
    metrics.md
    reproduction.md
    faq.md

  tests/
    test_stage1.py
    test_stage2.py
    test_eval.py
```


---

## Method Overview

### Stage 1: Inspection + Text Priors
Given an image, Stage 1 produces:
- `privacy_flag`: whether sensitive content exists
- `c_priv`: private caption (contains sensitive attributes)
- `c_pub`: public caption (removes/neutralizes sensitive attributes)
- `c_edit`: compact edit instruction for transformation

### Stage 2: Safe Generation
Condition a diffusion editor on textual priors (`c_pub`, `c_edit`) to generate a privacy-safe image while preserving non-private, task-relevant content.

---

## Evaluation Protocol
We follow four metric groups from the paper:
- **Quality**: realism / semantic alignment
- **Cheating**: unintended copying from source
- **Privacy**: leakage reduction and demographic anonymization behavior
- **Utility**: downstream task performance after training on anonymized data

Detailed definitions: [PLACEHOLDER_METRICS_DOC_URL](PLACEHOLDER_METRICS_DOC_URL)

---

## Quick Start

### 1) Environment
```bash
# pip install -r requirements.txt
```

### 2) Stage 1: Generate priors
```bash
python -m unsafe2safe.stage1.run --config configs/stage1/privacy_inspection.yaml
```

### 3) Stage 2: Train / Infer
```bash
bash scripts/stage2_train_editor.sh
```
```bash
bash scripts/stage2_infer_safe_images.sh
```

### 4) Evaluate
```bash
bash scripts/evaluate_all.sh
```

---

## Data Format
TODO
---

## Planned Releases
- [ ] Training code
- [ ] Inference code
- [ ] Evaluation suite
- [ ] Pretrained weights
- [ ] Processed anonymized datasets
- [ ] Reproducibility scripts

Release tracker: [PLACEHOLDER_RELEASE_TRACKER_URL](PLACEHOLDER_RELEASE_TRACKER_URL)

---

## Citation
```bibtex
@inproceedings{PLACEHOLDER_UNSAFE2SAFE_2026,
  title={Unsafe2Safe: Controllable Image Anonymization for Downstream Utility},
  author={PLACEHOLDER_AUTHORS},
  booktitle={CVPR},
  year={2026}
}
```

---

## Acknowledgements
Built with ideas and components from modern vision-language models and diffusion editing frameworks.

Detailed acknowledgements: [PLACEHOLDER_ACKNOWLEDGEMENTS_URL](PLACEHOLDER_ACKNOWLEDGEMENTS_URL)
