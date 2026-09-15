# Qwen3-VL OK-VQA

These scripts fine-tune and evaluate the Qwen3-VL-2B model on the OK-VQA
experiment used by the paper. The model, datasets, checkpoints, and generated
predictions stay outside this repository.

Install the Qwen3-VL training extras in a separate environment:

```bash
pip install peft trl bitsandbytes qwen-vl-utils
```

Build the LoRA adapter from the training split:

```bash
python unsafe2safe/adapters/vqa/train_qwen3_okvqa.py \
  --questions /path/to/OpenEnded_mscoco_train2014_questions.json \
  --annotations /path/to/mscoco_train2014_annotations.json \
  --image-root /path/to/coco \
  --safe-root /path/to/unsafe2safe-images \
  --safe-manifest /path/to/safe_images.csv \
  --private-manifest /path/to/private_images.csv \
  --output-dir /path/to/qwen3-okvqa-adapter
```

The safe and private manifests use a `file` column with COCO-relative image
paths. Private images without a safe counterpart are skipped. Omit both
manifests for the original-image baseline.

Generate predictions on the OK-VQA validation questions:

```bash
python unsafe2safe/adapters/vqa/evaluate_qwen3_okvqa.py \
  --questions /path/to/OpenEnded_mscoco_val2014_questions.json \
  --image-root /path/to/coco \
  --adapter /path/to/qwen3-okvqa-adapter \
  --output /path/to/okvqa_predictions.json
```

The evaluation script writes question-id to answer mappings. Use the official
OK-VQA evaluator to calculate the final VQA accuracy.
