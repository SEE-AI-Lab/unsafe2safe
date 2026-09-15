from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm.auto import tqdm

from pipeline.stage1.collect_captions import collect_captions
from pipeline.stage1.qwen_common import (
    build_text_generator,
    run_text_batch,
    write_caption_json,
)
from pipeline.stage1.internvl_common import (
    load_internvl_model_and_tokenizer,
    run_internvl_batch,
)


@dataclass
class Sample:
    rel_path: Path
    image_path: Path
    vars: dict[str, Any]


def build_effective_config(cfg, purpose, dataset):
    profile = cfg["purposes"][purpose]
    override = profile.get("dataset_overrides", {}).get(dataset, {})
    # Keep the paper's configuration shallow: shared run values, then local overrides.
    source = {**profile["source"], **override.get("source", {})}
    if "csv_path" in source:
        source["csv_path"] = source["csv_path"].format(dataset=dataset)
    return {"run": {**cfg["defaults"]["run"], **profile["run"], **override.get("run", {})}, "source": source, "dataset": cfg["datasets"][dataset], "output_dir": f"outputs/{dataset}/{purpose}"}


def _row_variables(row, source_cfg):
    """Expose row fields plus explicitly mapped prompt variables."""
    variables = row.to_dict()
    for variable, column in source_cfg.get("prompt_columns", {}).items():
        variables[variable] = row[column]
    return variables


def build_samples(dataset_cfg, source_cfg):
    source_type = source_cfg["type"]
    root_dir = dataset_cfg["root_dir"]

    if source_type == "csv":
        # Metadata-driven source: one row per image with optional extra columns for prompts.
        csv_path = source_cfg["csv_path"]
        df = pd.read_csv(csv_path)

        samples: list[Sample] = []
        for _, row in df.iterrows():
            rel = Path(str(row["file"]))
            image_path = Path(root_dir) / rel
            vars_dict = _row_variables(row, source_cfg)
            vars_dict["image_class"] = rel.parent.name
            samples.append(Sample(rel_path=rel, image_path=image_path, vars=vars_dict))
        return samples

    if source_type == "glob":
        # Raw folder source: enumerate files directly under dataset root.
        files = [p for p in Path(root_dir).glob("**/*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        samples = []
        for p in files:
            rel = p.relative_to(root_dir)
            vars_dict = {"image_class": p.parent.name, "file": rel}
            samples.append(Sample(rel_path=rel, image_path=p, vars=vars_dict))
        return samples

    raise ValueError(f"Unsupported source type: {source_type}")


def build_text_messages(system_prompt, prompt_template, batch):
    # Qwen receives one text-only chat conversation for each metadata row.
    messages = []
    for s in batch:
        user_text = prompt_template.format_map(s.vars)
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
        )
    return messages


def run_job(effective_cfg, purpose, dataset_name):
    run_cfg = effective_cfg["run"]
    source_cfg = effective_cfg["source"]
    dataset_cfg = effective_cfg["dataset"]

    backend = run_cfg["backend"]
    batch_size = run_cfg["batch_size"]
    max_new_tokens = run_cfg["max_new_tokens"]
    image_size = run_cfg["image_size"]
    prompt_text = Path(run_cfg["prompt_path"]).read_text(encoding="utf-8").strip()
    system_prompt = run_cfg["system_prompt"]

    output_dir = effective_cfg["output_dir"]

    samples = build_samples(dataset_cfg, source_cfg)
    # The paper uses InternVL for structured image responses and Qwen for text-only rewrites.
    if backend == "qwen_text":
        generator = build_text_generator(
            run_cfg["model_id"],
            cache_dir=run_cfg["cache_dir"],
        )
    elif backend == "internvl":
        model, tokenizer = load_internvl_model_and_tokenizer(
            run_cfg["model_id"],
            cache_dir=run_cfg["cache_dir"],
            device=run_cfg["device"],
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    desc = f"purpose={purpose} dataset={dataset_name} backend={backend}"
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        # Full-batch inference keeps throughput high and avoids fragmented GPU work.
        batch = samples[i : i + batch_size]
        save_paths = [Path(output_dir) / f"{s.rel_path.with_suffix('')}_caption.json" for s in batch]

        if backend == "qwen_text":
            msgs = build_text_messages(system_prompt, prompt_text, batch)
            outputs = run_text_batch(generator, msgs, max_new_tokens=max_new_tokens, batch_size=batch_size)
        elif backend == "internvl":
            image_paths = [s.image_path for s in batch]
            image_classes = [s.vars["image_class"] for s in batch]
            outputs = run_internvl_batch(
                model,
                tokenizer,
                image_paths,
                image_classes,
                prompt_text,
                system_prompt=system_prompt,
                image_size=image_size,
                max_new_tokens=max_new_tokens,
                device=run_cfg["device"],
            )
        for out_text, out_path in zip(outputs, save_paths):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_caption_json(out_path, out_text)

    # Keep one manifest per purpose so the next profile can consume it directly.
    output_column = {"generate_captions": "caption", "generate_flags": "caption", "generate_edit_instructions": "EDIT_INSTRUCTION", "combine_caption_and_edit": "COMBINED_CAPTION"}[purpose]
    collect_captions(output_dir, Path(output_dir).with_suffix(".csv"), parse_structured=backend == "internvl", output_column=output_column)


def main():
    parser = argparse.ArgumentParser(description="Unified Stage1 runner")
    parser.add_argument("--config", type=str, default="pipeline/stage1/config.yaml")
    parser.add_argument("--purpose", default="generate_captions", help="Stage 1 job to run")
    parser.add_argument("--dataset", default="mscoco", help="Dataset profile to run")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    purpose = args.purpose
    dataset_name = args.dataset
    effective_cfg = build_effective_config(cfg, purpose, dataset_name)
    run_job(effective_cfg, purpose, dataset_name)


if __name__ == "__main__":
    main()
