from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm.auto import tqdm

from pipeline.stage1.qwen_common import (
    build_text_generator,
    run_text_batch,
    write_caption_json,
)
from pipeline.stage1.internvl_common import (
    load_internvl_model_and_tokenizer,
    run_internvl_batch,
    run_internvl_pair_batch,
)


class SafeDict(dict):
    # Preserve unresolved placeholders instead of raising KeyError during format_map.
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class Sample:
    rel_path: Any
    image_path: Any
    vars: dict[str, Any]
    right_image_path: Any = None


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base, extra):
    # Recursive merge lets overrides patch only selected nested keys.
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def render_templates(value, context):
    # Expand placeholders like {dataset}/{purpose} through nested config objects.
    if isinstance(value, str):
        return value.format_map(SafeDict(context))
    if isinstance(value, list):
        return [render_templates(x, context) for x in value]
    if isinstance(value, dict):
        return {k: render_templates(v, context) for k, v in value.items()}
    return value


def build_effective_config(cfg, purpose, dataset):
    defaults = cfg.get("defaults", {})
    datasets = cfg.get("datasets", {})
    purposes = cfg.get("purposes", {})

    if dataset not in datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(datasets.keys())}")
    if purpose not in purposes:
        raise ValueError(f"Unknown purpose '{purpose}'. Available: {list(purposes.keys())}")

    purpose_cfg = purposes[purpose]
    dataset_cfg = datasets[dataset]
    merged = deep_merge(defaults, purpose_cfg)

    # Purpose config can include per-dataset overrides.
    purpose_ds_overrides = purpose_cfg.get("dataset_overrides", {}).get(dataset, {})
    merged = deep_merge(merged, purpose_ds_overrides)

    # Attach dataset settings, then render templates globally.
    merged["dataset"] = deep_merge(dataset_cfg, merged.get("dataset", {}))
    context = {"dataset": dataset, "purpose": purpose, **merged["dataset"]}
    merged = render_templates(merged, context)

    return merged


def read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_path(path, *, config_dir):
    """Resolve a config path from the working directory or the repository."""
    if path is None:
        return path
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        return str(candidate)

    search_roots = [Path.cwd(), Path(config_dir), Path(config_dir).parent.parent]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    # Keep non-existent output/cache paths relative to the caller's directory.
    return str((Path.cwd() / candidate).resolve())


def resolve_runtime_paths(effective_cfg, config_dir):
    """Resolve filesystem settings after template expansion."""
    cfg = copy.deepcopy(effective_cfg)
    for section, keys in {
        "run": ("cache_dir", "hf_home", "prompt_path"),
        "source": ("csv_path", "right_root_dir"),
        "output": ("output_dir",),
        "dataset": ("root_dir",),
    }.items():
        for key in keys:
            if key in cfg.get(section, {}):
                cfg[section][key] = resolve_path(cfg[section][key], config_dir=config_dir)
    return cfg


def apply_filters(df, filters):
    if not filters:
        return df
    out = df
    for f in filters:
        col = f["column"]
        mode = f.get("mode", "contains")
        value = f["value"]
        series = out[col]
        # Text matching modes cast to string; equality keeps original dtype behavior.
        if mode == "contains":
            out = out[series.astype("string").str.contains(value, na=False)]
        elif mode == "not_contains":
            out = out[~series.astype("string").str.contains(value, na=False)]
        elif mode == "equals":
            out = out[series == value]
        else:
            raise ValueError(f"Unsupported filter mode: {mode}")
    return out


def _require_columns(df, columns, source_name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def _relative_path(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"CSV file paths must stay relative: {path}")
    return path


def _row_variables(row, source_cfg):
    """Expose all row fields plus explicitly mapped prompt variables."""
    variables = row.to_dict()
    for variable, column in source_cfg.get("prompt_columns", {}).items():
        if column not in row:
            raise ValueError(f"Prompt variable {variable!r} refers to missing column {column!r}")
        variables[variable] = row[column]
    return variables


def build_samples(dataset_cfg, source_cfg):
    source_type = source_cfg["type"]
    root_dir = dataset_cfg["root_dir"]

    if source_type == "csv":
        # Metadata-driven source: one row per image with optional extra columns for prompts.
        csv_path = source_cfg["csv_path"]
        image_col = source_cfg.get("image_col", "file")
        df = pd.read_csv(csv_path)
        _require_columns(df, [image_col, *source_cfg.get("prompt_columns", {}).values()], csv_path)
        df = apply_filters(df, source_cfg.get("filters"))
        max_rows = source_cfg.get("max_rows")
        if max_rows:
            df = df.head(max_rows)

        samples: list[Sample] = []
        for _, row in df.iterrows():
            rel = _relative_path(row[image_col])
            image_path = Path(root_dir) / rel
            vars_dict = _row_variables(row, source_cfg)
            vars_dict["image_class"] = Path(rel).parent.name
            vars_dict["class_name"] = Path(rel).parent.name
            samples.append(Sample(rel_path=rel, image_path=image_path, vars=vars_dict))
        return samples

    if source_type == "glob":
        # Raw folder source: enumerate files directly under dataset root.
        pattern = source_cfg.get("pattern", "**/*")
        exts = tuple(source_cfg.get("exts", [".jpg", ".jpeg", ".png"]))
        files = [p for p in Path(root_dir).glob(pattern) if p.is_file() and p.suffix.lower() in exts]
        max_rows = source_cfg.get("max_rows")
        if max_rows:
            files = files[:max_rows]

        samples = []
        for p in files:
            rel = p.relative_to(root_dir)
            vars_dict = {"image_class": p.parent.name, "class_name": p.parent.name, "file": rel}
            samples.append(Sample(rel_path=rel, image_path=p, vars=vars_dict))
        return samples

    if source_type == "paired_csv":
        # Pair source: left/right image paths for comparison/evaluation purposes.
        csv_path = source_cfg["csv_path"]
        left_col = source_cfg["left_image_col"]
        right_col = source_cfg["right_image_col"]
        rel_col = source_cfg.get("rel_col", left_col)
        right_root = source_cfg.get("right_root_dir", root_dir)

        df = pd.read_csv(csv_path)
        _require_columns(df, [left_col, right_col, rel_col, *source_cfg.get("prompt_columns", {}).values()], csv_path)
        df = apply_filters(df, source_cfg.get("filters"))
        max_rows = source_cfg.get("max_rows")
        if max_rows:
            df = df.head(max_rows)

        samples: list[Sample] = []
        for _, row in df.iterrows():
            rel = _relative_path(row[rel_col])
            left_rel = _relative_path(row[left_col])
            right_rel = _relative_path(row[right_col])
            left_path = Path(root_dir) / left_rel
            right_path = Path(right_root) / right_rel
            vars_dict = _row_variables(row, source_cfg)
            vars_dict["image_class"] = Path(rel).parent.name
            vars_dict["class_name"] = Path(rel).parent.name
            samples.append(Sample(rel_path=rel, image_path=left_path, right_image_path=right_path, vars=vars_dict))
        return samples

    raise ValueError(f"Unsupported source type: {source_type}")


def output_path_for(sample, output_dir, suffix="_caption.json"):
    # Preserve relative folder hierarchy in outputs.
    stem = Path(sample.rel_path).with_suffix("")
    if stem.is_absolute() or ".." in stem.parts:
        raise ValueError(f"Sample path must be relative and contained: {sample.rel_path}")
    return Path(output_dir) / f"{stem}{suffix}"


def build_text_messages(system_prompt, prompt_template, batch):
    messages = []
    for s in batch:
        user_text = prompt_template.format_map(SafeDict(s.vars))
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
        )
    return messages


def build_qwen_vl_messages(system_prompt, prompt_template, batch):
    messages = []
    for s in batch:
        user_text = prompt_template.format_map(SafeDict(s.vars))
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image", "image": f"file://{s.image_path}"},
                    ],
                },
            ]
        )
    return messages



def run_job(effective_cfg, purpose, dataset_name, *, config_dir=None):
    effective_cfg = resolve_runtime_paths(effective_cfg, config_dir or Path.cwd())
    run_cfg = effective_cfg["run"]
    source_cfg = effective_cfg["source"]
    output_cfg = effective_cfg["output"]
    dataset_cfg = effective_cfg["dataset"]

    backend = run_cfg["backend"]
    batch_size = run_cfg.get("batch_size", 16)
    max_new_tokens = run_cfg.get("max_new_tokens", 512)
    image_size = run_cfg.get("image_size", 448)
    prompt_text = read_prompt(run_cfg["prompt_path"])
    system_prompt = run_cfg["system_prompt"]

    output_dir = output_cfg["output_dir"]
    suffix = output_cfg.get("filename_suffix", "_caption.json")

    samples = build_samples(dataset_cfg, source_cfg)
    if not samples:
        print("No samples found.")
        return

    if backend == "qwen_text":
        generator = build_text_generator(
            run_cfg["model_id"],
            hf_home=run_cfg.get("hf_home"),
            device_map=run_cfg.get("device_map", "cuda"),
            torch_dtype=run_cfg.get("torch_dtype", "auto"),
        )
    elif backend in {"internvl", "internvl_pair"}:
        model, tokenizer = load_internvl_model_and_tokenizer(
            run_cfg["model_id"],
            cache_dir=run_cfg.get("cache_dir", ".cache/huggingface"),
            device=run_cfg.get("device", "cuda"),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    desc = f"purpose={purpose} dataset={dataset_name} backend={backend}"
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        # Full-batch inference keeps throughput high and avoids fragmented GPU work.
        batch = samples[i : i + batch_size]
        infer_batch = batch
        save_paths = [output_path_for(s, output_dir, suffix=suffix) for s in infer_batch]

        if backend == "qwen_text":
            msgs = build_text_messages(system_prompt, prompt_text, infer_batch)
            outputs = run_text_batch(generator, msgs, max_new_tokens=max_new_tokens, batch_size=batch_size)
        elif backend == "internvl":
            image_paths = [s.image_path for s in infer_batch]
            image_classes = [s.vars.get("image_class", "") for s in infer_batch]
            outputs = run_internvl_batch(
                model,
                tokenizer,
                image_paths,
                image_classes,
                prompt_text,
                system_prompt=system_prompt,
                image_size=image_size,
                max_new_tokens=max_new_tokens,
                do_sample=run_cfg.get("do_sample", False),
                format_with_class=run_cfg.get("format_with_class", True),
                device=run_cfg.get("device", "cuda"),
            )
        else:  # internvl_pair
            left_paths = [s.image_path for s in infer_batch]
            right_paths = [s.right_image_path for s in infer_batch]
            outputs = run_internvl_pair_batch(
                model,
                tokenizer,
                left_paths,
                right_paths,
                prompt_text,
                system_prompt=system_prompt,
                image_size=image_size,
                max_new_tokens=max_new_tokens,
                do_sample=run_cfg.get("do_sample", False),
                pad_token_id=tokenizer.eos_token_id,
                device=run_cfg.get("device", "cuda"),
            )

        for out_text, out_path in zip(outputs, save_paths):
            ensure_parent(out_path)
            write_caption_json(out_path, out_text)


def main():
    parser = argparse.ArgumentParser(description="Unified Stage1 runner")
    parser.add_argument("--config", type=str, default="src/pipeline/stage1/configs/stage1.yaml")
    parser.add_argument("--purpose", type=str, default=None, help="Override purpose profile")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset profile")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(config_path)
    purpose = args.purpose or cfg.get("active_purpose")
    dataset_name = args.dataset or cfg.get("active_dataset")

    if not purpose or not dataset_name:
        raise ValueError("Both purpose and dataset must be set (via config or CLI).")

    effective_cfg = build_effective_config(cfg, purpose, dataset_name)
    run_job(effective_cfg, purpose, dataset_name, config_dir=config_path.parent)


if __name__ == "__main__":
    main()
