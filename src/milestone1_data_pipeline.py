"""Milestone 1 data pipeline for the fashion DDPM project.

This script validates that the Fashion Product Images Small dataset is loaded,
cleans usable rows, checks image availability, and writes small logged outputs
for the milestone demo.

Run:
    python src/milestone1_data_pipeline.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data/fashion_products")
STYLES_CSV = DATA_DIR / "styles.csv"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = Path("results")

TARGET_CLASSES = ["Apparel", "Accessories", "Footwear", "Personal Care"]


def validate_dataset() -> None:
    if not STYLES_CSV.exists():
        raise FileNotFoundError(
            "styles.csv not found. Run `python download_dataset.py` or manually place it in data/fashion_products/."
        )
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            "images/ folder not found. Place product images in data/fashion_products/images/."
        )


def build_metadata(max_rows_per_class: int = 250) -> pd.DataFrame:
    validate_dataset()
    df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")

    required_cols = ["id", "masterCategory", "subCategory", "articleType", "productDisplayName"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required_cols].dropna(subset=["id", "masterCategory"])
    df["image_path"] = df["id"].astype(str).apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

    df = df[(df["image_exists"]) & (df["masterCategory"].isin(TARGET_CLASSES))]

    balanced = (
        df.groupby("masterCategory", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max_rows_per_class), random_state=SEED))
        .reset_index(drop=True)
    )
    return balanced


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata()
    subset_path = RESULTS_DIR / "milestone1_metadata_subset.csv"
    summary_path = RESULTS_DIR / "milestone1_data_summary.json"

    metadata.to_csv(subset_path, index=False)

    summary = {
        "seed": SEED,
        "dataset": "Fashion Product Images Small",
        "rows_loaded_after_cleaning": int(len(metadata)),
        "target_classes": TARGET_CLASSES,
        "class_distribution": metadata["masterCategory"].value_counts().to_dict(),
        "subset_output": str(subset_path),
        "status": "data pipeline working and data loaded",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
