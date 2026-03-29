"""
LIAR dataset loader.
Loads the LIAR dataset (Wang, 2017) from HuggingFace as a substitute/supplement
for NELA-GT when source-credibility labels are needed.

LIAR contains ~12,800 short political statements from PolitiFact with 6 labels:
    pants-fire, false, barely-true, half-true, mostly-true, true

Label mappings:
    6-way  -> maps directly to credibility spectrum
    3-way  -> unreliable / mixed / reliable
    2-way  -> fake (pants-fire + false + barely-true) / true (half-true + mostly-true + true)

No extra dependencies required — downloads directly from UCSB via urllib.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np


# LIAR original 6 labels (ordered least to most credible)
LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

# Map LIAR label string -> numeric id for 6-way
LIAR_6WAY_MAP = {label: i for i, label in enumerate(LIAR_LABELS)}

# 3-way: unreliable / mixed / reliable
LIAR_3WAY_MAP = {
    "pants-fire":  (0, "unreliable"),
    "false":       (0, "unreliable"),
    "barely-true": (1, "mixed"),
    "half-true":   (1, "mixed"),
    "mostly-true": (2, "reliable"),
    "true":        (2, "reliable"),
}

# 2-way: fake / true
LIAR_2WAY_MAP = {
    "pants-fire":  (0, "fake"),
    "false":       (0, "fake"),
    "barely-true": (0, "fake"),
    "half-true":   (1, "true"),
    "mostly-true": (1, "true"),
    "true":        (1, "true"),
}


LIAR_TSV_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_ct",
    "false_ct", "half_true_ct", "mostly_true_ct", "pants_fire_ct", "context",
]

LIAR_DOWNLOAD_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
LIAR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "liar")


def _download_liar(cache_dir: str = LIAR_CACHE_DIR):
    """Download and extract LIAR TSV files if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train.tsv")

    if os.path.exists(train_path):
        return cache_dir

    zip_path = os.path.join(cache_dir, "liar_dataset.zip")
    print(f"  Downloading LIAR dataset...")
    urllib.request.urlretrieve(LIAR_DOWNLOAD_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)
    os.remove(zip_path)
    print(f"  Extracted to {cache_dir}")
    return cache_dir


def _load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", header=None,
        names=LIAR_TSV_COLUMNS, quoting=3,
    )


def load_liar(
    label_scheme: str = "2way",
    max_samples: int = None,
    split: str = "all",
    cache_dir: str = LIAR_CACHE_DIR,
) -> pd.DataFrame:
    """
    Load the LIAR dataset, downloading TSV files from UCSB if needed.

    Args:
        label_scheme: One of '2way', '3way', '6way'.
        max_samples: Maximum number of samples to return.
        split: 'train', 'validation', 'test', or 'all' (combines all splits).
        cache_dir: Where to store the downloaded files.

    Returns:
        DataFrame with 'text', 'label', 'label_id', 'source' columns.
    """
    print("Loading LIAR dataset...")
    cache_dir = _download_liar(cache_dir)

    split_files = {
        "train":      os.path.join(cache_dir, "train.tsv"),
        "validation": os.path.join(cache_dir, "valid.tsv"),
        "test":       os.path.join(cache_dir, "test.tsv"),
    }

    splits = list(split_files.keys()) if split == "all" else [split]
    dfs = []
    for s in splits:
        path = split_files.get(s)
        if path and os.path.exists(path):
            dfs.append(_load_tsv(path))

    if not dfs:
        raise FileNotFoundError(f"No LIAR TSV files found in {cache_dir}")

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  Raw samples loaded: {len(df_raw)}")

    # Build text from statement + subject for context
    df_raw["text"] = df_raw.apply(
        lambda row: f"{row['statement']} [{row['subject']}]"
        if pd.notna(row.get("subject")) and str(row.get("subject", "")).strip()
        else str(row["statement"]),
        axis=1,
    )

    # Apply label scheme
    if label_scheme == "6way":
        df_raw["label_id"] = df_raw["label"].map(LIAR_6WAY_MAP).fillna(-1).astype(int)
        df_raw["label"] = df_raw["label_id"].map(
            lambda i: LIAR_LABELS[i] if 0 <= i < len(LIAR_LABELS) else "unknown"
        )
    elif label_scheme == "3way":
        df_raw["label_id"] = df_raw["label"].map(lambda x: _resolve_label(x, LIAR_3WAY_MAP)[0])
        df_raw["label"]    = df_raw["label"].map(lambda x: _resolve_label(x, LIAR_3WAY_MAP)[1])
    else:  # 2way
        df_raw["label_id"] = df_raw["label"].map(lambda x: _resolve_label(x, LIAR_2WAY_MAP)[0])
        df_raw["label"]    = df_raw["label"].map(lambda x: _resolve_label(x, LIAR_2WAY_MAP)[1])

    df_raw["source"] = df_raw["speaker"].fillna("unknown")

    df = df_raw[["text", "label", "label_id", "source"]].copy()
    df = df.dropna(subset=["text", "label_id"])
    df = df[df["label_id"] >= 0]
    df = df[df["text"].str.strip().str.len() > 10].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  Final samples: {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def _resolve_label(raw_label, mapping: dict):
    return mapping.get(str(raw_label).strip(), (-1, "unknown"))


if __name__ == "__main__":
    for scheme in ["2way", "3way", "6way"]:
        print(f"\n{'='*50}")
        print(f"  LIAR — {scheme}")
        print(f"{'='*50}")
        df = load_liar(label_scheme=scheme, max_samples=3000)
        print(df.head(3).to_string())
