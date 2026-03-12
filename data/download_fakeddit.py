"""
Fakeddit dataset loader.
Loads and processes the Fakeddit dataset TSV files for political misinformation detection.

Expected directory structure:
    data/fakeddit/
        train.tsv
        validate.tsv
        test.tsv

Download from: https://github.com/entitize/Fakeddit
"""

import os
import pandas as pd
import numpy as np


# Fakeddit 6-way label mapping
LABEL_MAP_6WAY = {
    0: "true",
    1: "satire/parody",
    2: "misleading content",
    3: "imposter content",
    4: "false connection",
    5: "manipulated content",
}

LABEL_MAP_3WAY = {
    0: "true",
    1: "satire/parody",
    2: "fake",
    3: "fake",
    4: "fake",
    5: "fake",
}

LABEL_MAP_2WAY = {
    0: "true",
    1: "fake",
    2: "fake",
    3: "fake",
    4: "fake",
    5: "fake",
}

# Political subreddits / keywords for filtering
POLITICAL_SUBREDDITS = {
    "politics", "worldnews", "news", "politicalhumor",
    "conservative", "liberal", "democrats", "republican",
    "neutralpolitics", "politicaldiscussion", "uspolitics",
    "the_donald", "sandersforpresident", "libertarian",
    "socialism", "progressive", "geopolitics",
}

POLITICAL_KEYWORDS = [
    "president", "congress", "senate", "election", "vote",
    "democrat", "republican", "government", "policy", "law",
    "legislation", "campaign", "political", "politician",
    "trump", "biden", "obama", "clinton", "white house",
    "supreme court", "immigration", "healthcare", "tax",
    "military", "foreign policy", "trade", "sanctions",
]


def load_fakeddit(
    data_dir: str = "data/fakeddit",
    label_scheme: str = "6way",
    filter_political: bool = True,
    max_samples: int = None,
) -> dict:
    """
    Load the Fakeddit dataset.

    Args:
        data_dir: Directory containing the TSV files.
        label_scheme: One of '2way', '3way', '6way'.
        filter_political: Whether to filter for political content only.
        max_samples: Maximum number of samples per split (for memory constraints).

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames.
    """
    label_map = {
        "2way": LABEL_MAP_2WAY,
        "3way": LABEL_MAP_3WAY,
        "6way": LABEL_MAP_6WAY,
    }[label_scheme]

    splits = {}
    split_files = {
        "train": "train.tsv",
        "val": "validate.tsv",
        "test": "test.tsv",
    }

    for split_name, filename in split_files.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping {split_name} split.")
            continue

        print(f"Loading {split_name} split from {filepath}...")

        # Fakeddit TSV columns typically include:
        # id, title, subreddit, score, upvote_ratio, num_comments,
        # created_utc, 2_way_label, 3_way_label, 6_way_label
        try:
            df = pd.read_csv(
                filepath,
                sep="\t",
                usecols=lambda c: c in [
                    "clean_title", "title", "subreddit",
                    "2_way_label", "3_way_label", "6_way_label",
                ],
                nrows=max_samples * 5 if max_samples else None,  # Read extra for filtering
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            print("Trying alternative column format...")
            df = pd.read_csv(filepath, sep="\t", on_bad_lines="skip",
                             nrows=max_samples * 5 if max_samples else None)

        # Use clean_title if available, otherwise title
        if "clean_title" in df.columns:
            df["text"] = df["clean_title"]
        elif "title" in df.columns:
            df["text"] = df["title"]
        else:
            print(f"Warning: No text column found in {filepath}")
            continue

        # Map labels based on scheme
        label_col = f"{label_scheme.replace('way', '')}_way_label"
        if label_col not in df.columns:
            # Fallback: try to use whatever label column exists
            for col in ["6_way_label", "3_way_label", "2_way_label"]:
                if col in df.columns:
                    label_col = col
                    break

        if label_col in df.columns:
            df["label_id"] = df[label_col].astype(int)
            df["label"] = df["label_id"].map(label_map)
        else:
            print(f"Warning: No label column found in {filepath}")
            continue

        # Filter for political content
        if filter_political and "subreddit" in df.columns:
            subreddit_mask = df["subreddit"].str.lower().isin(POLITICAL_SUBREDDITS)
            keyword_mask = df["text"].str.lower().str.contains(
                "|".join(POLITICAL_KEYWORDS), na=False
            )
            df = df[subreddit_mask | keyword_mask].copy()
            print(f"  Filtered to {len(df)} political samples")

        # Clean up
        df = df[["text", "label", "label_id"]].dropna().reset_index(drop=True)

        # Limit samples
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        splits[split_name] = df
        print(f"  {split_name}: {len(df)} samples")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")

    return splits


if __name__ == "__main__":
    data = load_fakeddit(label_scheme="6way", filter_political=True, max_samples=5000)
    for split, df in data.items():
        print(f"\n{split}: {len(df)} samples")
        print(df["label"].value_counts())
