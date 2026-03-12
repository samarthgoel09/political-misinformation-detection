"""
NELA-GT dataset loader.
Loads and processes the NELA-GT (News Landscape) dataset for credibility classification.

Expected directory structure:
    data/nela/
        articles.db    (SQLite database with articles)
        -- OR --
        articles/      (directory of JSON files per source)
        labels.csv     (source-level credibility labels)

Download from: https://doi.org/10.7910/DVN/CHMUYZ
"""

import os
import json
import glob
import pandas as pd
import numpy as np


# NELA-GT source credibility mapping
# Aggregated labels based on Media Bias/Fact Check, NewsGuard, etc.
CREDIBILITY_MAP = {
    0: "reliable",
    1: "mixed",
    2: "unreliable",
}


def load_nela_from_json(
    data_dir: str = "data/nela",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load NELA-GT articles from JSON files.

    Args:
        data_dir: Directory containing NELA-GT data.
        max_samples: Maximum samples to load.

    Returns:
        DataFrame with 'text', 'label', 'label_id', 'source' columns.
    """
    articles_dir = os.path.join(data_dir, "articles")
    labels_file = os.path.join(data_dir, "labels.csv")

    if not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"Labels file not found at {labels_file}. "
            "Please download NELA-GT and place labels.csv in the data/nela/ directory."
        )

    # Load source-level credibility labels
    labels_df = pd.read_csv(labels_file)

    # The labels file typically has columns like:
    # source, aggregated_label (or similar)
    # Try to identify the label column
    label_col = None
    for col in labels_df.columns:
        if "label" in col.lower() or "aggregated" in col.lower() or "credibility" in col.lower():
            label_col = col
            break

    if label_col is None:
        # If can't find, try the last numeric column
        numeric_cols = labels_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            label_col = numeric_cols[-1]
        else:
            raise ValueError("Cannot identify label column in labels.csv")

    # Create source -> label mapping
    source_col = labels_df.columns[0]  # Usually the first column is the source name
    source_labels = dict(zip(
        labels_df[source_col].str.lower(),
        labels_df[label_col]
    ))

    print(f"Loaded {len(source_labels)} source credibility labels")

    # Load articles
    all_articles = []
    json_files = glob.glob(os.path.join(articles_dir, "**/*.json"), recursive=True)

    if not json_files:
        # Try loading from subdirectories named by source
        source_dirs = [d for d in os.listdir(articles_dir)
                       if os.path.isdir(os.path.join(articles_dir, d))]
        for source_name in source_dirs:
            source_path = os.path.join(articles_dir, source_name)
            json_files.extend(glob.glob(os.path.join(source_path, "*.json")))

    print(f"Found {len(json_files)} article files")

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                articles = json.load(f)
                if isinstance(articles, dict):
                    articles = [articles]
                for article in articles:
                    source = article.get("source", os.path.basename(os.path.dirname(filepath)))
                    source_lower = source.lower()

                    if source_lower in source_labels:
                        text = article.get("content", article.get("title", ""))
                        if text and len(text.strip()) > 20:
                            label_id = int(source_labels[source_lower])
                            all_articles.append({
                                "text": text[:1000],  # Truncate very long articles
                                "source": source,
                                "label_id": label_id,
                                "label": CREDIBILITY_MAP.get(label_id, "unknown"),
                            })
        except (json.JSONDecodeError, Exception) as e:
            continue

        if max_samples and len(all_articles) >= max_samples:
            break

    df = pd.DataFrame(all_articles)

    if len(df) == 0:
        print("Warning: No articles loaded. Check data directory structure.")
        return df

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} articles")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def load_nela_from_sqlite(
    db_path: str = "data/nela/nela-gt.db",
    labels_path: str = "data/nela/labels.csv",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Load NELA-GT articles from SQLite database.

    Args:
        db_path: Path to the SQLite database.
        labels_path: Path to the source credibility labels CSV.
        max_samples: Maximum samples to load.

    Returns:
        DataFrame with 'text', 'label', 'label_id', 'source' columns.
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # Load source labels
    labels_df = pd.read_csv(labels_path)
    source_col = labels_df.columns[0]
    label_col = None
    for col in labels_df.columns:
        if "label" in col.lower() or "aggregated" in col.lower():
            label_col = col
            break
    if label_col is None:
        label_col = labels_df.select_dtypes(include=[np.number]).columns[-1]

    source_labels = dict(zip(
        labels_df[source_col].str.lower(),
        labels_df[label_col]
    ))

    # Query articles
    query = "SELECT source, title, content FROM newsdata"
    if max_samples:
        query += f" LIMIT {max_samples * 3}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Map source to credibility label
    df["source_lower"] = df["source"].str.lower()
    df = df[df["source_lower"].isin(source_labels)].copy()
    df["label_id"] = df["source_lower"].map(source_labels).astype(int)
    df["label"] = df["label_id"].map(CREDIBILITY_MAP)

    # Use title + content or just content
    df["text"] = df.apply(
        lambda row: f"{row['title']}. {row['content'][:800]}"
        if pd.notna(row.get("content")) else str(row.get("title", "")),
        axis=1
    )

    df = df[["text", "label", "label_id", "source"]].dropna()
    df = df[df["text"].str.len() > 20].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} articles from SQLite")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def load_nela(data_dir: str = "data/nela", max_samples: int = None) -> pd.DataFrame:
    """
    Auto-detect NELA-GT format and load accordingly.
    """
    db_files = glob.glob(os.path.join(data_dir, "*.db"))
    labels_file = os.path.join(data_dir, "labels.csv")

    if db_files:
        return load_nela_from_sqlite(
            db_path=db_files[0],
            labels_path=labels_file,
            max_samples=max_samples,
        )
    elif os.path.exists(os.path.join(data_dir, "articles")):
        return load_nela_from_json(data_dir=data_dir, max_samples=max_samples)
    else:
        raise FileNotFoundError(
            f"No NELA-GT data found in {data_dir}. "
            "Please place either articles.db or an articles/ directory there."
        )


if __name__ == "__main__":
    df = load_nela(max_samples=5000)
    print(f"\nLoaded {len(df)} NELA-GT articles")
    print(df.head())
