# Data Loader — Load and prepare data for the pipeline
import pandas as pd
from pathlib import Path


def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found : {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Dataset loaded : {len(df)} rows | {path.name}")
    return df


def load_train_test(processed_dir: str = "../data/processed"):
    processed_dir = Path(processed_dir)
    train_df = load_dataset(processed_dir / "train_data.csv")
    test_df  = load_dataset(processed_dir / "test_data.csv")
    return train_df, test_df


def get_label_mappings():
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    return label2id, id2label