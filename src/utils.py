# Utils — Utility functions for the project
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral" : "#95a5a6"
}

def get_model_path(model_name: str) -> str:
    """Return the absolute path to a fine-tuned model folder."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_dirs = {
        "XLM-RoBERTa" : "xlm-roberta-base-finetuned",
        "mBERT"       : "mbert-finetuned",
        "DarijaBERT"  : "darijaBERT-finetuned"
    }
    if model_name not in model_dirs:
        raise ValueError(f"Unknown model : {model_name}")
    return str(BASE_DIR / "models" / model_dirs[model_name])

def sentiment_to_satisfaction(sentiment: str) -> str:
    mapping = {
        "positive": "Satisfied",
        "negative": "Unsatisfied",
        "neutral" : "Neutral"
    }
    return mapping.get(sentiment, "Unknown")


def get_satisfaction_rate(df: pd.DataFrame) -> float:
    """Calculate percentage of satisfied clients."""
    if len(df) == 0:
        return 0.0
    return round(
        (df["final_sentiment"] == "positive").sum() / len(df) * 100, 1
    )


def get_unsatisfaction_rate(df: pd.DataFrame) -> float:
    """Calculate percentage of unsatisfied clients."""
    if len(df) == 0:
        return 0.0
    return round(
        (df["final_sentiment"] == "negative").sum() / len(df) * 100, 1
    )


def plot_sentiment_distribution(df: pd.DataFrame, title: str = "Sentiment Distribution"):
    """Plot sentiment distribution from a dataframe."""
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[SENTIMENT_COLORS.get(s, "#bdc3c7") for s in counts.index],
        edgecolor="white", linewidth=1.5, width=0.5
    )
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1, str(v), ha="center", fontweight="bold"
        )
    plt.tight_layout()
    return fig


def save_results(results: dict, output_path: str):
    """Save prediction results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Results saved : {output_path}")