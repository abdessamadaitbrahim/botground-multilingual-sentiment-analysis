# Predict: load the fine-tuned models and predict sentiment
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessor import clean_text, validate_text

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

# Label mappings (same order as in the fine-tuning notebooks)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

BASE_DIR = Path(__file__).resolve().parent.parent
AVAILABLE_MODELS = {
    "XLM-RoBERTa": str(BASE_DIR / "models" / "xlm-roberta-base-finetuned"),
    "mBERT":       str(BASE_DIR / "models" / "mbert-finetuned"),
    "DarijaBERT":  str(BASE_DIR / "models" / "darijaBERT-finetuned"),
}


class SentimentPredictor:
    """Sentiment predictor supporting 3 fine-tuned models:
    XLM-RoBERTa, mBERT and DarijaBERT."""

    def __init__(self, model_name: str = "XLM-RoBERTa", max_length: int = 128):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from: {list(AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_path = Path(AVAILABLE_MODELS[model_name])
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model folder not found: {self.model_path}. "
                "See models/README.md to regenerate the weights."
            )

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded: {self.model_name} on {self.device}")

    def predict_single(self, text: str) -> dict:
        """Predict the sentiment of a single message."""
        text = clean_text(text)

        if not validate_text(text):
            return {
                "label": "neutral",
                "confidence": 0.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
                "color": SENTIMENT_COLORS["neutral"],
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_id = torch.argmax(probs).item()
        label = ID2LABEL[pred_id]

        return {
            "label": label,
            "confidence": round(probs[pred_id].item(), 4),
            "scores": {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))},
            "color": SENTIMENT_COLORS[label],
        }

    def predict_batch(self, texts: list) -> list:
        """Predict the sentiment of a list of messages (one forward pass each)."""
        return [self.predict_single(text) for text in texts]

    def predict_conversation(self, messages: list) -> dict:
        """Predict per-message sentiment and the overall conversation sentiment.

        Overall label: majority vote over the messages. On a tie, the label
        with the highest summed probability wins (deterministic).
        Satisfaction: share of positive messages, in percent.
        """
        if not messages:
            return {}

        message_results = []
        for msg in messages:
            result = self.predict_single(msg)
            result["text"] = msg
            message_results.append(result)

        labels = [r["label"] for r in message_results]
        counts = {lab: labels.count(lab) for lab in ID2LABEL.values()}
        prob_sum = {
            lab: sum(r["scores"][lab] for r in message_results) for lab in ID2LABEL.values()
        }
        final_label = max(counts, key=lambda lab: (counts[lab], prob_sum[lab]))

        return {
            "messages": message_results,
            "final_label": final_label,
            "final_color": SENTIMENT_COLORS[final_label],
            "satisfaction": round(counts["positive"] / len(labels) * 100, 1),
            "total_messages": len(messages),
            "positive_count": counts["positive"],
            "negative_count": counts["negative"],
            "neutral_count": counts["neutral"],
        }
