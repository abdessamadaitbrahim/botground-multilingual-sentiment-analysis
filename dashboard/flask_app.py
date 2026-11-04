# flask_app.py: Botground Sentiment Analysis, Flask backend
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# Add the project root to the path so that `src` can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import SentimentPredictor, AVAILABLE_MODELS
from database import (
    get_all_conversations, get_stats, save_conversation,
    delete_conversation, clear_all, get_problematic_conversations,
)

app = Flask(__name__)

# Cache loaded models to avoid reloading them on every request
_models = {}


def get_model(model_name: str) -> SentimentPredictor:
    if model_name not in _models:
        _models[model_name] = SentimentPredictor(model_name)
    return _models[model_name]


# PAGES

@app.route("/")
def index():
    date_from = request.args.get("date_from", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
    date_to = request.args.get("date_to", datetime.now().strftime("%Y-%m-%d"))
    stats = get_stats(date_from, date_to)
    recent, _ = get_all_conversations(date_from=date_from, date_to=date_to, limit=10)
    problematic = get_problematic_conversations(threshold=0.85, limit=5)
    return render_template(
        "index.html",
        stats=stats,
        recent=recent,
        problematic=problematic,
        date_from=date_from,
        date_to=date_to,
        models=list(AVAILABLE_MODELS.keys()),
    )


@app.route("/analyze")
def analyze():
    return render_template("analyze.html", models=list(AVAILABLE_MODELS.keys()))


@app.route("/conversations")
def conversations():
    sentiment = request.args.get("sentiment", "")
    language = request.args.get("language", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    search = request.args.get("search", "")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = 10

    convs, total_count = get_all_conversations(
        sentiment=sentiment or None,
        language=language or None,
        date_from=date_from or None,
        date_to=date_to or None,
        search=search or None,
        limit=per_page,
        offset=(page - 1) * per_page,
        order="desc",
    )

    total_pages = (total_count + per_page - 1) // per_page

    return render_template(
        "conversations.html",
        conversations=convs,
        filters={
            "sentiment": sentiment,
            "language": language,
            "date_from": date_from,
            "date_to": date_to,
            "search": search,
        },
        total=total_count,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@app.route("/analytics")
def analytics():
    date_from = request.args.get("date_from", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
    date_to = request.args.get("date_to", datetime.now().strftime("%Y-%m-%d"))
    stats = get_stats(date_from, date_to)
    return render_template("analytics.html", stats=stats, date_from=date_from, date_to=date_to)


# API

@app.route("/api/predict/single", methods=["POST"])
def api_predict_single():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        model_name = data.get("model", "XLM-RoBERTa")
        language = data.get("language", "auto")

        if not text:
            return jsonify({"error": "Text is required"}), 400
        if not language:
            return jsonify({"error": "Language is required"}), 400

        result = get_model(model_name).predict_single(text)

        save_conversation(
            text=text,
            sentiment=result["label"],
            confidence=result["confidence"],
            score_positive=result["scores"].get("positive", 0),
            score_negative=result["scores"].get("negative", 0),
            score_neutral=result["scores"].get("neutral", 0),
            language=language,
            model_used=model_name,
            conv_type="single",
        )

        return jsonify({
            "success": True,
            "text": text,
            "sentiment": result["label"],
            "confidence": round(result["confidence"] * 100, 1),
            "scores": {k: round(v * 100, 1) for k, v in result["scores"].items()},
            "color": result["color"],
            "language": language,
            "model": model_name,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/conversation", methods=["POST"])
def api_predict_conversation():
    try:
        data = request.get_json() or {}
        messages = data.get("messages", [])
        model_name = data.get("model", "XLM-RoBERTa")
        language = data.get("language", "auto")

        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        if not language:
            return jsonify({"error": "Language is required"}), 400

        result = get_model(model_name).predict_conversation(messages)
        msgs = result["messages"]
        n = result["total_messages"]

        # The whole conversation is saved as one record, with averaged scores
        save_conversation(
            text="\n".join(m["text"] for m in msgs),
            sentiment=result["final_label"],
            confidence=sum(m["confidence"] for m in msgs) / n,
            score_positive=sum(m["scores"].get("positive", 0) for m in msgs) / n,
            score_negative=sum(m["scores"].get("negative", 0) for m in msgs) / n,
            score_neutral=sum(m["scores"].get("neutral", 0) for m in msgs) / n,
            language=language,
            model_used=model_name,
            conv_type="conversation",
        )

        return jsonify({
            "success": True,
            "final_sentiment": result["final_label"],
            "final_color": result["final_color"],
            "satisfaction": result["satisfaction"],
            "total_messages": n,
            "positive_count": result["positive_count"],
            "negative_count": result["negative_count"],
            "neutral_count": result["neutral_count"],
            "model": model_name,
            "language": language,
            "messages": [
                {
                    "text": m["text"],
                    "sentiment": m["label"],
                    "confidence": round(m["confidence"] * 100, 1),
                    "color": m["color"],
                    "scores": {k: round(v * 100, 1) for k, v in m["scores"].items()},
                }
                for m in msgs
            ],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats(request.args.get("date_from"), request.args.get("date_to")))


@app.route("/api/conversations/delete/<int:conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    try:
        delete_conversation(conv_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/clear", methods=["DELETE"])
def api_clear_all():
    try:
        clear_all()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export")
def api_export():
    try:
        # get_all_conversations returns (rows, total_count)
        convs, _ = get_all_conversations(
            date_from=request.args.get("date_from"),
            date_to=request.args.get("date_to"),
        )
        df = pd.DataFrame(convs)

        output = io.StringIO()
        df.to_csv(output, index=False, encoding="utf-8")

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"botground_export_{datetime.now().strftime('%Y%m%d')}.csv",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local demo only. Set FLASK_DEBUG=1 to enable the debugger.
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=5000)
