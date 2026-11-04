# database.py — SQLite database management
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "botground.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            text            TEXT NOT NULL,
            sentiment       TEXT NOT NULL,
            confidence      REAL NOT NULL,
            score_positive  REAL NOT NULL,
            score_negative  REAL NOT NULL,
            score_neutral   REAL NOT NULL,
            language        TEXT NOT NULL,
            model_used      TEXT NOT NULL,
            conv_type       TEXT NOT NULL,
            timestamp       TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized")


def save_conversation(
    text, sentiment, confidence,
    score_positive, score_negative, score_neutral,
    language, model_used, conv_type
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (
            text, sentiment, confidence,
            score_positive, score_negative, score_neutral,
            language, model_used, conv_type, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        text, sentiment, round(confidence, 4),
        round(score_positive, 4),
        round(score_negative, 4),
        round(score_neutral, 4),
        language, model_used, conv_type,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_all_conversations(
    sentiment=None, language=None,
    date_from=None, date_to=None,
    search=None,
    limit=None, offset=0, order="desc"
):
    conn = get_connection()
    cursor = conn.cursor()

    query  = "SELECT * FROM conversations WHERE 1=1"
    params = []

    if sentiment:
        query += " AND sentiment = ?"
        params.append(sentiment)
    if language:
        query += " AND language = ?"
        params.append(language)
    if date_from:
        query += " AND timestamp >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        query += " AND timestamp <= ?"
        params.append(f"{date_to} 23:59:59")
    if search:
        query += " AND text LIKE ?"
        params.append(f"%{search}%")

    query += " ORDER BY id DESC" if order == "desc" else " ORDER BY id ASC"

    # Count total before limit
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    if limit:
        query += " LIMIT ? OFFSET ?"
        params = params + [int(limit), int(offset)]

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows, total_count


def get_problematic_conversations(threshold=0.85, limit=5):
    """Get highly negative conversations for alerts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM conversations
        WHERE sentiment = 'negative'
        AND confidence >= ?
        ORDER BY confidence DESC, timestamp DESC
        LIMIT ?
    """, (threshold, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_stats(date_from=None, date_to=None):
    conn   = get_connection()
    cursor = conn.cursor()

    params = []
    where  = "WHERE 1=1"

    if date_from:
        where += " AND timestamp >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where += " AND timestamp <= ?"
        params.append(f"{date_to} 23:59:59")

    # Total conversations
    cursor.execute(f"SELECT COUNT(*) FROM conversations {where}", params)
    total = cursor.fetchone()[0]

    # Sentiment counts
    cursor.execute(f"""
        SELECT sentiment, COUNT(*) as count
        FROM conversations {where}
        GROUP BY sentiment
    """, params)
    sentiment_counts = {row["sentiment"]: row["count"] for row in cursor.fetchall()}

    # Language counts
    cursor.execute(f"""
        SELECT language, COUNT(*) as count
        FROM conversations {where}
        GROUP BY language
    """, params)
    language_counts = {row["language"]: row["count"] for row in cursor.fetchall()}

    # Sentiment trend by day
    cursor.execute(f"""
        SELECT DATE(timestamp) as date, sentiment, COUNT(*) as count
        FROM conversations {where}
        GROUP BY DATE(timestamp), sentiment
        ORDER BY date
    """, params)
    trend = [dict(row) for row in cursor.fetchall()]

    # Sentiment by language
    cursor.execute(f"""
        SELECT language, sentiment, COUNT(*) as count
        FROM conversations {where}
        GROUP BY language, sentiment
    """, params)
    lang_sentiment = [dict(row) for row in cursor.fetchall()]

    # Average confidence
    cursor.execute(f"""
        SELECT AVG(confidence) as avg_conf FROM conversations {where}
    """, params)
    avg_confidence = cursor.fetchone()[0] or 0

    # Average scores
    cursor.execute(f"""
        SELECT 
            AVG(score_positive) as avg_pos,
            AVG(score_negative) as avg_neg,
            AVG(score_neutral)  as avg_neu
        FROM conversations {where}
    """, params)
    row = cursor.fetchone()
    avg_scores = {
        "positive": round((row["avg_pos"] or 0) * 100, 1),
        "negative": round((row["avg_neg"] or 0) * 100, 1),
        "neutral" : round((row["avg_neu"] or 0) * 100, 1)
    }

    conn.close()

    positive = sentiment_counts.get("positive", 0)
    negative = sentiment_counts.get("negative", 0)
    neutral  = sentiment_counts.get("neutral",  0)

    return {
        "total"            : total,
        "positive"         : positive,
        "negative"         : negative,
        "neutral"          : neutral,
        "satisfaction_rate": round(positive / total * 100, 1) if total else 0,
        "negative_rate"    : round(negative / total * 100, 1) if total else 0,
        "neutral_rate"     : round(neutral  / total * 100, 1) if total else 0,
        "language_counts"  : language_counts,
        "sentiment_counts" : sentiment_counts,
        "trend"            : trend,
        "lang_sentiment"   : lang_sentiment,
        "avg_confidence"   : round(avg_confidence * 100, 1),
        "avg_scores"       : avg_scores
    }


def delete_conversation(conv_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def clear_all():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


# Initialize on import
init_db()