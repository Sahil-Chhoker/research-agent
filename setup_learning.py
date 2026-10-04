"""Small SQLite store for learning records and quiz attempts.

The vector index can point back to ``learning_records.id`` later. SQLite remains
the source of truth for the record content and quiz history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


DEFAULT_DATABASE_PATH = "learning.db"


def utc_now() -> str:
    """Return a stable, UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


class LearningRecord(BaseModel):
    """The small amount of information needed to retrieve learned material."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    concepts: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("topic", "summary")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("concepts", "source_urls")
    @classmethod
    def strip_list_values(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class QuizQuestion(BaseModel):
    """A generated question and the answer recorded for it."""

    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    user_answer: str | None = None
    is_correct: bool | None = None
    learning_record_id: str | None = None

    @field_validator("question", "expected_answer")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class QuizAttempt(BaseModel):
    """A quiz and its results, kept as one compact JSON document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str = Field(min_length=1)
    questions: list[QuizQuestion] = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic cannot be empty")
        return value


def init_db(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Create the compact schema and return a configured connection."""
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS learning_records (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            concepts_json TEXT NOT NULL DEFAULT '[]',
            source_urls_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            questions_json TEXT NOT NULL,
            score REAL CHECK (score IS NULL OR score BETWEEN 0 AND 1),
            created_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _model(value: LearningRecord | dict[str, Any]) -> LearningRecord:
    return value if isinstance(value, LearningRecord) else LearningRecord.model_validate(value)


def save_learning_record(
    connection: sqlite3.Connection,
    record: LearningRecord | dict[str, Any],
) -> LearningRecord:
    """Validate and insert one learning record."""
    validated = _model(record)
    connection.execute(
        """
        INSERT INTO learning_records
            (id, topic, summary, concepts_json, source_urls_json,
             confidence, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            validated.id,
            validated.topic,
            validated.summary,
            json.dumps(validated.concepts),
            json.dumps(validated.source_urls),
            validated.confidence,
            validated.created_at,
            validated.updated_at,
        ),
    )
    connection.commit()
    return validated


def find_learning_records(
    connection: sqlite3.Connection,
    topic: str,
    limit: int = 10,
) -> list[LearningRecord]:
    """Find records matching a topic, prioritising lower-confidence material."""
    topic = topic.strip()
    if not topic:
        raise ValueError("topic cannot be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    rows = connection.execute(
        """
        SELECT id, topic, summary, concepts_json, source_urls_json,
               confidence, created_at, updated_at
        FROM learning_records
        WHERE topic LIKE ? OR summary LIKE ? OR concepts_json LIKE ?
        ORDER BY confidence ASC, updated_at DESC
        LIMIT ?
        """,
        (f"%{topic}%", f"%{topic}%", f"%{topic}%", limit),
    ).fetchall()
    return [
        LearningRecord(
            id=row["id"],
            topic=row["topic"],
            summary=row["summary"],
            concepts=json.loads(row["concepts_json"]),
            source_urls=json.loads(row["source_urls_json"]),
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def save_quiz_attempt(
    connection: sqlite3.Connection,
    attempt: QuizAttempt | dict[str, Any],
) -> QuizAttempt:
    """Validate and save a generated quiz and its answers."""
    validated = (
        attempt
        if isinstance(attempt, QuizAttempt)
        else QuizAttempt.model_validate(attempt)
    )
    connection.execute(
        """
        INSERT INTO quiz_attempts
            (id, topic, questions_json, score, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            validated.id,
            validated.topic,
            json.dumps([question.model_dump() for question in validated.questions]),
            validated.score,
            validated.created_at,
        ),
    )
    connection.commit()
    return validated


if __name__ == "__main__":
    with init_db() as connection:
        print(f"Learning database ready: {connection.execute('SELECT 1').fetchone()[0]}")
