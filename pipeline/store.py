"""Write classified records and aggregates into SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from config.settings import settings
from pipeline.metrics import friction_score, stakes_score

_SCHEMA = """
CREATE TABLE items (
    id TEXT PRIMARY KEY,
    source TEXT,
    source_url TEXT,
    source_detail TEXT,
    timestamp TEXT,
    region TEXT,
    rating REAL,
    text TEXT,
    is_retrieval_related INTEGER,
    filter_reason TEXT
);
CREATE TABLE extractions (
    id TEXT PRIMARY KEY,
    archetype TEXT,
    archetype_conf REAL,
    secondary_archetypes TEXT,
    failure_stage TEXT,
    contributing_factors TEXT,
    retrieval_surface TEXT,
    user_segment TEXT,
    emotion TEXT,
    user_workaround TEXT,
    outcome TEXT,
    friction_tier REAL,
    stakes_score REAL,
    stakes_level TEXT,
    device_platform TEXT,
    media_type TEXT,
    representative_quote TEXT,
    relevance_score REAL,
    extraction_confidence REAL,
    FOREIGN KEY (id) REFERENCES items(id)
);
CREATE TABLE anchors_retained (
    id TEXT,
    kind TEXT,
    value TEXT
);
CREATE TABLE anchors_forgotten (
    id TEXT,
    value TEXT
);
CREATE TABLE search_attempts (
    id TEXT,
    step TEXT,
    query_text TEXT
);
CREATE TABLE archetype_metrics (
    archetype TEXT PRIMARY KEY,
    count INTEGER,
    freq_score REAL,
    impact_score REAL,
    opportunity_score REAL,
    band TEXT,
    abandonment_rate REAL,
    avg_attempts REAL
);
CREATE TABLE segment_metrics (
    segment TEXT PRIMARY KEY,
    count INTEGER,
    problem_rate REAL,
    top_archetype TEXT,
    implication TEXT
);
CREATE TABLE surface_metrics (
    surface TEXT PRIMARY KEY,
    mentions INTEGER,
    frustration_pct REAL
);
CREATE TABLE trend_metrics (
    period TEXT,
    archetype TEXT,
    count INTEGER,
    avg_friction REAL
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def write_database(records: list[dict], aggregates: dict, path: Path | None = None) -> Path:
    path = path or settings.sqlite_path
    if path.exists():
        path.unlink()
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        for record in records:
            extraction = record.get("extraction") or {}
            conn.execute(
                """INSERT INTO items
                   (id, source, source_url, source_detail, timestamp, region, rating, text,
                    is_retrieval_related, filter_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (
                    record.get("id"),
                    record.get("source"),
                    record.get("source_url"),
                    record.get("source_detail"),
                    record.get("timestamp"),
                    extraction.get("region"),
                    record.get("rating"),
                    record.get("text"),
                    record.get("classification_note"),
                ),
            )
            conn.execute(
                """INSERT INTO extractions
                   (id, archetype, archetype_conf, secondary_archetypes, failure_stage,
                    contributing_factors, retrieval_surface, user_segment, emotion,
                    user_workaround, outcome, friction_tier, stakes_score, stakes_level,
                    device_platform, media_type, representative_quote, relevance_score,
                    extraction_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("id"),
                    record.get("archetype"),
                    record.get("archetype_confidence"),
                    json.dumps(record.get("secondary_archetypes") or []),
                    extraction.get("primary_failure_stage"),
                    json.dumps(extraction.get("contributing_factors") or []),
                    extraction.get("retrieval_surface"),
                    extraction.get("user_segment"),
                    extraction.get("emotion"),
                    extraction.get("user_workaround"),
                    extraction.get("outcome"),
                    friction_score(extraction),
                    stakes_score(extraction.get("stakes_level")),
                    extraction.get("stakes_level"),
                    extraction.get("device_platform"),
                    extraction.get("media_type"),
                    extraction.get("representative_quote"),
                    record.get("relevance_score"),
                    extraction.get("extraction_confidence"),
                ),
            )
            anchors = extraction.get("memory_anchors_retained") or {}
            for kind in ("temporal", "sensory", "emotional", "social"):
                for value in anchors.get(kind) or []:
                    if value:
                        conn.execute(
                            "INSERT INTO anchors_retained (id, kind, value) VALUES (?, ?, ?)",
                            (record.get("id"), kind, value),
                        )
            for value in extraction.get("information_forgotten") or []:
                if value:
                    conn.execute(
                        "INSERT INTO anchors_forgotten (id, value) VALUES (?, ?)",
                        (record.get("id"), value),
                    )
            formulation = extraction.get("search_formulation") or {}
            for step, key in (
                ("natural", "attempt_1_natural"),
                ("keywords", "attempt_2_keywords"),
                ("desperation", "attempt_3_desperation"),
            ):
                query = formulation.get(key)
                if query:
                    conn.execute(
                        "INSERT INTO search_attempts (id, step, query_text) VALUES (?, ?, ?)",
                        (record.get("id"), step, query),
                    )
        for row in aggregates.get("archetypes") or []:
            conn.execute(
                """INSERT INTO archetype_metrics
                   (archetype, count, freq_score, impact_score, opportunity_score, band,
                    abandonment_rate, avg_attempts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["archetype"], row["count"], row["freq_score"], row["impact_score"],
                    row["opportunity_score"], row["band"], row["abandonment_rate"],
                    row["avg_attempts"],
                ),
            )
        for row in aggregates.get("segments") or []:
            conn.execute(
                """INSERT INTO segment_metrics
                   (segment, count, problem_rate, top_archetype, implication)
                   VALUES (?, ?, ?, ?, ?)""",
                (row["segment"], row["count"], row["problem_rate"], row["top_archetype"], row["implication"]),
            )
        for row in aggregates.get("surfaces") or []:
            conn.execute(
                "INSERT INTO surface_metrics (surface, mentions, frustration_pct) VALUES (?, ?, ?)",
                (row["surface"], row["mentions"], row["frustration_pct"]),
            )
        for row in aggregates.get("trends") or []:
            conn.execute(
                "INSERT INTO trend_metrics (period, archetype, count, avg_friction) VALUES (?, ?, ?, ?)",
                (row["period"], row["archetype"], row["count"], row["avg_friction"]),
            )
        conn.commit()
    finally:
        conn.close()
    return path
