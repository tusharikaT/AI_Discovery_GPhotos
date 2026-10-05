"""Assign the six canonical archetypes from the extraction, with a consistency check."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings

ARCHETYPES = (
    "utility_document",
    "episodic_travel",
    "aesthetic_visual",
    "micro_moment",
    "disambiguation_comparative",
    "pre_verbal_sensory",
)

_DOCUMENT_MEDIA = {"document", "screenshot", "scan"}


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_record(record: dict) -> dict:
    """Return a copy with archetype, confidence, and secondary labels checked."""
    extraction = dict(record.get("extraction") or {})
    primary = extraction.get("primary_archetype")
    if primary not in ARCHETYPES:
        primary = "episodic_travel"
    secondary = [
        item for item in (extraction.get("secondary_archetypes") or []) if item in ARCHETYPES and item != primary
    ]
    confidence = float(extraction.get("extraction_confidence") or 0.5)
    media = extraction.get("media_type")
    note = "extraction"

    # A low-confidence aesthetic label on a document/screenshot is usually utility.
    if media in _DOCUMENT_MEDIA and primary in {"aesthetic_visual", "micro_moment"} and confidence < 0.45:
        secondary = [primary] + [item for item in secondary if item != "utility_document"]
        primary = "utility_document"
        confidence = max(confidence, 0.45)
        note = "document_media_override"

    out = dict(record)
    out["archetype"] = primary
    out["archetype_confidence"] = round(confidence, 3)
    out["secondary_archetypes"] = secondary
    out["classification_note"] = note
    return out


def classify_corpus(
    src: Path | None = None,
    dest: Path | None = None,
) -> tuple[Path, int]:
    settings.ensure_dirs()
    src = src or (settings.processed_dir / "extracted_records.jsonl")
    dest = dest or (settings.processed_dir / "classified.jsonl")
    rows = [classify_record(row) for row in _load(src)]
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest, len(rows)
