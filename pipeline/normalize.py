"""Normalize raw corpus text and tag language (Phase 2).

Strips markup, fixes encoding, collapses whitespace, and records the detected
language plus confidence. The strict English-only drop happens later in
``noise_filter``. Existing ids are preserved.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from config.settings import settings

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_REDDIT_BOILER_RE = re.compile(
    r"\s*submitted by\s+/u/\S+.*$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Return a single-spaced plain-text version of ``text``."""
    text = html.unescape(text or "")
    text = text.replace("\u00a0", " ")
    text = _TAG_RE.sub(" ", text)
    text = _REDDIT_BOILER_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def detect_language(text: str) -> tuple[str | None, float]:
    """Return ``(lang, confidence)``. Failures become ``(None, 0.0)``."""
    sample = text[:800]
    if len(sample.split()) < 4:
        return None, 0.0
    try:
        from langdetect import detect_langs
    except Exception:  # noqa: BLE001
        return None, 0.0
    try:
        langs = detect_langs(sample)
    except Exception:  # noqa: BLE001
        return None, 0.0
    if not langs:
        return None, 0.0
    top = langs[0]
    return top.lang, float(top.prob)


def normalize_record(record: dict) -> dict:
    text = clean_text(record.get("text") or "")
    lang, confidence = detect_language(text)
    out = dict(record)
    out["text"] = text
    out["lang"] = lang
    out["lang_confidence"] = round(confidence, 3)
    return out


def normalize_corpus(
    src: Path | None = None,
    dest: Path | None = None,
) -> tuple[Path, int]:
    """Write normalized records. Returns ``(path, count)``."""
    settings.ensure_dirs()
    src = src or (settings.raw_dir / "raw_corpus.jsonl")
    dest = dest or (settings.interim_dir / "normalized.jsonl")
    count = 0
    with src.open(encoding="utf-8") as handle, dest.open("w", encoding="utf-8") as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (record.get("text") or "").strip():
                continue
            out.write(json.dumps(normalize_record(record), ensure_ascii=False) + "\n")
            count += 1
    return dest, count
