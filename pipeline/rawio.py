"""Raw-store I/O: JSONL append, checkpoint/resume, rejects log, corpus merge."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings
from pipeline.prefilter import text_hash
from pipeline.sources.base import RawItem

MERGED_NAME = "raw_corpus.jsonl"
REJECTS_NAME = "rejects.json"


def source_path(source: str) -> Path:
    return settings.raw_dir / f"{source}.jsonl"


def load_seen(source: str) -> tuple[set[str], set[str]]:
    """Load already-written ids + normalized-text hashes for resume/dedup."""
    ids: set[str] = set()
    hashes: set[str] = set()
    path = source_path(source)
    if not path.exists():
        return ids, hashes
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                ids.add(rec["id"])
            if rec.get("text"):
                hashes.add(text_hash(rec["text"]))
    return ids, hashes


def append_item(source: str, item: RawItem) -> None:
    settings.ensure_dirs()
    with source_path(source).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def write_rejects(rejects: dict[str, dict[str, int]]) -> Path:
    """Merge this run's drop counts into the rejects log: {source: {reason: count}}."""
    settings.ensure_dirs()
    path = settings.raw_dir / REJECTS_NAME
    existing: dict[str, dict[str, int]] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    for source, reasons in rejects.items():
        bucket = existing.setdefault(source, {})
        for reason, count in reasons.items():
            bucket[reason] = int(bucket.get(reason, 0)) + int(count)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def merge_corpus() -> tuple[Path, int]:
    """Merge all per-source JSONL into one deduplicated raw_corpus.jsonl."""
    settings.ensure_dirs()
    out = settings.raw_dir / MERGED_NAME
    seen: set[str] = set()
    count = 0
    with out.open("w", encoding="utf-8") as w:
        for path in sorted(settings.raw_dir.glob("*.jsonl")):
            if path.name == MERGED_NAME:
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = rec.get("id")
                    if rid in seen:
                        continue
                    seen.add(rid)
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
    return out, count
