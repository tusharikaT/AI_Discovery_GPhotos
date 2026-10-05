"""Exact and near-duplicate removal across sources (Phase 2)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from config.settings import settings
from pipeline.prefilter import text_hash

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUM_PERM = 64


def _minhash(text: str) -> MinHash:
    tokens = _WORD_RE.findall((text or "").lower())
    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))]
    sketch = MinHash(num_perm=_NUM_PERM)
    for shingle in shingles[:400]:
        sketch.update(shingle.encode("utf-8", "ignore"))
    if not shingles:
        sketch.update(b"")
    return sketch


def _prefer(left: dict, right: dict) -> dict:
    """Keep the higher-scoring, then longer, record."""
    left_score = float(left.get("relevance_score") or 0)
    right_score = float(right.get("relevance_score") or 0)
    if right_score > left_score:
        return right
    if right_score < left_score:
        return left
    if len(right.get("text") or "") > len(left.get("text") or ""):
        return right
    return left


def dedup_corpus(
    src: Path | None = None,
    dest: Path | None = None,
) -> tuple[Path, dict]:
    """Write the deduped corpus. Returns ``(path, stats)``."""
    settings.ensure_dirs()
    src = src or (settings.interim_dir / "retrieval_candidates.jsonl")
    dest = dest or (settings.interim_dir / "filtered_corpus.jsonl")

    records = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    exact_dropped = 0
    by_hash: dict[str, dict] = {}
    for record in records:
        digest = text_hash(record.get("text") or "")
        current = by_hash.get(digest)
        if current is None:
            by_hash[digest] = record
        else:
            exact_dropped += 1
            by_hash[digest] = _prefer(current, record)

    unique = list(by_hash.values())
    threshold = settings.dedup_similarity_threshold
    # MinHash LSH speaks Jaccard. A slightly lower cutoff catches paraphrases
    # while the configured cosine-style threshold stays the documented target.
    lsh_threshold = min(0.8, max(0.5, threshold - 0.1))
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=_NUM_PERM)
    sketches: dict[str, MinHash] = {}
    for record in unique:
        sketches[record["id"]] = _minhash(record.get("text") or "")

    near_dropped = 0
    survivors: dict[str, dict] = {}
    for record in unique:
        sketch = sketches[record["id"]]
        matches = lsh.query(sketch)
        if not matches:
            lsh.insert(record["id"], sketch)
            survivors[record["id"]] = record
            continue
        kept_id = matches[0]
        kept = survivors.get(kept_id)
        if kept is None:
            lsh.insert(record["id"], sketch)
            survivors[record["id"]] = record
            continue
        winner = _prefer(kept, record)
        if winner["id"] != kept_id:
            survivors.pop(kept_id, None)
            survivors[winner["id"]] = winner
        near_dropped += 1

    kept_rows = list(survivors.values())
    with dest.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input": len(records),
        "exact_dropped": exact_dropped,
        "near_dropped": near_dropped,
        "kept": len(kept_rows),
    }
    return dest, stats
