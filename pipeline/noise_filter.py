"""Strict retrieval filter (Phase 2).

Cheap rules drop obvious non-retrieval text. Borderline items get a 0–1
relevance score from the configured LLM. Records at or above
``settings.relevance_score_threshold`` are kept.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.settings import settings

_STRONG_RETRIEVAL_RE = re.compile(
    r"("
    r"can'?t find|cannot find|couldn'?t find|unable to find|"
    r"search(?:ed|ing)? for|no results|zero results|"
    r"scroll(?:ing|ed)? through|ask photos|"
    r"looking for (?:the |my |a )?(?:photo|picture|pic|image|screenshot)|"
    r"trying to (?:find|locate)|"
    r"where (?:is|are) (?:the |my )?(?:photo|picture|pic)"
    r")",
    re.IGNORECASE,
)
_NON_RETRIEVAL_RE = re.compile(
    r"("
    r"\b(?:magic eraser|unblur|collage|printing|video export)\b|"
    r"\b(?:storage|quota|subscription|google one)\b|"
    r"\b(?:backup|sync|syncing|upload)\b|"
    r"\b(?:crash|crashes|freezes|lag|battery)\b|"
    r"\b(?:edit|editing)\b"
    r")",
    re.IGNORECASE,
)
_PROMO_RE = re.compile(
    r"(promo code|coupon|buy now|click here|download now|whatsapp plus|"
    r"our (?:tool|software|app) can recover|data recovery)",
    re.IGNORECASE,
)
_COMPETITOR_RE = re.compile(
    r"\b(apple photos|icloud photos|samsung gallery|amazon photos)\b",
    re.IGNORECASE,
)
_GOOGLE_PHOTOS_RE = re.compile(r"google photos", re.IGNORECASE)
_LETTER_RE = re.compile(r"[A-Za-z]")
_NON_ASCII_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_LLM_BATCH = 6


def _english_ratio(text: str) -> float:
    letters = _NON_ASCII_LETTER_RE.findall(text)
    if not letters:
        return 0.0
    ascii_letters = _LETTER_RE.findall(text)
    return len(ascii_letters) / len(letters)


def rule_decision(record: dict) -> tuple[str, str, float | None]:
    """Return ``(drop|keep|llm, reason, score_or_none)``."""
    text = record.get("text") or ""
    words = text.split()
    lang = record.get("lang")
    lang_conf = float(record.get("lang_confidence") or 0)

    if lang and lang != "en" and lang_conf >= 0.85 and len(words) >= 8:
        return "drop", "non_english", None
    if len(text) > 40 and _english_ratio(text) < 0.55:
        return "drop", "non_english", None

    if _PROMO_RE.search(text):
        return "drop", "promo_or_bot", None

    if (
        _COMPETITOR_RE.search(text)
        and not _GOOGLE_PHOTOS_RE.search(text)
        and not settings.keep_competitor_items
    ):
        return "drop", "competitor_only", None

    strong = bool(_STRONG_RETRIEVAL_RE.search(text))
    other = bool(_NON_RETRIEVAL_RE.search(text))

    if strong and not other:
        return "keep", "rule_retrieval", 0.9
    if other and not strong:
        return "drop", "photos_not_retrieval", None
    if not strong and len(words) < 18:
        return "drop", "low_information", None
    return "llm", "borderline", None


def _llm_request(items: list[dict]) -> list[dict]:
    """Score a batch. Each item is ``{id, text}``. Returns ``[{id, score}]``."""
    payload_items = [
        {"id": item["id"], "text": (item["text"] or "")[:1200]} for item in items
    ]
    prompt = (
        "Score whether each text is about someone failing to retrieve a photo "
        "they believe exists (vague memory, bad search, endless scrolling, "
        "Ask Photos failure, can't find a screenshot/receipt/face).\n"
        "0 = not retrieval (praise, storage, billing, backup/sync, editing, "
        "crashes, printing, sharing how-to).\n"
        "1 = clear retrieval failure.\n"
        "Reply with ONLY a JSON array of "
        '{"id": "...", "score": 0.0} objects, one per input, scores between 0 and 1.\n\n'
        + json.dumps(payload_items, ensure_ascii=False)
    )
    body = json.dumps(
        {
            "model": settings.llm_model or "claude-haiku-4-5-20251001",
            "max_tokens": 400,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        method="POST",
    )
    req.add_header("x-api-key", settings.anthropic_api_key)
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in model response")
    parsed = json.loads(text[start : end + 1])
    scores = []
    for row in parsed:
        score = float(row["score"])
        scores.append({"id": str(row["id"]), "score": max(0.0, min(1.0, score))})
    return scores


def _score_batch(items: list[dict]) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            return _llm_request(items)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
            last_error = exc
            time.sleep(settings.request_backoff_base_s * (attempt + 1))
    raise RuntimeError(f"LLM batch failed: {last_error}")


def _load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row and "score" in row:
                scores[row["id"]] = float(row["score"])
    return scores


def _append_scores(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def filter_corpus(
    src: Path | None = None,
    dest: Path | None = None,
    report_path: Path | None = None,
) -> tuple[Path, dict]:
    """Filter normalized records. Returns ``(candidates_path, report)``."""
    settings.ensure_dirs()
    src = src or (settings.interim_dir / "normalized.jsonl")
    dest = dest or (settings.interim_dir / "retrieval_candidates.jsonl")
    report_path = report_path or (settings.interim_dir / "filter_report.json")
    score_path = settings.interim_dir / "llm_scores.jsonl"

    records = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    dropped: dict[str, int] = {}
    dropped_by_source: dict[str, dict[str, int]] = {}
    llm_queue: list[dict] = []
    kept: list[dict] = []

    def _drop(record: dict, reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1
        bucket = dropped_by_source.setdefault(record.get("source") or "unknown", {})
        bucket[reason] = bucket.get(reason, 0) + 1

    for record in records:
        decision, reason, score = rule_decision(record)
        if decision == "drop":
            _drop(record, reason)
        elif decision == "keep":
            row = dict(record)
            row["is_retrieval_related"] = True
            row["relevance_score"] = score
            row["filter_reason"] = reason
            kept.append(row)
        else:
            llm_queue.append(record)

    known_scores = _load_scores(score_path)
    pending = [row for row in llm_queue if row["id"] not in known_scores]
    print(
        f"  rules kept={len(kept)} dropped={sum(dropped.values())} "
        f"llm_needed={len(llm_queue)} llm_remaining={len(pending)}",
        flush=True,
    )

    if pending:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for borderline scoring")
        batches = [
            pending[i : i + _LLM_BATCH] for i in range(0, len(pending), _LLM_BATCH)
        ]
        workers = max(1, settings.llm_concurrency)
        done = 0
        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    scored = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [llm] batch failed ({exc}); retrying one by one", flush=True)
                    scored = []
                    for item in batch:
                        try:
                            scored.extend(_score_batch([item]))
                        except Exception as inner:  # noqa: BLE001
                            print(f"  [llm] drop {item['id']}: {inner}", flush=True)
                            scored.append({"id": item["id"], "score": 0.0})
                with write_lock:
                    _append_scores(score_path, scored)
                for row in scored:
                    known_scores[row["id"]] = row["score"]
                done += len(batch)
                if done % 120 == 0 or done == len(pending):
                    print(f"  ... llm scored {done}/{len(pending)}", flush=True)

    threshold = settings.relevance_score_threshold
    for record in llm_queue:
        score = known_scores.get(record["id"])
        if score is None:
            _drop(record, "llm_missing")
            continue
        if score >= threshold:
            row = dict(record)
            row["is_retrieval_related"] = True
            row["relevance_score"] = round(score, 3)
            row["filter_reason"] = "llm_retrieval"
            kept.append(row)
        else:
            _drop(record, "below_relevance_threshold")

    with dest.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": len(records),
        "kept_before_dedup": len(kept),
        "dropped_by_reason": dropped,
        "dropped_by_source": dropped_by_source,
        "threshold": threshold,
        "sample_kept": [
            {
                "source": row.get("source"),
                "score": row.get("relevance_score"),
                "reason": row.get("filter_reason"),
                "text": (row.get("text") or "")[:240],
            }
            for row in kept[:30]
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest, report
