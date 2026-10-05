"""Label each kept review with the pains the page will show.

The question is the one approved in docs/discovery_dashboard_plan.md §4.1.
A review may carry more than one pain. The reply is saved, so a repeat run
does not send the review again.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.settings import settings
from pipeline.extract.extractor import _load_jsonl, _parse_json_object
from pipeline.extract.llm_client import LLMClient, LLMError

PAINS = (
    "search_returns_nothing",
    "wrong_photo",
    "faces",
    "document",
    "ask_followup",
    "library_flood",
    "vague_memory",
    "burst",
    "memories_albums",
    "search_worked",
    "does_not_fit",
)
SCREENS = (
    "Search",
    "Ask Photos",
    "Memories",
    "Albums",
    "Timeline",
    "Faces",
    "Lens",
    "not_named",
)
LOOKING = (
    "a person",
    "a place or trip",
    "a document or receipt",
    "a specific object",
    "a screenshot",
    "an old event",
    "not_said",
)
REMEMBERS = ("a rough time", "how it looked", "who was there", "how it felt")
MISSING = (
    "the exact date",
    "the place name",
    "the person's name",
    "the object's name",
    "the album name",
)
WHO = (
    "parent",
    "traveler",
    "large_library",
    "keeps_documents",
    "older_library",
    "student",
    "not_said",
)
FEELING = (
    "frustrated",
    "anxious",
    "worn_down",
    "no_longer_trusts_search",
    "grieving",
    "gave_up",
    "relieved",
    "not_clear",
)
DID_NEXT = (
    "scrolled",
    "asked_someone",
    "rebuilt_albums",
    "said_they_gave_up",
    "not_described",
)

PAIN_LABELS = {
    "search_returns_nothing": "Search returns nothing",
    "wrong_photo": "Search returns the wrong photo",
    "faces": "Faces are missing or the wrong person",
    "document": "A receipt, label, or serial number cannot be found",
    "ask_followup": "Ask Photos forgets the follow-up",
    "library_flood": "The library is so large that any result is still a flood",
    "vague_memory": "They remember a time, a color, or a feeling, and search only accepts a noun or a date",
    "burst": "Near-identical photos, and they cannot pick the right one",
    "memories_albums": "Memories or albums show the wrong moment",
    "search_worked": "Search actually worked",
    "does_not_fit": "Not about finding a photo",
}

PROMPT = """You are labeling one Google Photos review for a product manager. Use only the allowed values. If the review does not say something, leave it empty or use the not-said value. Do not guess. Do not invent a label that is not on the list.

Pains. Include every one the review actually describes. Allowed:
search_returns_nothing, wrong_photo, faces, document, ask_followup, library_flood, vague_memory, burst, memories_albums, search_worked.
If the review is about price, storage, backup, sync, crashes, or editing, and not about finding a photo, pains is ["does_not_fit"].

Screen. One of: Search, Ask Photos, Memories, Albums, Timeline, Faces, Lens, not_named. "Search doesn't work" is Search. If they never name a screen, use not_named.

Looking for. One of: a person, a place or trip, a document or receipt, a specific object, a screenshot, an old event, not_said.

Still remembers. Any of: a rough time, how it looked, who was there, how it felt. Empty if they do not say.

Missing. Any of: the exact date, the place name, the person's name, the object's name, the album name. Empty if they do not say. Do not assume they forgot the date.

Who. One of: parent, traveler, large_library, keeps_documents, older_library, student, not_said. Only if the text says so. "Thousands of photos" is large_library.

Feeling. One of: frustrated, anxious, worn_down, no_longer_trusts_search, grieving, gave_up, relieved, not_clear. Anxious means they needed the photo for something that could not wait. no_longer_trusts_search means they say it used to work.

Did next. One of: scrolled, asked_someone, rebuilt_albums, said_they_gave_up, not_described. Only if they describe that action. Failing to find the photo is not itself gave up.

Sentence. Copy one span from the review, at most 220 characters, that shows why you chose the pains. The span must appear in the review.

Example. Review: "Search returns nothing for last summer. I scrolled for an hour."
{{"pains":["search_returns_nothing","library_flood"],"screen":"Search","looking_for":"an old event","still_remembers":["a rough time"],"missing":[],"who":"not_said","feeling":"frustrated","did_next":"scrolled","sentence":"Search returns nothing for last summer."}}

Example. Review: "Too expensive and the backup is stuck."
{{"pains":["does_not_fit"],"screen":"not_named","looking_for":"not_said","still_remembers":[],"missing":[],"who":"not_said","feeling":"not_clear","did_next":"not_described","sentence":"Too expensive and the backup is stuck."}}

Reply in JSON only, with the keys pains, screen, looking_for, still_remembers, missing, who, feeling, did_next, sentence.

Source: {source}. Stars: {stars}.
Review:
{text}
"""


def _as_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item]


def _one(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    if text in allowed:
        return text
    lowered = text.lower().replace(" ", "_")
    for item in allowed:
        if item.lower().replace(" ", "_") == lowered:
            return item
    return default


def _sentence(value: str, source: str) -> str:
    cleaned = " ".join((value or "").split())
    source_flat = " ".join(source.split())
    if cleaned and cleaned in source_flat:
        return cleaned[:220]
    return source_flat[:220]


def validate_label(payload: dict, review_text: str) -> dict:
    pains = [item for item in _as_list(payload.get("pains")) if item in PAINS]
    if "does_not_fit" in pains:
        pains = ["does_not_fit"]
    if not pains:
        pains = ["does_not_fit"]
    return {
        "pains": pains,
        "screen": _one(payload.get("screen"), SCREENS, "not_named"),
        "looking_for": _one(payload.get("looking_for"), LOOKING, "not_said"),
        "still_remembers": [item for item in _as_list(payload.get("still_remembers")) if item in REMEMBERS],
        "missing": [item for item in _as_list(payload.get("missing")) if item in MISSING],
        "who": _one(payload.get("who"), WHO, "not_said"),
        "feeling": _one(payload.get("feeling"), FEELING, "not_clear"),
        "did_next": _one(payload.get("did_next"), DID_NEXT, "not_described"),
        "sentence": _sentence(str(payload.get("sentence") or ""), review_text),
    }


def _prompt_for(record: dict) -> str:
    rating = record.get("rating")
    stars = "none" if rating is None else str(rating)
    return PROMPT.format(
        source=record.get("source") or "unknown",
        stars=stars,
        text=record.get("text") or "",
    )


def _label_one(client: LLMClient, record: dict) -> tuple[dict | None, dict, str | None]:
    prompt = _prompt_for(record)
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_hit": False}
    last_error = "unknown"
    review = record.get("text") or ""
    for _attempt in range(2):
        try:
            raw, usage = client.chat(prompt, max_tokens=500)
        except (LLMError, TimeoutError, ValueError) as exc:
            return None, usage_total, str(exc)
        usage_total["input_tokens"] += int(usage.get("input_tokens") or 0)
        usage_total["output_tokens"] += int(usage.get("output_tokens") or 0)
        usage_total["cache_hit"] = bool(usage.get("cache_hit"))
        try:
            label = validate_label(_parse_json_object(raw), review)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)[:400]
            client.discard(prompt)
            prompt = (
                f"{prompt}\n\nThe previous reply was invalid ({last_error}). "
                "Reply again with ONLY the JSON object."
            )
            continue
        saved = {
            "id": record.get("id"),
            "source": record.get("source"),
            "source_url": record.get("source_url"),
            "source_detail": record.get("source_detail"),
            "timestamp": record.get("timestamp"),
            "rating": record.get("rating"),
            "text": review,
            "extraction": record.get("extraction") or {},
            "label": label,
        }
        return saved, usage_total, None
    return None, usage_total, last_error


def _client() -> LLMClient:
    client = LLMClient()
    client.provider = "anthropic"
    if not client.model or "claude" not in client.model:
        client.model = "claude-haiku-4-5-20251001"
    return client


def label_corpus(limit: int | None = None) -> dict:
    settings.ensure_dirs()
    src = settings.processed_dir / "classified.jsonl"
    dest = settings.processed_dir / "labeled.jsonl"
    quarantine_path = settings.processed_dir / "label_quarantine.jsonl"
    records = _load_jsonl(src)
    done = {row["id"] for row in _load_jsonl(dest) if row.get("id")}
    pending = [row for row in records if row.get("id") not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"label {len(pending)} pending / {len(records)} total", flush=True)
    client = _client()
    kept = quarantined = cache_hits = input_tokens = output_tokens = 0
    workers = max(1, settings.llm_concurrency)
    with dest.open("a", encoding="utf-8") as out, quarantine_path.open("a", encoding="utf-8") as bad:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_label_one, client, row): row for row in pending}
            finished = 0
            for future in as_completed(futures):
                record = futures[future]
                saved, usage, error = future.result()
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                cache_hits += int(bool(usage.get("cache_hit")))
                if saved is None:
                    quarantined += 1
                    bad.write(json.dumps({
                        "id": record.get("id"),
                        "error": error,
                    }, ensure_ascii=False) + "\n")
                    bad.flush()
                else:
                    kept += 1
                    out.write(json.dumps(saved, ensure_ascii=False) + "\n")
                    out.flush()
                finished += 1
                if finished % 25 == 0 or finished == len(pending):
                    print(
                        f"  ... labeled {finished}/{len(pending)} kept={kept} quarantined={quarantined}",
                        flush=True,
                    )
    report = {
        "kept": kept,
        "quarantined": quarantined,
        "cache_hits": cache_hits,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": client.model,
    }
    print(
        f"label kept {kept}, quarantined {quarantined}, "
        f"tokens in={input_tokens} out={output_tokens}",
        flush=True,
    )
    return report


def feelings_by_id(labeled_path: Path | None = None) -> dict[str, str]:
    """Feelings already stored on labeled.jsonl, keyed by review id."""
    path = labeled_path or (settings.processed_dir / "labeled.jsonl")
    feelings: dict[str, str] = {}
    for row in _load_jsonl(path):
        review_id = row.get("id")
        if not review_id:
            continue
        feelings[review_id] = _one((row.get("label") or {}).get("feeling"), FEELING, "not_clear")
    return feelings


def attach_feelings(paths: list[Path] | None = None, labeled_path: Path | None = None) -> dict:
    """Copy label.feeling onto each extracted row by id. No model call."""
    feelings = feelings_by_id(labeled_path)
    targets = paths or [
        settings.processed_dir / "extracted_records.jsonl",
        settings.processed_dir / "classified.jsonl",
    ]
    attached = missing = 0
    counts: dict[str, int] = {}
    for path in targets:
        records = _load_jsonl(path)
        for row in records:
            feeling = feelings.get(row.get("id") or "")
            if feeling is None:
                missing += 1
                feeling = "not_clear"
            row["feeling"] = feeling
            attached += 1
            counts[feeling] = counts.get(feeling, 0) + 1
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    # Counts are per file. Report one file's distribution.
    per_file = attached // len(targets) if targets else 0
    report = {
        "files": len(targets),
        "rows_per_file": per_file,
        "missing_ids": missing // len(targets) if targets else 0,
        "feelings": {name: counts.get(name, 0) // len(targets) if targets else 0 for name in FEELING},
    }
    print(
        f"attached feeling on {per_file} rows x {len(targets)} files, "
        f"unmatched {report['missing_ids']}",
        flush=True,
    )
    return report
