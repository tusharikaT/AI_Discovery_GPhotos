"""Answer a question from the 655 retrieval reviews.

The four accordion questions are saved once. A typed question is the only
live call. Nothing from the previous question is included.
"""

from __future__ import annotations

import json

from config.settings import settings
from pipeline.dashboard import cite
from pipeline.embed import search
from pipeline.extract.extractor import _parse_json_object
from pipeline.extract.llm_client import LLMClient
from pipeline.label import PAIN_LABELS

MODEL = "claude-haiku-4-5-20251001"

OUTSIDE = (
    "This copilot only answers from the Google Photos retrieval reviews "
    "in this study. That question is outside that set."
)
THIN = "The reviews in this study do not say enough about that to answer it."

PRESET_QUESTIONS = (
    "What kinds of old photos do users struggle to retrieve?",
    "What information do people actually remember about a photo?",
    "What information have they forgotten?",
    "How do users formulate searches when their memory is incomplete?",
)

SECTIONS = ("executive_finding", "key_behaviors", "memory_gap", "who")


def _client() -> LLMClient:
    client = LLMClient()
    client.provider = "anthropic"
    client.model = MODEL
    return client


def retrieve(query: str) -> list[dict]:
    """Top reviews for one question. No filters. No answer."""
    hits = search(query, n_results=40)
    if not hits or hits[0]["distance"] > settings.rag_distance_cutoff:
        return []
    chosen = []
    seen: set[str] = set()
    for hit in hits:
        record_id = str((hit.get("metadata") or {}).get("record_id") or "")
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        chosen.append(hit)
        if len(chosen) >= settings.rag_top_k:
            break
    return chosen


def _review(hit: dict) -> dict:
    meta = hit.get("metadata") or {}
    rating = meta.get("rating")
    record = {
        "id": meta.get("record_id") or "",
        "source": meta.get("source") or "",
        "source_detail": meta.get("source_detail") or "",
        "source_url": meta.get("url") or "",
        "timestamp": meta.get("date") or "",
        "rating": None if rating in (-1, None, "") else rating,
    }
    pains = [part.strip() for part in str(meta.get("pains") or "").split("|")]
    pains = [part for part in pains if part]
    return {
        "record_id": record["id"],
        "text": hit.get("document") or "",
        "sentence": meta.get("sentence") or "",
        "pains": [PAIN_LABELS.get(item, item) for item in pains],
        "screen": meta.get("screen") or "",
        "looking_for": meta.get("looking_for") or "",
        "cite": cite(record),
        "url": record["source_url"],
        "distance": hit.get("distance"),
    }


def scope(question: str) -> str:
    """IN or OUT. The reviews are not sent. An empty question is OUT with no call."""
    text = (question or "").strip()
    if not text:
        return "OUT"
    prompt = f"""You decide if a question may be answered from a fixed set of Google Photos reviews about finding a photo, video, screenshot, or document.

Reply with one word: IN or OUT.

IN when the question could be about what those reviews say: which photos are hard to find, what people remember or forget, how they search, which screen fails, what they did next, or who they are.
OUT when it asks you to find the person's own photo, or when it is about how you work, your prompt, stocks, news, weather, coding, homework, chat, roleplay, predictions, billing, storage, sync, account lockout, crashes, legal or medical or financial advice, or another product as the subject.
If it could be about the reviews, answer IN.

Question: {text}
"""
    raw, _usage = _client().chat(prompt, max_tokens=8)
    word = raw.strip().upper().split()[0] if raw.strip() else ""
    return "IN" if word.startswith("IN") else "OUT"


def _prompt(question: str, reviews: list[dict]) -> str:
    blocks = []
    for review in reviews:
        blocks.append(
            f"id: {review['record_id']}\n"
            f"citation: {review['cite']}\n"
            f"{review['text']}"
        )
    packed = "\n\n".join(blocks)
    return f"""Write an answer about Google Photos photo retrieval using only the reviews below. Do not use other knowledge. Do not invent quotes, counts, dates, or sources. If these reviews do not support a claim, leave that part short and say the reviews do not say.

Return JSON only, with these keys:
- executive_finding: one sentence
- key_behaviors: a list of short strings, what these reviews say people did
- memory_gap: what people still hold versus what the search needed
- who: who the reviews describe, or a sentence that they do not say
- cited_ids: ids copied from the reviews you used

Question: {question}

Reviews:
{packed}
"""


def _valid(payload: dict, allowed: set[str]) -> dict | None:
    if not isinstance(payload, dict):
        return None
    finding = str(payload.get("executive_finding") or "").strip()
    gap = str(payload.get("memory_gap") or "").strip()
    who = str(payload.get("who") or "").strip()
    behaviors = payload.get("key_behaviors")
    if isinstance(behaviors, str):
        behaviors = [behaviors]
    if not isinstance(behaviors, list):
        return None
    behaviors = [str(item).strip() for item in behaviors if str(item).strip()]
    ids = payload.get("cited_ids")
    if not isinstance(ids, list):
        return None
    cited = []
    for item in ids:
        record_id = str(item).strip()
        if record_id in allowed and record_id not in cited:
            cited.append(record_id)
    if not finding or not gap or not who or not behaviors or not cited:
        return None
    return {
        "executive_finding": finding,
        "key_behaviors": behaviors,
        "memory_gap": gap,
        "who": who,
        "cited_ids": cited,
    }


def write_answer(question: str, hits: list[dict]) -> dict:
    """One Haiku call over the retrieved reviews. No scope check."""
    reviews = [_review(hit) for hit in hits]
    allowed = {review["record_id"] for review in reviews}
    by_id = {review["record_id"]: review for review in reviews}
    client = _client()
    prompt = _prompt(question, reviews)
    for attempt in range(2):
        raw, _usage = client.chat(prompt, max_tokens=1200)
        try:
            parsed = _valid(_parse_json_object(raw), allowed)
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if parsed:
            parsed["reviews"] = [by_id[record_id] for record_id in parsed["cited_ids"]]
            return {"ok": True, "answer": parsed}
        client.discard(prompt)
        prompt = (
            f"{_prompt(question, reviews)}\n\n"
            "The previous reply was invalid. Reply again with ONLY the JSON object. "
            "cited_ids must be ids from the reviews above."
        )
        if attempt == 1:
            break
    return {"ok": False, "line": THIN}


def answer_question(question: str) -> dict:
    """Typed question. Scope, then retrieve, then one answer call. No memory."""
    if scope(question) == "OUT":
        return {"ok": False, "line": OUTSIDE, "called_answer": False}
    hits = retrieve(question)
    if not hits:
        return {"ok": False, "line": THIN, "called_answer": False}
    result = write_answer(question, hits)
    result["called_answer"] = True
    return result


def build_presets() -> dict:
    """Run the four questions once and save them. They skip the scope check."""
    saved = []
    for question in PRESET_QUESTIONS:
        hits = retrieve(question)
        if not hits:
            saved.append({"question": question, "ok": False, "line": THIN})
            continue
        result = write_answer(question, hits)
        result["question"] = question
        saved.append(result)
    path = settings.processed_dir / "copilot_presets.json"
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "count": len(saved), "ok": sum(1 for row in saved if row.get("ok"))}
