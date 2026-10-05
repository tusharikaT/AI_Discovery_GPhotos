"""Count the labels into the file the four pages read. No model call."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from config.settings import settings
from pipeline.extract.extractor import _load_jsonl
from pipeline.derived import (
    SEGMENT_IMPLICATION,
    friction_band,
    is_retrieval_problem,
    memory_phrase,
    missing_detail,
    reads_frustrated,
    review_friction,
    segment_name,
    star_sentiment,
)
from pipeline.label import PAIN_LABELS
from pipeline.metrics import (
    frequency_score,
    friction_score,
    impact_score,
    opportunity_band,
    stakes_score,
)

SOURCE_LABELS = {
    "playstore": "Play Store",
    "appstore": "App Store",
    "reddit": "Reddit",
    "web": "Web",
    "youtube": "YouTube",
}
WHO_LABELS = {
    "parent": "Parents",
    "traveler": "Travelers",
    "large_library": "Large libraries",
    "keeps_documents": "People who keep documents",
    "older_library": "Older libraries",
    "student": "Students",
    "not_said": "Not said",
}
FEELING_LABELS = {
    "frustrated": "Frustrated",
    "anxious": "Anxious",
    "worn_down": "Worn down",
    "no_longer_trusts_search": "No longer trusts search",
    "grieving": "Grieving a photo",
    "gave_up": "Gave up",
    "relieved": "Relieved",
    "not_clear": "Not clear",
}
NEXT_LABELS = {
    "scrolled": "Scrolled",
    "asked_someone": "Asked someone else",
    "rebuilt_albums": "Rebuilt albums",
    "said_they_gave_up": "Said they gave up",
    "not_described": "Not described",
}
_LOCALE = {
    "us": "North America",
    "ca": "North America",
    "gb": "UK and Ireland",
    "uk": "UK and Ireland",
    "au": "Australia and New Zealand",
    "nz": "Australia and New Zealand",
    "in": "South Asia",
}


def region_for(record: dict) -> str:
    source = (record.get("source") or "").lower()
    if source == "reddit":
        return "Reddit, no country"
    if source == "youtube":
        return "YouTube"
    if source == "web":
        return "Web"
    detail = (record.get("source_detail") or "").lower()
    locale = detail.split("/")[-1] if "/" in detail else ""
    return _LOCALE.get(locale, locale.upper() or "Unknown")


def cite(record: dict) -> str:
    parts = [SOURCE_LABELS.get(record.get("source") or "", record.get("source") or "Source")]
    parts.append(region_for(record))
    stamp = (record.get("timestamp") or "")[:10]
    if stamp:
        parts.append(stamp)
    rating = record.get("rating")
    if rating is not None:
        try:
            number = float(rating)
            shown = int(number) if number == int(number) else round(number, 1)
            parts.append(f"{shown} stars")
        except (TypeError, ValueError):
            pass
    return " · ".join(parts)


def _year(stamp: str | None) -> str | None:
    if stamp and len(stamp) >= 4 and stamp[:4].isdigit():
        return stamp[:4]
    return None


def _described_query(record: dict) -> bool:
    formulation = (record.get("extraction") or {}).get("search_formulation") or {}
    return any(formulation.get(key) for key in (
        "attempt_1_natural", "attempt_2_keywords", "attempt_3_desperation",
    ))


def _best(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    def score(row: dict) -> tuple:
        text = row.get("text") or ""
        try:
            rating = float(row["rating"]) if row.get("rating") is not None else 5
        except (TypeError, ValueError):
            rating = 5
        return (len(text) < 80, rating, -len(text))

    return sorted(rows, key=score)[0]


def _pain_row(pain: str, group: list[dict], total: int) -> dict:
    extractions = [row.get("extraction") or {} for row in group]
    stakes = [stakes_score(item.get("stakes_level")) for item in extractions] or [5]
    frictions = [friction_score(item) for item in extractions] or [3]
    failed = sum(1 for item in extractions if item.get("outcome") == "abandoned")
    rate = failed / len(group)
    freq = frequency_score(len(group), total)
    impact = impact_score(sum(stakes) / len(stakes), rate, sum(frictions) / len(frictions))
    chosen = _best(group)
    feelings = Counter((row.get("label") or {}).get("feeling") or "not_clear" for row in group)
    nxt = Counter((row.get("label") or {}).get("did_next") or "not_described" for row in group)
    regions = Counter(region_for(row) for row in group)
    who = Counter((row.get("label") or {}).get("who") or "not_said" for row in group)
    specific = [(name, count) for name, count in who.most_common() if name != "not_said"]
    return {
        "id": pain,
        "label": PAIN_LABELS.get(pain, pain),
        "count": len(group),
        "share": round(len(group) / total, 4) if total else 0,
        "frequency": freq,
        "impact": impact,
        "opportunity": round(freq * impact, 2),
        "severity": round(impact * 10, 1),
        "quadrant": opportunity_band(freq, impact),
        "failed_share": round(rate, 4),
        "top_region": regions.most_common(1)[0][0],
        "feelings": feelings.most_common(),
        "did_next": nxt.most_common(1)[0][0],
        "who": specific[0][0] if specific else "not_said",
        "sentence": (chosen.get("label") or {}).get("sentence") if chosen else "",
        "cite": cite(chosen) if chosen else "",
        "url": chosen.get("source_url") if chosen else None,
    }


def build_dashboard(records: list[dict]) -> dict:
    total = len(records)
    by_pain: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        pains = (row.get("label") or {}).get("pains") or ["does_not_fit"]
        for pain in pains:
            by_pain[pain].append(row)
    pains = [_pain_row(pain, group, total) for pain, group in by_pain.items()]
    by_opportunity = sorted(pains, key=lambda row: row["opportunity"], reverse=True)
    by_count = sorted(pains, key=lambda row: row["count"], reverse=True)
    failures = [row for row in by_opportunity if row["id"] not in {"does_not_fit", "search_worked"}]

    sources: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        sources[row.get("source") or "unknown"].append(row)
    source_rows = []
    for source, group in sources.items():
        years = [year for year in (_year(row.get("timestamp")) for row in group) if year]
        peak = Counter(years).most_common(1)
        source_rows.append({
            "source": source,
            "label": SOURCE_LABELS.get(source, source),
            "count": len(group),
            "share": round(len(group) / total, 4) if total else 0,
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "peak_year": peak[0][0] if peak else None,
        })
    source_rows.sort(key=lambda row: row["count"], reverse=True)

    year_counts = Counter(year for year in (_year(row.get("timestamp")) for row in records) if year)
    rated = []
    for row in records:
        if row.get("rating") is None:
            continue
        try:
            rated.append(float(row["rating"]))
        except (TypeError, ValueError):
            continue
    histogram = [
        {"star": star, "count": sum(1 for value in rated if int(round(value)) == star)}
        for star in range(1, 6)
    ]

    screens = Counter()
    screen_hard = Counter()
    hard_feelings = {"frustrated", "anxious", "gave_up"}
    for row in records:
        screen = (row.get("label") or {}).get("screen") or "not_named"
        if screen == "not_named":
            continue
        screens[screen] += 1
        feeling = (row.get("label") or {}).get("feeling")
        failed = (row.get("extraction") or {}).get("outcome") == "abandoned"
        if feeling in hard_feelings or failed:
            screen_hard[screen] += 1
    screen_rows = []
    for screen, count in screens.most_common():
        named = [
            row for row in records
            if (row.get("label") or {}).get("screen") == screen
        ]
        frustrated = [row for row in named if reads_frustrated(row)]
        quoted = next(
            (row for row in frustrated if (row.get("label") or {}).get("sentence")),
            None,
        )
        if quoted is None:
            quoted = next(
                (row for row in named if (row.get("label") or {}).get("sentence")),
                None,
            )
        screen_rows.append({
            "screen": screen,
            "count": count,
            "hard_share": round(screen_hard[screen] / count, 4) if count else 0,
            "frustrated_share": round(len(frustrated) / count, 4) if count else 0,
            "sentence": (quoted.get("label") or {}).get("sentence") if quoted else "",
            "cite": cite(quoted) if quoted else "",
            "url": quoted.get("source_url") if quoted else None,
        })

    looking = Counter((row.get("label") or {}).get("looking_for") or "not_said" for row in records)
    who = Counter((row.get("label") or {}).get("who") or "not_said" for row in records)
    feelings = Counter((row.get("label") or {}).get("feeling") or "not_clear" for row in records)
    did_next = Counter((row.get("label") or {}).get("did_next") or "not_described" for row in records)
    remembers = Counter()
    missing = Counter()
    pairs = Counter()
    for row in records:
        cues = (row.get("label") or {}).get("still_remembers") or []
        for cue in cues:
            remembers[cue] += 1
        for index, left in enumerate(cues):
            for right in cues[index + 1 :]:
                pairs[tuple(sorted((left, right)))] += 1
        for cue in (row.get("label") or {}).get("missing") or []:
            missing[cue] += 1

    described = [row for row in records if _described_query(row)]
    steps = Counter()
    for row in described:
        formulation = (row.get("extraction") or {}).get("search_formulation") or {}
        chain = ["start"]
        if formulation.get("attempt_1_natural"):
            chain.append("a natural phrase")
        if formulation.get("attempt_2_keywords"):
            chain.append("keywords")
        if formulation.get("attempt_3_desperation"):
            chain.append("a short last try")
        chain.append((row.get("extraction") or {}).get("outcome") or "unknown")
        for left, right in zip(chain, chain[1:]):
            steps[(left, right)] += 1

    dated = sorted(
        [row for row in records if row.get("timestamp")],
        key=lambda row: row.get("timestamp") or "",
        reverse=True,
    )
    recent = [
        {
            "sentence": (row.get("label") or {}).get("sentence") or "",
            "cite": cite(row),
            "url": row.get("source_url"),
            "pains": [PAIN_LABELS.get(item, item) for item in (row.get("label") or {}).get("pains") or []],
        }
        for row in dated[:8]
    ]
    year_2026 = year_counts.get("2026", 0)
    unnamed_screen = sum(1 for row in records if (row.get("label") or {}).get("screen") == "not_named")

    sentiment = Counter(star_sentiment(row.get("rating")) for row in records)
    scores = [review_friction(row) for row in records]
    bands = Counter(friction_band(score) for score in scores)
    band_order = ("0-20", "20-40", "40-60", "60-80", "80-100")
    pain_scores: dict[str, list[int]] = defaultdict(list)
    for row, score in zip(records, scores):
        for pain in (row.get("label") or {}).get("pains") or []:
            if pain in {"does_not_fit", "search_worked"}:
                continue
            pain_scores[pain].append(score)
    missing_counts: Counter = Counter()
    phrase_counts: Counter = Counter()
    phrase_kinds: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        extraction = row.get("extraction") or {}
        seen_missing = {
            label
            for label in (missing_detail(value) for value in extraction.get("information_forgotten") or [])
            if label
        }
        missing_counts.update(seen_missing)
        anchors = extraction.get("memory_anchors_retained") or {}
        seen_phrases = set()
        for kind in ("temporal", "sensory", "emotional", "social"):
            for value in anchors.get(kind) or []:
                phrase = memory_phrase(str(value))
                if not phrase or phrase in seen_phrases:
                    continue
                seen_phrases.add(phrase)
                phrase_counts[phrase] += 1
                phrase_kinds[phrase][kind] += 1
    by_segment: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        name = segment_name(row)
        if name:
            by_segment[name].append(row)
    segment_rows = []
    for name, group in by_segment.items():
        problems = sum(1 for row in group if is_retrieval_problem(row))
        pain_counter = Counter(
            pain
            for row in group
            for pain in (row.get("label") or {}).get("pains") or []
            if pain not in {"does_not_fit", "search_worked"}
        )
        top = pain_counter.most_common(1)
        segment_rows.append({
            "label": name,
            "count": len(group),
            "problem_rate": round(problems / len(group), 4) if group else 0,
            "top_pain": PAIN_LABELS.get(top[0][0], top[0][0]) if top else "",
            "implication": SEGMENT_IMPLICATION.get(name, ""),
        })
    segment_rows.sort(key=lambda row: row["count"], reverse=True)

    return {
        "total": total,
        "pains_by_opportunity": by_opportunity,
        "pains_by_count": by_count,
        "failures_by_opportunity": failures,
        "top_pain": failures[0] if failures else None,
        "search_worked": next((row for row in pains if row["id"] == "search_worked"), None),
        "does_not_fit": next((row for row in pains if row["id"] == "does_not_fit"), None),
        "sources": source_rows,
        "years": [{"year": year, "count": count} for year, count in sorted(year_counts.items())],
        "ratings": {
            "average": round(sum(rated) / len(rated), 2) if rated else None,
            "rated_count": len(rated),
            "unrated_count": total - len(rated),
            "histogram": histogram,
        },
        "screens": screen_rows,
        "unnamed_screen": unnamed_screen,
        "looking_for": [
            {"label": name, "count": count}
            for name, count in looking.most_common()
            if name != "not_said"
        ],
        "who": [
            {"id": name, "label": WHO_LABELS.get(name, name), "count": count}
            for name, count in who.most_common()
            if name != "not_said"
        ],
        "who_not_said": who.get("not_said", 0),
        "feelings": [
            {"id": name, "label": FEELING_LABELS.get(name, name), "count": count}
            for name, count in feelings.most_common()
            if name != "not_clear"
        ],
        "feeling_not_clear": feelings.get("not_clear", 0),
        "did_next": [
            {"id": name, "label": NEXT_LABELS.get(name, name), "count": count}
            for name, count in did_next.most_common()
            if name != "not_described"
        ],
        "did_next_not_described": did_next.get("not_described", 0),
        "remembers": [{"label": name, "count": count} for name, count in remembers.most_common()],
        "missing": [{"label": name, "count": count} for name, count in missing.most_common()],
        "pairs": [
            {"label": " + ".join(pair), "count": count}
            for pair, count in pairs.most_common(6)
        ],
        "described_query": len(described),
        "query_steps": [
            {"source": left, "target": right, "value": count}
            for (left, right), count in steps.most_common()
        ],
        "recent": recent,
        "coverage": {
            "year_2026": year_2026,
            "year_2026_share": round(year_2026 / total, 4) if total else 0,
            "play_store": sum(1 for row in records if row.get("source") == "playstore"),
        },
        "raw_ingested": 17803,
        "sentiment": [
            {"label": name, "count": sentiment.get(name, 0)}
            for name in ("Frustrated", "Neutral", "Positive")
        ],
        "friction": {
            "average": round(sum(scores) / len(scores)) if scores else 0,
            "high_count": sum(1 for score in scores if score >= 60),
            "bands": [{"label": name, "count": bands.get(name, 0)} for name in band_order],
            "by_pain": [
                {
                    "label": PAIN_LABELS.get(pain, pain),
                    "average": round(sum(values) / len(values)) if values else 0,
                    "count": len(values),
                }
                for pain, values in sorted(pain_scores.items(), key=lambda item: -sum(item[1]) / len(item[1]))
            ],
        },
        "missing_details": [
            {"label": name, "count": count}
            for name, count in missing_counts.most_common()
        ],
        "named_phrases": [
            {
                "label": name,
                "kind": phrase_kinds[name].most_common(1)[0][0] if phrase_kinds.get(name) else "",
                "count": count,
            }
            for name, count in phrase_counts.most_common()
            if count >= 3
        ][:15],
        "segments": segment_rows,
        "segment_unplaced": total - sum(row["count"] for row in segment_rows),
    }


def suggest_builds(payload: dict) -> list[dict]:
    """One call. The lines stay off the page until you have seen them; the file keeps them as suggestions."""
    from pipeline.extract.llm_client import LLMClient

    rows = []
    for row in payload.get("failures_by_opportunity") or []:
        rows.append(
            f"{row['id']}: {row['label']} — {row['count']} reviews, severity {row['severity']}. "
            f"Quote: {row.get('sentence') or ''}"
        )
    prompt = (
        "Write one cause and one thing Google Photos should build for each pain. "
        "Plain sentences. Do not invent a count. Reply with a JSON list of objects "
        "with keys id, cause, build, why.\n\n" + "\n".join(rows)
    )
    client = LLMClient()
    client.provider = "anthropic"
    if not client.model or "claude" not in client.model:
        client.model = "claude-haiku-4-5-20251001"
    raw, _usage = client.chat(prompt, max_tokens=1200)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    return json.loads(raw[start : end + 1])


def write_dashboard() -> dict:
    records = _load_jsonl(settings.processed_dir / "labeled.jsonl")
    payload = build_dashboard(records)
    try:
        payload["builds"] = suggest_builds(payload)
    except Exception as exc:  # noqa: BLE001
        payload["builds"] = []
        payload["builds_error"] = str(exc)[:300]
    path = settings.processed_dir / "dashboard.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"dashboard {payload['total']} reviews -> {path}", flush=True)
    return payload
