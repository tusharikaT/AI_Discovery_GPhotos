"""Opportunity scores and the aggregates the dashboard reads.

Impact = 0.4·Stakes + 0.3·Abandonment + 0.3·FrictionTier
Opportunity = Frequency × Impact

Vocabulary contrast is lexical. Embedding distance is Phase 5.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from config.settings import settings

FORMULA = "Impact = 0.4*Stakes + 0.3*Abandonment + 0.3*Friction; Opportunity = Frequency * Impact"

STAKES_SCORE = {
    "medical_financial_legal": 10.0,
    "sentimental": 7.0,
    "trivial": 3.0,
    "unknown": 5.0,
}

_STOP = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "my", "i",
    "it", "is", "was", "that", "this", "with", "from", "at", "be", "me",
}
_SYSTEM_TAGS = {
    "person", "face", "dog", "cat", "car", "beach", "food", "bottle", "table",
    "chair", "tree", "flower", "building", "screenshot", "document", "receipt",
}
_ATMOSPHERIC = {
    "cozy", "gloomy", "messy", "warm", "rainy", "vibe", "soft", "pastel",
    "funny", "candid", "blurry", "nostalgic", "happy", "sad",
}
_WORD = re.compile(r"[a-z0-9']+")
_MINUTES = re.compile(r"(\d+)\s*min", re.IGNORECASE)

SEGMENT_IMPLICATION = {
    "heavy_shooter": "Large libraries make flooding worse. Prioritize sub-filters after a face or object hit.",
    "traveler": "Trip memory is relative time plus place vibe, not a calendar date.",
    "parent": "Retrieval is often a specific child moment buried in near-identical bursts.",
    "student": "Notes, whiteboards, and slides are utility searches with weak filenames.",
    "document_keeper": "Receipts and labels fail when OCR or the exact noun is missing.",
    "older_user": "Timeline scrolling is the fallback. Search must accept rough era language.",
    "general": "Most people describe a life moment, then strip the query down to one noun.",
    "unknown": "Segment was not stated in the text.",
    "platform_ios": "iOS reviews show whether Ask Photos and the search bar fail the same way as Android.",
    "platform_android": "Android volume is the baseline for search-bar and timeline failure.",
}


def stakes_score(level: str | None) -> float:
    return STAKES_SCORE.get(level or "unknown", 5.0)


def friction_score(extraction: dict) -> float:
    """Map outcome, attempt count, and workaround text onto 1–10."""
    outcome = extraction.get("outcome") or "unknown"
    workaround = (extraction.get("user_workaround") or "").lower()
    attempts = int(extraction.get("num_search_attempts") or 0)
    if outcome in {"abandoned", "social_offload"} or attempts >= 3:
        return 9.0
    if any(word in workaround for word in ("scroll", "album", "manual", "timeline")):
        return 6.0
    return 3.0


def workaround_bucket(extraction: dict) -> str:
    text = (extraction.get("user_workaround") or "").lower()
    outcome = extraction.get("outcome") or "unknown"
    if outcome == "social_offload" or any(word in text for word in ("friend", "family", "asked", "whatsapp")):
        return "social_offload"
    if any(word in text for word in ("scroll", "timeline", "manual")):
        return "manual_scroll"
    if "album" in text or "organiz" in text:
        return "album_reorg"
    if outcome == "abandoned":
        return "permanent_loss"
    return "other"


def opportunity_band(freq: float, impact: float) -> str:
    """Name the bet from frequency and impact, not from the 60-point cutoff.

    Frequency is share-of-voice times 10, so the largest of six archetypes
    lands near 4–5. Multiplying by impact cannot reach 60. The names follow
    the brief's definitions instead: a common and painful problem is the
    strategic bet, a common but less severe one is the safety net.
    """
    if freq >= 3 and impact >= 6.5:
        return "Core Strategic Bet"
    if freq >= 2 and impact >= 5.5:
        return "Critical Safety Net"
    if freq >= 3 and impact < 5:
        return "Everyday Papercut"
    return "Low-Priority Nuance"


def impact_score(stakes: float, abandonment_rate: float, friction: float) -> float:
    """Abandonment rate is 0–1. Result is clamped to 1–10."""
    raw = 0.4 * stakes + 0.3 * (abandonment_rate * 10) + 0.3 * friction
    return round(max(1.0, min(10.0, raw)), 2)


def frequency_score(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total * 10, 2)


def _quarter(timestamp: str | None) -> str | None:
    if not timestamp or len(timestamp) < 7:
        return None
    try:
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
    except ValueError:
        return None
    return f"{year}-Q{(month - 1) // 3 + 1}"


def _tokens(text: str) -> list[str]:
    return [word for word in _WORD.findall((text or "").lower()) if word not in _STOP and len(word) > 2]


def build_aggregates(records: list[dict], raw_total: int) -> dict:
    total = len(records)
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_arch[record.get("archetype") or "episodic_travel"].append(record)

    archetype_rows = []
    for archetype, group in by_arch.items():
        extractions = [row.get("extraction") or {} for row in group]
        stakes = [stakes_score(item.get("stakes_level")) for item in extractions]
        frictions = [friction_score(item) for item in extractions]
        abandoned = sum(1 for item in extractions if item.get("outcome") == "abandoned")
        attempts = [int(item.get("num_search_attempts") or 0) for item in extractions]
        mean_stakes = sum(stakes) / len(stakes)
        mean_friction = sum(frictions) / len(frictions)
        abandonment_rate = abandoned / len(group)
        freq = frequency_score(len(group), total)
        impact = impact_score(mean_stakes, abandonment_rate, mean_friction)
        opportunity = round(freq * impact, 2)
        ranked = sorted(
            group,
            key=lambda row: (
                (row.get("extraction") or {}).get("outcome") == "abandoned",
                float((row.get("extraction") or {}).get("extraction_confidence") or 0),
            ),
            reverse=True,
        )
        quote = next(
            (
                (row.get("extraction") or {}).get("representative_quote")
                for row in ranked
                if (row.get("extraction") or {}).get("representative_quote")
            ),
            "",
        )
        archetype_rows.append({
            "archetype": archetype,
            "count": len(group),
            "freq_score": freq,
            "share": round(len(group) / total, 4) if total else 0,
            "impact_score": impact,
            "opportunity_score": opportunity,
            "band": opportunity_band(freq, impact),
            "abandonment_rate": round(abandonment_rate, 4),
            "avg_attempts": round(sum(attempts) / len(attempts), 2) if attempts else 0,
            "mean_stakes": round(mean_stakes, 2),
            "mean_friction": round(mean_friction, 2),
            "proof_quote": quote,
        })
    archetype_rows.sort(key=lambda row: row["opportunity_score"], reverse=True)

    abandoned_n = sum(
        1 for row in records if (row.get("extraction") or {}).get("outcome") == "abandoned"
    )
    attempt_values = [
        int((row.get("extraction") or {}).get("num_search_attempts") or 0) for row in records
    ]
    buckets = Counter(workaround_bucket(row.get("extraction") or {}) for row in records)
    scroll_minutes = []
    for row in records:
        workaround = (row.get("extraction") or {}).get("user_workaround") or ""
        match = _MINUTES.search(workaround)
        if match:
            scroll_minutes.append(int(match.group(1)))
    scroll_minutes.sort()
    median_minutes = scroll_minutes[len(scroll_minutes) // 2] if scroll_minutes else None

    retained = Counter()
    forgotten = Counter()
    kind_pairs = Counter()
    kinds = ("temporal", "sensory", "emotional", "social")
    for row in records:
        anchors = (row.get("extraction") or {}).get("memory_anchors_retained") or {}
        present = []
        for kind in kinds:
            values = [value for value in (anchors.get(kind) or []) if value]
            if values:
                present.append(kind)
                retained[kind] += len(values)
        for left_i, left in enumerate(present):
            for right in present[left_i + 1 :]:
                kind_pairs[f"{left}+{right}"] += 1
        for value in (row.get("extraction") or {}).get("information_forgotten") or []:
            if value:
                forgotten[str(value).strip().lower()] += 1

    sankey = Counter()
    for row in records:
        formulation = (row.get("extraction") or {}).get("search_formulation") or {}
        steps = []
        if formulation.get("attempt_1_natural"):
            steps.append("natural")
        if formulation.get("attempt_2_keywords"):
            steps.append("keywords")
        if formulation.get("attempt_3_desperation"):
            steps.append("desperation")
        outcome = (row.get("extraction") or {}).get("outcome") or "unknown"
        chain = ["start", *steps, outcome]
        for left, right in zip(chain, chain[1:]):
            sankey[(left, right)] += 1

    user_vocab = Counter()
    system_hits = Counter()
    atmospheric_hits = Counter()
    for row in records:
        formulation = (row.get("extraction") or {}).get("search_formulation") or {}
        blob = " ".join(
            str(formulation.get(key) or "")
            for key in ("attempt_1_natural", "attempt_2_keywords", "attempt_3_desperation")
        )
        words = _tokens(blob)
        user_vocab.update(words)
        system_hits.update(word for word in words if word in _SYSTEM_TAGS)
        atmospheric_hits.update(word for word in words if word in _ATMOSPHERIC)
    vocab_denom = sum(system_hits.values()) + sum(atmospheric_hits.values())
    atmospheric_share = (
        round(sum(atmospheric_hits.values()) / vocab_denom, 4) if vocab_denom else 0
    )

    surfaces = Counter()
    surface_frustrated = Counter()
    for row in records:
        extraction = row.get("extraction") or {}
        surface = extraction.get("retrieval_surface") or "unknown"
        surfaces[surface] += 1
        if extraction.get("emotion") in {"frustration", "anxiety"} or extraction.get("outcome") == "abandoned":
            surface_frustrated[surface] += 1
    surface_rows = []
    for surface, count in surfaces.most_common():
        surface_rows.append({
            "surface": surface,
            "mentions": count,
            "frustration_pct": round(100 * surface_frustrated[surface] / count, 1),
        })

    effort = Counter()
    for row in records:
        extraction = row.get("extraction") or {}
        effort[(int(extraction.get("num_search_attempts") or 0), workaround_bucket(extraction))] += 1

    trends = Counter()
    trend_friction: dict[tuple[str, str], list[float]] = defaultdict(list)
    unknown_dates = 0
    for row in records:
        period = _quarter(row.get("timestamp"))
        archetype = row.get("archetype") or "episodic_travel"
        if period is None:
            unknown_dates += 1
            continue
        trends[(period, archetype)] += 1
        trend_friction[(period, archetype)].append(friction_score(row.get("extraction") or {}))
    trend_rows = []
    for (period, archetype), count in sorted(trends.items()):
        values = trend_friction[(period, archetype)]
        trend_rows.append({
            "period": period,
            "archetype": archetype,
            "count": count,
            "avg_friction": round(sum(values) / len(values), 2),
        })

    segments = Counter()
    segment_abandoned = Counter()
    segment_arch: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        extraction = row.get("extraction") or {}
        segment = extraction.get("user_segment") or "unknown"
        segments[segment] += 1
        segment_arch[segment][row.get("archetype") or "episodic_travel"] += 1
        if extraction.get("outcome") == "abandoned":
            segment_abandoned[segment] += 1
        platform = extraction.get("device_platform") or "unknown"
        if platform in {"ios", "android"}:
            key = f"platform_{platform}"
            segments[key] += 1
            segment_arch[key][row.get("archetype") or "episodic_travel"] += 1
            if extraction.get("outcome") == "abandoned":
                segment_abandoned[key] += 1
    segment_rows = []
    for segment, count in segments.most_common():
        top = segment_arch[segment].most_common(1)
        segment_rows.append({
            "segment": segment,
            "count": count,
            "problem_rate": round(segment_abandoned[segment] / count, 4) if count else 0,
            "top_archetype": top[0][0] if top else None,
            "implication": SEGMENT_IMPLICATION.get(segment, "No segment note."),
        })

    top = archetype_rows[0] if archetype_rows else None
    return {
        "formula": FORMULA,
        "hero": {
            "total_extracted": total,
            "raw_ingested": raw_total,
            "retrieval_discussion_rate": round(total / raw_total, 4) if raw_total else 0,
            "mean_abandonment_rate": round(abandoned_n / total, 4) if total else 0,
            "avg_query_attempts": round(sum(attempt_values) / len(attempt_values), 2) if attempt_values else 0,
            "manual_scroll_share": round(buckets.get("manual_scroll", 0) / total, 4) if total else 0,
            "manual_scroll_median_minutes": median_minutes,
            "unknown_timestamp_count": unknown_dates,
        },
        "top_problem": top,
        "archetypes": archetype_rows,
        "anchors_retained": dict(retained),
        "anchors_forgotten": forgotten.most_common(20),
        "anchor_cooccurrence": [
            {"pair": pair, "count": count} for pair, count in kind_pairs.most_common()
        ],
        "sankey": [
            {"source": left, "target": right, "value": count}
            for (left, right), count in sankey.most_common()
        ],
        "user_vocabulary": user_vocab.most_common(30),
        "system_tag_hits": system_hits.most_common(),
        "atmospheric_hits": atmospheric_hits.most_common(),
        "atmospheric_share_of_query_terms": atmospheric_share,
        "semantic_distance_method": "lexical_overlap_pending_embeddings",
        "surfaces": surface_rows,
        "workarounds": dict(buckets),
        "effort_curve": [
            {"attempts": attempts, "workaround": bucket, "count": count}
            for (attempts, bucket), count in sorted(effort.items())
        ],
        "trends": trend_rows,
        "segments": segment_rows,
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_aggregates(records: list[dict]) -> tuple[Path, dict]:
    raw_total = _count_lines(settings.raw_dir / "raw_corpus.jsonl")
    payload = build_aggregates(records, raw_total)
    settings.ensure_dirs()
    path = settings.aggregates_path
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, payload
