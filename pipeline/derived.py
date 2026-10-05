"""Turn fields we already stored into the charts. No model call."""

from __future__ import annotations

from pipeline.label import PAINS
from pipeline.metrics import friction_score

# 1 and 2 stars are a complaint. 3, and a review with no stars, are neutral.
# 4 and 5 stars are a good rating.
SENTIMENT = ("Frustrated", "Neutral", "Positive")

FEELING_POINTS = {
    "gave_up": 100,
    "anxious": 90,
    "no_longer_trusts_search": 85,
    "frustrated": 80,
    "worn_down": 75,
    "grieving": 65,
    "not_clear": 40,
    "relieved": 10,
}
FRUSTRATED_FEELINGS = {
    "frustrated",
    "anxious",
    "gave_up",
    "worn_down",
    "no_longer_trusts_search",
}
FAILURE_PAINS = set(PAINS) - {"does_not_fit", "search_worked"}

# First matching rule wins, so "photo date" stays a date and "photo location" stays a place.
_MISSING_RULES = (
    ("The exact date", ("date", "when taken", "when they were", "when photos")),
    ("The place", ("location", "place")),
    ("The album", ("album", "folder")),
    ("The file name", ("file name", "filename", "file type")),
    ("Which photos are gone", ("deleted", "missing", "number of photos")),
    ("What the photo shows", ("content", "subject", "photo", "screenshot", "visual", "video")),
    ("A time period", ("time period", "event")),
    ("The search words", ("search term",)),
    ("Who sent it", ("sender",)),
)
_PRODUCT_WORDS = ("frustrat", "disappoint", "organization", "system", " app", "bug", "update", " of use")
_PHRASE_ALIAS = {
    "friend": "Friends",
    "friends": "Friends",
    "person": "People",
    "people": "People",
    "people in photos": "People",
    "kid": "Children",
    "kids": "Children",
    "child": "Children",
    "children": "Children",
    "son": "Children",
    "years": "Years ago",
    "years ago": "Years ago",
    "old": "Old photos",
    "old photos": "Old photos",
}
_SEGMENT = {
    "large_library": "Large libraries",
    "heavy_shooter": "Large libraries",
    "keeps_documents": "People who keep documents",
    "document_keeper": "People who keep documents",
    "parent": "Parents",
    "traveler": "Travelers",
    "older_library": "Older libraries",
    "older_user": "Older libraries",
    "student": "Students",
}
SEGMENT_IMPLICATION = {
    "Large libraries": "A big library turns one hit into a flood. Search needs a way to narrow after the first result.",
    "People who keep documents": "Receipts and labels fail when the exact word or the scan is missing.",
    "Parents": "The photo they want is one moment of a child, buried in near-identical shots.",
    "Travelers": "They remember the trip as a season or a place, not a calendar date.",
    "Older libraries": "Years of photos make timeline scrolling the fallback. Search has to accept a rough era.",
    "Students": "Notes, slides, and screenshots are the search, and the file name is weak.",
}
ANCHOR_KIND = {
    "temporal": "a time",
    "sensory": "how it looked",
    "emotional": "how it felt",
    "social": "who was there",
}


def _star(rating: object) -> int | None:
    if rating is None or rating == "":
        return None
    try:
        star = int(round(float(rating)))
    except (TypeError, ValueError):
        return None
    if star < 1 or star > 5:
        return None
    return star


def star_sentiment(rating: object) -> str:
    star = _star(rating)
    if star is None or star == 3:
        return "Neutral"
    if star <= 2:
        return "Frustrated"
    return "Positive"


def review_friction(record: dict) -> int:
    """0–100. Average of the star rating, the stored feeling, and the search-effort tier."""
    extraction = record.get("extraction") or {}
    feeling = (record.get("label") or {}).get("feeling") or "not_clear"
    star = _star(record.get("rating"))
    star_points = 50 if star is None else {1: 100, 2: 80, 3: 50, 4: 25, 5: 10}[star]
    points = (
        star_points,
        FEELING_POINTS.get(feeling, 40),
        int(round(friction_score(extraction) * 10)),
    )
    return int(round(sum(points) / len(points)))


def friction_band(score: int) -> str:
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def missing_detail(phrase: str) -> str | None:
    text = (phrase or "").strip().lower()
    if not text:
        return None
    for label, needles in _MISSING_RULES:
        if any(needle in text for needle in needles):
            return label
    return None


def memory_phrase(text: str) -> str | None:
    raw = " ".join((text or "").split()).strip().lower()
    if not raw or any(word in raw for word in _PRODUCT_WORDS):
        return None
    if raw in _PHRASE_ALIAS:
        return _PHRASE_ALIAS[raw]
    return raw[:1].upper() + raw[1:]


def segment_name(record: dict) -> str | None:
    """Prefer the later label. Fall back to the extraction segment when that label is blank."""
    who = (record.get("label") or {}).get("who") or "not_said"
    if who in _SEGMENT:
        return _SEGMENT[who]
    segment = (record.get("extraction") or {}).get("user_segment") or ""
    return _SEGMENT.get(segment)


def is_retrieval_problem(record: dict) -> bool:
    pains = (record.get("label") or {}).get("pains") or []
    return any(pain in FAILURE_PAINS for pain in pains)


def reads_frustrated(record: dict) -> bool:
    feeling = (record.get("label") or {}).get("feeling") or ""
    return feeling in FRUSTRATED_FEELINGS


_DOC_WHO = {"keeps_documents", "student", "document_keeper"}
_MEM_WHO = {"parent", "traveler", "older_library", "older_user"}
_OBJ_WHO = {"large_library", "heavy_shooter"}
_DOC_LOOK = {"a document or receipt", "a screenshot"}
_MEM_LOOK = {"an old event", "a person", "a place or trip"}
_OBJ_LOOK = {"a specific object"}
_DOC_WORDS = (
    "receipt", "serial", "screenshot", "prescription", "invoice",
    "whiteboard", "scan of", "document",
)
_MEM_WORDS = (
    "mom", "dad", "wedding", "birthday", "trip", "vacation",
    "my son", "my daughter", "my kid", "my child",
)
_MEM_PAINS = {"faces", "memories_albums", "vague_memory"}


def job_segment(record: dict) -> str | None:
    """One of memories, documents, objects, or None when the review is not about finding a photo."""
    label = record.get("label") or {}
    extraction = record.get("extraction") or {}
    who = label.get("who") or "not_said"
    segment = extraction.get("user_segment") or ""
    looking = label.get("looking_for") or "not_said"
    pains = set(label.get("pains") or [])
    text = (record.get("text") or "").lower()
    if (
        who in _DOC_WHO
        or segment in _DOC_WHO
        or looking in _DOC_LOOK
        or "document" in pains
        or any(word in text for word in _DOC_WORDS)
    ):
        return "documents"
    if (
        who in _MEM_WHO
        or segment in _MEM_WHO
        or looking in _MEM_LOOK
        or pains & _MEM_PAINS
        or any(word in text for word in _MEM_WORDS)
    ):
        return "memories"
    if (
        who in _OBJ_WHO
        or segment in _OBJ_WHO
        or looking in _OBJ_LOOK
        or "library_flood" in pains
    ):
        return "objects"
    if pains <= {"does_not_fit", "search_worked"} or not pains:
        return None
    return "objects"
