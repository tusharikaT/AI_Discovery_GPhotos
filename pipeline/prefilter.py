"""Inline pre-filter (Phase 1) — cheap, high-precision junk removal.

Applied to every ``RawItem`` *before* it is written to the raw store. Only drops
what we are near-certain is noise (anything dropped here is gone without
re-scraping); the strict, context-aware filtering happens in Phase 2.

Rules (see implementation_plan.md Phase 1):
  1. length/emptiness gate
  2. language gate (lenient English-only)
  3. exact-duplicate gate (normalized-text hash)
  4. hard-noise keyword gate WITH retrieval-signal override (mixed items kept)
  5. spam/promo gate
  6. relevance-seed gate (broad sources only)
"""

from __future__ import annotations

import hashlib
import re

from config.settings import (
    NOISE_LEXICON,
    PHOTOS_ANCHOR_TERMS,
    RETRIEVAL_SIGNAL_TERMS,
    settings,
)
from pipeline.sources.base import RawItem

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_LETTERS_RE = re.compile(r"[a-zA-Z]")

# Sources considered already on-topic (skip the relevance-seed gate).
# Reddit is handled specially via subreddit (see _is_on_topic).
_ON_TOPIC_SOURCES = {"playstore", "appstore", "youtube"}

# Cheap promo/spam signals.
_SPAM_PATTERNS = re.compile(
    r"(promo\s*code|coupon|referral|discount code|buy now|"
    r"click here to win|free gift|whatsapp \+?\d)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs, collapse whitespace — for hashing/matching."""
    text = _URL_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8", "ignore")).hexdigest()


def _term_pattern(term: str) -> re.Pattern:
    """Match a term on word boundaries, allowing a simple plural or -ing/-ed."""
    words = [re.escape(part) + r"(?:s|es|ing|ed)?" for part in term.split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"\b")


_TERM_CACHE: dict[str, re.Pattern] = {}


def _contains_any(text_low: str, terms: list[str]) -> bool:
    for term in terms:
        pattern = _TERM_CACHE.get(term)
        if pattern is None:
            pattern = _term_pattern(term)
            _TERM_CACHE[term] = pattern
        if pattern.search(text_low):
            return True
    return False


def _is_on_topic(item: RawItem) -> bool:
    if item.source in _ON_TOPIC_SOURCES:
        return True
    detail = (item.source_detail or "").lower()
    if item.source == "reddit" and "googlephotos" in detail:
        return True
    return False


def _detect_english(text: str) -> tuple[bool, float]:
    """Return (is_english, confidence). Missing lib -> (True, 0.0) i.e. keep."""
    try:
        from langdetect import detect_langs  # lazy import
    except Exception:  # noqa: BLE001
        return True, 0.0
    try:
        langs = detect_langs(text)
    except Exception:  # noqa: BLE001
        return True, 0.0
    if not langs:
        return True, 0.0
    top = langs[0]
    return (top.lang == "en"), float(top.prob)


def prefilter(item: RawItem, seen_hashes: set[str]) -> tuple[bool, str | None]:
    """Return ``(keep, drop_reason)``. Mutates ``seen_hashes`` when kept.

    ``drop_reason`` is ``None`` when the item is kept.
    """
    text = item.text or ""
    text_low = normalize_text(text)
    words = text_low.split()

    # 1) length / emptiness -------------------------------------------------
    if len(words) < settings.prefilter_min_words:
        return False, "too_short_words"
    if len(text.strip()) < settings.prefilter_min_chars:
        return False, "too_short_chars"
    # emoji / punctuation / URL-only (almost no letters left)
    if len(_LETTERS_RE.findall(text_low)) < 10:
        return False, "no_text_content"

    # 2) language (lenient) -------------------------------------------------
    is_en, conf = _detect_english(text)
    if (not is_en) and conf >= settings.lang_confidence_threshold and len(words) >= 6:
        return False, "non_english"

    # 3) exact-duplicate ----------------------------------------------------
    h = text_hash(text)
    if h in seen_hashes:
        return False, "duplicate"

    # 4) hard-noise keyword gate WITH retrieval-signal override -------------
    has_signal = _contains_any(text_low, RETRIEVAL_SIGNAL_TERMS)
    if not has_signal:
        for category, terms in NOISE_LEXICON.items():
            if _contains_any(text_low, terms):
                return False, f"noise_{category}"

    # 5) spam / promo -------------------------------------------------------
    if len(_URL_RE.findall(text)) >= 3 or _SPAM_PATTERNS.search(text):
        return False, "spam_promo"

    # 6) relevance-seed gate (broad sources only) ---------------------------
    if not _is_on_topic(item):
        has_anchor = _contains_any(text_low, PHOTOS_ANCHOR_TERMS)
        if not (has_anchor and has_signal):
            return False, "off_topic_seed"

    # kept
    seen_hashes.add(h)
    return True, None
