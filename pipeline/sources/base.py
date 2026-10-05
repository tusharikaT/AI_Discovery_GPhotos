"""Scraper base class and the canonical ``RawItem`` schema.

Every scraper yields ``RawItem`` instances built via :meth:`RawItem.create`,
which enforces the fixed-schema / null-policy contract from the implementation
plan: all keys are always present, unknowns are explicit ``None``/``[]`` (never
missing), ``id`` is a stable hash, ``text`` is required, and a ``missing_fields``
list records which optional fields were unavailable.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Optional fields whose absence is tracked (but never an error).
_OPTIONAL_FIELDS = (
    "source_url",
    "source_detail",
    "author_hash",
    "timestamp",
    "title",
    "rating",
    "region",
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RawItem:
    """A single scraped item in canonical form (pre-extraction)."""

    id: str
    source: str
    text: str
    source_url: str | None = None
    source_detail: str | None = None
    author_hash: str | None = None
    timestamp: str | None = None          # ISO-8601 date/datetime string
    title: str | None = None
    rating: float | None = None
    region: str | None = None
    scraped_at: str = field(default_factory=_iso_now)
    missing_fields: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        source: str,
        text: str,
        source_id: str | None = None,
        source_url: str | None = None,
        source_detail: str | None = None,
        author: str | None = None,
        timestamp: str | None = None,
        title: str | None = None,
        rating: float | None = None,
        region: str | None = None,
    ) -> "RawItem":
        text = (text or "").strip()

        # Stable id: prefer source + source_id, else hash of source + text.
        basis = f"{source}:{source_id}" if source_id else f"{source}:{text}"
        item_id = _sha1(basis)[:20]

        author_hash = _sha1(author)[:16] if author else None

        item = cls(
            id=item_id,
            source=source,
            text=text,
            source_url=source_url or None,
            source_detail=source_detail or None,
            author_hash=author_hash,
            timestamp=timestamp or None,
            title=(title.strip() if title else None),
            rating=rating,
            region=region or None,
        )
        item.missing_fields = [
            f for f in _OPTIONAL_FIELDS if getattr(item, f) in (None, "")
        ]
        return item

    def to_dict(self) -> dict:
        return asdict(self)


class Scraper(ABC):
    """Abstract per-source scraper. Subclasses yield :class:`RawItem`."""

    name: str = "base"

    def __init__(self, limit: int | None = None) -> None:
        # ``limit`` is a soft cap per source; ``None`` means unbounded.
        self.limit = limit

    @abstractmethod
    def scrape(self) -> Iterator[RawItem]:
        """Yield RawItem instances. Must be resumable/idempotent-friendly."""
        raise NotImplementedError
