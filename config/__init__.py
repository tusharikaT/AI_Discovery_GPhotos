"""Configuration package for the AI Discovery Engine."""

from config.settings import (  # noqa: F401
    BASE_DIR,
    NOISE_LEXICON,
    ON_TOPIC_SOURCES,
    PHOTOS_ANCHOR_TERMS,
    RETRIEVAL_SIGNAL_TERMS,
    Settings,
    settings,
)

__all__ = [
    "settings",
    "Settings",
    "BASE_DIR",
    "NOISE_LEXICON",
    "RETRIEVAL_SIGNAL_TERMS",
    "PHOTOS_ANCHOR_TERMS",
    "ON_TOPIC_SOURCES",
]
