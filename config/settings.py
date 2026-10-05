"""Central, env-driven configuration for the AI Discovery Engine.

All tunable knobs (paths, model names, thresholds, concurrency, rate limits) and
the pre-filter lexicons live here so later phases never hard-code values.

Scalar values can be overridden via a local ``.env`` file (see ``.env.example``).
Complex defaults (lexicons, seed lists) are defined in Python and can be edited
directly or overridden programmatically.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Project root (…/AI_discovery)
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings. Env vars (or .env) override the scalar defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- paths --
    data_dir: Path = BASE_DIR / "data"

    # ---------------------------------------------------------- app / build --
    app_title: str = "AI-Powered Discovery Engine"
    random_seed: int = 42

    # ---------------------------------------------------------- LLM (Phase 3) #
    # Provider/model are decided at Phase 3; keys come from the environment.
    llm_provider: str = "openai"          # openai | anthropic | local  (TBD)
    llm_model: str = ""                    # e.g. gpt-4o-mini / claude-… (TBD)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_concurrency: int = 4
    llm_temperature: float = 0.0
    llm_max_retries: int = 4

    # ------------------------------------------------- embeddings (Phase 5) --
    # Free local model (MIT). Locked after measuring the Phase 4 corpus.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_chunk_size: int = 300       # words; Phase 8, fits the 512-token model
    embedding_chunk_overlap: int = 40
    rag_top_k: int = 15
    # Cosine distance. Lower is closer. The four accordion questions land at
    # 0.24–0.42. "purple stapler from the moon base" lands at 0.47.
    rag_distance_cutoff: float = 0.45

    # --------------------------------------------------- scraping (Phase 1) --
    # 0 == unbounded (web-first, no hard cap). Override per source via env.
    scrape_max_items_per_source: int = 0
    scrape_smoke_limit: int = 30          # tiny cap for the smoke-test run
    request_timeout_s: int = 20
    request_backoff_base_s: float = 1.5
    request_max_retries: int = 5
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    reddit_subreddits: list[str] = Field(
        default_factory=lambda: [
            "googlephotos",
            "Android",
            "apple",
            "photography",
        ]
    )

    # Retrieval-intent search queries that drive the web-first workhorse.
    web_search_queries: list[str] = Field(
        default_factory=lambda: [
            "can't find old photo google photos",
            "google photos search doesn't work",
            "google photos can't find picture i took",
            "scrolling forever to find a photo google photos",
            "google photos search not finding old photos",
            "ask photos can't find",
            "google photos find screenshot from years ago",
            "google photos search medicine receipt document",
            "google photos search failed to find photo",
            "google photos I remember the photo but can't find it",
            "google photos scrolling through years to find a picture",
            "google photos search receipts prescriptions documents",
            "google photos face search missing person",
            "google photos search by description doesn't work",
            "\"ask photos\" useless OR frustrating OR \"can't find\"",
            "site:support.google.com/photos/thread can't find photo",
            "site:support.google.com/photos/thread search not working",
            "site:support.google.com/photos/thread old photos missing search",
            "site:support.google.com/photos/community can't find picture",
            "site:www.quora.com google photos can't find old photo",
            "site:www.quora.com google photos search doesn't work",
            "site:xdaforums.com google photos search find photo",
            "site:forums.androidcentral.com google photos can't find",
            "site:forums.macrumors.com google photos search",
            "site:www.androidpolice.com google photos search find",
            "site:www.reddit.com/r/googlephotos can't find photo search",
            "google photos \"had to scroll\" photo OR picture",
            "google photos \"searched for\" photo nothing OR \"no results\"",
        ]
    )

    # Seed URLs for open-web scraping (expanded at runtime via search).
    web_seed_urls: list[str] = Field(default_factory=list)

    # Max search result pages/URLs to pull per web query (0/None -> default).
    web_results_per_query: int = 50

    # YouTube video URLs whose comment threads to scrape (Google Photos
    # search/retrieval/Ask Photos videos). Populated for the full run.
    youtube_video_urls: list[str] = Field(default_factory=list)

    # --------------------------------------------- inline pre-filter (P1) --
    prefilter_min_words: int = 5
    prefilter_min_chars: int = 25
    lang_confidence_threshold: float = 0.75   # lenient at fetch time

    # ------------------------------------------- strict filter (Phase 2) --
    relevance_score_threshold: float = 0.55   # semantic retrieval-signal gate
    dedup_similarity_threshold: float = 0.90  # near-duplicate cosine cutoff
    keep_competitor_items: bool = False       # False -> drop; True -> tag only

    # ------------------------------------------------------ derived paths --
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "discovery.sqlite"

    @property
    def aggregates_path(self) -> Path:
        return self.processed_dir / "aggregates.json"

    def ensure_dirs(self) -> None:
        """Create the data directory tree if it does not exist."""
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.db_dir,
            self.vectorstore_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Pre-filter lexicons (Phase 1). Editable here; used by the inline pre-filter.
# --------------------------------------------------------------------------- #
NOISE_LEXICON: dict[str, list[str]] = {
    "billing_pricing": [
        "price", "pricing", "subscription", "refund", "charged", "bill",
        "invoice", "google one", "expensive", "cancel my",
    ],
    "storage_quota": [
        "storage", "gb", "quota", "full", "run out of space", "upgrade plan",
    ],
    "sync_backup": [
        "backup", "sync", "syncing", "upload stuck", "won't upload", "sd card",
    ],
    "account_security": [
        "password", "log in", "login", "2fa", "locked out",
        "recover my account", "hacked",
    ],
    "hardware_stability": [
        "crash", "crashes", "freezes", "lag", "battery", "overheating",
        "won't install", "update broke", "black screen",
    ],
}

# If any of these appear, the item is KEPT even when noise terms are present.
RETRIEVAL_SIGNAL_TERMS: list[str] = [
    "search", "find", "can't find", "cant find", "look for", "locate",
    "retrieve", "remember", "memory", "scroll", "scrolling", "photo of",
    "picture of", "screenshot", "album", "tag", "faces", "years ago",
    "that trip", "where is",
]

# For broad sources, require at least one Photos anchor before writing.
PHOTOS_ANCHOR_TERMS: list[str] = [
    "google photos", "gallery", "photo", "picture", "image", "screenshot",
]

# Sources considered already on-topic (skip the relevance-seed gate).
ON_TOPIC_SOURCES: set[str] = {"googlephotos", "playstore", "appstore"}


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #
settings = Settings()

__all__ = [
    "settings",
    "Settings",
    "BASE_DIR",
    "NOISE_LEXICON",
    "RETRIEVAL_SIGNAL_TERMS",
    "PHOTOS_ANCHOR_TERMS",
    "ON_TOPIC_SOURCES",
]
