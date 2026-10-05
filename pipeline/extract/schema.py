"""Pydantic schema for one cognitive extraction (Phase 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Archetype = Literal[
    "utility_document",
    "episodic_travel",
    "aesthetic_visual",
    "micro_moment",
    "disambiguation_comparative",
    "pre_verbal_sensory",
]
FailureStage = Literal[
    "semantic_mismatch",
    "ocr_failure",
    "ranking_flooding",
    "expression_failure",
]
Outcome = Literal[
    "abandoned",
    "retrieved_with_friction",
    "social_offload",
    "unknown",
]
Segment = Literal[
    "heavy_shooter",
    "traveler",
    "parent",
    "student",
    "document_keeper",
    "older_user",
    "general",
    "unknown",
]
Surface = Literal[
    "search_bar",
    "ask_photos",
    "memories",
    "albums",
    "timeline",
    "faces",
    "lens",
    "unknown",
]
Emotion = Literal[
    "anxiety",
    "frustration",
    "nostalgia_loss",
    "resignation",
    "relief",
    "unknown",
]
Platform = Literal["ios", "android", "web", "unknown"]
MediaType = Literal["photo", "video", "screenshot", "document", "scan", "unknown"]
Stakes = Literal["medical_financial_legal", "sentimental", "trivial", "unknown"]

_ARCHETYPE_ALIASES = {
    "utility/document": "utility_document",
    "utility": "utility_document",
    "document": "utility_document",
    "episodic/travel": "episodic_travel",
    "episodic": "episodic_travel",
    "travel": "episodic_travel",
    "milestone": "episodic_travel",
    "personal milestone": "episodic_travel",
    "aesthetic/visual": "aesthetic_visual",
    "aesthetic": "aesthetic_visual",
    "visual": "aesthetic_visual",
    "micro-moment": "micro_moment",
    "micro moment": "micro_moment",
    "needle": "micro_moment",
    "disambiguation/comparative": "disambiguation_comparative",
    "comparative": "disambiguation_comparative",
    "disambiguation": "disambiguation_comparative",
    "pre-verbal/sensory": "pre_verbal_sensory",
    "pre-verbal": "pre_verbal_sensory",
    "lexical void": "pre_verbal_sensory",
    "thingamajig": "pre_verbal_sensory",
}
_FAILURE_ALIASES = {
    "interpretation": "semantic_mismatch",
    "interpretation (semantic mismatch)": "semantic_mismatch",
    "semantic mismatch": "semantic_mismatch",
    "ocr": "ocr_failure",
    "ocr failure": "ocr_failure",
    "ranking flooding": "ranking_flooding",
    "flooding": "ranking_flooding",
    "expression": "expression_failure",
    "expression failure": "expression_failure",
}
_OUTCOME_ALIASES = {
    "abandoned": "abandoned",
    "retrieved with extreme friction": "retrieved_with_friction",
    "retrieved_with_extreme_friction": "retrieved_with_friction",
    "social offload": "social_offload",
    "social offload (asked a friend)": "social_offload",
}
_SEGMENT_ALIASES = {
    "heavy shooter": "heavy_shooter",
    "document keeper": "document_keeper",
    "older user": "older_user",
}
_SURFACE_ALIASES = {
    "search": "search_bar",
    "search bar": "search_bar",
    "ask photos": "ask_photos",
    "timeline scroll": "timeline",
    "face groups": "faces",
    "face": "faces",
}
_EMOTION_ALIASES = {
    "nostalgia": "nostalgia_loss",
    "nostalgia-loss": "nostalgia_loss",
    "loss": "nostalgia_loss",
}
_STAKES_ALIASES = {
    "medical": "medical_financial_legal",
    "financial": "medical_financial_legal",
    "legal": "medical_financial_legal",
    "medical/financial/legal": "medical_financial_legal",
    "medical-financial-legal": "medical_financial_legal",
}


def _norm_token(value: object, aliases: dict[str, str]) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = text.replace("/", "_")
    if text in aliases:
        return aliases[text]
    spaced = str(value or "").strip().lower()
    return aliases.get(spaced, text)


class MemoryAnchors(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temporal: list[str] = Field(default_factory=list)
    sensory: list[str] = Field(default_factory=list)
    emotional: list[str] = Field(default_factory=list)
    social: list[str] = Field(default_factory=list)

    @field_validator("temporal", "sensory", "emotional", "social", mode="before")
    @classmethod
    def _list(cls, value: object) -> list:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class SearchFormulation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attempt_1_natural: str | None = None
    attempt_2_keywords: str | None = None
    attempt_3_desperation: str | None = None

    @field_validator(
        "attempt_1_natural",
        "attempt_2_keywords",
        "attempt_3_desperation",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class Extraction(BaseModel):
    """Structured cognitive read of one review or post."""

    model_config = ConfigDict(extra="ignore")

    primary_archetype: Archetype
    secondary_archetypes: list[Archetype] = Field(default_factory=list)
    memory_anchors_retained: MemoryAnchors = Field(default_factory=MemoryAnchors)
    information_forgotten: list[str] = Field(default_factory=list)
    search_formulation: SearchFormulation = Field(default_factory=SearchFormulation)
    primary_failure_stage: FailureStage
    contributing_factors: list[FailureStage] = Field(default_factory=list)
    user_workaround: str | None = None
    outcome: Outcome = "unknown"
    user_segment: Segment = "unknown"
    retrieval_surface: Surface = "unknown"
    emotion: Emotion = "unknown"
    region: str | None = None
    device_platform: Platform = "unknown"
    media_type: MediaType = "unknown"
    stakes_level: Stakes = "unknown"
    num_search_attempts: int = 0
    representative_quote: str = ""
    extraction_confidence: float = 0.5

    @field_validator("primary_archetype", mode="before")
    @classmethod
    def _archetype(cls, value: object) -> str:
        return _norm_token(value, _ARCHETYPE_ALIASES)

    @field_validator("secondary_archetypes", mode="before")
    @classmethod
    def _secondary(cls, value: object) -> list:
        if not value:
            return []
        items = [value] if isinstance(value, str) else list(value)
        return [_norm_token(item, _ARCHETYPE_ALIASES) for item in items if item]

    @field_validator("primary_failure_stage", mode="before")
    @classmethod
    def _failure(cls, value: object) -> str:
        return _norm_token(value, _FAILURE_ALIASES)

    @field_validator("contributing_factors", mode="before")
    @classmethod
    def _factors(cls, value: object) -> list:
        if not value:
            return []
        items = [value] if isinstance(value, str) else list(value)
        return [_norm_token(item, _FAILURE_ALIASES) for item in items if item]

    @field_validator("outcome", mode="before")
    @classmethod
    def _outcome(cls, value: object) -> str:
        return _norm_token(value, _OUTCOME_ALIASES) or "unknown"

    @field_validator("user_segment", mode="before")
    @classmethod
    def _segment(cls, value: object) -> str:
        return _norm_token(value, _SEGMENT_ALIASES) or "unknown"

    @field_validator("retrieval_surface", mode="before")
    @classmethod
    def _surface(cls, value: object) -> str:
        return _norm_token(value, _SURFACE_ALIASES) or "unknown"

    @field_validator("emotion", mode="before")
    @classmethod
    def _emotion(cls, value: object) -> str:
        return _norm_token(value, _EMOTION_ALIASES) or "unknown"

    @field_validator("stakes_level", mode="before")
    @classmethod
    def _stakes(cls, value: object) -> str:
        return _norm_token(value, _STAKES_ALIASES) or "unknown"

    @field_validator("device_platform", "media_type", mode="before")
    @classmethod
    def _simple(cls, value: object) -> str:
        return _norm_token(value, {}) or "unknown"

    @field_validator("information_forgotten", mode="before")
    @classmethod
    def _forgotten(cls, value: object) -> list:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("extraction_confidence", mode="before")
    @classmethod
    def _conf(cls, value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, number))

    @model_validator(mode="after")
    def _derive(self) -> "Extraction":
        attempts = [
            self.search_formulation.attempt_1_natural,
            self.search_formulation.attempt_2_keywords,
            self.search_formulation.attempt_3_desperation,
        ]
        self.num_search_attempts = sum(1 for attempt in attempts if attempt)
        self.representative_quote = (self.representative_quote or "").strip()[:400]
        if self.primary_archetype in self.secondary_archetypes:
            self.secondary_archetypes = [
                item for item in self.secondary_archetypes if item != self.primary_archetype
            ]
        if self.primary_failure_stage in self.contributing_factors:
            self.contributing_factors = [
                item for item in self.contributing_factors if item != self.primary_failure_stage
            ]
        return self
