"""Provider-agnostic chat client with an on-disk response cache.

Cache key is (model, prompt hash). A repeat extraction does not call the API.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings


class LLMError(RuntimeError):
    pass


def _cache_path(model: str, prompt: str) -> Path:
    digest = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
    folder = settings.processed_dir / "llm_cache"
    folder.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    return folder / f"{safe_model}_{digest[:24]}.json"


class LLMClient:
    def __init__(self) -> None:
        self.provider = (settings.llm_provider or "anthropic").lower()
        self.model = settings.llm_model or "claude-haiku-4-5-20251001"

    def discard(self, prompt: str) -> None:
        """Drop a cached reply that failed schema validation."""
        path = _cache_path(self.model, prompt)
        if path.exists():
            path.unlink()

    def chat(self, prompt: str, *, max_tokens: int = 900) -> tuple[str, dict]:
        """Return ``(text, usage)``. ``usage`` has input_tokens and output_tokens."""
        path = _cache_path(self.model, prompt)
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            usage = dict(cached.get("usage") or {})
            usage["cache_hit"] = True
            return cached.get("text") or "", usage

        if self.provider == "anthropic":
            text, usage = self._anthropic(prompt, max_tokens)
        else:
            raise LLMError(f"Unsupported LLM provider: {self.provider}")

        path.write_text(
            json.dumps({"text": text, "usage": usage}, ensure_ascii=False),
            encoding="utf-8",
        )
        usage["cache_hit"] = False
        return text, usage

    @retry(
        retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, LLMError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _anthropic(self, prompt: str, max_tokens: int) -> tuple[str, dict]:
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            method="POST",
        )
        request.add_header("x-api-key", settings.anthropic_api_key)
        request.add_header("anthropic-version", "2023-06-01")
        request.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens") or 0),
            "output_tokens": int(usage_raw.get("output_tokens") or 0),
        }
        return text, usage
