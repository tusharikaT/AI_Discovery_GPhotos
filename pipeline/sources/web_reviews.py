"""Open-web scraper — the web-first workhorse.

Discovers pages with Bing (paginated) over retrieval-intent queries, then
extracts main content, including comment sections, with trafilatura. Also
processes any configured seed URLs. Yields one RawItem per page.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import settings
from pipeline.sources.base import RawItem, Scraper

_MAX_CHARS = 8000  # cap per-page text to keep records manageable
_PAGE_SIZE = 10
_SKIP_HOST_SUFFIXES = (
    "bing.com",
    "microsoft.com",
    "duckduckgo.com",
    "wikipedia.org",
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "play.google.com",
    "apps.apple.com",
)


def _unwrap_bing(href: str) -> str | None:
    """Turn a Bing tracking redirect into the real destination URL."""
    if "bing.com/ck/" not in href:
        return href
    token = (parse_qs(urlparse(href).query).get("u") or [""])[0]
    if token.startswith("a1"):
        token = token[2:]
    padding = "=" * (-len(token) % 4)
    try:
        target = base64.urlsafe_b64decode(token + padding).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return None
    return target if target.startswith("http") else None


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or "web"
    except Exception:  # noqa: BLE001
        return "web"


def _skip_url(url: str) -> bool:
    """Skip pages that are not user discussions (homepages, encyclopedias)."""
    host = _domain(url).lower()
    path = urlparse(url).path or "/"
    if any(host == suffix or host.endswith("." + suffix) for suffix in _SKIP_HOST_SUFFIXES):
        return True
    if host.endswith("wikipedia.org"):
        return True
    if host.startswith("www.google.") and path in ("", "/"):
        return True
    if host == "photos.google.com" and path in ("", "/"):
        return True
    return False


class WebReviewsScraper(Scraper):
    name = "web"

    def _known_urls(self) -> set[str]:
        """URLs already stored, so a rerun does not re-download them."""
        path = settings.raw_dir / "web.jsonl"
        known: set[str] = set()
        if not path.exists():
            return known
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    url = json.loads(line).get("source_url")
                except json.JSONDecodeError:
                    continue
                if url:
                    known.add(url)
        return known

    def _search_urls(self, query: str, max_results: int) -> list[tuple[str, str]]:
        """Return up to ``max_results`` organic (url, title) pairs from Bing."""
        headers = {
            "User-Agent": settings.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        first = 1
        while len(found) < max_results and first <= max_results:
            try:
                resp = requests.get(
                    "https://www.bing.com/search",
                    params={"q": query, "count": _PAGE_SIZE, "first": first},
                    headers=headers,
                    timeout=settings.request_timeout_s,
                )
            except requests.RequestException as exc:
                print(f"  [web] search error for '{query}': {exc}", flush=True)
                break
            if resp.status_code != 200:
                print(f"  [web] search HTTP {resp.status_code} for '{query}'", flush=True)
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            added = 0
            for anchor in soup.select("li.b_algo h2 a, h2 a"):
                href = _unwrap_bing((anchor.get("href") or "").split("#")[0])
                if not href or not href.startswith("http") or _skip_url(href) or href in seen:
                    continue
                seen.add(href)
                title = anchor.get_text(" ", strip=True)
                found.append((href, title))
                added += 1
                if len(found) >= max_results:
                    break
            if added == 0:
                break
            first += _PAGE_SIZE
            time.sleep(0.6)
        return found

    def _extract(self, url: str) -> str | None:
        try:
            import trafilatura  # lazy import
        except Exception as e:  # noqa: BLE001
            print(f"  [web] trafilatura unavailable: {e}")
            return None
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            text = trafilatura.extract(
                downloaded, include_comments=True, include_tables=False,
                favor_recall=True,
            )
            return text
        except Exception:  # noqa: BLE001
            return None

    def _emit(self, url: str, title: str, seen_urls: set[str]) -> RawItem | None:
        if not url or url in seen_urls or _skip_url(url):
            return None
        seen_urls.add(url)
        text = self._extract(url)
        if not text:
            return None
        text = text.strip()[:_MAX_CHARS]
        if len(text.split()) < settings.prefilter_min_words:
            return None
        return RawItem.create(
            source="web",
            text=text,
            source_id=url,
            source_url=url,
            source_detail=_domain(url),
            title=title or None,
        )

    def scrape(self) -> Iterator[RawItem]:
        yielded = 0
        seen_urls = self._known_urls()
        per_query = settings.web_results_per_query
        queries = list(settings.web_search_queries)

        for url in settings.web_seed_urls:
            if self.limit and yielded >= self.limit:
                return
            item = self._emit(url, "", seen_urls)
            if item:
                yielded += 1
                yield item
            time.sleep(0.4)

        for index, query in enumerate(queries, start=1):
            if self.limit and yielded >= self.limit:
                return
            print(f"  [web] query {index}/{len(queries)}: {query}", flush=True)
            for url, title in self._search_urls(query, per_query):
                if self.limit and yielded >= self.limit:
                    return
                item = self._emit(url, title, seen_urls)
                if not item:
                    continue
                yielded += 1
                yield item
                time.sleep(0.4)
