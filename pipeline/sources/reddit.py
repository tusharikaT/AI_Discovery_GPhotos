"""Reddit scraper — anonymous public RSS feeds (no API key).

Reddit's ``.json`` endpoints now return 403 to non-authenticated clients, but
the public Atom/RSS feeds (``search.rss`` / ``new.rss``) remain accessible.
We search retrieval-intent queries within the configured subreddits and pull
the newest submissions from r/googlephotos. Each entry (title + body) becomes a
RawItem.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import requests
from bs4 import BeautifulSoup

from config.settings import settings
from pipeline.sources.base import RawItem, Scraper

_QUERIES = [
    "can't find photo",
    "can't find old photo",
    "search not working",
    "find old picture",
    "scrolling to find photo",
    "ask photos",
    "search doesn't find",
    "find screenshot",
    "locate photo",
]

_BASE = "https://www.reddit.com"
_ATOM = "{http://www.w3.org/2005/Atom}"


class RedditScraper(Scraper):
    name = "reddit"

    def _get_feed(self, url: str, params: dict) -> str | None:
        headers = {"User-Agent": settings.user_agent, "Accept": "application/atom+xml"}
        for attempt in range(settings.request_max_retries):
            try:
                resp = requests.get(
                    url, params=params, headers=headers,
                    timeout=settings.request_timeout_s,
                )
                if resp.status_code == 200 and "xml" in resp.headers.get(
                    "content-type", ""
                ):
                    return resp.text
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(settings.request_backoff_base_s * (attempt + 1))
                    continue
                return None
            except requests.RequestException:
                time.sleep(settings.request_backoff_base_s * (attempt + 1))
        return None

    def _parse_entries(self, xml_text: str, subreddit: str) -> Iterator[RawItem]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return
        for entry in root.findall(f"{_ATOM}entry"):
            title_el = entry.find(f"{_ATOM}title")
            content_el = entry.find(f"{_ATOM}content")
            link_el = entry.find(f"{_ATOM}link")
            id_el = entry.find(f"{_ATOM}id")
            published_el = entry.find(f"{_ATOM}published")
            author_el = entry.find(f"{_ATOM}author/{_ATOM}name")

            title = title_el.text if title_el is not None else ""
            body_html = content_el.text if content_el is not None else ""
            body = BeautifulSoup(body_html or "", "html.parser").get_text(" ").strip()
            text = (f"{title}\n{body}".strip()) if body else (title or "")
            if not text:
                continue

            url = link_el.get("href") if link_el is not None else None
            source_id = id_el.text if id_el is not None else url
            ts = None
            if published_el is not None and published_el.text:
                ts = published_el.text.split("T")[0]  # ISO date part
            author = author_el.text if author_el is not None else None

            yield RawItem.create(
                source="reddit",
                text=text,
                source_id=source_id,
                source_url=url,
                source_detail=f"r/{subreddit}",
                author=author,
                timestamp=ts,
                title=title,
            )

    def scrape(self) -> Iterator[RawItem]:
        yielded = 0
        seen_ids: set[str] = set()

        def _emit(item: RawItem):
            nonlocal yielded
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                yielded += 1
                return True
            return False

        for sub in settings.reddit_subreddits:
            for q in _QUERIES:
                if self.limit and yielded >= self.limit:
                    return
                xml_text = self._get_feed(
                    f"{_BASE}/r/{sub}/search.rss",
                    {"q": q, "restrict_sr": 1, "sort": "relevance", "limit": 100},
                )
                if not xml_text:
                    continue
                for item in self._parse_entries(xml_text, sub):
                    if self.limit and yielded >= self.limit:
                        return
                    if _emit(item):
                        yield item
                time.sleep(0.8)

        # newest from r/googlephotos
        if not (self.limit and yielded >= self.limit):
            xml_text = self._get_feed(f"{_BASE}/r/googlephotos/new.rss", {"limit": 100})
            if xml_text:
                for item in self._parse_entries(xml_text, "googlephotos"):
                    if self.limit and yielded >= self.limit:
                        return
                    if _emit(item):
                        yield item
