"""YouTube comment scraper (youtube-comment-downloader).

Scrapes comment threads from the configured Google Photos search/retrieval
videos. Uses ``settings.youtube_video_urls`` when set; otherwise discovers a
small set of videos via web search.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import requests

from config.settings import settings
from pipeline.sources.base import RawItem, Scraper

_VIDEO_ID_RE = re.compile(r"watch\?v=([A-Za-z0-9_-]{11})")
_DISCOVERY_QUERIES = [
    "google photos search not working",
    "google photos can't find old photos",
    "ask photos google photos",
]


class YouTubeScraper(Scraper):
    name = "youtube"

    def _discover_urls(self, max_urls: int = 6) -> list[str]:
        """Find a few Google Photos retrieval videos when none are configured."""
        found: list[str] = []
        seen: set[str] = set()
        headers = {"User-Agent": settings.user_agent}
        for query in _DISCOVERY_QUERIES:
            try:
                resp = requests.get(
                    "https://www.youtube.com/results",
                    params={"search_query": query},
                    headers=headers,
                    timeout=settings.request_timeout_s,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [youtube] search error for '{query}': {e}")
                continue
            for video_id in _VIDEO_ID_RE.findall(resp.text):
                if video_id in seen:
                    continue
                seen.add(video_id)
                found.append(f"https://www.youtube.com/watch?v={video_id}")
                if len(found) >= max_urls:
                    return found
        return found

    def scrape(self) -> Iterator[RawItem]:
        urls = list(settings.youtube_video_urls) or self._discover_urls()
        if not urls:
            print("  [youtube] no video URLs found — skipping")
            return
        print(f"  [youtube] {len(urls)} video(s)")

        try:
            from youtube_comment_downloader import (  # lazy import
                SORT_BY_POPULAR,
                YoutubeCommentDownloader,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [youtube] youtube-comment-downloader unavailable: {e}")
            return

        downloader = YoutubeCommentDownloader()
        yielded = 0
        per_video = None
        if self.limit:
            per_video = max(1, self.limit // len(urls))

        for url in urls:
            got = 0
            try:
                stream = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
            except Exception as e:  # noqa: BLE001
                print(f"  [youtube] {url} error: {e}")
                continue

            for c in stream:
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                yield RawItem.create(
                    source="youtube",
                    text=text,
                    source_id=c.get("cid"),
                    source_url=url,
                    source_detail="youtube",
                    author=c.get("author"),
                    timestamp=c.get("time_parsed") and None,  # relative -> null
                )
                yielded += 1
                got += 1
                if self.limit and yielded >= self.limit:
                    return
                if per_video and got >= per_video:
                    break
