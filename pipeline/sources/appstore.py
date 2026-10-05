"""Apple App Store scraper for Google Photos reviews.

Uses Apple's public RSS customer-reviews JSON feed (reliable, no key) instead of
the fragile amp-api that ``app-store-scraper`` relies on. Pages 1..N across
several storefronts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import requests

from config.settings import settings
from pipeline.sources.base import RawItem, Scraper

_APP_ID = 962194608
_COUNTRIES = ["us", "gb", "in", "au", "ca"]
_MAX_PAGES = 10  # Apple caps the RSS feed around 10 pages / ~500 reviews


class AppStoreScraper(Scraper):
    name = "appstore"

    def _get_page(self, country: str, page: int) -> list[dict]:
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={_APP_ID}/sortby=mostrecent/json"
        )
        headers = {"User-Agent": settings.user_agent}
        for attempt in range(settings.request_max_retries):
            try:
                r = requests.get(url, headers=headers, timeout=settings.request_timeout_s)
                if r.status_code == 200:
                    data = r.json()
                    return data.get("feed", {}).get("entry", []) or []
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(settings.request_backoff_base_s * (attempt + 1))
                    continue
                return []
            except (requests.RequestException, ValueError):
                time.sleep(settings.request_backoff_base_s * (attempt + 1))
        return []

    def scrape(self) -> Iterator[RawItem]:
        yielded = 0
        per_country = None
        if self.limit:
            per_country = max(1, self.limit // len(_COUNTRIES))

        for country in _COUNTRIES:
            got_country = 0
            for page in range(1, _MAX_PAGES + 1):
                if self.limit and yielded >= self.limit:
                    return
                entries = self._get_page(country, page)
                if not entries:
                    break

                for e in entries:
                    # The first entry is app metadata (no im:rating) — skip it.
                    rating_obj = e.get("im:rating")
                    if not rating_obj:
                        continue
                    content = (e.get("content", {}).get("label") or "").strip()
                    title = (e.get("title", {}).get("label") or "").strip()
                    text = (title + "\n" + content).strip() if title else content
                    if not text:
                        continue
                    try:
                        rating = float(rating_obj.get("label"))
                    except (TypeError, ValueError):
                        rating = None
                    ts = (e.get("updated", {}).get("label") or "").split("T")[0] or None
                    author = e.get("author", {}).get("name", {}).get("label")
                    rid = e.get("id", {}).get("label")

                    yield RawItem.create(
                        source="appstore",
                        text=text,
                        source_id=rid,
                        source_url=f"https://apps.apple.com/{country}/app/id{_APP_ID}",
                        source_detail=f"appstore/{country}",
                        author=author,
                        timestamp=ts,
                        title=title or None,
                        rating=rating,
                        region=country.upper(),
                    )
                    yielded += 1
                    got_country += 1
                    if self.limit and yielded >= self.limit:
                        return
                    if per_country and got_country >= per_country:
                        break

                if per_country and got_country >= per_country:
                    break
                time.sleep(0.5)
