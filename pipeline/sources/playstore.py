"""Google Play Store scraper for Google Photos reviews (google-play-scraper).

Pulls newest and most-relevant reviews across multiple locales, paging via
continuation token.
"""

from __future__ import annotations

from collections.abc import Iterator

from pipeline.sources.base import RawItem, Scraper

_APP_ID = "com.google.android.apps.photos"
_COUNTRIES = ["us", "gb", "in", "au", "ca"]


class PlayStoreScraper(Scraper):
    name = "playstore"

    def scrape(self) -> Iterator[RawItem]:
        try:
            from google_play_scraper import Sort, reviews  # lazy import
        except Exception as e:  # noqa: BLE001
            print(f"  [playstore] google-play-scraper unavailable: {e}")
            return

        yielded = 0
        per_country = None
        if self.limit:
            per_country = max(1, self.limit // len(_COUNTRIES))

        # Newest catches fresh complaints; most-relevant catches longer reviews.
        for sort in (Sort.NEWEST, Sort.MOST_RELEVANT):
            for country in _COUNTRIES:
                for item in self._pull_country(reviews, country, sort, per_country):
                    yield item
                    yielded += 1
                    if self.limit and yielded >= self.limit:
                        return

    def _pull_country(self, reviews, country, sort, per_country) -> Iterator[RawItem]:
        token = None
        fetched_country = 0
        while True:
            batch = 100
            if per_country:
                batch = min(100, per_country - fetched_country)
                if batch <= 0:
                    return
            try:
                result, token = reviews(
                    _APP_ID,
                    lang="en",
                    country=country,
                    sort=sort,
                    count=batch,
                    continuation_token=token,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [playstore] {country} error: {e}")
                return

            if not result:
                return

            for r in result:
                content = (r.get("content") or "").strip()
                if not content:
                    continue
                at = r.get("at")
                ts = at.date().isoformat() if at else None
                yield RawItem.create(
                    source="playstore",
                    text=content,
                    source_id=r.get("reviewId"),
                    source_url=(
                        f"https://play.google.com/store/apps/details?"
                        f"id={_APP_ID}&reviewId={r.get('reviewId')}"
                    ),
                    source_detail=f"playstore/{country}",
                    author=r.get("userName"),
                    timestamp=ts,
                    rating=r.get("score"),
                    region=country.upper(),
                )
                fetched_country += 1
                if per_country and fetched_country >= per_country:
                    return

            if token is None:
                return
