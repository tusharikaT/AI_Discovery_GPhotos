"""CLI orchestrator for the offline pipeline.

Phase 1 implements the ``scrape`` stage. Later stages are wired as stubs and
filled in their respective phases.

Usage:
    python -m pipeline.run --stage scrape [--sources reddit,web,...] [--smoke] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from config.settings import settings
from pipeline.prefilter import prefilter
from pipeline.rawio import (
    append_item,
    load_seen,
    merge_corpus,
    write_rejects,
)
from pipeline.sources.appstore import AppStoreScraper
from pipeline.sources.playstore import PlayStoreScraper
from pipeline.sources.reddit import RedditScraper
from pipeline.sources.web_reviews import WebReviewsScraper
from pipeline.sources.youtube import YouTubeScraper

SCRAPERS = {
    "web": WebReviewsScraper,
    "reddit": RedditScraper,
    "playstore": PlayStoreScraper,
    "appstore": AppStoreScraper,
    "youtube": YouTubeScraper,
}

# Kept-item caps. Web and Play Store stop at 15k.
# Every other source stops at 10k.
SOURCE_CAPS = {
    "web": 15000,
    "playstore": 15000,
    "reddit": 10000,
    "appstore": 10000,
    "youtube": 10000,
}


def _resolve_limit(args, source: str) -> int | None:
    if args.smoke:
        return settings.scrape_smoke_limit
    if args.limit is not None:
        return args.limit
    if source in SOURCE_CAPS:
        return SOURCE_CAPS[source]
    val = settings.scrape_max_items_per_source
    return None if val in (0, None) else val


def run_scrape(sources: list[str], args) -> None:
    settings.ensure_dirs()
    rejects: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals = {"kept": 0, "dropped": 0}

    for source in sources:
        scraper_cls = SCRAPERS.get(source)
        if not scraper_cls:
            print(f"[skip] unknown source '{source}'")
            continue

        limit = _resolve_limit(args, source)
        print(f"\n=== Scraping: {source} (limit={limit if limit else 'unbounded'}) ===")
        seen_ids, seen_hashes = load_seen(source)
        kept = 0
        dropped = 0
        try:
            # Scraper limit is a fetch ceiling. The kept-count break below is
            # the real cap, so drops don't stop us short of the target.
            fetch_cap = None if limit is None else limit * 4
            scraper = scraper_cls(limit=fetch_cap)
            for item in scraper.scrape():
                if item.id in seen_ids:
                    continue
                keep, reason = prefilter(item, seen_hashes)
                if not keep:
                    dropped += 1
                    rejects[source][reason or "unknown"] += 1
                    continue
                append_item(source, item)
                seen_ids.add(item.id)
                kept += 1
                if kept % 50 == 0:
                    print(f"  ... {source} kept {kept}", flush=True)
                if limit is not None and kept >= limit:
                    print(f"  [cap] {source} reached {limit}", flush=True)
                    break
        except KeyboardInterrupt:
            print(f"  [interrupted] {source}: saved {kept} so far")
            break
        except Exception as e:  # noqa: BLE001  (isolate per-source failures)
            print(f"  [error] {source}: {e}")

        totals["kept"] += kept
        totals["dropped"] += dropped
        top_reasons = sorted(rejects[source].items(), key=lambda x: -x[1])[:5]
        print(f"  kept={kept}  dropped={dropped}  top_drops={top_reasons}")

    payload = {s: dict(r) for s, r in rejects.items()}
    if getattr(args, "no_merge", False):
        # Parallel runs must not rewrite the shared corpus or rejects log.
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        for source, reasons in payload.items():
            path = settings.raw_dir / f"rejects_{source}.json"
            path.write_text(json.dumps({source: reasons}, indent=2), encoding="utf-8")
        print("\n=== SCRAPE SUMMARY (no merge; parallel run) ===")
        print(f"  total kept   : {totals['kept']}")
        print(f"  total dropped: {totals['dropped']}")
        return

    # Persist rejects log + merge corpus
    write_rejects(payload)
    merged_path, merged_count = merge_corpus()

    print("\n=== SCRAPE SUMMARY ===")
    print(f"  total kept   : {totals['kept']}")
    print(f"  total dropped: {totals['dropped']}")
    print(f"  merged corpus: {merged_count} items -> {merged_path}")


def run_filter() -> None:
    from pipeline.noise_filter import filter_corpus
    from pipeline.normalize import normalize_corpus

    path, count = normalize_corpus()
    print(f"normalized {count} -> {path}", flush=True)
    dest, report = filter_corpus()
    print(
        f"filter kept {report['kept_before_dedup']} / {report['input']} -> {dest}",
        flush=True,
    )
    print("dropped:", report["dropped_by_reason"], flush=True)


def run_dedup() -> None:
    import json as _json

    from pipeline.dedup import dedup_corpus

    dest, stats = dedup_corpus()
    report_path = settings.interim_dir / "filter_report.json"
    report = {}
    if report_path.exists():
        report = _json.loads(report_path.read_text(encoding="utf-8"))
    report["dedup"] = stats
    report_path.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"dedup kept {stats['kept']} "
        f"(exact -{stats['exact_dropped']}, near -{stats['near_dropped']}) -> {dest}",
        flush=True,
    )


def run_classify() -> None:
    from pipeline.classify import classify_corpus

    path, count = classify_corpus()
    print(f"classified {count} -> {path}", flush=True)


def run_metrics() -> None:
    import json as _json

    from pipeline.metrics import write_aggregates
    from pipeline.store import write_database

    src = settings.processed_dir / "classified.jsonl"
    records = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    agg_path, aggregates = write_aggregates(records)
    db_path = write_database(records, aggregates)
    top = aggregates.get("top_problem") or {}
    print(
        f"metrics {len(records)} records -> {agg_path}",
        flush=True,
    )
    print(
        f"top {top.get('archetype')} opportunity={top.get('opportunity_score')} "
        f"band={top.get('band')} db={db_path}",
        flush=True,
    )


def run_extract(resume: bool = True) -> None:
    from pipeline.extract.extractor import extract_corpus

    extract_corpus(resume=resume)


def run_embed() -> None:
    from pipeline.embed import build_index

    stats = build_index()
    print(
        f"index {stats['chunks']} chunks from {stats['records']} records "
        f"model={stats['model']} collection={stats['collection']}",
        flush=True,
    )


def _not_implemented(stage: str) -> None:
    print(f"[stage '{stage}'] not implemented yet — will be built in a later phase.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Discovery Engine pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["scrape", "filter", "dedup", "extract", "classify",
                 "metrics", "embed", "label", "dashboard", "all"],
    )
    parser.add_argument(
        "--sources",
        default=",".join(SCRAPERS.keys()),
        help="comma-separated sources (default: all)",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny per-source cap")
    parser.add_argument("--limit", type=int, default=None, help="per-source cap")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="skip shared corpus merge (use when scraping sources in parallel)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip items already present in extracted_records.jsonl",
    )
    args = parser.parse_args()

    if args.stage == "scrape":
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        run_scrape(sources, args)
    elif args.stage == "filter":
        run_filter()
    elif args.stage == "dedup":
        run_dedup()
    elif args.stage == "extract":
        run_extract(resume=True)
    elif args.stage == "classify":
        run_classify()
    elif args.stage == "metrics":
        run_metrics()
    elif args.stage == "embed":
        run_embed()
    elif args.stage == "label":
        from pipeline.label import label_corpus

        label_corpus(limit=args.limit)
    elif args.stage == "dashboard":
        from pipeline.dashboard import write_dashboard

        write_dashboard()
    else:
        _not_implemented(args.stage)


if __name__ == "__main__":
    main()
