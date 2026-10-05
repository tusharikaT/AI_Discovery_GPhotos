"""Checkpointed cognitive extraction over the filtered corpus."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import ValidationError

from config.settings import settings
from pipeline.extract.llm_client import LLMClient, LLMError
from pipeline.extract.prompts import PROMPT_VERSION, build_prompt
from pipeline.extract.schema import Extraction

def _parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model did not return a JSON object")
    return json.loads(text[start : end + 1])


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _done_ids(path: Path) -> set[str]:
    return {row["id"] for row in _load_jsonl(path) if row.get("id")}


def _extract_one(client: LLMClient, record: dict) -> tuple[dict | None, dict, str | None]:
    """Return ``(saved_row_or_none, usage, error)``."""
    prompt = build_prompt(record.get("text") or "")
    prompt = f"prompt_version={PROMPT_VERSION}\n{prompt}"
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_hit": False}
    last_error = "unknown"
    for _attempt in range(2):
        try:
            raw, usage = client.chat(prompt)
        except (LLMError, TimeoutError, ValueError) as exc:
            return None, usage_total, str(exc)
        usage_total["input_tokens"] += int(usage.get("input_tokens") or 0)
        usage_total["output_tokens"] += int(usage.get("output_tokens") or 0)
        usage_total["cache_hit"] = bool(usage.get("cache_hit"))
        try:
            payload = _parse_json_object(raw)
            extraction = Extraction.model_validate(payload)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)[:400]
            client.discard(prompt)
            prompt = (
                f"{prompt}\n\nThe previous reply was invalid ({last_error}). "
                "Reply again with ONLY the JSON object."
            )
            continue
        quote = extraction.representative_quote
        source_text = record.get("text") or ""
        if quote and quote not in source_text:
            extraction.representative_quote = source_text[:240].strip()
        saved = {
            "id": record.get("id"),
            "source": record.get("source"),
            "source_url": record.get("source_url"),
            "source_detail": record.get("source_detail"),
            "timestamp": record.get("timestamp"),
            "rating": record.get("rating"),
            "text": source_text,
            "relevance_score": record.get("relevance_score"),
            "extraction": extraction.model_dump(),
        }
        return saved, usage_total, None
    return None, usage_total, last_error


def extract_corpus(resume: bool = True) -> dict:
    """Extract every filtered item. Returns a small run report."""
    settings.ensure_dirs()
    src = settings.interim_dir / "filtered_corpus.jsonl"
    dest = settings.processed_dir / "extracted_records.jsonl"
    quarantine_path = settings.processed_dir / "quarantine.jsonl"
    records = _load_jsonl(src)
    done = _done_ids(dest) if resume else set()
    pending = [row for row in records if row.get("id") not in done]
    print(
        f"extract {len(pending)} pending / {len(records)} total "
        f"(resume={resume}, already_done={len(done)})",
        flush=True,
    )

    client = LLMClient()
    kept = 0
    quarantined = 0
    cache_hits = 0
    input_tokens = 0
    output_tokens = 0
    workers = max(1, settings.llm_concurrency)

    with dest.open("a", encoding="utf-8") as out, quarantine_path.open("a", encoding="utf-8") as bad:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one, client, row): row for row in pending}
            finished = 0
            for future in as_completed(futures):
                record = futures[future]
                saved, usage, error = future.result()
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                if usage.get("cache_hit"):
                    cache_hits += 1
                if saved is None:
                    quarantined += 1
                    bad.write(json.dumps({
                        "id": record.get("id"),
                        "source": record.get("source"),
                        "error": error,
                    }, ensure_ascii=False) + "\n")
                    bad.flush()
                else:
                    kept += 1
                    out.write(json.dumps(saved, ensure_ascii=False) + "\n")
                    out.flush()
                finished += 1
                if finished % 50 == 0 or finished == len(pending):
                    print(
                        f"  ... extracted {finished}/{len(pending)} "
                        f"kept={kept} quarantined={quarantined}",
                        flush=True,
                    )

    total_done = len(done) + kept
    report = {
        "input": len(records),
        "already_done": len(done),
        "extracted_this_run": kept,
        "quarantined_this_run": quarantined,
        "pass_rate_this_run": round(kept / len(pending), 4) if pending else 1.0,
        "corpus_extracted": total_done,
        "cache_hits": cache_hits,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
    }
    report_path = settings.processed_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"extraction kept {kept}, quarantined {quarantined}, "
        f"pass_rate={report['pass_rate_this_run']}, "
        f"tokens in={input_tokens} out={output_tokens}",
        flush=True,
    )
    return report
