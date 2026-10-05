# AI-Powered Discovery Engine

An AI system that analyzes public user feedback about **photo retrieval in Google Photos** at scale, and turns it into **comparable retrieval-problem opportunity areas backed by cited, real-user evidence**.

> Goes beyond sentiment/summarization: it extracts cognitive memory anchors, tracks search-formulation degradation, categorizes retrieval-failure archetypes, and mathematically compares opportunity areas.

Internal planning/spec docs live in `docs/` (not committed): `context.md` (what & why), `architecture.md` (how), and `implementation_plan.md` (phased build).

---

## Architecture at a glance

**Offline pipeline → committed artifacts → read-only Streamlit app.**

```
scrape -> filter -> extract (LLM) -> metrics/opportunity -> embed
      \-> SQLite + Chroma + aggregates.json ->  Streamlit dashboard (5 views + RAG copilot)
```

## Project layout

```
config/     env-driven settings + pre-filter lexicons
data/        raw / interim / processed / db / vectorstore  (artifacts committed)
pipeline/    sources (scrapers), normalize, noise_filter, dedup, extract, classify, metrics, embed, store, run
app/         streamlit_app, data_access, rag, charts, views/
tests/       unit tests (filter / dedup / metrics)
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env      # optional; fill keys when needed (Phase 3+)
```

Verify config loads:

```bash
python -c "from config.settings import settings; print(settings.app_title)"
```

## Running the pipeline (later phases)

```bash
python -m pipeline.run --stage scrape      # Phase 1
python -m pipeline.run --stage filter      # Phase 2
python -m pipeline.run --stage dedup       # Phase 2
python -m pipeline.run --stage extract     # Phase 3
python -m pipeline.run --stage classify    # Phase 4
python -m pipeline.run --stage metrics     # Phase 4
python -m pipeline.run --stage embed       # Phase 5
python -m pipeline.run --stage all
```

## Launching the dashboard (later phases)

```bash
streamlit run app/streamlit_app.py
```

---

*Build status: **Phase 0 — Scaffold & Configuration** complete. Subsequent phases proceed only on explicit instruction.*
