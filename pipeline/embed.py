"""Phase 8 retrieval index.

Steps, in order: ingest, chunk, probe the embedding model, embed, store.
Queries use QUERY_PREFIX inside ``search``. Stored passages do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.settings import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "discovery"
CORPUS_NAME = "rag_corpus.jsonl"
CHUNKS_NAME = "rag_chunks.jsonl"
VECTORS_NAME = "rag_vectors.npy"
IDS_NAME = "rag_vector_ids.json"


def _processed(name: str) -> Path:
    return settings.processed_dir / name


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _words(text: str) -> list[str]:
    return (text or "").split()


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = _words(text)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    pieces = []
    start = 0
    while start < len(words):
        piece = words[start : start + size]
        if not piece:
            break
        pieces.append(" ".join(piece))
        if start + size >= len(words):
            break
        start += step
    return pieces


def _join(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value if item)
    return "" if value is None else str(value)


def _kept(record: dict) -> bool:
    pains = (record.get("label") or {}).get("pains") or []
    return bool(pains) and "does_not_fit" not in pains


def ingest() -> dict:
    """Step 1. Retrieval reviews only. No chunking and no model."""
    settings.ensure_dirs()
    source = settings.processed_dir / "labeled.jsonl"
    rows = []
    for record in _read_jsonl(source):
        if not _kept(record):
            continue
        label = record.get("label") or {}
        rows.append({
            "id": record.get("id"),
            "source": record.get("source"),
            "source_url": record.get("source_url"),
            "source_detail": record.get("source_detail"),
            "timestamp": record.get("timestamp"),
            "rating": record.get("rating"),
            "text": record.get("text") or "",
            "label": {
                "pains": label.get("pains") or [],
                "screen": label.get("screen") or "",
                "looking_for": label.get("looking_for") or "",
                "still_remembers": label.get("still_remembers") or [],
                "missing": label.get("missing") or [],
                "who": label.get("who") or "",
                "did_next": label.get("did_next") or "",
                "sentence": label.get("sentence") or "",
            },
        })
    _write_jsonl(_processed(CORPUS_NAME), rows)
    unfit = sum(1 for row in rows if "does_not_fit" in (row["label"]["pains"]))
    return {"records": len(rows), "does_not_fit": unfit}


def chunk() -> dict:
    """Step 2. Word windows. No model."""
    settings.ensure_dirs()
    size = settings.embedding_chunk_size
    overlap = settings.embedding_chunk_overlap
    records = _read_jsonl(_processed(CORPUS_NAME))
    chunks = []
    split = 0
    missing = 0
    for record in records:
        pieces = _word_chunks(record.get("text") or "", size, overlap)
        if not pieces:
            missing += 1
            continue
        if len(pieces) > 1:
            split += 1
        label = record.get("label") or {}
        for index, piece in enumerate(pieces):
            chunks.append({
                "record_id": record.get("id"),
                "chunk_index": index,
                "text": piece,
                "source": record.get("source"),
                "source_url": record.get("source_url"),
                "source_detail": record.get("source_detail"),
                "timestamp": record.get("timestamp"),
                "rating": record.get("rating"),
                "pains": label.get("pains") or [],
                "screen": label.get("screen") or "",
                "looking_for": label.get("looking_for") or "",
                "still_remembers": label.get("still_remembers") or [],
                "missing": label.get("missing") or [],
                "who": label.get("who") or "",
                "did_next": label.get("did_next") or "",
                "sentence": label.get("sentence") or "",
            })
    _write_jsonl(_processed(CHUNKS_NAME), chunks)
    return {
        "records": len(records),
        "chunks": len(chunks),
        "split_reviews": split,
        "missing_text": missing,
        "chunk_words": size,
    }


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def probe_model() -> dict:
    """Step 3. Encode one sentence. Do not read or write the chunk file."""
    before = _processed(CHUNKS_NAME).stat().st_mtime_ns
    model = load_model()
    vector = model.encode(
        "a photo of a receipt I cannot find",
        normalize_embeddings=True,
    )
    after = _processed(CHUNKS_NAME).stat().st_mtime_ns
    array = np.asarray(vector)
    return {
        "dimensions": int(array.shape[-1]),
        "chunks_unchanged": before == after,
        "prefix": QUERY_PREFIX,
    }


def embed_chunks() -> dict:
    """Step 4. Vectors on disk. No Chroma write."""
    chunks = _read_jsonl(_processed(CHUNKS_NAME))
    model = load_model()
    vectors = model.encode(
        [row["text"] for row in chunks],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    np.save(_processed(VECTORS_NAME), matrix)
    ids = [f"{row['record_id']}#{row['chunk_index']}" for row in chunks]
    _processed(IDS_NAME).write_text(json.dumps(ids), encoding="utf-8")
    return {
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "chunks": len(chunks),
        "min_norm": float(norms.min()) if len(norms) else 0.0,
        "max_norm": float(norms.max()) if len(norms) else 0.0,
    }


def _chroma():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings.ensure_dirs()
    return chromadb.PersistentClient(
        path=str(settings.vectorstore_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def store_vectors() -> dict:
    """Step 5. Load the saved matrix. Do not embed again."""
    chunks = _read_jsonl(_processed(CHUNKS_NAME))
    matrix = np.load(_processed(VECTORS_NAME))
    ids = json.loads(_processed(IDS_NAME).read_text(encoding="utf-8"))
    if not (len(chunks) == len(ids) == len(matrix)):
        raise RuntimeError("chunk, id, and vector counts differ")
    client = _chroma()
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    batch = 256
    for start in range(0, len(ids), batch):
        stop = start + batch
        collection.upsert(
            ids=ids[start:stop],
            documents=[row["text"] for row in chunks[start:stop]],
            embeddings=matrix[start:stop].tolist(),
            metadatas=[_metadata(row) for row in chunks[start:stop]],
        )
    reloaded = _chroma().get_collection(COLLECTION)
    return {"stored": int(reloaded.count()), "chunks": len(chunks)}


def _metadata(row: dict) -> dict:
    rating = row.get("rating")
    return {
        "source": _join(row.get("source")),
        "date": _join(row.get("timestamp"))[:10],
        "url": _join(row.get("source_url")),
        "record_id": _join(row.get("record_id")),
        "chunk_index": int(row.get("chunk_index") or 0),
        "pains": _join(row.get("pains")),
        "screen": _join(row.get("screen")),
        "looking_for": _join(row.get("looking_for")),
        "remembers": _join(row.get("still_remembers")),
        "missing": _join(row.get("missing")),
        "who": _join(row.get("who")),
        "did_next": _join(row.get("did_next")),
        "sentence": _join(row.get("sentence"))[:500],
        "rating": -1 if rating is None else int(rating),
        "source_detail": _join(row.get("source_detail")),
    }


def build_index() -> dict:
    """Run steps 1–5. Used by ``python -m pipeline.run --stage embed``."""
    ingested = ingest()
    chunked = chunk()
    probed = probe_model()
    embedded = embed_chunks()
    stored = store_vectors()
    return {
        "records": ingested["records"],
        "chunks": stored["stored"],
        "model": settings.embedding_model,
        "collection": COLLECTION,
        "dimensions": probed["dimensions"],
        "embedded_rows": embedded["rows"],
        "split_reviews": chunked["split_reviews"],
    }


def search(query: str, n_results: int = 15) -> list[dict]:
    """Nearest chunks. The query prefix is applied here only."""
    client = _chroma()
    collection = client.get_collection(COLLECTION)
    model = load_model()
    vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()
    found = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    docs = (found.get("documents") or [[]])[0]
    metas = (found.get("metadatas") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]
    for document, metadata, distance in zip(docs, metas, distances):
        hits.append({
            "document": document,
            "metadata": metadata or {},
            "distance": float(distance),
        })
    return hits


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    actions = {
        "ingest": ingest,
        "chunk": chunk,
        "probe": probe_model,
        "embed": embed_chunks,
        "store": store_vectors,
        "all": build_index,
    }
    if step not in actions:
        raise SystemExit(f"unknown step {step}")
    print(actions[step]())
