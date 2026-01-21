#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, pickle
from pathlib import Path
from typing import Dict, List, Tuple

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

from src.jprag.bm25_logic import char_ngrams
# Ensure BM25 class is available for pickle loading even if we don't instantiate it
from src.jprag.bm25_logic import BM25

def load_chunk_map(chunks_path: Path) -> Dict[str, Dict]:
    m = {}
    for obj in read_jsonl(chunks_path):
        m[obj["chunk_id"]] = obj
    return m

def topk(scores: List[float], k: int) -> List[int]:
    # returns indices of top-k scores (descending)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25_pkl", type=str, default="artifacts/bm25.pkl")
    ap.add_argument("--chunk_ids", type=str, default="artifacts/bm25_chunk_ids.json")
    ap.add_argument("--chunks_jsonl", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--snippet", type=int, default=220)
    args = ap.parse_args()

    bm25_pkl = Path(args.bm25_pkl)
    chunk_ids_path = Path(args.chunk_ids)
    chunks_path = Path(args.chunks_jsonl)

    data = pickle.loads(bm25_pkl.read_bytes())
    bm25 = data["bm25"]
    ngram = int(data.get("ngram", 3))

    chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
    chunk_map = load_chunk_map(chunks_path)

    q_tokens = char_ngrams(args.query, n=ngram)
    scores = bm25.get_scores(q_tokens)
    idxs = topk(scores, args.k)

    print(f"\nQuery: {args.query}\nTop-{args.k} results:\n")
    for rank, i in enumerate(idxs, start=1):
        cid = chunk_ids[i]
        meta = chunk_map.get(cid, {})
        doc_id = meta.get("doc_id", "?")
        source = meta.get("source", "?")
        page = meta.get("page", "?")
        text = (meta.get("text", "") or "").replace("\n", " ")
        snip = text[: args.snippet] + ("..." if len(text) > args.snippet else "")
        print(f"[{rank}] score={scores[i]:.4f} | {doc_id} | {source} | page={page} | {cid}")
        print(f"    {snip}\n")

if __name__ == "__main__":
    main()
