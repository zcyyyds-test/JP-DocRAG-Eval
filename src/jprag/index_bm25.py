#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BM25 indexing for Japanese using character n-grams (no tokenizer needed).

Input:
  artifacts/chunks.jsonl  (each line has chunk_id, doc_id, source, page, text)

Output:
  artifacts/bm25.pkl
  artifacts/bm25_chunk_ids.json
"""

from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

from src.jprag.bm25_logic import BM25, char_ngrams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_chunks", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--out_pkl", type=str, default="artifacts/bm25.pkl")
    ap.add_argument("--out_ids", type=str, default="artifacts/bm25_chunk_ids.json")
    ap.add_argument("--ngram", type=int, default=3)
    args = ap.parse_args()

    in_chunks = Path(args.in_chunks)
    out_pkl = Path(args.out_pkl)
    out_ids = Path(args.out_ids)

    chunk_ids: List[str] = []
    corpus_tokens: List[List[str]] = []

    for obj in tqdm(read_jsonl(in_chunks), desc="Tokenize chunks"):
        cid = obj["chunk_id"]
        txt = obj.get("text", "") or ""
        toks = char_ngrams(txt, n=args.ngram)
        chunk_ids.append(cid)
        corpus_tokens.append(toks)

    bm25 = BM25(corpus_tokens)

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump({"bm25": bm25, "ngram": args.ngram}, f)

    out_ids.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] BM25 index -> {out_pkl}")
    print(f"[OK] chunk_ids  -> {out_ids} (N={len(chunk_ids)})")

if __name__ == "__main__":
    main()
