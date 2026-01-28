#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BM25 Indexing Script.
Reads artifacts/chunks.jsonl, builds BM25 index, and saves to artifacts/bm25.pkl.
"""

import argparse
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict

from src.jprag.bm25_logic import BM25, char_ngrams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--out_pkl", type=str, default="artifacts/bm25.pkl")
    ap.add_argument("--out_ids", type=str, default="artifacts/bm25_chunk_ids.json")
    ap.add_argument("--ngram", type=int, default=3)
    args = ap.parse_args()

    chunks_path = Path(args.chunks_jsonl)
    out_pkl = Path(args.out_pkl)
    out_ids = Path(args.out_ids)

    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return

    logger.info(f"Reading chunks from {chunks_path}...")
    
    chunk_ids = []
    corpus_tokens = []
    
    count = 0
    for obj in read_jsonl(chunks_path):
        text = obj.get("text", "")
        cid = obj["chunk_id"]
        
        # Tokenize (Char N-grams)
        tokens = char_ngrams(text, n=args.ngram)
        
        chunk_ids.append(cid)
        corpus_tokens.append(tokens)
        count += 1

    logger.info(f"Building BM25 index for {count} chunks (ngram={args.ngram})...")
    bm25 = BM25(corpus_tokens)

    logger.info(f"Saving to {out_pkl}...")
    data = {
        "bm25": bm25,
        "ngram": args.ngram
    }
    
    with out_pkl.open("wb") as f:
        pickle.dump(data, f)
        
    logger.info(f"Saving chunk IDs to {out_ids}...")
    out_ids.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    logger.info("Done.")

if __name__ == "__main__":
    main()
