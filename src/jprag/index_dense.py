#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dense indexing using SentenceTransformers and FAISS.

Input:
  artifacts/chunks.jsonl

Output:
  artifacts/dense/embeddings.npy
  artifacts/dense/faiss.index
  artifacts/dense/index_meta.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("Please install requirements: pip install sentence-transformers faiss-cpu numpy")
    raise e

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--out_dir", type=str, default="artifacts/dense")
    ap.add_argument("--model_name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    chunks_path = Path(args.chunks_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return

    logger.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # 1. Load chunks (text only)
    logger.info(f"Reading chunks from {chunks_path}")
    texts = []
    chunk_ids = []
    
    # We might want to persist chunk_ids order for retrieval mapping
    # Assuming corpus is static during indexing/retrieval cycle
    for obj in read_jsonl(chunks_path):
        # Format text for embedding: combine title/meta if beneficial, here simple text
        # You might want to prepend title if available in metadata
        text = obj.get("text", "")
        texts.append(text)
        chunk_ids.append(obj["chunk_id"])

    if not texts:
        logger.warning("No texts found.")
        return

    logger.info(f"Encoding {len(texts)} chunks...")
    # Normalize for cosine similarity if using IP index
    # Ensure explicit normalization
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    
    # faiss.normalize_L2(embeddings) # Already normalized by sentence-transformers

    # 2. Build FAISS index
    d = embeddings.shape[1]
    logger.info(f"Building FAISS index (d={d})...")
    # IndexFlatIP is exact inner product search. Since normalized, it's cosine similarity.
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # 3. Save artifacts
    emb_path = out_dir / "embeddings.npy"
    idx_path = out_dir / "faiss.index"
    meta_path = out_dir / "index_meta.json"
    ids_path = out_dir / "chunk_ids.json"

    np.save(emb_path, embeddings)
    faiss.write_index(index, str(idx_path))
    
    meta = {
        "model_name": args.model_name,
        "count": len(texts),
        "dimension": d,
        "metric": "inner_product (normalized)"
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Done.")
    logger.info(f"  Embeddings -> {emb_path}")
    logger.info(f"  Index      -> {idx_path}")
    logger.info(f"  Chunk IDs  -> {ids_path}")

if __name__ == "__main__":
    main()
