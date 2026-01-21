#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Retriever combining BM25 and Dense (FAISS) using RRF.

Usage:
  python -m src.jprag.retrieve --queries "query1" "query2" --method hybrid
"""

import argparse
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Helpers
from src.jprag.bm25_logic import BM25, char_ngrams
from src.jprag.normalize import normalize_query

def topk(scores: List[float], k: int) -> List[int]:
    # Returns indices of top-k scores
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

from src.jprag.alias import DOC_ALIASES

def get_doc_filter(query: str) -> Optional[List[str]]:
    """Returns list of doc_ids if query matches aliases, else None"""
    candidates = set()
    matched = False
    for alias, doc_ids in DOC_ALIASES.items():
        if alias in query:
            matched = True
            candidates.update(doc_ids)
    
    if matched:
        return list(candidates)
    return None

class Retriever:
    def __init__(self, 
                 method: str = "hybrid",
                 bm25_path: Path = Path("artifacts/bm25.pkl"),
                 bm25_chunk_ids_path: Path = Path("artifacts/bm25_chunk_ids.json"),
                 dense_idx_path: Path = Path("artifacts/dense/faiss.index"),
                 dense_chunk_ids_path: Path = Path("artifacts/dense/chunk_ids.json"),
                 dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.method = method
        self.bm25 = None
        self.bm25_chunk_ids = []
        self.dense_index = None
        self.dense_model = None
        self.dense_chunk_ids = []

        # Load BM25
        if method in ["bm25", "hybrid"]:
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.bm25_ngram = data.get("ngram", 3)
                if bm25_chunk_ids_path.exists():
                    self.bm25_chunk_ids = json.loads(bm25_chunk_ids_path.read_text(encoding="utf-8"))
            else:
                print(f"Warning: BM25 pickle not found at {bm25_path}")

        # Load Dense
        if method in ["dense", "hybrid"]:
            if dense_idx_path.exists():
                self.dense_index = faiss.read_index(str(dense_idx_path))
                self.dense_model = SentenceTransformer(dense_model_name)
                if dense_chunk_ids_path.exists():
                    self.dense_chunk_ids = json.loads(dense_chunk_ids_path.read_text(encoding="utf-8"))
            else:
                print(f"Warning: Dense index not found at {dense_idx_path}")

    def search_bm25(self, query: str, top_k: int = 100, filter_doc_ids: List[str] = None) -> Dict[str, float]:
        if not self.bm25:
            return {}
        # Normalize inside search to ensure consistency
        query = normalize_query(query)
        tokens = char_ngrams(query, n=self.bm25_ngram)
        scores = self.bm25.get_scores(tokens)
        
        res = {}
        # Optimization: only check top indices if N is large? 
        # For small N, iteration is fast.
        # But BM25.get_scores returns full list.
        # Let's prune by score > 0
        
        # Or just take top K.
        indices = topk(scores, top_k * 2 if filter_doc_ids else top_k)
        for i in indices:
            s = scores[i]
            if s > 0:
                cid = self.bm25_chunk_ids[i]
                # Filter check
                if filter_doc_ids:
                    doc_id = cid.split(":")[0]
                    if doc_id not in filter_doc_ids:
                        continue
                res[cid] = float(s)
        return res

    def search_dense(self, query: str, top_k: int = 100, filter_doc_ids: List[str] = None) -> Dict[str, float]:
        if not self.dense_index:
            return {}
        
        
        # Normalize: normalize_embeddings=True ensures consistent behavior
        q_emb = self.dense_model.encode([normalize_query(query)], convert_to_numpy=True, normalize_embeddings=True)
        # faiss.normalize_L2(q_emb) # Redundant if normalize_embeddings=True
        
        # If filtering, we might need to retrieve more props to filter down
        D, I = self.dense_index.search(q_emb, top_k * 5 if filter_doc_ids else top_k)
        
        res = {}
        row_indices = I[0]
        row_scores = D[0]
        
        for idx, score in zip(row_indices, row_scores):
            if idx < 0: continue
            if idx < len(self.dense_chunk_ids):
                cid = self.dense_chunk_ids[idx]
                if filter_doc_ids:
                    doc_id = cid.split(":")[0]
                    if doc_id not in filter_doc_ids:
                        continue
                res[cid] = float(score)
        return res

    def fusion_rrf(self, results_list: List[List[Tuple[str, float]]], k: int = 60, weights: List[float] = None) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion with weights.
        score = sum( w_i * (1 / (k + rank_i)) )
        """
        if weights is None:
            weights = [1.0] * len(results_list)
        
        fused_scores = {}
        for r_list, w in zip(results_list, weights):
            for rank, (cid, _) in enumerate(r_list):
                if cid not in fused_scores:
                    fused_scores[cid] = 0.0
                fused_scores[cid] += w * (1.0 / (k + rank + 1))
        
        sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        # Auto-detect Doc Filter
        doc_filter = get_doc_filter(query)
        if doc_filter:
            # print(f"  [Filter] Restricting to: {doc_filter}")
            pass

        if self.method == "bm25":
            res = self.search_bm25(query, top_k=k*2, filter_doc_ids=doc_filter)
            return sorted(res.items(), key=lambda x: x[1], reverse=True)[:k]
            
        elif self.method == "dense":
            res = self.search_dense(query, top_k=k*2, filter_doc_ids=doc_filter)
            return sorted(res.items(), key=lambda x: x[1], reverse=True)[:k]
            
        elif self.method == "hybrid":
            # For Reciprocal Rank Fusion, we need deeper pools
            b_res = self.search_bm25(query, top_k=100, filter_doc_ids=doc_filter)
            d_res = self.search_dense(query, top_k=100, filter_doc_ids=doc_filter)
            # Weighted RRF: BM25 (2.0) > Dense (1.0) because on this dataset BM25 is much stronger.
            return self.fusion_rrf([sorted(b_res.items(), key=lambda x: x[1], reverse=True), 
                                    sorted(d_res.items(), key=lambda x: x[1], reverse=True)], 
                                   k=60, weights=[2.0, 1.0])[:k]
            
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--method", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    retriever = Retriever(method=args.method)
    query = normalize_query(args.query)
    results = retriever.search(query, k=args.k)

    print(f"\nQuery: {args.query} (Normalized: {query}) (Method: {args.method})")
    for i, (cid, score) in enumerate(results, start=1):
        print(f"[{i}] {cid} | score={score:.4f}")

if __name__ == "__main__":
    main()
