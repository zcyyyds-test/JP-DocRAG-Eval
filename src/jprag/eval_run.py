#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Evaluation Script.
Runs evaluation for BM25, Dense, and Hybrid retrievers.

Outputs:
  reports/results.csv: Summary metrics
  reports/failure_cases.md: Detailed analysis of failures
  reports/eval_details.jsonl: Full per-query log
"""

import argparse
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict

from src.jprag.retrieve import Retriever
from src.jprag.eval_metrics import calc_metrics_at_k, get_gold_evidence, check_hit
from src.jprag.bm25_logic import char_ngrams # Needed if we want to debug tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_chunk_map(path: Path) -> Dict[str, Dict]:
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m[obj["chunk_id"]] = obj
    return m

def run_eval(args):
    gold_path = Path(args.gold)
    chunks_path = Path(args.chunks_jsonl)
    
    logger.info(f"Loading gold data from {gold_path}")
    gold_data = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_data.append(json.loads(line))
                
    logger.info(f"Loading chunk map from {chunks_path}")
    chunk_map = load_chunk_map(chunks_path)
    
    methods = args.methods.split(",")
    all_metrics = []
    
    # Prepare detailed log
    details_path = Path("reports/eval_details.jsonl")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    f_details = details_path.open("w", encoding="utf-8")
    
    # Prepare failure cases content
    failures_content = ["# Failure Cases Analysis\n"]
    
    for method in methods:
        method = method.strip()
        logger.info(f"--- Evaluating Method: {method} ---")
        
        retriever = Retriever(method=method)
        results = []
        
        failures_content.append(f"## Method: {method}\n")
        
        for i, item in enumerate(gold_data):
            qid = item.get("qid", str(i))
            query = item["question"]
            
            # Retrieve
            search_res = retriever.search(query, k=args.k) # List of (cid, score)
            
            # Format preds for metrics
            preds = []
            for cid, score in search_res:
                meta = chunk_map.get(cid, {})
                preds.append({
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id"),
                    "page": meta.get("page"),
                    "score": score,
                    "text": meta.get("text", "")
                })
            
            # Log detail
            log_entry = {
                "method": method,
                "qid": qid,
                "query": query,
                "gold": item["gold"],
                "preds": [{"chunk_id": p["chunk_id"], "score": p["score"], "doc_id": p["doc_id"], "page": p["page"]} for p in preds]
            }
            f_details.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            results.append({"qid": qid, "preds": preds})
            
            # Check for failure (Top-1 miss?)
            # Let's log failures for Recall@5 (or args.k if small)
            g_chunk_ids, g_doc_pages = get_gold_evidence(item)
            
            # Find rank of first hit
            hit_rank = None
            for r, p in enumerate(preds, 1):
                if check_hit(p, g_chunk_ids, g_doc_pages):
                    hit_rank = r
                    break
            
            # If miss or rank > 5, detailed log
            if hit_rank is None or hit_rank > 5:
                failures_content.append(f"### {qid}: {query}\n")
                failures_content.append(f"- **Gold**: {item['gold']}\n")
                failures_content.append(f"- **Hit Rank**: {hit_rank if hit_rank else 'Not in Top-' + str(args.k)}\n")
                failures_content.append(f"- **Top Predictions**:\n")
                for j, p in enumerate(preds[:3], 1):
                    snip = p['text'].replace('\n', ' ')[:100] + "..."
                    failures_content.append(f"  {j}. [{p['score']:.4f}] {p['doc_id']} p.{p['page']} ({p['chunk_id']}): {snip}\n")
                failures_content.append("\n")

        # Compute metrics
        m = calc_metrics_at_k(results, gold_data, k_list=[1, 3, 5, 10])
        m_flat = {"method": method}
        m_flat.update(m)
        all_metrics.append(m_flat)
        logger.info(f"Metrics: {m}")

    f_details.close()
    
    # Save failure cases
    with open("reports/failure_cases.md", "w", encoding="utf-8") as f:
        f.write("".join(failures_content))
    logger.info("Saved reports/failure_cases.md")
    
    # Save CSV
    csv_path = Path("reports/results.csv")
    fieldnames = ["method"] + [f"Recall@{k}" for k in [1, 3, 5, 10]] + [f"MRR@{k}" for k in [1, 3, 5, 10]]
    
    # Check if we need to append or write new. 
    # For now, let's just write new table for this run.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            # Filter keys that are in fieldnames
            w_row = {k: row.get(k, 0.0) for k in fieldnames}
            writer.writerow(w_row)
            
    logger.info(f"Saved {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default="data/qa/gold_qa.jsonl")
    ap.add_argument("--chunks_jsonl", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--methods", type=str, default="bm25,dense,hybrid", help="Comma separated methods")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    
    run_eval(args)

if __name__ == "__main__":
    main()
