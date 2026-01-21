#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Sweep Script for RAG Pipeline.

Iterates over configurations (Chunking parameters, Retrieval parameters)
and logs performance metrics.

Usage:
  python3 src/jprag/sweep.py --ablate_chunking
"""

import argparse
import subprocess
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd # Optional, but good for collecting results

def run_cmd(cmd: List[str], log_path: Path = None):
    """Run a subprocess command."""
    print(f"Running: {' '.join(cmd)}")
    if log_path:
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)

def parse_metrics(csv_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read results.csv and merge with config info."""
    if not csv_path.exists():
        return []
    
    # Read CSV
    # method,Recall@1,Recall@3,Recall@5,Recall@10,MRR@1,MRR@3,MRR@5,MRR@10
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Flatten config into row
            new_row = dict(row)
            for k, v in config.items():
                new_row[f"param_{k}"] = v
            rows.append(new_row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate_chunking", action="store_true", help="If set, re-run chunking and indexing.")
    ap.add_argument("--fast", action="store_true", help="Run a tiny grid for testing.")
    args = ap.parse_args()
    
    # Define Grid
    # 1. Chunking Params
    if args.ablate_chunking:
        if args.fast:
             chunk_configs = [
                {"max_chars": 400, "overlap": 50, "min_chars": 50},
                {"max_chars": 800, "overlap": 100, "min_chars": 100},
            ]
        else:
            chunk_configs = [
                {"max_chars": 400, "overlap": 50, "min_chars": 50},
                {"max_chars": 800, "overlap": 100, "min_chars": 100},
                {"max_chars": 1200, "overlap": 150, "min_chars": 200}, # Baseline
            ]
    else:
        # Default (No re-chunking, just assume current chunks are baseline or whatever is there)
        # We can't really sweep chunking if we don't re-run it.
        # But for now let's just use one dummy config if not ablating.
        chunk_configs = [{"max_chars": "CURRENT", "overlap": "CURRENT", "min_chars": "CURRENT"}]

    # 2. Retrieval Params (Currently we just evaluate all methods: bm25, dense, hybrid)
    # But maybe we want to ablate Dense vs BM25 weights? 
    # For now, eval_run.py runs all 3. So we just collect those.
    
    results = []
    
    sweep_dir = Path("reports/sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    for i, c_conf in enumerate(chunk_configs):
        print(f"\n=== Configuration {i+1}/{len(chunk_configs)}: {c_conf} ===")
        
        # 1. Chunking
        if args.ablate_chunking:
            print(">> Running Chunking...")
            run_cmd([
                "python3", "src/jprag/chunk.py",
                "--max_chars", str(c_conf["max_chars"]),
                "--overlap_chars", str(c_conf["overlap"]),
                "--min_chars", str(c_conf["min_chars"])
            ], log_path=sweep_dir / f"chunk_{i}.log")
            
            # 2. Indexing
            print(">> Running BM25 Indexing...")
            run_cmd(["python3", "src/jprag/index_bm25.py"], log_path=sweep_dir / f"index_bm25_{i}.log")
            
            print(">> Running Dense Indexing...")
            run_cmd(["python3", "src/jprag/index_dense.py"], log_path=sweep_dir / f"index_dense_{i}.log")
            
        # 3. Evaluation
        print(">> Running Evaluation...")
        # Point to the expanded QA
        run_cmd([
            "python3", "src/jprag/eval_run.py",
            "--gold", "data/qa/generated_qa.jsonl",
            "--k", "10"
        ], log_path=sweep_dir / f"eval_{i}.log")
        
        # 4. Collect results
        res_csv = Path("reports/results.csv")
        batch_results = parse_metrics(res_csv, c_conf)
        results.extend(batch_results)
        
    # Save Final Sweep Report
    final_csv = Path("reports/sweep_results.csv")
    if results:
        headers = list(results[0].keys())
        with final_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nSweep Completed. Results saved to {final_csv}")
        
        # Print summary
        try:
            # Simple pretty print if tabulate exists or just simple loop
            print("\nSummary (Method | Recall@10 | Params):")
            for r in results:
                print(f"{r.get('method')} | {r.get('Recall@10')} | {r.get('param_max_chars')}/{r.get('param_overlap')}")
        except:
            pass
            
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()
