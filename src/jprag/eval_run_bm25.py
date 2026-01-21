#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.jprag.bm25_logic import BM25, char_ngrams

import argparse, json, pickle, csv
from pathlib import Path
from typing import Dict, List, Tuple

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def topk(scores: List[float], k: int) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

def load_chunk_map(chunks_path: Path) -> Dict[str, Dict]:
    m = {}
    for obj in read_jsonl(chunks_path):
        m[obj["chunk_id"]] = obj
    return m

def hit_at_k(pred_chunk_ids: List[str], chunk_map: Dict[str, Dict], gold: List[Dict], k: int) -> Tuple[bool, int]:
    """
    Return (hit?, first_rank) where first_rank is 1-indexed, or 0 if no hit.
    A prediction hits if:
      - chunk_id matches any gold chunk_id (if provided), OR
      - (doc_id, page) matches any gold evidence
    """
    gold_chunk_ids = {g.get("chunk_id") for g in gold if g.get("chunk_id")}
    gold_doc_pages = {(g.get("doc_id"), int(g.get("page"))) for g in gold if g.get("doc_id") and g.get("page") is not None}

    for r, cid in enumerate(pred_chunk_ids[:k], start=1):
        if cid in gold_chunk_ids:
            return True, r
        meta = chunk_map.get(cid, {})
        dp = (meta.get("doc_id"), int(meta.get("page")))
        if dp in gold_doc_pages:
            return True, r
    return False, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default="data/qa/gold_qa.jsonl")
    ap.add_argument("--bm25_pkl", type=str, default="artifacts/bm25.pkl")
    ap.add_argument("--chunk_ids", type=str, default="artifacts/bm25_chunk_ids.json")
    ap.add_argument("--chunks_jsonl", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--ks", type=str, default="1,3,5,10")
    ap.add_argument("--out_csv", type=str, default="reports/results.csv")
    ap.add_argument("--out_fail", type=str, default="reports/failure_cases.md")
    ap.add_argument("--top_show", type=int, default=3)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    gold_path = Path(args.gold)
    bm25_pkl = Path(args.bm25_pkl)
    chunk_ids_path = Path(args.chunk_ids)
    chunks_path = Path(args.chunks_jsonl)

    data = pickle.loads(bm25_pkl.read_bytes())
    bm25 = data["bm25"]
    ngram = int(data.get("ngram", 3))
    chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
    chunk_map = load_chunk_map(chunks_path)

    qa = list(read_jsonl(gold_path))
    if not qa:
        raise ValueError("gold_qa.jsonl is empty.")

    # metrics accumulators
    hit_cnt = {k: 0 for k in ks}
    mrr_sum = {k: 0.0 for k in ks}
    failures = []

    for ex in qa:
        qid = ex["qid"]
        q = ex["question"]
        gold = ex.get("gold", [])

        q_tokens = char_ngrams(q, n=ngram)
        scores = bm25.get_scores(q_tokens)
        idxs = topk(scores, max(ks))
        pred_chunk_ids = [chunk_ids[i] for i in idxs]

        # compute metrics
        for k in ks:
            hit, first_rank = hit_at_k(pred_chunk_ids, chunk_map, gold, k)
            if hit:
                hit_cnt[k] += 1
                mrr_sum[k] += 1.0 / first_rank

        # failure for k=max(ks)
        hit_max, _ = hit_at_k(pred_chunk_ids, chunk_map, gold, max(ks))
        if not hit_max:
            top3 = pred_chunk_ids[: args.top_show]
            top3_snip = []
            for cid in top3:
                meta = chunk_map.get(cid, {})
                t = (meta.get("text", "") or "").replace("\n", " ")
                top3_snip.append({
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id"),
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "snippet": (t[:220] + ("..." if len(t) > 220 else ""))
                })
            failures.append({"qid": qid, "question": q, "gold": gold, "top": top3_snip})

    n = len(qa)
    recalls = {k: hit_cnt[k] / n for k in ks}
    mrrs = {k: mrr_sum[k] / n for k in ks}

    # write results.csv (append-friendly)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["retriever","ngram","N"] + [f"Recall@{k}" for k in ks] + [f"MRR@{k}" for k in ks]
    row = ["bm25", str(ngram), str(n)] + [f"{recalls[k]:.4f}" for k in ks] + [f"{mrrs[k]:.4f}" for k in ks]
    write_header = not out_csv.exists()
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    # write failure_cases.md (top-3 + gold page)
    out_fail = Path(args.out_fail)
    out_fail.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Failure cases (retriever=bm25, ngram={ngram})\n")
    lines.append(f"- N={n}\n")
    lines.append(f"- Failures@{max(ks)}: {len(failures)}\n\n")

    for fc in failures:
        lines.append(f"## {fc['qid']}\n")
        lines.append(f"**Q:** {fc['question']}\n\n")
        lines.append("**Gold evidence:**\n")
        for g in fc["gold"]:
            lines.append(f"- doc_id={g.get('doc_id')} page={g.get('page')} chunk_id={g.get('chunk_id','')}\n")
        lines.append("\n**Top predictions:**\n")
        for t in fc["top"]:
            lines.append(f"- {t['doc_id']} | {t['source']} | page={t['page']} | {t['chunk_id']}\n")
            lines.append(f"  - {t['snippet']}\n")
        lines.append("\n")
    out_fail.write_text("".join(lines), encoding="utf-8")

    print("[OK] results appended to:", out_csv)
    print("[OK] failure cases ->:", out_fail)
    print("Recalls:", recalls)
    print("MRRs   :", mrrs)

if __name__ == "__main__":
    main()
