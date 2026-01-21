#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/qa/gold_qa.jsonl")
    ap.add_argument("--pages", default="artifacts/pages.jsonl")  # 用未clean的也行
    args = ap.parse_args()

    gold = list(read_jsonl(Path(args.gold)))

    # build set of valid (doc_id, page)
    valid = set()
    for x in read_jsonl(Path(args.pages)):
        valid.add((x["doc_id"], int(x["page"])))

    bad = []
    total = 0
    for ex in gold:
        for g in ex.get("gold", []):
            total += 1
            key = (g.get("doc_id"), int(g.get("page")))
            if key not in valid:
                bad.append({"qid": ex.get("qid"), "question": ex.get("question"), "gold": g})

    print(f"[OK] valid (doc_id,page) pairs: {len(valid)}")
    print(f"[OK] gold evidences: {total}")
    print(f"[WARN] invalid evidences: {len(bad)}")
    for b in bad[:20]:
        print(" -", b)

if __name__ == "__main__":
    main()
