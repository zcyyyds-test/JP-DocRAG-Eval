#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunking: pages_clean.jsonl -> chunks.jsonl

Input JSONL:
  {"doc_id","source","page","text"}

Output JSONL:
  {
    "chunk_id": "doc001:p012:c003",
    "doc_id": "doc001",
    "source": "xxx.pdf",
    "page": 12,
    "text": "...",
    "char_len": 1234
  }
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List

SENT_SPLIT = re.compile(r"(?<=[。！？\.\!\?])\s*")
MULTI_NL = re.compile(r"\n{2,}")

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def split_to_units(text: str) -> List[str]:
    # prefer paragraphs first
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in MULTI_NL.split(text) if p.strip()]
    units: List[str] = []
    for p in paras:
        # if paragraph too long, split by sentence punctuation
        if len(p) > 1500:
            sents = [s.strip() for s in SENT_SPLIT.split(p) if s.strip()]
            units.extend(sents if sents else [p])
        else:
            units.append(p)
    return units

def make_chunks_from_units(units: List[str], max_chars: int, overlap_chars: int, min_chars: int) -> List[str]:
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        t = buf.strip()
        if len(t) >= min_chars:
            chunks.append(t)
        buf = ""

    for u in units:
        if not buf:
            buf = u
            continue
        # try append
        cand = buf + "\n\n" + u
        if len(cand) <= max_chars:
            buf = cand
        else:
            flush()
            buf = u

    flush()

    # add overlap (character-level tail overlap)
    if overlap_chars > 0 and len(chunks) >= 2:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            overlapped.append((tail + "\n" + chunks[i]).strip())
        chunks = overlapped

    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pages", type=str, default="artifacts/pages_clean.jsonl")
    ap.add_argument("--out_chunks", type=str, default="artifacts/chunks.jsonl")
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--overlap_chars", type=int, default=150)
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    in_pages = Path(args.in_pages)
    out_chunks = Path(args.out_chunks)
    if out_chunks.exists():
        out_chunks.unlink()

    total_chunks = 0
    for obj in read_jsonl(in_pages):
        doc_id = obj["doc_id"]
        source = obj["source"]
        page = int(obj["page"])
        text = obj.get("text", "") or ""
        units = split_to_units(text)
        chunks = make_chunks_from_units(units, args.max_chars, args.overlap_chars, args.min_chars)

        for ci, ch in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}:p{page:03d}:c{ci:03d}"
            write_jsonl(out_chunks, {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": source,
                "page": page,
                "text": ch,
                "char_len": len(ch),
            })
            total_chunks += 1

    print(f"[OK] Wrote chunks -> {out_chunks} (chunks={total_chunks})")

if __name__ == "__main__":
    main()
