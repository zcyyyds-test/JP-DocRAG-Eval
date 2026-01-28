#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean pages.jsonl -> pages_clean.jsonl

Operations (safe defaults for Japanese PDFs):
- Normalize whitespace (including full-width spaces)
- Join hard line-breaks inside paragraphs
- Remove very short/noisy lines (page numbers, lone symbols)
- Optional: remove repeated headers/footers by frequency across pages (per doc)

Input JSONL line format:
  {"doc_id": "...", "source": "...", "page": 1, "text": "..."}
Output keeps the same keys, but cleaned text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

RE_MULTISPACE = re.compile(r"[ \t]+")
RE_FULLWIDTH_SPACE = re.compile("\u3000+")
RE_LINE_JUNK = re.compile(r"^[\s\-–—_=~•・●◆◇□■※\.\,，。:：;；\(\)\[\]【】「」『』]+$")
RE_PAGE_NUM = re.compile(r"^\s*(\d+|-\s*\d+\s*-)\s*$")
RE_URL = re.compile(r"https?://\S+")
RE_EMAIL = re.compile(r"\b\S+@\S+\b")

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

from jprag.normalize import normalize_text as normalize_unicode

def normalize_text(text: str) -> str:
    # 1. Unicode Normalization (NFKC, Circled Numbers, etc.)
    text = normalize_unicode(text)
    
    # 2. Whitespace Normalization
    # full-width spaces -> single space (NFKC handles this usually, but good to be safe)
    text = RE_FULLWIDTH_SPACE.sub(" ", text)
    # normalize spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(RE_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n"))
    # remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def is_noise_line(ln: str, min_len: int) -> bool:
    if not ln:
        return True
    if len(ln) < min_len:
        # allow short but meaningful Japanese headings? keep if contains Kanji/Hiragana/Katakana
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", ln):
            return False
        return True
    if RE_LINE_JUNK.match(ln):
        return True
    if RE_PAGE_NUM.match(ln):
        return True
    return False

def join_lines(lines: List[str]) -> str:
    """
    Join hard line breaks.
    Simple heuristic:
    - Keep blank lines as paragraph breaks
    - Otherwise, join with space only when needed
    """
    paras: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            # join without forcing spaces (Japanese doesn't need spaces)
            # but insert one if previous ends with ASCII letter/number and next starts with ASCII
            out = ""
            for part in buf:
                if not out:
                    out = part
                else:
                    if (out[-1].isascii() and out[-1].isalnum()) and (part and part[0].isascii() and part[0].isalnum()):
                        out += " " + part
                    else:
                        out += part
            paras.append(out.strip())
            buf = []

    for ln in lines:
        if ln.strip() == "":
            flush()
            paras.append("")  # keep paragraph break
        else:
            buf.append(ln.strip())
    flush()

    # compress multiple paragraph breaks
    text = "\n\n".join([p for p in paras if p != ""] if "" not in paras else (
        # rebuild while keeping breaks
        (lambda ps: [x for i, x in enumerate(ps) if not (x == "" and i > 0 and ps[i-1] == "")])(paras)
    ))
    # the lambda returns list; join respecting empty entries
    if isinstance(text, list):
        rebuilt = []
        for p in text:
            if p == "":
                rebuilt.append("\n\n")
            else:
                rebuilt.append(p)
        return "".join(rebuilt).strip()
    return text.strip()

RE_DIGITS = re.compile(r"\d+")

def collect_header_footer_candidates(pages: List[Dict], top_k: int = 2, bottom_k: int = 2) -> Counter:
    c = Counter()
    for p in pages:
        lines = [ln.strip() for ln in normalize_text(p.get("text", "")).split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        
        candidates = lines[:top_k] + lines[-bottom_k:]
        for ln in candidates:
            masked_ln = RE_DIGITS.sub("#", ln)
            c[masked_ln] += 1
    return c

def remove_headers_footers(lines: List[str], frequent: set) -> List[str]:
    out = []
    for ln in lines:
        masked_ln = RE_DIGITS.sub("#", ln.strip())
        if masked_ln in frequent:
            continue
        out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pages", type=str, default="artifacts/pages.jsonl")
    ap.add_argument("--out_pages", type=str, default="artifacts/pages_clean.jsonl")
    ap.add_argument("--out_report", type=str, default="artifacts/clean_report.json")
    ap.add_argument("--min_line_len", type=int, default=2)
    ap.add_argument("--hf_threshold_ratio", type=float, default=0.6, help="line appears on >= ratio of pages in a doc => header/footer")
    args = ap.parse_args()

    in_pages = Path(args.in_pages)
    out_pages = Path(args.out_pages)
    out_report = Path(args.out_report)

    if out_pages.exists():
        out_pages.unlink()

    # Load pages grouped by doc_id
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    total_in_chars = 0
    for obj in read_jsonl(in_pages):
        by_doc[obj["doc_id"]].append(obj)
        total_in_chars += len(obj.get("text",""))

    removed_hf_total = 0
    removed_noise_lines_total = 0
    total_out_chars = 0
    doc_reports = []

    for doc_id, pages in by_doc.items():
        pages_sorted = sorted(pages, key=lambda x: int(x["page"]))
        hf_counter = collect_header_footer_candidates(pages_sorted)
        n_pages = len(pages_sorted)
        frequent = {ln for ln, cnt in hf_counter.items() if n_pages > 0 and (cnt / n_pages) >= args.hf_threshold_ratio}

        doc_removed_hf = 0
        doc_removed_noise = 0

        for p in pages_sorted:
            raw = normalize_text(p.get("text",""))
            lines = raw.split("\n")

            # remove headers/footers
            before = len(lines)
            lines = remove_headers_footers(lines, frequent)
            doc_removed_hf += (before - len(lines))

            # remove noisy lines
            cleaned_lines = []
            for ln in lines:
                ln2 = ln.strip()
                if is_noise_line(ln2, args.min_line_len):
                    doc_removed_noise += 1
                    continue
                cleaned_lines.append(ln2)

            # join lines into paragraphs
            joined = join_lines(cleaned_lines)

            out_obj = {
                "doc_id": p["doc_id"],
                "source": p["source"],
                "page": p["page"],
                "text": joined
            }
            write_jsonl(out_pages, out_obj)
            total_out_chars += len(joined)

        removed_hf_total += doc_removed_hf
        removed_noise_lines_total += doc_removed_noise

        doc_reports.append({
            "doc_id": doc_id,
            "pages": n_pages,
            "header_footer_candidates_removed": doc_removed_hf,
            "noise_lines_removed": doc_removed_noise,
            "frequent_hf_examples": list(sorted(list(frequent)))[:8],
        })

    report = {
        "input_chars": total_in_chars,
        "output_chars": total_out_chars,
        "removed_header_footer_lines": removed_hf_total,
        "removed_noise_lines": removed_noise_lines_total,
        "docs": doc_reports,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Cleaned pages -> {out_pages}")
    print(f"[OK] Report       -> {out_report}")
    print(f"     chars: {total_in_chars} -> {total_out_chars}")

if __name__ == "__main__":
    main()
