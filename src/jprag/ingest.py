#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingestion: PDFs -> pages.jsonl (one line per page), keeping doc_id + page traceability.

Input:
  - data/docs/docs_list.csv with columns: doc_id, filename, source_url, pages
  - PDFs under data/docs/

Output:
  - artifacts/pages.jsonl: {"doc_id","source","page","text"}
  - artifacts/ingest_log.jsonl: parse warnings/errors (empty page, exceptions)
  - artifacts/ingest_stats.json: summary stats
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pdfplumber
from tqdm import tqdm


def read_docs_list(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"docs_list.csv not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"doc_id", "filename", "source_url", "pages"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"docs_list.csv must contain columns: {sorted(required)}")

        for r in reader:
            doc_id = (r.get("doc_id") or "").strip()
            filename = (r.get("filename") or "").strip()
            source_url = (r.get("source_url") or "").strip()
            pages = (r.get("pages") or "").strip()

            if not doc_id or not filename:
                continue

            rows.append({
                "doc_id": doc_id,
                "filename": filename,
                "source_url": source_url,
                "pages": pages,
            })
    if not rows:
        raise ValueError("docs_list.csv is empty or invalid.")
    return rows


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ingest_one_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      pages: list of page objects for pages.jsonl
      logs : list of warning/error logs
    """
    pages_out: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):  # page is 1-indexed
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    pages_out.append({"page": i, "text": text})

                    if not text:
                        logs.append({
                            "level": "WARN",
                            "type": "empty_page",
                            "page": i,
                            "message": "extract_text() returned empty",
                        })
                except Exception as e:
                    pages_out.append({"page": i, "text": ""})
                    logs.append({
                        "level": "ERROR",
                        "type": "page_exception",
                        "page": i,
                        "message": repr(e),
                    })
    except Exception as e:
        logs.append({
            "level": "ERROR",
            "type": "pdf_open_exception",
            "page": None,
            "message": repr(e),
        })

    return pages_out, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_csv", type=str, default="data/docs/docs_list.csv")
    ap.add_argument("--docs_dir", type=str, default="data/docs")
    ap.add_argument("--out_pages", type=str, default="artifacts/pages.jsonl")
    ap.add_argument("--out_log", type=str, default="artifacts/ingest_log.jsonl")
    ap.add_argument("--out_stats", type=str, default="artifacts/ingest_stats.json")
    args = ap.parse_args()

    docs_csv = Path(args.docs_csv)
    docs_dir = Path(args.docs_dir)
    out_pages = Path(args.out_pages)
    out_log = Path(args.out_log)
    out_stats = Path(args.out_stats)

    # reset outputs (avoid appending repeatedly by accident)
    for p in [out_pages, out_log]:
        if p.exists():
            p.unlink()

    docs = read_docs_list(docs_csv)

    total_pages_written = 0
    total_empty_pages = 0
    doc_stats: List[Dict[str, Any]] = []

    for d in tqdm(docs, desc="Ingest PDFs"):
        doc_id = d["doc_id"]
        filename = d["filename"]
        pdf_path = docs_dir / filename

        if not pdf_path.exists():
            write_jsonl(out_log, {
                "level": "ERROR",
                "type": "missing_pdf",
                "doc_id": doc_id,
                "source": filename,
                "page": None,
                "message": f"PDF not found at {pdf_path}",
            })
            continue

        pages, logs = ingest_one_pdf(pdf_path)

        # write per-page outputs: {"doc_id","source","page","text"}  (文档要求格式)  :contentReference[oaicite:4]{index=4}
        empty_cnt = 0
        for p in pages:
            if not p["text"]:
                empty_cnt += 1
            write_jsonl(out_pages, {
                "doc_id": doc_id,
                "source": filename,
                "page": p["page"],
                "text": p["text"],
            })

        # write logs with doc_id/source
        for lg in logs:
            lg2 = dict(lg)
            lg2.update({"doc_id": doc_id, "source": filename})
            write_jsonl(out_log, lg2)

        total_pages_written += len(pages)
        total_empty_pages += empty_cnt

        doc_stats.append({
            "doc_id": doc_id,
            "source": filename,
            "pages_written": len(pages),
            "empty_pages": empty_cnt,
            "empty_ratio": (empty_cnt / len(pages)) if pages else 1.0,
        })

    out_stats.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "docs": len(docs),
        "pages_written": total_pages_written,
        "empty_pages": total_empty_pages,
        "empty_ratio": (total_empty_pages / total_pages_written) if total_pages_written else 1.0,
        "per_doc": sorted(doc_stats, key=lambda x: x["empty_ratio"], reverse=True),
    }
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] pages.jsonl -> {out_pages} ({total_pages_written} pages)")
    print(f"[OK] logs       -> {out_log}")
    print(f"[OK] stats      -> {out_stats}")
    print(f"     empty pages: {total_empty_pages} ({stats['empty_ratio']:.2%})")
    print("Tip: if empty_ratio is high, those PDFs might be scanned (need OCR, but v1 forbids OCR).")


if __name__ == "__main__":
    main()
