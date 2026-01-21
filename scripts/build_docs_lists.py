#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-generate docs_list.csv for PDFs under data/docs/.

Output columns:
  doc_id, filename, source_url, pages

Features:
- Incremental update: if output CSV exists, keep existing doc_id assignments.
- Page counting via pypdf (preferred) or PyPDF2 fallback.
- Optional URL map file (CSV/JSON) to fill source_url.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

DOCID_RE = re.compile(r"^(?P<prefix>[a-zA-Z]+)(?P<num>\d+)$")


def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    # Try pypdf first, then PyPDF2
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return None


def load_existing_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: filename -> row dict (doc_id, filename, source_url, pages)
    """
    if not csv_path.exists():
        return {}

    out: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            out[fn] = {
                "doc_id": (row.get("doc_id") or "").strip(),
                "filename": fn,
                "source_url": (row.get("source_url") or "").strip(),
                "pages": (row.get("pages") or "").strip(),
            }
    return out


def parse_doc_id(doc_id: str, prefix: str) -> Optional[int]:
    doc_id = doc_id.strip()
    m = DOCID_RE.match(doc_id)
    if not m:
        return None
    if m.group("prefix") != prefix:
        return None
    try:
        return int(m.group("num"))
    except ValueError:
        return None


def infer_next_id(existing: Dict[str, Dict[str, str]], prefix: str, start: int) -> int:
    max_id = start - 1
    for row in existing.values():
        n = parse_doc_id(row.get("doc_id", ""), prefix)
        if n is not None:
            max_id = max(max_id, n)
    return max_id + 1


def load_url_map(url_map_path: Optional[Path]) -> Dict[str, str]:
    """
    Supports:
      - CSV with columns: filename, source_url (header required)
      - JSON dict: { "filename.pdf": "https://..." , ... }
    """
    if not url_map_path:
        return {}
    if not url_map_path.exists():
        raise FileNotFoundError(f"url_map not found: {url_map_path}")

    if url_map_path.suffix.lower() == ".json":
        data = json.loads(url_map_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON url_map must be a dict: {filename: url}")
        return {str(k): str(v) for k, v in data.items()}

    if url_map_path.suffix.lower() in (".csv", ".tsv"):
        delim = "," if url_map_path.suffix.lower() == ".csv" else "\t"
        out: Dict[str, str] = {}
        with url_map_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            if "filename" not in reader.fieldnames or "source_url" not in reader.fieldnames:
                raise ValueError("CSV/TSV url_map must have headers: filename, source_url")
            for row in reader:
                fn = (row.get("filename") or "").strip()
                url = (row.get("source_url") or "").strip()
                if fn and url:
                    out[fn] = url
        return out

    raise ValueError("url_map must be .csv/.tsv/.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", type=str, default="data/docs", help="Directory containing PDFs")
    ap.add_argument("--out_csv", type=str, default="data/docs/docs_list.csv", help="Output CSV path")
    ap.add_argument("--prefix", type=str, default="doc", help="doc_id prefix, e.g., doc")
    ap.add_argument("--start", type=int, default=1, help="starting numeric id for doc_id")
    ap.add_argument("--width", type=int, default=3, help="zero-pad width for numeric id, e.g., 3 -> doc001")
    ap.add_argument("--url_map", type=str, default=None, help="Optional CSV/JSON map to fill source_url")
    ap.add_argument("--recursive", action="store_true", help="Scan PDFs recursively under docs_dir")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    out_csv = Path(args.out_csv)
    url_map_path = Path(args.url_map) if args.url_map else None

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    # Scan PDFs
    pdf_paths: List[Path] = []
    if args.recursive:
        pdf_paths = sorted(docs_dir.rglob("*.pdf"))
    else:
        pdf_paths = sorted(docs_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"[WARN] No PDFs found under: {docs_dir}")
        return

    # Load existing docs_list.csv (for stable doc_id)
    existing = load_existing_csv(out_csv)

    # Optional URL map
    url_map = load_url_map(url_map_path)

    # Determine next id
    next_id = infer_next_id(existing, args.prefix, args.start)

    rows: List[Dict[str, str]] = []

    # Keep only basename as filename (recommended)
    # If you prefer relative paths, adapt here.
    for p in pdf_paths:
        filename = p.name

        if filename in existing and existing[filename].get("doc_id"):
            doc_id = existing[filename]["doc_id"]
            source_url = existing[filename].get("source_url", "")
        else:
            doc_id = f"{args.prefix}{next_id:0{args.width}d}"
            next_id += 1
            source_url = ""

        # If url_map provides it, fill (but don't override an existing non-empty URL)
        if (not source_url) and filename in url_map:
            source_url = url_map[filename]

        pages = get_pdf_page_count(p)
        pages_str = "" if pages is None else str(pages)

        rows.append({
            "doc_id": doc_id,
            "filename": filename,
            "source_url": source_url,
            "pages": pages_str,
        })

    # Sort by doc_id numeric then filename for readability
    def sort_key(r: Dict[str, str]) -> Tuple[int, str]:
        n = parse_doc_id(r["doc_id"], args.prefix)
        return (n if n is not None else 10**9, r["filename"])

    rows.sort(key=sort_key)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "filename", "source_url", "pages"])
        writer.writeheader()
        writer.writerows(rows)

    total_pages = sum(int(r["pages"]) for r in rows if r["pages"].isdigit())
    print(f"[OK] Wrote: {out_csv}")
    print(f"     PDFs: {len(rows)} | Total pages (known): {total_pages}")
    missing_pages = [r for r in rows if not r["pages"].isdigit()]
    if missing_pages:
        print(f"[WARN] Page count failed for {len(missing_pages)} PDFs:")
        for r in missing_pages:
            print(f"       - {r['filename']}")


if __name__ == "__main__":
    main()
