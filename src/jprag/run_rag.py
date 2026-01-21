#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Pipeline:
1. Retrieve (Hybrid/BM25)
2. Merge Chunks by Page (De-duplication)
3. Construct Prompt with [doc:page] tags
4. Generate Answer via LLM
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from src.jprag.retrieve import Retriever
from src.jprag.llm import LLMGenerator
from src.jprag.normalize import normalize_query

def load_chunks_map(path: Path) -> Dict[str, Dict]:
    """Load chunks.jsonl into a dict for quick text lookup."""
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["chunk_id"]] = obj
    return mapping

def merge_chunks_to_pages(results: List[str], chunk_map: Dict[str, Dict]) -> List[Dict]:
    """
    Group retrieved chunk_ids by (doc_id, page).
    Return list of "Page Contexts" with merged text.
    """
    page_groups = defaultdict(list) # (doc_id, page) -> list of chunks in order
    
    for cid in results:
        if cid not in chunk_map:
            continue
        c = chunk_map[cid]
        key = (c["doc_id"], c["page"])
        page_groups[key].append(c)
        
    # Reconstruct text for each page
    # Since we chunked by sliding window, simple "join" might duplicate text if overlap is high.
    # But since we want "Context", providing the full page text (or joined chunks) is fine.
    # A smart way: If we have multiple chunks from same page, just take the 'text' of the chunks 
    # and join them with '...'. Or if we had original page text map, use that.
    # For now, let's join chunk texts with ' [...] ' to indicate breaks, or just newlines.
    
    merged_pages = []
    for (doc_id, page), chunks in page_groups.items():
        # Sort chunks by chunk index logic (last part of chunk_id)
        # chunk_id format: doc:p:c
        chunks.sort(key=lambda x: x["chunk_id"])
        
        # Simple merge: join texts. 
        # Ideally we should merge overlapping strings, but that's complex.
        # Concatenation is safe context.
        merged_text = "\n...\n".join([c["text"] for c in chunks])
        
        merged_pages.append({
            "ref_id": f"[{doc_id}:p{page}]",
            "doc_id": doc_id,
            "page": page,
            "text": merged_text
        })
        
    return merged_pages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()
    
    # 1. Init
    print("Loading Retriever...")
    retriever = Retriever(method="hybrid") # Use strongest method
    chunk_map = load_chunks_map(Path("artifacts/chunks.jsonl"))
    llm = LLMGenerator()
    
    # 2. Retrieve
    print(f"Retrieving for: {args.query}")
    # Normalize handled in retriever, but explicit is good
    q_norm = normalize_query(args.query)
    results = retriever.search(q_norm, k=args.top_k) # returns keys
    
    # 3. Merge & Dedupe
    # results is list of (cid, score) tuple
    top_cids = [cid for cid, score in results]
    pages = merge_chunks_to_pages(top_cids, chunk_map)
    
    # 4. Construct Prompt
    context_str = ""
    for p in pages:
        context_str += f"Source {p['ref_id']}:\n{p['text']}\n\n"
        
    system_prompy = """
あなたは日本の電力インフラ・規制の専門家です。
与えられた「参考資料」ダケに基づいて、ユーザーの質問に答えてください。

【重要ルール】
1. 回答には必ず【根拠】として、参考資料のIDを明記してください。形式: [docXX:pYY]
2. 参考資料に答えが含まれていない場合は、正直に「提供された資料にはその情報が含まれていません」と答えてください。推測で答えないでください。
3. 文体は簡潔で専門的な「だ・である」調としてください。
    """
    
    user_prompt = f"""
【質問】
{args.query}

【参考資料】
{context_str}

【回答】
"""
    
    # 5. Generate
    print("Generating Answer...")
    # print(f"DEBUG PROMPT:\n{user_prompt}\n")
    
    ans = llm.generate(system_prompy, user_prompt)
    
    print("\n" + "="*40)
    print(f"Question: {args.query}")
    print("="*40)
    print(ans)
    print("="*40)

if __name__ == "__main__":
    main()
