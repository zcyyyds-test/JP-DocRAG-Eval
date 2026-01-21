from typing import List, Dict, Tuple, Set

def get_gold_evidence(gold_item: Dict) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    """
    Extract gold evidence sets from a single gold item.
    Returns:
        gold_chunk_ids: set of chunk_ids
        gold_doc_pages: set of (doc_id, page_number) tuples
    """
    gold_chunk_ids = {g.get("chunk_id") for g in gold_item.get("gold", []) if g.get("chunk_id")}
    
    gold_doc_pages = set()
    for g in gold_item.get("gold", []):
        did = g.get("doc_id")
        pg = g.get("page")
        if did and pg is not None:
            gold_doc_pages.add((did, int(pg)))
            
    return gold_chunk_ids, gold_doc_pages

def check_hit(pred_chunk: Dict, gold_chunk_ids: Set[str], gold_doc_pages: Set[Tuple[str, int]]) -> bool:
    """
    Check if a single prediction hits the gold evidence.
    pred_chunk should contain 'chunk_id', 'doc_id', 'page'.
    """
    cid = pred_chunk.get("chunk_id")
    if cid and cid in gold_chunk_ids:
        return True
        
    did = pred_chunk.get("doc_id")
    pg = pred_chunk.get("page")
    if did and pg is not None:
        if (did, int(pg)) in gold_doc_pages:
            return True
            
    return False

def calc_metrics_at_k(results: List[Dict], gold_data: List[Dict], k_list: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Calculate Recall@k and MRR@k for a list of query results.
    
    Args:
        results: List of result dicts. Each dict must have 'qid' or be aligned by index with gold_data if sorted.
                 Ideally, provide a dict mapping qid -> list of pred_chunks.
                 Here we assume `results` is a list of {qid: ..., preds: [chunk_meta, ...]}
        gold_data: List of gold items {qid: ..., gold: [...]}.
        k_list: List of k values to compute metrics for.
        
    Returns:
        Dictionary of metric names to values, e.g., "Recall@1": 0.5
    """
    # Map qid to gold
    qid2gold = {item["qid"]: item for item in gold_data}
    
    metrics = {}
    for k in k_list:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"MRR@{k}"] = 0.0
    
    count = 0
    
    for res in results:
        qid = res["qid"]
        if qid not in qid2gold:
            continue
            
        preds = res["preds"] # List of chunk metas
        g_item = qid2gold[qid]
        g_chunk_ids, g_doc_pages = get_gold_evidence(g_item)
        
        # Calculate hits for this query for max k
        max_k = max(k_list)
        hits_at_rank = [] # 1-based ranks of hits
        
        for rank, p in enumerate(preds[:max_k], start=1):
            if check_hit(p, g_chunk_ids, g_doc_pages):
                hits_at_rank.append(rank)
        
        for k in k_list:
            # Recall@k: is there any hit <= k?
            # Note: This is binary recall (hit or miss). For multi-evidence recall, logic differs.
            # Assuming "Success if ANY gold found" for now (Reciprocal Rank style).
            has_hit = any(r <= k for r in hits_at_rank)
            if has_hit:
                metrics[f"Recall@{k}"] += 1.0
                
            # MRR@k: 1 / rank of first hit
            first_hit_rank = next((r for r in hits_at_rank if r <= k), None)
            if first_hit_rank:
                metrics[f"MRR@{k}"] += 1.0 / first_hit_rank
                
        count += 1
        
    if count > 0:
        for k in metrics:
            metrics[k] /= count
            
    return metrics
