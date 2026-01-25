# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder

class JapaneseCrossEncoder:
    def __init__(self, model_name: str = "hotchpotch/japanese-reranker-cross-encoder-small-v1"):
        """
        Initialize the CrossEncoder model.
        max_length=512 is set to avoid IndexError (position embedding overflow).
        """
        print(f"Loading CrossEncoder: {model_name} ...")
        # Enforce max_length to prevent position embedding errors with long inputs
        self.model = CrossEncoder(model_name, max_length=512)

    def predict(self, inputs: List[List[str]]) -> np.ndarray:
        return self.model.predict(inputs)

    def rerank(self, query: str, chunk_ids: List[str], chunk_map: Dict[str, Dict], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Rerank a list of chunk_ids based on query and text content in chunk_map.
        
        Args:
            query: The search query
            chunk_ids: List of candidate chunk_ids (pre-filtered by retriever)
            chunk_map: Dictionary mapping chunk_id to chunk data (must contain 'text')
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
        """
        if not chunk_ids:
            return []
            
        # Prepare pairs for model
        input_pairs = []
        valid_cids = []
        
        for cid in chunk_ids:
            if cid in chunk_map:
                text = chunk_map[cid].get("text", "")
                input_pairs.append([query, text])
                valid_cids.append(cid)
                
        if not input_pairs:
            return []
            
        # Predict scores for all pairs
        scores = self.model.predict(input_pairs)
        
        # Combine cid and score
        results = []
        for i, score in enumerate(scores):
            results.append((valid_cids[i], float(score)))
            
        # Re-sort
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
            
        return results
