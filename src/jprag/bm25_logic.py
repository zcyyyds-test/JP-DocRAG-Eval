# -*- coding: utf-8 -*-
from typing import List, Dict
import math
from collections import Counter, defaultdict

def char_ngrams(text: str, n: int = 3) -> List[str]:
    # Remove spaces but keep Japanese punctuation; BM25 works fine with this.
    s = (text or "").replace("\u3000", " ").replace(" ", "")
    if len(s) <= n:
        return [s] if s else []
    return [s[i:i+n] for i in range(0, len(s)-n+1)]

class BM25:
    # lightweight BM25 (Okapi) implementation
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        df = defaultdict(int)
        for toks in corpus_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf = {}
        for term, dfi in df.items():
            # classic BM25 idf
            self.idf[term] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

        self.tf = [Counter(toks) for toks in corpus_tokens]

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_const = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            tfi = self.tf[i]
            s = 0.0
            for t in query_tokens:
                if t not in tfi:
                    continue
                freq = tfi[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (freq * (self.k1 + 1)) / (freq + denom_const)
            scores[i] = s
        return scores
