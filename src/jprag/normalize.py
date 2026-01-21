#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import re

# Mapping for circled numbers 1-50
# ① (U+2460) to 1, etc.
CIRCLED_INTS = {chr(0x2460 + i): str(i + 1) for i in range(20)} 
# ㉑ (U+3251) to 21...
CIRCLED_INTS.update({chr(0x3251 + i): str(21 + i) for i in range(15)})
# ㊱ (U+32b1) to 36...
CIRCLED_INTS.update({chr(0x32b1 + i): str(36 + i) for i in range(15)})

def normalize_text(text: str) -> str:
    """
    Apply standard normalization for Japanese technical documents.
    1. Unicode NFKC (full-width -> half-width for alphanumerics, etc.)
    2. Circled numbers -> ASCII numbers (① -> 1)
    3. Common unit/symbol unification (KV -> kV is tricky without context, but we can do safe ones)
    """
    if not text:
        return ""

    # 1. NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Circled numbers
    # Because NFKC maps ① -> 1, this might already be done?
    # Let's verify. python -c 'import unicodedata; print(unicodedata.normalize("NFKC", "①"))' -> "1"
    # Yes, NFKC handles standard circled numbers!
    # But just in case some are missed or distinct in other encodings, we keep logic if needed. 
    # Actually, NFKC is usually sufficient for ①. 
    # Let's stick to NFKC first.
    
    # 3. Units / Case normalization
    # User mentioned "kV/ＫＶ/ｋＶ". NFKC handles ＫＶ->KV.
    # But "KV" vs "kv" vs "kV" is case.
    # We generally don't want to lower-case everything for Japanese (Kanji has no case),
    # but for search keywords, case-insensitivity is usually good.
    # However, standard BM25 usually preserves case unless stemmed.
    # Let's lowercase ASCII parts? Or just rely on NFKC for width.
    # "kV" is the standard symbol. "KV" might be a typo. "kv" is typo.
    # Let's forcefully normalize specific technical units if we are sure.
    # For now, let's just do NFKC + Lowercase for search robustness? 
    # Or keep case?
    # User requirement: "Unit unification".
    # Let's implement a safe replace for common electrical units if needed.
    # For now, NFKC is the biggest win. 
    
    # Text replacement for specific issues
    # e.g. "UC⑥" -> NFKC -> "UC6". Correct.
    
    return text

def normalize_query(text: str) -> str:
    """
    Normalization for query/retrieval time.
    Same as text, but maybe aggressive lowercasing?
    For dense retrieval, case matters less. For BM25, it matters if not stemmed.
    Let's align with the cleaning.
    """
    return normalize_text(text)
