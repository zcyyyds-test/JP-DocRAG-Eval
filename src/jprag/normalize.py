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
    # NFKC usually handles circled numbers (e.g., ① -> 1).
    
    # 3. Units / Case normalization
    # NFKC handles width normalization (ＫＶ -> KV). 
    
    # Text replacement for specific issues if needed.
    
    return text

def normalize_query(text: str) -> str:
    """
    Normalization for query/retrieval time.
    Consistent with text normalization.
    """
    return normalize_text(text)
