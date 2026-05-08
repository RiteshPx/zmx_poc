"""
services/coordinate_matcher.py

Matches LLM-extracted field values back to word-level bounding boxes
from the PDF, so we know exactly where to highlight.
"""

import re
from typing import List, Dict


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^\w]", "", text.lower())


def find_coordinates_for_value(value: str, words: List[Dict]) -> List[Dict]:
    """
    Find all bounding boxes in the word list that correspond to `value`.

    Handles:
    - Single-word values: "Ritesh"
    - Multi-word values : "Ritesh Parmar"
    - Values with special chars: "INV-1023", "₹45,000"

    Returns list of matching word dicts (may span multiple consecutive words).
    """
    if not value or not words:
        return []

    value_str   = str(value).strip()
    value_words = value_str.split()
    n           = len(value_words)
    matches     = []

    for i in range(len(words) - n + 1):
        # Compare a window of n words
        window = words[i : i + n]

        # All must be on the same page
        if len(set(w["page"] for w in window)) > 1:
            continue

        window_text = " ".join(w["word"] for w in window)

        if _normalize(window_text) == _normalize(value_str):
            # Merge the window into one bounding box
            merged = {
                "value": value_str,
                "page" : window[0]["page"],
                "x0"   : min(w["x0"] for w in window),
                "y0"   : min(w["y0"] for w in window),
                "x1"   : max(w["x1"] for w in window),
                "y1"   : max(w["y1"] for w in window),
            }
            matches.append(merged)

    return matches


def match_all_fields(extracted_fields: dict, words: List[Dict]) -> Dict[str, List[Dict]]:
    """
    For every extracted field value, find its bounding boxes in the PDF.

    Returns:
        {
            "invoice_number": [{"value": "INV-101", "page": 0, "x0":..., ...}],
            "vendor_name"   : [{"value": "ABC Tech", "page": 0, ...}],
            ...
        }
    """
    result = {}
    for field, value in extracted_fields.items():
        if value is None:
            result[field] = []
            continue

        coords = find_coordinates_for_value(str(value), words)
        result[field] = coords

    return result