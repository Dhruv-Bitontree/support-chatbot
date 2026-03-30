"""Helpers for recognizing and normalizing order references."""

from __future__ import annotations

import re

CANONICAL_ORDER_ID_RE = re.compile(r"^ORD-[A-Z0-9]{3,20}$")

_ORD_PREFIX_RE = re.compile(r"\bORD[\s-]*([A-Z0-9]{3,20})\b", re.IGNORECASE)
_ORDER_WORDING_RE = re.compile(
    r"\border\s*(?:id|number|#|no\.?)?\s*[:#-]?\s*(\d{4,12})\b",
    re.IGNORECASE,
)
_HASH_ORDER_RE = re.compile(r"#\s*(\d{4,12})\b")
_NUMERIC_ONLY_RE = re.compile(r"^\s*(\d{4,12})\s*$")


def normalize_order_id(raw: str | None) -> str | None:
    """Normalize user-entered order references into canonical ORD-... form."""
    if not raw:
        return None

    text = raw.strip()
    if not text:
        return None

    direct_match = CANONICAL_ORDER_ID_RE.fullmatch(text.upper())
    if direct_match:
        return text.upper()

    prefixed_match = _ORD_PREFIX_RE.search(text)
    if prefixed_match:
        return f"ORD-{prefixed_match.group(1).upper()}"

    wording_match = _ORDER_WORDING_RE.search(text)
    if wording_match:
        return f"ORD-{wording_match.group(1)}"

    hash_match = _HASH_ORDER_RE.search(text)
    if hash_match:
        return f"ORD-{hash_match.group(1)}"

    numeric_match = _NUMERIC_ONLY_RE.fullmatch(text)
    if numeric_match:
        return f"ORD-{numeric_match.group(1)}"

    return None


def looks_like_explicit_order_reference(raw: str | None) -> bool:
    """Return True when text explicitly resembles an order reference."""
    if not raw:
        return False

    text = raw.strip()
    if not text:
        return False

    if normalize_order_id(text) is None:
        return False

    lowered = text.lower()
    return (
        bool(_NUMERIC_ONLY_RE.fullmatch(text))
        or bool(_ORD_PREFIX_RE.search(text))
        or bool(_HASH_ORDER_RE.search(text))
        or "order" in lowered
    )
