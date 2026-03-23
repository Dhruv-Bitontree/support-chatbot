"""Intent classification for incoming messages."""

import logging
import re

from app.models.chat import Intent

logger = logging.getLogger(__name__)

# Keyword-based patterns for fast classification (LLM fallback available)
INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.GREETING: [
        r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings)\b",
    ],
    Intent.ORDER_TRACKING: [
        r"\b(order|tracking|shipment|delivery|shipped|delivered|package|where\s+is)\b",
        r"\b(ORD-\w+)\b",
        r"\b(order\s*(number|id|#|no))\b",
        r"\b(track|status)\b",
    ],
    Intent.COMPLAINT: [
        r"\b(complain|complaint|unhappy|dissatisfied|terrible|awful|worst|angry|furious)\b",
        r"\b(refund|return|broken|damaged|defective|wrong\s+item|missing)\b",
        r"\b(escalate|manager|supervisor|unacceptable)\b",
        r"\b(never\s+again|rip\s*off|scam|fraud)\b",
    ],
    Intent.FAQ: [
        r"\b(how\s+(do|can|to)|what\s+(is|are)|when\s+(do|does|can|will))\b",
        r"\b(policy|policies|return\s+policy|shipping|warranty|hours|contact)\b",
        r"\b(faq|help|information|explain)\b",
    ],
}


def classify_intent_keyword(text: str) -> Intent:
    """Fast keyword-based intent classification."""
    text_lower = text.lower().strip()

    # Score each intent
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores[intent] += 1

    # Return highest scoring, or GENERAL if no matches
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return Intent.GENERAL

    return best


async def classify_intent_llm(text: str, llm) -> Intent:
    """LLM-based intent classification for ambiguous cases."""
    try:
        categories = [i.value for i in Intent]
        result = await llm.classify(text, categories)
        return Intent(result)
    except (ValueError, Exception) as e:
        logger.warning(f"LLM intent classification failed: {e}, falling back to keyword")
        return classify_intent_keyword(text)


def classify_intent(text: str) -> Intent:
    """Primary classification using keywords. Use classify_intent_llm for complex cases."""
    return classify_intent_keyword(text)
