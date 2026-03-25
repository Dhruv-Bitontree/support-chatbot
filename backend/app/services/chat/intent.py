"""Intent classification for incoming messages.

STEP 3 REWRITE: Added sarcasm detection, frustration signals, and multi-intent
conflict resolution with explicit priority hierarchy.
"""

import logging
import re
from enum import Enum

from app.models.chat import Intent

logger = logging.getLogger(__name__)


class FrustrationLevel(str, Enum):
    """Frustration intensity detected in user message."""
    NONE = "none"
    MILD = "mild"          # Slight disappointment, polite complaint
    MODERATE = "moderate"  # Clear frustration, needs attention
    HIGH = "high"          # Angry, demands escalation

# ── Intent patterns ──────────────────────────────────────────────────────────
# STEP 3: Separated polite transactional keywords from angry complaint keywords

INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.GREETING: [
        r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings)\b",
    ],
    Intent.ORDER_TRACKING: [
        r"\b(where\s+is\s+(my\s+)?(order|package|shipment))\b",
        r"\b(track(ing)?\s+(my\s+)?(order|package|shipment)?)\b",
        r"\b(order|delivery|shipping)\s+status\b",
        r"\b(find|check)\s+(my\s+)?(order|package)\b",
        r"\b(ORD-[A-Z0-9]+)\b",
        r"\b(order\s*(number|id|#|no))\b",
    ],
    # COMPLAINT patterns now focus on ANGER/FRUSTRATION signals, not just keywords
    Intent.COMPLAINT: [
        r"\b(complain|complaint|unhappy|dissatisfied|terrible|awful|worst|angry|furious)\b",
        r"\b(escalate|manager|supervisor|unacceptable|ridiculous|outrageous)\b",
        r"\b(never\s+again|rip\s*off|scam|fraud|disgusting|pathetic)\b",
        r"\b(legal\s+action|lawsuit|lawyer|sue|court\s+case)\b",
        r"\b(not\s+placing\s+(any\s+)?order|won'?t\s+order\s+again)\b",
        r"\b(never\s+getting\s+my\s+order|not\s+on\s+time|late\s+again)\b",
        r"\b(lost\s+(my\s+)?(order|package)|missing\s+(order|package))\b",
    ],
    # FAQ patterns include polite transactional questions
    Intent.FAQ: [
        r"\b(how\s+(do|can|to)|what\s+(is|are)|when\s+(do|does|can|will))\b",
        r"\b(policy|policies|return\s+policy|shipping|warranty|hours|contact)\b",
        r"\b(faq|help|information|explain)\b",
    ],
}

# ── Frustration signals ──────────────────────────────────────────────────────
# STEP 3: Detect frustration even without explicit complaint words

FRUSTRATION_SIGNALS = {
    "high": [
        r"\b(furious|livid|outraged|disgusting|pathetic|ridiculous|unbelievable)\b",
        r"\b(never\s+again|worst\s+ever|absolutely\s+unacceptable)\b",
        r"\b(legal\s+action|lawsuit|lawyer|sue|court\s+case)\b",
        r"[A-Z]{4,}",  # Multiple words in ALL CAPS
        r"!{2,}",      # Multiple exclamation marks
    ],
    "moderate": [
        r"\b(frustrated|disappointed|let\s+down|expected\s+better|not\s+okay)\b",
        r"\b(seriously\??|really\??|again\??|still\s+waiting|third\s+time)\b",
        r"\b(nobody\s+is\s+helping|keep\s+telling\s+me|how\s+many\s+times)\b",
        r"\b(not\s+on\s+time|late\s+delivery|delivery\s+delay(ed)?)\b",
        r"\b(never\s+getting\s+my\s+order|lost\s+(my\s+)?(order|package))\b",
        r"\b(not\s+placing\s+(any\s+)?order|won'?t\s+order\s+again)\b",
    ],
    "mild": [
        r"\b(concerned|worried|confused|unclear|unsure)\b",
        r"\b(hoping|would\s+like|prefer|wish)\b",
    ],
}

# ── Sarcasm signals ──────────────────────────────────────────────────────────
# STEP 3: Detect sarcastic praise combined with negative context

SARCASM_SIGNALS = [
    r"\b(oh\s+great|oh\s+wonderful|oh\s+fantastic|yeah\s+right|thanks\s+for\s+nothing)\b",
    r"\b(love\s+how|sure\s+i(?:'| )?ll|i\s+guess|apparently)\b",
]

SARCASM_POSITIVE_WORDS = re.compile(
    r"\b(good|great|awesome|amazing|fantastic|excellent|wonderful|love)\b",
    re.IGNORECASE,
)
SARCASM_NEGATIVE_CONTEXT = re.compile(
    r"\b(never|not|late|delay|delayed|again|anymore|worst|bad)\b",
    re.IGNORECASE,
)
SARCASM_LAUGHTER = re.compile(r"\b(haha|hahaha|lol|lmao)\b", re.IGNORECASE)

ORDER_LOOKUP_PATTERNS = [
    r"\b(where\s+is\s+(my\s+)?(order|package|shipment))\b",
    r"\b(track(ing)?\s+(my\s+)?(order|package|shipment)?)\b",
    r"\b(order|delivery|shipping)\s+status\b",
    r"\b(find|check)\s+(my\s+)?(order|package)\b",
    r"\b(order\s*(number|id|#|no))\b",
]

# ── Polite transactional keywords ────────────────────────────────────────────
# STEP 3: Words that indicate a polite question, not a complaint

POLITE_INDICATORS = [
    r"\b(please|thank\s+you|thanks|could\s+you|would\s+you|may\s+I)\b",
    r"\b(just\s+wondering|curious|question\s+about)\b",
    r"^\s*(hi|hello|hey)",  # Starts with greeting
]


def detect_sarcasm(text: str) -> bool:
    """Detect sarcastic tone in user message.
    
    Sarcasm indicators: ironic praise ("oh great"), quoted words, phrases like
    "I guess" or "apparently" combined with negative context.
    """
    text_lower = text.lower()
    for pattern in SARCASM_SIGNALS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    # Mixed-tone sarcasm: positive praise + negative context and/or laughter.
    has_positive = bool(SARCASM_POSITIVE_WORDS.search(text))
    has_negative_context = bool(SARCASM_NEGATIVE_CONTEXT.search(text))
    has_laughter = bool(SARCASM_LAUGHTER.search(text))
    if has_positive and (has_negative_context or has_laughter):
        return True

    return False


def detect_frustration(text: str) -> FrustrationLevel:
    """Detect frustration level in user message.
    
    Returns HIGH, MODERATE, MILD, or NONE based on language intensity.
    """
    text_lower = text.lower()
    
    # Check high frustration first
    for pattern in FRUSTRATION_SIGNALS["high"]:
        if re.search(pattern, text, re.IGNORECASE if pattern.startswith(r"\b") else 0):
            return FrustrationLevel.HIGH
    
    # Check moderate frustration
    for pattern in FRUSTRATION_SIGNALS["moderate"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return FrustrationLevel.MODERATE
    
    # Check mild frustration
    for pattern in FRUSTRATION_SIGNALS["mild"]:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return FrustrationLevel.MILD
    
    return FrustrationLevel.NONE


def is_polite_tone(text: str) -> bool:
    """Check if message has polite, non-confrontational tone."""
    text_lower = text.lower()
    for pattern in POLITE_INDICATORS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def is_order_lookup_request(text: str) -> bool:
    """Return True when a message explicitly asks to track/find an order."""
    text_lower = text.lower().strip()
    if re.search(r"\bORD-[A-Z0-9]+\b", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{4,6}", text_lower):
        return True
    for pattern in ORDER_LOOKUP_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def classify_intent_keyword(text: str) -> tuple[Intent, FrustrationLevel, bool]:
    """Fast keyword-based intent classification with frustration and sarcasm detection.
    
    STEP 3: Returns (Intent, FrustrationLevel, sarcasm_detected) tuple.
    
    Priority hierarchy when multiple intents match:
    1. COMPLAINT (if frustration detected OR explicit complaint words)
    2. ORDER_TRACKING
    3. FAQ
    4. GREETING
    5. GENERAL (fallback)
    
    Special rules:
    - Sarcasm overrides positive sentiment → treat as complaint
    - Polite tone + "refund"/"return" → FAQ, not COMPLAINT
    - Frustration + any other intent → COMPLAINT wins
    """
    text_lower = text.lower().strip()
    
    # Detect sarcasm and frustration first
    sarcasm = detect_sarcasm(text)
    frustration = detect_frustration(text)
    is_polite = is_polite_tone(text)

    # Score each intent
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores[intent] += 1

    # PRIORITY RULE 1: High frustration OR sarcasm → always COMPLAINT
    if frustration in (FrustrationLevel.HIGH, FrustrationLevel.MODERATE) or sarcasm:
        return Intent.COMPLAINT, frustration, sarcasm
    
    # PRIORITY RULE 2: Polite tone + transactional keywords → FAQ not COMPLAINT
    # Example: "Hi, I returned my item, when will I get my refund?"
    if is_polite and scores[Intent.FAQ] > 0 and scores[Intent.COMPLAINT] > 0:
        return Intent.FAQ, frustration, sarcasm
    
    # PRIORITY RULE 3: COMPLAINT keywords present → COMPLAINT wins over other intents
    if scores[Intent.COMPLAINT] > 0:
        return Intent.COMPLAINT, frustration, sarcasm
    
    # PRIORITY RULE 4: ORDER_TRACKING only when request is explicit/actionable
    if scores[Intent.ORDER_TRACKING] > 0 and is_order_lookup_request(text):
        return Intent.ORDER_TRACKING, frustration, sarcasm
    
    # PRIORITY RULE 5: FAQ beats GREETING and GENERAL
    if scores[Intent.FAQ] > 0:
        return Intent.FAQ, frustration, sarcasm
    
    # PRIORITY RULE 6: GREETING
    if scores[Intent.GREETING] > 0:
        return Intent.GREETING, frustration, sarcasm
    
    # Fallback to GENERAL
    return Intent.GENERAL, frustration, sarcasm


def classify_intent(text: str) -> tuple[Intent, FrustrationLevel, bool]:
    """Primary classification using keywords.
    
    STEP 3: Now returns (Intent, FrustrationLevel, sarcasm_detected) tuple.
    Use classify_intent_llm for complex cases.
    """
    return classify_intent_keyword(text)


async def classify_intent_llm(text: str, llm) -> tuple[Intent, FrustrationLevel, bool]:
    """LLM-based intent classification for ambiguous cases.
    
    STEP 3: Returns (Intent, FrustrationLevel, sarcasm_detected) tuple.
    Falls back to keyword classification if LLM fails.
    """
    try:
        categories = [i.value for i in Intent]
        result = await llm.classify(text, categories)
        intent = Intent(result)
        # Still run frustration/sarcasm detection even with LLM classification
        frustration = detect_frustration(text)
        sarcasm = detect_sarcasm(text)
        return intent, frustration, sarcasm
    except (ValueError, Exception) as e:
        logger.warning(f"LLM intent classification failed: {e}, falling back to keyword")
        return classify_intent_keyword(text)
