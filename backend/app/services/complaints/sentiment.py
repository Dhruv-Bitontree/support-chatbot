"""Sentiment analysis service using VADER."""

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.models.complaint import SentimentResult

logger = logging.getLogger(__name__)

_analyzer: SentimentIntensityAnalyzer | None = None


def get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze sentiment of text using VADER.

    Returns a SentimentResult with score (-1 to 1), label, and confidence.
    
    STEP 1 FIX: Removed fabricated confidence formula (min(1.0, abs(compound) + 0.3))
    which made neutral messages appear 30% confident. Now uses honest mapping based
    on the actual strength of the sentiment signal.
    """
    analyzer = get_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Honest confidence mapping based on sentiment strength
    abs_score = abs(compound)
    if abs_score >= 0.6:
        confidence = 0.9  # high confidence
    elif abs_score >= 0.3:
        confidence = 0.7  # medium confidence
    else:
        confidence = 0.5  # low confidence (near neutral)

    return SentimentResult(
        score=compound,
        label=label,
        confidence=round(confidence, 3),
    )
