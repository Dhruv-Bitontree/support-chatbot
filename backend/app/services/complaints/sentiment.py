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

    # Confidence based on how far from neutral
    confidence = min(1.0, abs(compound) + 0.3)

    return SentimentResult(
        score=compound,
        label=label,
        confidence=round(confidence, 3),
    )
