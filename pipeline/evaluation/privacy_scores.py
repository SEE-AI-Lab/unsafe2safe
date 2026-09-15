"""Formula helpers for the paper's text and demographic privacy scores."""

from collections import Counter
from math import log

from rapidfuzz.fuzz import token_set_ratio


RACE_CATEGORIES = ("White", "Black", "Asian", "Hispanic", "Other")


def token_set_similarity(original_text: str, anonymized_text: str) -> float:
    """Return the paper's token-set TextSim score in the range [0, 1]."""
    return token_set_ratio(original_text, anonymized_text) / 100.0


def normalized_race_entropy(predicted_races, categories=RACE_CATEGORIES) -> float:
    """Return normalized demographic entropy for one image's VLM predictions."""
    counts = Counter(race for race in predicted_races if race in categories)
    total = sum(counts.values())
    entropy = -sum((count / total) * log(count / total) for count in counts.values())
    return entropy / log(len(categories))
