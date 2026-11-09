# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

Pipeline:
  - Preprocess the text into tokens (words, emoticons, emojis)
  - Look for positive and negative signals, flipping them after a negator
  - Compute a numeric score
  - Convert that score (plus which signals fired) into a mood label
"""

import re
from typing import List, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the polarity of the next sentiment word ("not happy").
NEGATORS = {"not", "no", "never", "nothing", "hardly", "aint", "nor"}

# How many tokens after a negator can still be flipped.
# 3 covers "not very happy" and "not really that good" without reaching
# too far into the rest of the sentence.
NEGATION_WINDOW = 3

# Emoticons are matched before punctuation is thrown away, otherwise ":)"
# would be stripped down to nothing.
_EMOTICON = r"[:;=]-?[()dp/\\|]"
_WORD = r"[a-z0-9']+"
_SYMBOL = r"[^\w\s]"  # any other single non-word character (emojis live here)
_TOKEN_RE = re.compile(f"{_EMOTICON}|{_WORD}|{_SYMBOL}")


def _is_negator(token: str) -> bool:
    return token in NEGATORS or token.endswith("n't")


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

          - Lowercases and normalizes curly apostrophes (’ -> ')
          - Keeps emoticons like ":)" and ":(" as single tokens
          - Keeps each emoji as its own token
          - Drops ordinary ASCII punctuation ("!", ",", ".")
          - Shortens stretched words: "soooo" -> "soo", and if the
            single-letter version is a known word, uses it ("haaappy" -> "happy")
        """
        cleaned = text.strip().lower().replace("’", "'")

        tokens: List[str] = []
        for tok in _TOKEN_RE.findall(cleaned):
            # Plain ASCII punctuation carries no mood signal on its own.
            if len(tok) == 1 and tok.isascii() and not tok.isalnum():
                continue
            # Emoji variation selectors and skin-tone modifiers are not
            # meaningful on their own.
            if tok == "️" or "\U0001F3FB" <= tok <= "\U0001F3FF":
                continue
            if re.fullmatch(_WORD, tok):
                tok = self._normalize_repeats(tok)
            tokens.append(tok)

        return tokens

    def _normalize_repeats(self, token: str) -> str:
        """Collapse runs of 3+ identical letters ("goooood" -> "good")."""
        doubled = re.sub(r"(.)\1{2,}", r"\1\1", token)
        if doubled in self.positive_words or doubled in self.negative_words:
            return doubled
        single = re.sub(r"(.)\1{2,}", r"\1", token)
        if single in self.positive_words or single in self.negative_words:
            return single
        return doubled

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Shared scoring pass used by score_text, predict_label and explain.

        Returns (score, positive_hits, negative_hits). Hits are recorded by
        their *effective* polarity, so "not happy" lands in negative_hits
        as "not happy".
        """
        tokens = self.preprocess(text)

        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negation_left = 0  # tokens remaining in the current negation window
        negator = ""

        for token in tokens:
            if _is_negator(token):
                negation_left = NEGATION_WINDOW
                negator = token
                continue

            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1
            else:
                polarity = 0

            label = token
            if polarity != 0 and negation_left > 0:
                polarity = -polarity
                label = f"{negator} {token}"
                negation_left = 0  # a negator flips only one sentiment word
            elif negation_left > 0:
                negation_left -= 1

            if polarity > 0:
                positive_hits.append(label)
            elif polarity < 0:
                negative_hits.append(label)
            score += polarity

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words add 1, negative words subtract 1.
        Enhancement: negation handling. A negator ("not", "never",
        any "n't" word, ...) flips the polarity of the next sentiment
        word within NEGATION_WINDOW tokens, so "not happy" scores -1
        and "not bad" scores +1.
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the score for a piece of text into a mood label.

          - Both positive AND negative signals present -> "mixed"
            (e.g. "tired but hopeful"), regardless of the net score
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - otherwise  -> "neutral"
        """
        score, positive_hits, negative_hits = self._analyze(text)
        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example: 'score=0 positive=['hopeful'] negative=['tired']'
        """
        score, positive_hits, negative_hits = self._analyze(text)
        return f"score={score} positive={positive_hits} negative={negative_hits}"
