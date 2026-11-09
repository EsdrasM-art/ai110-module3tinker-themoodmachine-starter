"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Added in Part 2: common plain-English positive words
    "hopeful",
    "proud",
    "glad",
    "best",
    "nice",
    "fantastic",
    "grateful",
    "enjoy",
    "enjoyed",
    # Added in Part 3 (fix: emojis had no effect). Only emojis/emoticons whose
    # meaning is fairly stable. Deliberately NOT added: 🙂 (often passive-
    # aggressive), 😭 (crying OR laughing), 💀 ("I'm dead" = laughing).
    ":)",
    ":d",
    "😊",
    "😀",
    "😁",
    "😄",
    "😍",
    "🥰",
    "❤",
    "🎉",
    "🔥",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Added in Part 2: common plain-English negative words
    "miss",
    "worst",
    "annoyed",
    "lonely",
    "scared",
    "worried",
    "ugh",
    "exhausted",
    # Added in Part 3 (emoji fix, see note in POSITIVE_WORDS)
    ":(",
    "😢",
    "😞",
    "😩",
    "😡",
    "😠",
    "💔",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- Added in Part 1: realistic language (slang, emojis, sarcasm, mixed) ---
    "Lowkey stressed but kind of proud of myself",
    "I absolutely love getting stuck in traffic for an hour",
    "this new album is fire 🔥",
    "I'm fine 🙂",
    "Got the job!!! 😭😭",
    "ugh my phone died again 😩",
    "Not bad at all, actually pretty good",
    "I miss my friends but college is going great",
    "Just woke up, gonna grab coffee",
    "Exam tomorrow and I haven't even started studying 💀",
    # --- Added in Part 4: more data to see how each model reacts ---
    "Best birthday ever, thank you all 🎉",
    "im so mad rn my roommate ate my leftovers",
    "the tacos here are bussin no cap",
    "Traffic was awful but the beach was worth it",
    "Package arrives Tuesday",
    "I'm so happy for you!!",
    "Honestly kinda lonely this weekend",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # --- Added in Part 1 ---
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "negative",  # "I absolutely love getting stuck in traffic for an hour" (sarcasm)
    "positive",  # "this new album is fire 🔥" (slang)
    "neutral",   # "I'm fine 🙂" (debatable: could read as passive-aggressive/negative)
    "positive",  # "Got the job!!! 😭😭" (tears of joy, not sadness)
    "negative",  # "ugh my phone died again 😩"
    "positive",  # "Not bad at all, actually pretty good" (negated negative)
    "mixed",     # "I miss my friends but college is going great"
    "neutral",   # "Just woke up, gonna grab coffee"
    "negative",  # "Exam tomorrow and I haven't even started studying 💀"
    # --- Added in Part 4 ---
    "positive",  # "Best birthday ever, thank you all 🎉"
    "negative",  # "im so mad rn my roommate ate my leftovers"
    "positive",  # "the tacos here are bussin no cap"
    "mixed",     # "Traffic was awful but the beach was worth it"
    "neutral",   # "Package arrives Tuesday"
    "positive",  # "I'm so happy for you!!"
    "negative",  # "Honestly kinda lonely this weekend"
]

# Sanity check: every post needs exactly one label.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS has {len(SAMPLE_POSTS)} items but TRUE_LABELS has {len(TRUE_LABELS)}"
)

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
