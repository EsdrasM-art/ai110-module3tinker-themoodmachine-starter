"""
Part 3 stress test: "breaker" sentences designed to confuse the rule based
Mood Machine. These are NOT part of the evaluation set in dataset.py; they
are a separate probe so we can watch failure patterns without tuning the
model to them.

Run:  python breakers.py
"""

from mood_analyzer import MoodAnalyzer

# (sentence, label a human would most likely give)
BREAKERS = {
    "sarcasm": [
        ("I love getting stuck in traffic", "negative"),
        ("Oh great, another Monday", "negative"),
        ("Wow, thanks for nothing", "negative"),
    ],
    "slang": [
        ("That party was sick", "positive"),
        ("This beat is fire", "positive"),
        ("That game was wicked", "positive"),
        ("That outfit is bad, I need it", "positive"),
    ],
    "emojis": [
        ("I'm fine 🙂", "neutral"),
        ("Love you ❤️", "positive"),
        ("Rough day :(", "negative"),
        ("😭😭😭 this is so funny", "positive"),
        ("I'm dead 💀💀", "positive"),
        ("this place is literally on fire 🔥 call 911", "negative"),
    ],
    "mixed": [
        ("I'm exhausted but proud of myself", "mixed"),
        ("Happy for her, sad for me", "mixed"),
    ],
}

# Same meaning, different language style. If the informal / regional /
# dialect version fails while the "standard" version passes, that is
# overt bias: the error depends on WHO wrote it, not WHAT they meant.
STYLE_PAIRS = [
    # (standard phrasing, informal / regional / non-standard phrasing, label)
    ("I am very happy today", "im hella happyyy today", "positive"),
    ("That concert was amazing", "that concert was bussin fr", "positive"),
    ("That concert was amazing", "that concert slapped no cap", "positive"),
    ("I am so angry right now", "im so mad rn", "negative"),
    ("I am so angry right now", "I'm proper vexed", "negative"),
    ("This food is really good", "dis food is gud", "positive"),
    ("This food is really good", "this food is wicked good", "positive"),
    ("I am exhausted", "I'm knackered", "negative"),
    ("I am not happy about this", "I ain't happy bout this", "negative"),
    ("I feel terrible", "feelin some type of way ngl", "negative"),
]


def run() -> None:
    analyzer = MoodAnalyzer()

    def show(text: str, expected: str) -> bool:
        pred = analyzer.predict_label(text)
        ok = pred == expected
        mark = "  " if ok else "X "
        print(f'{mark}"{text}" -> {pred} (expected {expected}; {analyzer.explain(text)})')
        return ok

    for category, items in BREAKERS.items():
        print(f"\n=== Breakers: {category} ===")
        passed = sum(show(t, e) for t, e in items)
        print(f"   {passed}/{len(items)} correct")

    print("\n=== Style pairs (standard vs informal/regional) ===")
    std_ok = inf_ok = 0
    for standard, informal, expected in STYLE_PAIRS:
        std_ok += show(standard, expected)
        inf_ok += show(informal, expected)
        print()
    n = len(STYLE_PAIRS)
    print(f"Standard phrasing correct: {std_ok}/{n}")
    print(f"Informal/regional phrasing correct: {inf_ok}/{n}")


if __name__ == "__main__":
    run()
