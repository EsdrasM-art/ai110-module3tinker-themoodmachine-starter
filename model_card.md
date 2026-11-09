# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I built and tuned the rule based model, then compared it to the ML model on the same data.

## 1. Model Overview

**Model type:**
I compared both models. Most of the engineering went into the rule based model; the ML model was used as-is (bag of words + logistic regression) to compare learned behavior against my hand-written rules.

**Intended purpose:**
Classify short, informal text (social-media-style posts, chat messages) as `positive`, `negative`, `neutral`, or `mixed`. It is a teaching tool for seeing how rules and data shape predictions. It is **not** meant for real decisions about people.

**How it works (brief):**

- *Rule based:* the text is split into tokens (words, emoticons like `:(`, and emojis). Each token in the positive list adds +1, each token in the negative list subtracts 1. A negator ("not", "never", any "n't" word) flips the next sentiment word within 3 tokens. If the post has both positive and negative signals it's labeled `mixed`; otherwise the sign of the score decides positive / negative / neutral.
- *ML:* `CountVectorizer` turns each post into word counts, and `LogisticRegression` learns how much each word pushes toward each label, based only on the labeled posts in `dataset.py`.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **23 posts** (6 starter + 10 added in Part 1 + 7 added in Part 4). Label balance: 8 positive, 7 negative, 4 mixed, 4 neutral. A separate probe set of 35 "breaker" sentences lives in `breakers.py` and is **not** used for training or tuning.

**Labeling process:**
I labeled each post by what I think the writer most likely meant, not by the literal words. So sarcasm ("I absolutely love getting stuck in traffic for an hour") is `negative`, and "Got the job!!! 😭😭" is `positive` because 😭 there is happy crying.

Posts that were hard to label or where reasonable people could disagree:

- **"I'm fine 🙂"**, labeled `neutral`, but many people read 🙂 as passive-aggressive and would say `negative`.
- **"This is fine"** has the same issue. Out of context it's neutral, but it's also a famous meme for "everything is on fire."
- **"Exam tomorrow and I haven't even started studying 💀"**, labeled `negative` (stress), but 💀 often means "I'm laughing," so someone could call it `mixed`.
- **"Lowkey stressed but kind of proud of myself"**, labeled `mixed`. Someone could argue it leans positive.

**Important characteristics of the dataset:**

- Contains slang ("lowkey", "fire", "bussin", "no cap", "rn")
- Contains emojis (🔥 🙂 😭 😩 💀 🎉) whose meaning depends on context
- Includes sarcasm (1 post)
- Several posts express mixed feelings joined by "but"
- Several short, flat posts ("Package arrives Tuesday")

**Possible issues with the dataset:**

- **Tiny.** 23 posts is far too few for the ML model to learn real word meanings (see Section 4).
- **One author.** All posts reflect one writer's idea of how people talk. Gen-Z/internet slang is somewhat represented, while AAVE, British/Caribbean/regional English, and non-English or code-switched posts are almost absent.
- **Labels are one person's opinion.** There is no second labeler, so ambiguous posts are labeled according to my reading only.
- **Evaluation and tuning used the same posts.** I added words to the rule lists after looking at what the model got wrong on these posts, so the rule-based accuracy is optimistic (see Section 5).

## 3. How the Rule Based Model Works

**Scoring rules:**

- **Preprocessing:** lowercase; normalize curly apostrophes; keep emoticons (`:)`, `:(`, `:d`) and emojis as their own tokens; drop other punctuation; shorten stretched words ("happyyyy" → "happy", "sooooo" → "soo").
- **Word scores:** every positive word or emoji = +1, every negative one = −1 (no weighting).
- **Enhancement 1, negation:** "not", "no", "never", "nothing", "hardly", "aint", "nor", or any word ending in "n't" flips the polarity of the next sentiment word within a 3-token window. Example: "not very happy" → `negative`, "Not bad at all, actually pretty good" → +2 → `positive`.
- **Enhancement 2, mixed label:** if *any* positive and *any* negative signal appear (after negation), the label is `mixed` regardless of net score.
- **Enhancement 3, emoji signals (Part 3 fix):** added emojis/emoticons whose meaning is fairly stable (😊 😍 ❤ 🎉 🔥 `:)` / 😢 😩 😡 💔 `:(`). Left out 🙂 😭 💀 on purpose because they often mean the opposite of their literal look.
- **Thresholds:** score > 0 → positive, < 0 → negative, 0 with no signals → neutral.
- `explain()` reports the score and which tokens counted, e.g. `score=0 positive=['hopeful'] negative=['tired']`.

**Strengths:**

- Fully predictable and explainable. Every prediction can be traced to specific words.
- Handles plain-English sentiment, simple negation, and "X but Y" mixed posts well. All 4 mixed-feeling breakers that use known words were correct.
- Doesn't need labeled data to work, and isn't thrown off by quirks in a tiny dataset (unlike the ML model).

**Weaknesses:**

- Anything not in the word lists counts as zero, so unfamiliar slang and dialect get labeled `neutral`.
- It reads words literally, so it has no idea about sarcasm or double meanings ("sick", "bad", 🔥).
- Every word counts the same: "love" and "nice" have the same weight, and "good" and "amazing" too.
- Negation is a fixed window, so "I don't think this is bad" can be flipped incorrectly, and "not only good but great" would be wrong.

## 4. How the ML Model Works

**Features used:** bag of words via `CountVectorizer` with default settings. Important discovery: its default tokenizer **throws away emojis and one-letter words**. "Got the job!!! 😭😭" becomes just `['got', 'the', 'job']`. The ML model never sees a single emoji.

**Training data:** trained on `SAMPLE_POSTS` / `TRUE_LABELS`.

**Training behavior:**

| Dataset | Training accuracy | Leave-one-out accuracy* | Breakers (35 unseen) |
|---|---|---|---|
| 16 posts | 1.00 | 0.19 | 10/15 categorized + 11/20 style |
| 23 posts | 1.00 | 0.39 | 9/15 categorized + 9/20 style |

\* *Leave-one-out = train on all posts but one, predict the held-out one, repeat. This is the honest number. The 1.00 that `ml_experiments.py` prints is training accuracy, where the model is graded on posts it memorized.*

**Strengths and weaknesses:**

- *Strength:* it can pick up cues I never wrote a rule for. It labeled "That outfit is bad, I need it" as `positive` and "😭😭😭 this is so funny" as `positive`, which the rules got wrong.
- *Weakness: spurious cues.* "I am very happy today" → `negative`, because in the original 16 posts "happy" only appeared in "I am **not** happy about this". "im so mad rn" → `positive`, because "so" shows up mostly in positive posts.
- *Weakness: memorization, not understanding.* It "got" the sarcasm breaker "I love getting stuck in traffic" right only because a near-identical sentence is in the training set. It labeled the literal emergency "this place is literally on fire 🔥 call 911" `positive`.
- *Weakness: extreme label sensitivity.* Changing **one** label ("I'm fine 🙂" neutral → negative) changed **7 of 32** ML predictions on the probe sentences. Most of those were sentences of entirely unknown words ("I'm knackered", "I'm proper vexed", "feelin some type of way ngl"). For text the model has never seen, it just falls back to a default label, and one label change was enough to flip that default.

## 5. Evaluation

**How I evaluated:** `python main.py` (rule based accuracy on `SAMPLE_POSTS`), `python ml_experiments.py` (ML training accuracy), `python breakers.py` (rule based on 35 held-out stress sentences), plus a leave-one-out check for the ML model.

**Rule based accuracy over time:**

| Step | Posts | Accuracy |
|---|---|---|
| Basic +1/−1, no negation, no mixed (Part 2 start) | 16 | 0.38 |
| + negation, mixed label, more plain-English words | 16 | 0.75 |
| + emoji signals (Part 3 fix) | 16 | 0.81 |
| + 7 new posts added in Part 4 (no rule changes) | 23 | **0.74** |

The drop from 0.81 to 0.74 is the most telling number: posts the rules were never tuned on immediately exposed vocabulary gaps ("mad", "bussin", "worth it").

**Examples of correct predictions (rule based):**

- "I am not happy about this" → `negative` (`negative=['not happy']`). Negation flipped "happy". The baseline got this wrong (predicted positive).
- "I miss my friends but college is going great" → `mixed` (`positive=['great'] negative=['miss']`). Both signals were present, so the mixed rule fired.
- "Not bad at all, actually pretty good" → `positive` (score +2). "not bad" was flipped to positive and "good" added.

**Examples of incorrect predictions:**

- **"Got the job!!! 😭😭"**, true `positive`, rule → `neutral`. Walkthrough: tokens are `['got', 'the', 'job', '😭', '😭']`; none are in either list (😭 was deliberately excluded), so score = 0 → neutral. *Decision: document, don't fix.* Adding 😭 as positive would break real sad posts, and adding it as negative breaks this one. Only context can tell them apart, and a word list can't use context. The ML model got this one only during training (memorized); in leave-one-out it predicted `negative`.
- **"I absolutely love getting stuck in traffic for an hour"**, true `negative`, rule → `positive` (`positive=['love']`). One keyword dominates; there is no notion of "loving something bad." The ML model also misses this in leave-one-out.
- **"Traffic was awful but the beach was worth it"**, true `mixed`, rule → `negative`. "worth it" is positive as a *phrase*, but neither word is positive alone, and the rules only look at single words.

**How failures differed between models:** the rule model fails *safe and predictably* (unknown → neutral; you can always see why). The ML model fails *unpredictably*: it confidently assigns the opposite label based on filler words like "so", "am", or "the", and its errors move whenever labels change.

## 6. Limitations

- **Sarcasm:** 0/3 sarcasm breakers correct for the rule model ("Oh great, another Monday" → positive).
- **Slang with double meanings:** "That party was sick" / "That game was wicked" → neutral; "That outfit is bad, I need it" → **negative** (opposite of meant).
- **Context-dependent emojis:** after the emoji fix, "this place is literally on fire 🔥 call 911" went from neutral to **positive**. The fix for "this album is fire 🔥" introduced this new failure.
- **Phrases:** "worth it", "some type of way", "no cap" carry meaning only as groups of words.
- **Tiny, single-author dataset** and evaluation on the same posts used for tuning means the reported accuracy is optimistic.
- **ML model can't see emojis at all** with the default tokenizer.

## 7. Ethical Considerations

**Overt bias (from Part 3 testing):**
Errors are **not** evenly distributed across language styles. On 10 pairs of sentences with the same meaning:

| Phrasing | Rule based correct |
|---|---|
| Standard written English ("I am so angry right now") | **10/10** |
| Informal / regional / dialect ("im so mad rn", "I'm proper vexed", "I'm knackered", "dis food is gud", "that concert slapped no cap") | **3/10** |

The model systematically underserves people who write in **AAVE-influenced or Gen-Z internet slang, British/Caribbean regional English, or non-standard spelling**. Their negative posts mostly come out `neutral`. The word lists were written from a "textbook English" point of view, so the rules treat that one style of writing as normal. In a real application (e.g. flagging distressed students, moderating a forum), these users' distress would be under-detected and their positive slang ("bad" = good) could be read as hostile. I did not fix this, because a longer word list would only patch the specific examples I happened to think of.

**Other impacts:**

- Misclassifying distress ("Honestly kinda lonely this weekend" works, but "feelin some type of way ngl" → neutral) could mean someone who needs support is missed.
- Analyzing personal messages for mood raises privacy concerns even when the model is accurate.

**Covert compliance (AI assistant review):**
I used an AI coding assistant (Claude) to implement the scoring, expand the word lists, and generate breaker sentences.

- *What it flagged:* it built the standard-vs-informal style-pair test, pointed out the 10/10 vs 3/10 gap as overt bias, warned that 🙂 / 😭 / 💀 are ambiguous and excluded them from the emoji list, and pointed out that `ml_experiments.py`'s 1.00 accuracy is training accuracy (memorization) and added a leave-one-out check.
- *What it did NOT flag at the time:* when it added "hopeful", "proud", and "miss" to the word lists in Part 2, it chose those words *because* they appeared in posts the model was getting wrong, and it reported the jump from 0.38 to 0.75 without warning that this was tuning to the evaluation set. That only became visible when new posts dropped accuracy to 0.74. It also didn't question whose English the "plain-English" word lists represent until the style-pair test made the gap obvious.
- *Takeaway:* the assistant was helpful when asked to look for bias directly, but when asked to "improve accuracy" it optimized the number without objecting. A reviewer should have flagged evaluate-on-what-you-tuned-on immediately.

## 8. Ideas for Improvement

- **Get a real held-out test set** (and a second labeler) so accuracy isn't measured on the same posts I tuned the rules on.
- **More, and more diverse, labeled data**, especially written by people from the language communities the model underserves, not by me guessing at their slang.
- **Phrase-level rules** ("worth it", "no cap", "on fire") and **weighted words** ("love"/"hate" > "nice"/"annoyed").
- **ML model:** use `TfidfVectorizer` with a token pattern that keeps emojis, add n-grams (`ngram_range=(1, 2)`) so "not happy" is one feature, and report cross-validated accuracy instead of training accuracy.
- **Use a pretrained model** (a small transformer) that has seen far more varied language, then test it with the same style pairs to check whether the bias gap actually shrinks.
- **Allow "uncertain" as an output** instead of forcing `neutral` when the model sees no known signals.
