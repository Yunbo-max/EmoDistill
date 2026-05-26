"""
Emotion taxonomies for DQN-new and baseline experiments.

Three configurations are supported:

1. EKMAN_7  — the original EvoEmo set (Ekman 6 + neutral). Kept for backward
   compatibility with the existing baselines/ directory.

2. IZARD_10 — Izard's Differential Emotions Scale (DES). Considered a complete
   non-reducible set of fundamental emotions in affective psychology. Negotiation
   research (Van Kleef, Olekalns) frequently maps to a subset of these.
   Source: Izard, C.E. "Patterns of Emotions" (1972).

3. GOEMOTIONS_27 — Google Research's fine-grained taxonomy from the GoEmotions
   dataset (Demszky et al., ACL 2020). Largest empirically validated emotion set
   suitable for text. Useful for future ablations.
   Source: https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/

Switch active taxonomy via TAXONOMY env var or by calling set_active_taxonomy().
"""

from typing import Dict, List, Tuple
import os


# -------- Taxonomies --------

EKMAN_7: List[str] = [
    "happy", "surprising", "angry", "sad", "disgust", "fear", "neutral",
]

IZARD_10: List[str] = [
    "interest",   # engaged, curious
    "joy",        # positive, enthusiastic
    "surprise",   # unexpected
    "sadness",    # disappointed, downcast
    "anger",      # firm, assertive
    "disgust",    # disapproval, professional disappointment
    "contempt",   # condescending superiority
    "fear",       # cautious, anxious
    "shame",      # acknowledging fault, conciliatory
    "guilt",      # apologetic, taking responsibility
]

GOEMOTIONS_27: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


# -------- Prompts per emotion --------
#
# Uniform "Respond in a {ADJ} tone." prompt structure for all emotions.
# This keeps prompt length / detail identical across emotions so any
# downstream signal (mean_R variation across emotions) is attributable to
# the EMOTION itself, not to prompt-engineering bias.

# Adjective form used in the uniform prompt template.
EMOTION_ADJECTIVE: Dict[str, str] = {
    # Ekman 7 (legacy taxonomy)
    "happy": "happy",
    "surprising": "surprised",
    "angry": "angry",
    "sad": "sad",
    "disgust": "disgusted",
    "fear": "fearful",
    "neutral": "neutral",
    # Izard 10 additions
    "interest": "interested",
    "joy": "joyful",
    "surprise": "surprised",
    "sadness": "sad",
    "anger": "angry",
    "contempt": "contemptuous",
    "shame": "ashamed",
    "guilt": "guilty",
    # GoEmotions 27 + neutral
    "admiration": "admiring",
    "amusement": "amused",
    "annoyance": "annoyed",
    "approval": "approving",
    "caring": "caring",
    "confusion": "confused",
    "curiosity": "curious",
    "desire": "desiring",
    "disappointment": "disappointed",
    "disapproval": "disapproving",
    "embarrassment": "embarrassed",
    "excitement": "excited",
    "gratitude": "grateful",
    "grief": "grieving",
    "love": "loving",
    "nervousness": "nervous",
    "optimism": "optimistic",
    "pride": "proud",
    "realization": "discerning",
    "relief": "relieved",
    "remorse": "remorseful",
}

# Uniform 3-sentence template — keeps prompt structure constant so signal
# differences across emotions are attributable to EMOTION content, not prompt
# length / detail variation.
#
#   Sentence 1: "Respond with {a|an} {ADJ} tone."         (template, fixed)
#   Sentence 2: "{Affective description}."                (per-emotion)
#   Sentence 3: "Use language that {behavioral hint}."    (per-emotion)


def _article(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


# Per-emotion (affective_description, behavioral_hint) pairs. Together with the
# adjective, each row uniquely characterizes the emotion in a negotiation
# context. All three components rotate via the SAME template — no prompt
# engineering bias toward any single emotion.
EMOTION_DESCRIPTORS: Dict[str, tuple] = {
    "admiration":     ("Your words convey genuine respect for the other party's reasoning",
                       "recognizes their merits while still pressing your position"),
    "amusement":      ("Your words convey light playfulness about the back-and-forth",
                       "injects subtle humor without dismissing the matter"),
    "anger":          ("Your words convey strong displeasure with the current state of affairs",
                       "is firm, assertive, and signals urgency"),
    "angry":          ("Your words convey strong displeasure with the current state of affairs",
                       "is firm, assertive, and signals urgency"),
    "annoyance":      ("Your words convey mild frustration with the slow progress",
                       "is sharp and impatient without escalating into outright anger"),
    "approval":       ("Your words convey clear agreement with elements of the other party's position",
                       "affirms shared ground before reintroducing your ask"),
    "caring":         ("Your words convey concern for the other party's wellbeing beyond the transaction",
                       "is warm, supportive, and centered on mutual interest"),
    "confusion":      ("Your words convey uncertainty about the other party's reasoning",
                       "asks for clarification and probes their stated rationale"),
    "curiosity":      ("Your words convey genuine interest in the other party's underlying interests",
                       "asks open-ended questions and invites them to share more"),
    "desire":         ("Your words convey strong wanting for a particular outcome",
                       "emphasizes what you seek and the value of reaching agreement"),
    "disappointment": ("Your words convey measured letdown at the current offer",
                       "signals that the proposal falls noticeably short of expectations"),
    "disapproval":    ("Your words convey firm rejection of the current proposal",
                       "explicitly states the offer is unacceptable as stated"),
    "disgust":        ("Your words convey strong distaste for the current direction",
                       "signals that the proposal is fundamentally objectionable"),
    "disgusted":      ("Your words convey strong distaste for the current direction",
                       "signals that the proposal is fundamentally objectionable"),
    "embarrassment":  ("Your words convey self-consciousness about your own position",
                       "hedges and softens your demands while still pursuing them"),
    "excitement":     ("Your words convey high energy about the prospect of a deal",
                       "is enthusiastic and momentum-building toward agreement"),
    "fear":           ("Your words convey anxiety about potential negative outcomes",
                       "is cautious and stresses risks of the negotiation collapsing"),
    "gratitude":      ("Your words convey sincere thanks for the other party's flexibility so far",
                       "acknowledges their concessions and invites further reciprocity"),
    "grief":          ("Your words convey heavy loss over how things have unfolded",
                       "is somber and reflects on what could have been"),
    "happy":          ("Your words convey genuine positivity about the progress being made",
                       "is upbeat and confidence-building toward agreement"),
    "interest":       ("Your words convey active engagement with the other party's perspective",
                       "asks probing questions to understand what truly matters to them"),
    "joy":            ("Your words convey genuine delight at the prospect of a mutual deal",
                       "is warm, enthusiastic, and frames the negotiation as opportunity"),
    "love":           ("Your words convey deep care for the long-term relationship",
                       "emphasizes partnership and shared future beyond this transaction"),
    "nervousness":    ("Your words convey unease about the negotiation's trajectory",
                       "is tentative, hedging, and signals openness to compromise"),
    "optimism":       ("Your words convey confidence that an agreement is well within reach",
                       "is forward-looking and solution-focused"),
    "pride":          ("Your words convey confidence and standing in your position",
                       "is assertive about your value without being dismissive of theirs"),
    "realization":    ("Your words convey a moment of insight about what is really at stake",
                       "signals deeper comprehension and a sharper read of the situation"),
    "relief":         ("Your words convey easing tension as progress finally emerges",
                       "acknowledges the difficulty before moving forward"),
    "remorse":        ("Your words convey regret for prior friction in the negotiation",
                       "takes responsibility and seeks to repair the working relationship"),
    "sad":            ("Your words convey somber disappointment about the impasse",
                       "is downcast and seeks empathy from the other side"),
    "sadness":        ("Your words convey somber disappointment about the impasse",
                       "is downcast and seeks empathy from the other side"),
    "shame":          ("Your words convey acknowledgment of your side's possible miscalculation",
                       "concedes ground while seeking common path forward"),
    "guilt":          ("Your words convey responsibility for prior friction on your side",
                       "is apologetic and offers concrete ways to make amends"),
    "contempt":       ("Your words convey cool superiority over the other party's stance",
                       "signals that they are missing the obvious better path"),
    "surprise":       ("Your words convey genuine astonishment at the other party's position",
                       "reflects an unexpected shift and reopens the conversation"),
    "surprising":     ("Your words convey deliberately unexpected framing of the situation",
                       "introduces a fresh angle to break the current stalemate"),
    "neutral":        ("Your words convey balanced professionalism without affective coloring",
                       "is direct, factual, and free of emotional emphasis"),
}


def _build_prompt(adj: str, descriptor: tuple) -> str:
    affect, behavior = descriptor
    return (
        f"Respond with {_article(adj)} {adj} tone. "
        f"{affect}. "
        f"Use language that {behavior}."
    )


PROMPT_BANK: Dict[str, str] = {
    emo: _build_prompt(adj, EMOTION_DESCRIPTORS.get(emo, ("Your words convey a measured perspective",
                                                            "is professional and direct")))
    for emo, adj in EMOTION_ADJECTIVE.items()
}

DEFAULT_PROMPT = _build_prompt("neutral", EMOTION_DESCRIPTORS["neutral"])


# -------- Active-taxonomy mechanism --------

_ACTIVE_NAME = (os.environ.get("EVOEMO_TAXONOMY", "ekman7") or "ekman7").lower()
_TAXONOMIES: Dict[str, List[str]] = {
    "ekman7": EKMAN_7,
    "izard10": IZARD_10,
    "goemotions27": GOEMOTIONS_27,
}


def set_active_taxonomy(name: str) -> None:
    """Set active emotion taxonomy by name: 'ekman7' | 'izard10' | 'goemotions27'."""
    global _ACTIVE_NAME
    name_l = name.lower()
    if name_l not in _TAXONOMIES:
        raise ValueError(f"Unknown taxonomy {name!r}; expected one of {list(_TAXONOMIES)}")
    _ACTIVE_NAME = name_l


def get_active_taxonomy_name() -> str:
    return _ACTIVE_NAME


def get_emotions() -> List[str]:
    """Return active emotion list."""
    return _TAXONOMIES[_ACTIVE_NAME]


def n_emotions() -> int:
    return len(get_emotions())


def emotion_to_idx() -> Dict[str, int]:
    return {e: i for i, e in enumerate(get_emotions())}


def prompt_for(emotion: str) -> str:
    return PROMPT_BANK.get(emotion, DEFAULT_PROMPT)


def normalize_emotion_idx(idx: int) -> float:
    """Map an integer emotion idx to [0, 1] for use in state vectors."""
    n = n_emotions()
    if n <= 1:
        return 0.0
    return float(idx) / float(n - 1)


def parse_emotion_str(s: str) -> int:
    """Parse a raw emotion string to an index in the active taxonomy.

    Falls back to a 'neutral'/'neutral-like' index if unknown.
    """
    s = (s or "").strip().lower()
    mapping = emotion_to_idx()
    if s in mapping:
        return mapping[s]
    # Common alias collapses across taxonomies
    aliases = {
        "happy": ("joy",),
        "joy": ("happy",),
        "sad": ("sadness",),
        "sadness": ("sad",),
        "surprising": ("surprise",),
        "surprise": ("surprising",),
        "anger": ("angry",),
        "angry": ("anger",),
    }
    for alt in aliases.get(s, ()):
        if alt in mapping:
            return mapping[alt]
    # Default fallback
    for fallback in ("neutral", "joy", "approval"):
        if fallback in mapping:
            return mapping[fallback]
    return 0
