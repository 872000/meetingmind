"""Heuristic information extraction over parsed transcripts.

These are rule-based heuristics, not machine learning: they catch common
meeting phrasing reliably but will miss reworded commitments and can
misattribute an assignee. The README states this limitation plainly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .nlp import split_sentences, tokenize

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

# Matches lines like "[00:01:12] ALEX: hello" or "ALEX: hello".
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[(?P<ts>[^\]]+)\]\s*)?(?P<speaker>[A-Za-z][A-Za-z .'-]{1,30}?):\s*(?P<text>.*)$"
)

# Timestamps sprinkled inside a line, e.g. "(00:12)".
_INLINE_TS = re.compile(r"[\[\(]\d{1,3}:\d{2}(?::\d{2})?[\]\)]")


@dataclass
class Utterance:
    speaker: str
    text: str
    timestamp: str | None = None


@dataclass
class Sentence:
    speaker: str
    text: str


def parse_transcript(raw: str) -> list[Utterance]:
    """Parse a transcript into utterances.

    Consecutive lines from the same speaker are merged into one utterance.
    Lines without a speaker label continue the previous utterance.
    """
    utterances: list[Utterance] = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            speaker = m.group("speaker").strip().upper()
            text = _INLINE_TS.sub("", m.group("text")).strip()
            ts = m.group("ts")
            if utterances and utterances[-1].speaker == speaker and utterances[-1].timestamp == ts:
                utterances[-1].text += " " + text
            else:
                utterances.append(Utterance(speaker=speaker, text=text, timestamp=ts))
        elif utterances:
            # continuation line
            utterances[-1].text += " " + _INLINE_TS.sub("", line).strip()
    return [u for u in utterances if u.text]


def sentences_with_speakers(utterances: list[Utterance]) -> list[Sentence]:
    """Flatten utterances into speaker-attributed sentences."""
    out: list[Sentence] = []
    for u in utterances:
        for s in split_sentences(u.text):
            out.append(Sentence(speaker=u.speaker, text=s))
    return out


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

# Commitment verbs / modal phrasing that usually introduce an action item.
_COMMITMENT = re.compile(
    r"\b(will|shall|'ll|need to|needs to|must|has to|have to|going to|"
    r"let me|i'll|i will)\b",
    re.IGNORECASE,
)

# "Alex will send the report", "Lisa shall coordinate ...", "I will draft ..."
_ASSIGNEE_WILL = re.compile(
    r"\b(?P<name>[A-Z][a-z]+)\s+(?P<modal>will|shall)\b"
)
# "Sam, you will draft the FAQ" / "Priya, you will collect ..." — vocative delegation.
_VOCATIVE_YOU = re.compile(r"^(?P<name>[A-Z][a-z]+),\s+you\s+(will|shall)\b", re.IGNORECASE)
_I_WILL = re.compile(r"\bI\s+(will|shall|'ll)\b", re.IGNORECASE)

# Due-date phrasing: "by Friday", "by next Monday", "by EOD", "by October 27th" ...
_DUE_DATE = re.compile(
    r"\bby\s+(?P<due>"
    r"EOD|end of (?:the )?(?:day|week)|"
    r"(?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"tomorrow|today|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?"
    r")",
    re.IGNORECASE,
)

# Imperative openers: "Send the report by Friday." (name optional)
_IMPERATIVE_VERBS = frozenset(
    "send draft write prepare create update review collect schedule book call email "
    "share fix investigate research coordinate own take handle finish complete submit "
    "deliver provide build test run grant set add document circulate".split()
)


@dataclass
class ActionItem:
    text: str
    assignee: str | None = None
    due: str | None = None
    speaker: str = ""


def _clean_action_text(text: str) -> str:
    text = text.strip()
    # strip leading discourse fillers
    text = re.sub(r"^(okay|ok|so|well|alright|great|thanks)[,.\s]+", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(".")


def extract_action_items(utterances: list[Utterance]) -> list[ActionItem]:
    """Heuristic action-item extraction.

    A sentence becomes an action item when it contains commitment phrasing
    ("will/shall/need to/...") or starts with an imperative verb. The assignee
    is the named person before "will/shall", "I" resolved to the speaker, or
    "you" resolved to the previous speaker when it reads like a delegation.
    Due dates are pulled from "by <date>" phrases.
    """
    items: list[ActionItem] = []
    prev_speaker: str | None = None
    for u in utterances:
        for s in split_sentences(u.text):
            low = s.lower()
            has_commitment = bool(_COMMITMENT.search(s))
            first_word = low.split(" ")[0].rstrip(",.") if low else ""
            is_imperative = first_word in _IMPERATIVE_VERBS
            if not (has_commitment or is_imperative):
                continue

            assignee: str | None = None
            m = _ASSIGNEE_WILL.search(s)
            v = _VOCATIVE_YOU.match(s)
            if v:
                assignee = v.group("name").upper()
            elif m and m.group("name").lower() not in {"i", "you", "we", "they"}:
                assignee = m.group("name").upper()
            elif _I_WILL.search(s):
                assignee = u.speaker

            due_m = _DUE_DATE.search(s)
            due = due_m.group("due") if due_m else None

            text = _clean_action_text(s)
            if len(tokenize(text)) < 4:
                # Too short to be a real commitment ("I'll start", ...).
                continue
            items.append(
                ActionItem(
                    text=text,
                    assignee=assignee,
                    due=due,
                    speaker=u.speaker,
                )
            )
        prev_speaker = u.speaker

    # de-duplicate identical texts, keep first
    seen: set[str] = set()
    unique: list[ActionItem] = []
    for it in items:
        key = it.text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

_DECISION_PATTERNS = [
    r"\bwe\s+decided\b",
    r"\bwe'?ve\s+decided\b",
    r"\bdecision\s+(made|is|was)\b",
    r"\bwe\s+agreed\b",
    r"\bagreed\b",
    r"\blet'?s\s+go\s+with\b",
    r"\bwe\s+will\s+go\s+with\b",
    r"\bi\s+propose\b",
    r"\bwe\s+landed\s+on\b",
    r"\bfinal\s+call\b",
]
_DECISION_RE = re.compile("|".join(_DECISION_PATTERNS), re.IGNORECASE)


@dataclass
class Decision:
    text: str
    speaker: str = ""


def extract_decisions(utterances: list[Utterance]) -> list[Decision]:
    """Return sentences that state a decision ("we decided", "agreed", ...)."""
    decisions: list[Decision] = []
    for u in utterances:
        for s in split_sentences(u.text):
            if _DECISION_RE.search(s) and len(tokenize(s)) >= 3:
                decisions.append(Decision(text=s.strip(), speaker=u.speaker))
    # de-duplicate
    seen: set[str] = set()
    unique: list[Decision] = []
    for d in decisions:
        key = d.text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Talk time
# ---------------------------------------------------------------------------

@dataclass
class SpeakerStats:
    speaker: str
    words: int
    share: float
    utterances: int = 0


def talk_time(utterances: list[Utterance]) -> list[SpeakerStats]:
    """Word counts and fractional share per speaker, sorted by share desc."""
    counts: dict[str, int] = {}
    utt_counts: dict[str, int] = {}
    for u in utterances:
        n = len(tokenize(u.text))
        counts[u.speaker] = counts.get(u.speaker, 0) + n
        utt_counts[u.speaker] = utt_counts.get(u.speaker, 0) + 1
    total = sum(counts.values()) or 1
    stats = [
        SpeakerStats(
            speaker=sp,
            words=w,
            share=w / total,
            utterances=utt_counts[sp],
        )
        for sp, w in counts.items()
    ]
    stats.sort(key=lambda s: s.share, reverse=True)
    return stats
