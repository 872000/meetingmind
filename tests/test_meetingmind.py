"""Tests for MeetingMind: TextRank, action items, decisions, talk time."""

import math

from meetingmind import nlp
from meetingmind.extract import (
    extract_action_items,
    extract_decisions,
    parse_transcript,
    talk_time,
)

TINY_TRANSCRIPT = """\
ALEX: The launch date is the most important decision today.
PRIYA: I agree, the launch date matters more than anything else.
ALEX: Alex will send the report by Friday.
PRIYA: We decided to launch on October 27th.
"""


def test_textrank_picks_key_sentences():
    """On a tiny transcript about a launch date, the top-ranked sentence
    should be one of the sentences about the launch date."""
    utterances = parse_transcript(TINY_TRANSCRIPT)
    sentences = []
    for u in utterances:
        sentences.extend(nlp.split_sentences(u.text))
    top = nlp.summarize_sentences(sentences, top_n=2)
    assert len(top) == 2
    joined = " ".join(top).lower()
    assert "launch date" in joined


def test_textrank_returns_document_order():
    sentences = [
        "The cat sat on the mat.",
        "Dogs are loyal animals that bark loudly.",
        "The cat sat on the mat near the door.",
    ]
    idx = nlp.textrank(sentences, top_n=2)
    assert idx == sorted(idx)  # original order preserved
    assert len(idx) == 2


def test_action_item_regex_catches_named_commitment():
    utterances = parse_transcript(TINY_TRANSCRIPT)
    items = extract_action_items(utterances)
    matched = [i for i in items if "send the report" in i.text.lower()]
    assert matched, f"expected an action item about the report, got: {[i.text for i in items]}"
    item = matched[0]
    assert item.assignee == "ALEX"
    assert item.due and "friday" in item.due.lower()


def test_action_item_i_will_resolves_to_speaker():
    utterances = parse_transcript("SAM: I will draft the pricing FAQ by Wednesday.\n")
    items = extract_action_items(utterances)
    assert len(items) == 1
    assert items[0].assignee == "SAM"
    assert "wednesday" in (items[0].due or "").lower()


def test_decision_detection():
    utterances = parse_transcript(TINY_TRANSCRIPT)
    decisions = extract_decisions(utterances)
    assert any("october 27th" in d.text.lower() for d in decisions)


def test_talk_time_shares_sum_to_one():
    utterances = parse_transcript(TINY_TRANSCRIPT)
    stats = talk_time(utterances)
    assert math.isclose(sum(s.share for s in stats), 1.0, rel_tol=1e-9)
    speakers = {s.speaker for s in stats}
    assert speakers == {"ALEX", "PRIYA"}


def test_talk_time_sorted_descending():
    utterances = parse_transcript(TINY_TRANSCRIPT)
    stats = talk_time(utterances)
    shares = [s.share for s in stats]
    assert shares == sorted(shares, reverse=True)


def test_parse_handles_timestamps_and_continuations():
    raw = "[00:00:05] ALEX: Hello there.\ncontinuing the thought here.\n[00:01:00] PRIYA: Hi Alex.\n"
    utterances = parse_transcript(raw)
    assert len(utterances) == 2
    assert utterances[0].speaker == "ALEX"
    assert "continuing the thought" in utterances[0].text
    assert utterances[0].timestamp == "00:00:05"


def test_empty_inputs_do_not_crash():
    assert nlp.summarize_sentences([], top_n=3) == []
    assert nlp.textrank([], top_n=3) == []
    assert parse_transcript("") == []
    assert talk_time([]) == []
    assert extract_action_items([]) == []
    assert extract_decisions([]) == []


def test_cli_reports_missing_transcript_cleanly():
    """A missing transcript must exit 2, not raise FileNotFoundError."""
    from meetingmind import cli

    assert cli.main(["summarize", "no-such-transcript.txt"]) == 2
    assert cli.main(["notes", "no-such-transcript.txt"]) == 2
