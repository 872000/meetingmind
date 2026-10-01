#!/usr/bin/env python3
"""MeetingMind CLI.

    python -m meetingmind.cli summarize <transcript>
    python -m meetingmind.cli notes <transcript> [--out DIR]
    python -m meetingmind.cli demo
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from . import nlp
from .extract import parse_transcript, sentences_with_speakers
from .notes import analyze, render_markdown, write_notes

TRANSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcripts")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _title(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("-", " ").replace("_", " ").title()


def cmd_summarize(args: argparse.Namespace) -> int:
    raw = _read(args.transcript)
    utterances = parse_transcript(raw)
    sentences = [s.text for s in sentences_with_speakers(utterances)]
    for i, s in enumerate(nlp.summarize_sentences(sentences, top_n=args.sentences), 1):
        print(f"{i}. {s}")
    return 0


def cmd_notes(args: argparse.Namespace) -> int:
    raw = _read(args.transcript)
    title = _title(args.transcript)
    report = analyze(raw, summary_sentences=args.sentences)
    if args.out:
        paths = write_notes(raw, title, args.out, summary_sentences=args.sentences)
        print(f"Wrote {paths['markdown']}")
        print(f"Wrote {paths['chart']}")
    else:
        print(render_markdown(report, title))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = args.out or os.path.join(os.getcwd(), "demo-output")
    files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
    if not files:
        print("No bundled transcripts found.", file=sys.stderr)
        return 1
    print(f"MeetingMind demo — processing {len(files)} transcripts -> {out_dir}\n")
    for path in files:
        raw = _read(path)
        title = _title(path)
        paths = write_notes(raw, title, out_dir)
        report = analyze(raw)
        print(f"## {title}")
        print(f"   summary sentences : {len(report['summary'])}")
        print(f"   action items      : {len(report['actions'])}")
        print(f"   decisions         : {len(report['decisions'])}")
        top = report["talk_time"][0] if report["talk_time"] else None
        if top:
            print(f"   top speaker       : {top.speaker.title()} ({top.share:.0%})")
        print(f"   notes             : {paths['markdown']}")
        print(f"   chart             : {paths['chart']}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meetingmind",
        description="Transcript intelligence: summaries, action items, decisions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="Print an extractive TextRank summary.")
    p_sum.add_argument("transcript", help="Path to a .txt transcript")
    p_sum.add_argument("-n", "--sentences", type=int, default=5, help="Summary length")
    p_sum.set_defaults(func=cmd_summarize)

    p_notes = sub.add_parser("notes", help="Generate Markdown meeting notes (+ chart).")
    p_notes.add_argument("transcript", help="Path to a .txt transcript")
    p_notes.add_argument("-n", "--sentences", type=int, default=5, help="Summary length")
    p_notes.add_argument("--out", default=None, help="Write notes + chart PNG into DIR")
    p_notes.set_defaults(func=cmd_notes)

    p_demo = sub.add_parser("demo", help="End-to-end run over bundled transcripts.")
    p_demo.add_argument("--out", default=None, help="Output directory (default: ./demo-output)")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
