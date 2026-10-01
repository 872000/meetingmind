#!/usr/bin/env python3
"""Render a generated MeetingMind notes Markdown file as a styled document PNG.

This is a *rendered preview* of the real generated notes (notes-launch-plan.md),
drawn with PIL — not a browser screenshot.
"""

import os
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "demo-output", "notes-launch-plan.md")
DST = os.path.join(BASE, "assets", "notes-screenshot.png")

W = 1000
MARGIN = 56
BG = (250, 250, 248)
INK = (28, 28, 30)
MUTED = (110, 110, 115)
ACCENT = (31, 90, 160)
RULE = (225, 225, 228)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        f"/usr/share/fonts/truetype/dejavu/{name}",
        f"/usr/share/fonts/{name}",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse(md: str):
    """Very small markdown subset parser -> list of (kind, text) blocks."""
    blocks = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            blocks.append(("h1", s[2:]))
        elif s.startswith("## "):
            blocks.append(("h2", s[3:]))
        elif s.startswith("- "):
            blocks.append(("li", s[2:]))
        elif s.startswith("|") and "---" not in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            blocks.append(("tr", cells))
        elif s.startswith("_") and s.endswith("_") and len(s) > 4:
            blocks.append(("note", s.strip("_")))
        elif s == "---":
            blocks.append(("rule", ""))
    return blocks


def clean_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    return text


def main():
    with open(SRC, encoding="utf-8") as f:
        blocks = parse(f.read())

    f_h1 = font(34, bold=True)
    f_h2 = font(21, bold=True)
    f_body = font(16)
    f_note = font(14)

    # First pass: lay out wrapped lines to measure height.
    laid = []  # (kind, [lines], extra_after)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    max_w = W - 2 * MARGIN

    def wrap(text, fnt, indent=0):
        width = max_w - indent
        words, lines, cur = text.split(), [], ""
        for w_ in words:
            trial = (cur + " " + w_).strip()
            if tmp.textlength(trial, font=fnt) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w_
        if cur:
            lines.append(cur)
        return lines

    for kind, payload in blocks:
        if kind == "h1":
            laid.append(("h1", wrap(clean_md(payload), f_h1), 14))
        elif kind == "h2":
            laid.append(("h2", wrap(clean_md(payload), f_h2), 22))
        elif kind == "li":
            laid.append(("li", wrap(clean_md(payload), f_body, indent=22), 10))
        elif kind == "tr":
            laid.append(("tr", payload, 4))
        elif kind == "note":
            laid.append(("note", wrap(clean_md(payload), f_note), 10))
        elif kind == "rule":
            laid.append(("rule", [], 14))

    line_h = lambda fnt: fnt.size + 10  # noqa: E731
    y = MARGIN + 10
    # header band height
    for kind, payload, after in laid:
        if kind == "h1":
            y += len(payload) * line_h(f_h1) + after
        elif kind == "h2":
            y += len(payload) * line_h(f_h2) + after
        elif kind == "li":
            y += len(payload) * line_h(f_body) + after
        elif kind == "tr":
            y += 30 + after
        elif kind == "note":
            y += len(payload) * line_h(f_note) + after
        elif kind == "rule":
            y += 24 + after
    H = y + MARGIN

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Top accent bar
    d.rectangle([0, 0, W, 8], fill=ACCENT)

    y = MARGIN + 10
    col_w = [max_w * 0.45, max_w * 0.25, max_w * 0.30]
    header_done = False
    for kind, payload, after in laid:
        if kind == "h1":
            for ln in payload:
                d.text((MARGIN, y), ln, font=f_h1, fill=INK)
                y += line_h(f_h1)
            y += after
        elif kind == "h2":
            for ln in payload:
                d.text((MARGIN, y), ln, font=f_h2, fill=ACCENT)
                y += line_h(f_h2)
            y += after
        elif kind == "li":
            first = True
            for ln in payload:
                x = MARGIN + 22
                if first:
                    d.text((MARGIN + 2, y), "•", font=f_body, fill=ACCENT)
                    first = False
                d.text((x, y), ln, font=f_body, fill=INK)
                y += line_h(f_body)
            y += after
        elif kind == "tr":
            cells = payload
            if cells[0].lower() == "speaker" and not header_done:
                xx = MARGIN
                for c, cw in zip(cells, col_w):
                    d.text((xx, y + 5), c, font=f_body, fill=MUTED)
                    xx += cw
                header_done = True
            else:
                xx = MARGIN
                for c, cw in zip(cells, col_w):
                    d.text((xx, y + 5), c, font=f_body, fill=INK)
                    xx += cw
            d.line([MARGIN, y + 30, W - MARGIN, y + 30], fill=RULE)
            y += 30 + after
        elif kind == "note":
            for ln in payload:
                d.text((MARGIN, y), ln, font=f_note, fill=MUTED)
                y += line_h(f_note)
            y += after
        elif kind == "rule":
            d.line([MARGIN, y + 10, W - MARGIN, y + 10], fill=RULE)
            y += 24 + after

    img.save(DST)
    print(f"Wrote {DST} ({W}x{H})")


if __name__ == "__main__":
    main()
