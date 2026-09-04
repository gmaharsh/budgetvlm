"""Video-MME prompt + answer parsing."""
from __future__ import annotations

import re
from typing import Sequence


LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def build_mcq_prompt(
    question: str,
    options: Sequence[str],
    *,
    with_subtitle: bool = False,
    subtitle_text: str = "",
) -> str:
    """Standard Video-MME-style multiple-choice prompt (no subtitles by default)."""
    lines: list[str] = []
    if with_subtitle and subtitle_text:
        lines.append("This video's subtitles are listed below:")
        lines.append(subtitle_text)
        lines.append("")
    lines.append(
        "Select the best answer to the following multiple-choice question based on the video."
    )
    lines.append("Respond with only the letter (A, B, C, or D) of the correct option.")
    lines.append(f"Question: {question}")
    for opt in options:
        lines.append(opt if opt.strip().startswith(("A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)")) else opt)
    return "\n".join(lines)


def extract_letter(text: str) -> str:
    """Parse model output into A/B/C/D. Empty string if none found."""
    if not text:
        return ""
    text = text.strip()
    # Prefer explicit patterns
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*([A-D])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\s*([A-D])\b", text, re.I)
    if m:
        return m.group(1).upper()
    found = LETTER_RE.findall(text)
    if found:
        return found[-1].upper()
    return ""


def is_correct(prediction: str, ground_truth: str) -> bool:
    return extract_letter(prediction) == extract_letter(ground_truth) and bool(extract_letter(ground_truth))
