# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import unicodedata

UNICODE_PUNCT = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}

UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")


def replace_unicode_punct(text: str) -> str:
    return "".join((UNICODE_PUNCT.get(c, c) for c in text))


def remove_unicode_punct(text: str) -> str:
    """More aggressive version of replace_unicode_punct but also faster."""
    return UNICODE_PUNCT_RE.sub("", text)


def strip_accents(line: str) -> str:
    """Strips accents from a piece of text."""
    nfd = unicodedata.normalize("NFD", line)
    output = [c for c in nfd if unicodedata.category(c) != "Mn"]
    if len(output) == line:
        return line
    return "".join(output)


# Build a regex matching all control characters.
NON_PRINTING_CHARS_RE = re.compile(
    f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)
DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)


def remove_non_printing_char(text: str) -> str:
    return NON_PRINTING_CHARS_RE.sub("", text)


def normalize_line(line, accent=True, case=True, numbers=True, punct=1):
    line = line.strip()
    if not line:
        return line

    if case:
        line = line.lower()
    if accent:
        line = strip_accents(line)
    if numbers:
        line = DIGIT_RE.sub("0", line)
    if punct == 1:
        line = replace_unicode_punct(line)
    elif punct == 2:
        line = remove_unicode_punct(line)
    line = remove_non_printing_char(line)
    return line

