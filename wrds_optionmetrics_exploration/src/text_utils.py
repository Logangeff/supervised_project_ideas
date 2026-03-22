from __future__ import annotations

import re


# Compatibility shim for the frozen claim detector bundle serialized from the
# numerical_claim_detection_project `src.text_utils` module.
NORMALIZE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

TOKEN_PATTERN = re.compile(
    r"\$?\d+(?:[.,]\d+)*(?:-\$?\d+(?:[.,]\d+)*)?%?|\b[a-z]+(?:['-][a-z]+)*\b"
)


def normalize_text(text: str) -> str:
    normalized = str(text).translate(NORMALIZE_TRANSLATION).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))
