from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def normalize_ticker(value: str) -> str:
    cleaned = (value or "").strip().upper()
    cleaned = cleaned.replace("/", ".")
    cleaned = cleaned.replace("-", ".")
    return cleaned


def ticker_variants(value: str) -> set[str]:
    normalized = normalize_ticker(value)
    variants = {normalized}
    if "." in normalized:
        variants.add(normalized.replace(".", "-"))
        variants.add(normalized.replace(".", ""))
    if "-" in normalized:
        variants.add(normalized.replace("-", "."))
        variants.add(normalized.replace("-", ""))
    return {variant for variant in variants if variant}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_universe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def csv_quote_join(values: list[str]) -> str:
    quoted = []
    for value in values:
        escaped = value.replace("'", "''")
        quoted.append(f"'{escaped}'")
    return ", ".join(quoted)
