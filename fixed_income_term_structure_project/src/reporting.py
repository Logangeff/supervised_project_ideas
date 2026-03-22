from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)

