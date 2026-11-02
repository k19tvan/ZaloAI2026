"""JSONL helpers shared by `inference.py` and `evaluate.py`.

Line-per-record with flush-on-write is what makes a run resumable: an
interrupted run leaves a readable prefix, never a truncated JSON document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TextIO


def read(path: str | Path) -> list[dict]:
    """All records in `path`, or an empty list if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_records(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_line(fh: TextIO, record: dict) -> None:
    """Append one record and flush, so a killed run keeps what it finished."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def write_all(path: str | Path, records: list[dict]) -> None:
    """Replace `path` with `records` (used to regenerate derived files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            write_line(fh, record)
