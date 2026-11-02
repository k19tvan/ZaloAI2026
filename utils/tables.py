"""Access to one business' data files.

Every shop ships a different bookkeeping layout (flat CSV, Excel with a report
title in row 1, one sheet or file per month, a POS SQLite). Rather than guess a
canonical schema per shop, this module exposes the files *as written*:

- `raw(key)`   -> the sheet as a grid with no header interpretation.
- `table(key)` -> the same grid with one row promoted to the header.

The generated snippet sees the first rows of every grid in its prompt, so it can
pick the real header row itself (`header_row=3` for a report-title sheet). That
keeps header/fill-down/dirty-cell handling in the per-question snippet, where
the question's own needs are known, instead of in a shared parser that would
have to be right for all 44 shops at once.

Grids are cached as pickles because a 14 MB XLSX takes ~30 s to parse and the
same sheet is re-read by every question of that shop.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import pandas as pd

#: File types that hold business data (anything else, e.g. README.md, is prose).
DATA_SUFFIXES = (".csv", ".xlsx", ".sqlite")

#: Separates file name from sheet / SQLite table name inside a table key.
KEY_SEP = "::"


class Business:
    """The data files of one `business_id`, addressed by table key.

    A table key is ``"<file>"`` for a CSV, ``"<file>::<sheet>"`` for an Excel
    sheet and ``"<file>::<table>"`` for a SQLite table.
    """

    def __init__(self, data_dir: str | Path, cache_dir: str | Path | None = None):
        self.dir = Path(data_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"business data dir not found: {self.dir}")
        self.cache_dir = Path(cache_dir) if cache_dir else self.dir / ".cache"

    @property
    def readme(self) -> str:
        """The shop's own description of its files, or "" if it ships none."""
        path = self.dir / "README.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @property
    def files(self) -> list[Path]:
        return sorted(p for p in self.dir.iterdir() if p.suffix.lower() in DATA_SUFFIXES)

    def keys(self) -> list[str]:
        """Every readable table key of this business, in file order."""
        out: list[str] = []
        for path in self.files:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                out.append(path.name)
            elif suffix == ".xlsx":
                out += [f"{path.name}{KEY_SEP}{s}" for s in _sheet_names(path)]
            elif suffix == ".sqlite":
                out += [f"{path.name}{KEY_SEP}{t}" for t in _sqlite_tables(path)]
        return out

    def raw(self, key: str) -> pd.DataFrame:
        """The grid behind `key`, with integer column labels and no header row.

        CSV cells come back as strings (a text file has no types); XLSX cells
        keep the types openpyxl reports (so an Excel date stays a Timestamp and
        a serial date stays a number); a SQLite table is already typed and
        keeps its real column names.
        """
        cached = self.cache_dir / (_cache_name(key) + ".pkl")
        if cached.exists():
            with cached.open("rb") as fh:
                return pickle.load(fh)

        frame = self._read(key)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(frame, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cached)  # atomic: concurrent workers never read a half file
        return frame

    def table(self, key: str, header_row: int = 0) -> pd.DataFrame:
        """`raw(key)` with row `header_row` promoted to column names.

        Rows above the header are dropped, the index is reset, and column names
        are stripped. SQLite tables already have names and are returned as-is.
        """
        frame = self.raw(key)
        if _is_sqlite_key(key):
            return frame.copy()
        header = frame.iloc[header_row]
        out = frame.iloc[header_row + 1 :].reset_index(drop=True)
        out.columns = [str(c).strip() for c in header]
        return out

    def sql(self, query: str, sqlite_file: str | None = None) -> pd.DataFrame:
        """Run a read query against this shop's SQLite file.

        `sqlite_file` is only needed if the shop ships more than one.
        """
        candidates = [p for p in self.files if p.suffix.lower() == ".sqlite"]
        if sqlite_file:
            candidates = [p for p in candidates if p.name == sqlite_file]
        if not candidates:
            raise FileNotFoundError(f"no SQLite file in {self.dir}")
        with sqlite3.connect(f"file:{candidates[0]}?mode=ro", uri=True) as conn:
            return pd.read_sql_query(query, conn)

    def _read(self, key: str) -> pd.DataFrame:
        name, _, member = key.partition(KEY_SEP)
        path = self.dir / name
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, header=None, dtype=object)
        if suffix == ".xlsx":
            return pd.read_excel(path, sheet_name=member, header=None, engine="openpyxl")
        if suffix == ".sqlite":
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                return pd.read_sql_query(f'SELECT * FROM "{member}"', conn)
        raise ValueError(f"unsupported table key: {key}")

    def preview(self, key: str, rows: int = 6, max_cell: int = 40) -> str:
        """Shape plus the first `rows` raw rows of `key`, as prompt text."""
        frame = self.raw(key)
        head = frame.head(rows)
        lines = [f"### {key}  ({len(frame)} rows x {frame.shape[1]} cols)"]
        lines.append("cols: " + " | ".join(str(c) for c in frame.columns))
        for idx, row in head.iterrows():
            cells = [_short(v, max_cell) for v in row.tolist()]
            lines.append(f"[{idx}] " + " | ".join(cells))
        return "\n".join(lines)

    def profile(self, rows: int = 6) -> str:
        """README plus a preview of every table: the schema half of the prompt."""
        parts = []
        if self.readme:
            parts.append("## The shop's own README.md\n" + self.readme.strip())
        parts.append(
            "## Tables (shown RAW: no header applied, column labels are positions)\n"
            + "\n\n".join(self.preview(k, rows) for k in self.keys())
        )
        return "\n\n".join(parts)


def _short(value: object, limit: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\n", " ").replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_sqlite_key(key: str) -> bool:
    return key.partition(KEY_SEP)[0].lower().endswith(".sqlite")


def _cache_name(key: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in key)


def _sheet_names(path: Path) -> list[str]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(book.sheetnames)
    finally:
        book.close()


def _sqlite_tables(path: Path) -> list[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]
