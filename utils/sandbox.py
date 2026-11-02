"""Runs a model-written pandas snippet and collects its answer.

The snippet runs in a separate process so that a crash, an endless loop or a
memory blow-up costs one question instead of the whole run, and so that a
timeout is enforceable. Communication is deliberately narrow: the snippet gets
the names listed in `RUNTIME_API` (the same text the prompt shows the model) and
must assign `ANSWER`; the runner prints it as one JSON line.

This is an execution boundary, not a security boundary — the snippet comes from
our own model and can touch the filesystem like any local script.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Names the snippet may use, as shown to the model. Keep in sync with `_RUNNER`.
RUNTIME_API = """\
tables()                      -> list[str]: key of every table ("file.csv", "file.xlsx::sheet",
                                 "pos.sqlite::table")
raw(key)                      -> RAW DataFrame: no header applied, columns labelled 0,1,2...
table(key, header_row=0)      -> DataFrame using row header_row as column names, rows above dropped
sql(query)                    -> DataFrame; only for a shop that ships a .sqlite file
AS_OF                         -> datetime.date: the question's "today"
CALENDAR                      -> lunar calendar DataFrame: date, weekday, lunar_day, lunar_month,
                                 lunar_year, lunar_leap, holiday (date is a 'YYYY-MM-DD' string)
README                        -> str: the shop's own README.md
pd, np, dt, re, math          -> pandas, numpy, datetime, re, math, already imported
"""

#: Marks the JSON result line in the subprocess' stdout; anything the snippet
#: prints itself stays visible above it for debugging.
RESULT_PREFIX = "__ANSWER__"

_RUNNER = '''\
import datetime as dt, json, math, re, sys
sys.path.insert(0, {root!r})
import numpy as np, pandas as pd
from utils.sandbox import to_jsonable
from utils.tables import Business

_biz = Business({data_dir!r}, {cache_dir!r})
tables, raw, table, sql = _biz.keys, _biz.raw, _biz.table, _biz.sql
README = _biz.readme
AS_OF = dt.date.fromisoformat({as_of!r})
CALENDAR = pd.read_csv({calendar!r}, dtype={{"date": str}})
ANSWER = None

# ---------------- model-written snippet ----------------
{snippet}
# -------------------------------------------------------

print({prefix!r} + json.dumps(to_jsonable(ANSWER), ensure_ascii=False))
'''


def project_root() -> Path:
    """Directory holding `inference.py` and the `utils` package."""
    return Path(__file__).resolve().parents[1]


@dataclass
class ExecResult:
    """Outcome of one snippet run."""

    #: Parsed `ANSWER`, or None when the snippet failed.
    answer: object | None
    #: Traceback / timeout / protocol error to feed back to the model, or None.
    error: str | None
    #: Whatever the snippet printed, kept for debugging.
    output: str = ""
    #: Where the executed script was written, so a failure can be re-run by hand.
    script: Path | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def to_jsonable(value: object) -> object:
    """Convert pandas/numpy values to plain JSON types, or raise.

    Raising matters: a silent `str()` fallback would turn a Timestamp into
    `"2026-05-20 00:00:00"` and lose the point on format. A TypeError instead
    reaches the model as a repairable error.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TypeError(f"ANSWER contains a non-finite number: {value!r}")
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date)):
        raise TypeError(
            f"ANSWER contains a date object ({value!r}); a date answer must be the "
            "string 'YYYY-MM-DD' (or 'YYYY-MM' for a month question)"
        )

    import numpy as np
    import pandas as pd

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if value is pd.NaT or value is pd.NA:
        raise TypeError("ANSWER contains a missing value (NaT/NA)")
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"ANSWER contains a {type(value).__name__}; convert it to a plain Python "
            "number, string, list or dict first"
        )
    raise TypeError(f"ANSWER contains an unsupported value of type {type(value).__name__}")


def run_snippet(
    snippet: str,
    data_dir: Path,
    calendar: Path,
    as_of: str,
    cache_dir: Path,
    script_path: Path,
    timeout: int = 180,
) -> ExecResult:
    """Execute `snippet` against one business and return its `ANSWER`.

    `script_path` is written (and kept) so a failing question can be re-run with
    `python <script_path>`.
    """
    root = project_root()
    script = _RUNNER.format(
        root=str(root),
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        calendar=str(calendar),
        as_of=as_of,
        snippet=snippet,
        prefix=RESULT_PREFIX,
    )
    script_path = script_path.resolve()
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")

    # Run from the project root with relative paths, so neither the interpreter
    # nor the script shows up as an absolute path in `ps` on a shared machine.
    try:
        proc = subprocess.run(
            [_relative_to(Path(sys.executable), root), _relative_to(script_path, root)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            None, f"TimeoutError: snippet ran longer than {timeout}s", script=script_path
        )

    if proc.returncode != 0:
        return ExecResult(None, _last_lines(proc.stderr), proc.stdout, script_path)

    result_lines = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_PREFIX)]
    if not result_lines:
        return ExecResult(
            None, "the snippet finished without producing ANSWER", proc.stdout, script_path
        )
    payload = result_lines[-1][len(RESULT_PREFIX) :]
    try:
        return ExecResult(json.loads(payload), None, proc.stdout, script_path)
    except json.JSONDecodeError as exc:
        return ExecResult(None, f"ANSWER was not JSON-serialisable: {exc}", proc.stdout, script_path)


def _relative_to(path: Path, root: Path) -> str:
    """`path` relative to `root` when it lives inside it, else unchanged."""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _last_lines(text: str, limit: int = 40) -> str:
    """Trim a traceback to its tail: the frames nearest the error matter most."""
    lines = text.strip().splitlines()
    return "\n".join(lines[-limit:]) if lines else "the snippet failed without a message"
