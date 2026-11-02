"""Answer format gate and the offline Penalized Accuracy scorer (rulebook §9).

Two jobs, kept together because both encode the same section of the rulebook:

- `validate(answer)` rejects a payload the grader would mark wrong on format
  alone (a money string, a percent written as a ratio, a bare string where a
  list is required). A rejected answer becomes an abstain, which scores 0
  instead of −1.
- `is_correct(prediction, reference)` reproduces the grader's comparison, so
  `datamate score` reports the same number the leaderboard would.

Penalized Accuracy: correct +1, wrong −1, abstain 0, divided by N.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

#: Types a submission line may carry. `abstain` always scores 0.
ANSWER_TYPES = (
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
)

#: Percent and non-money averages are compared at two decimals (§9).
DECIMALS = 2

_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
_NON_TEXT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


# --------------------------------------------------------------------------
# format gate
# --------------------------------------------------------------------------


def validate(answer: dict) -> str | None:
    """Return None if `answer` is a well-formed submission payload, else why not.

    Checks shape only. A number with the right format but the wrong value
    passes here and is caught by the grader.
    """
    kind = answer.get("type")
    if kind not in ANSWER_TYPES:
        return f"unknown type {kind!r}"
    if kind in ("unanswerable", "abstain"):
        return None

    if "answer" not in answer:
        return f"type {kind} needs an 'answer' field"
    value = answer["answer"]

    if kind in ("number", "percent"):
        return None if _is_number(value) else f"{kind} must be a JSON number, got {value!r}"
    if kind == "string":
        return None if isinstance(value, str) and value.strip() else "string must be non-empty text"
    if kind == "date":
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return "date must be 'YYYY-MM-DD' or 'YYYY-MM'"
        return None
    if kind == "boolean":
        return None if isinstance(value, bool) else "boolean must be JSON true/false"
    if kind == "list":
        if not isinstance(value, list) or not value:
            return "list must be a non-empty JSON array"
        bad = [v for v in value if not (_is_number(v) or isinstance(v, str))]
        return f"list items must be numbers or strings, got {bad[:1]!r}" if bad else None
    if kind == "table":
        return _validate_table(value)
    return None


def _validate_table(value: object) -> str | None:
    if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
        return "table must be {'columns': [...], 'rows': [[...]]}"
    columns, rows = value["columns"], value["rows"]
    if not isinstance(columns, list) or not columns:
        return "table needs a non-empty 'columns' list"
    if not isinstance(rows, list):
        return "table 'rows' must be a list of rows"
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            return f"every row needs {len(columns)} cells, got {row!r}"
    return None


def _is_number(value: object) -> bool:
    # bool is an int subclass in Python; the grader treats the two as distinct.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# value matching
# --------------------------------------------------------------------------


def normalize_text(value: object) -> str:
    """Grader's string normalization: NFC, lowercase, punctuation to space.

    Diacritics are preserved, so "Sting dâu" and "Sting dầu" stay different.
    """
    text = unicodedata.normalize("NFC", str(value)).lower()
    return " ".join(_NON_TEXT_RE.sub(" ", text).split())


def numbers_match(a: object, b: object) -> bool:
    """Equality at two decimals, tolerating float representation error only."""
    if not (_is_number(a) and _is_number(b)):
        return False
    if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in (a, b)):
        return False
    return abs(round(float(a), DECIMALS) - round(float(b), DECIMALS)) < 1e-9


def is_correct(prediction: dict, reference: dict) -> bool:
    """Whether `prediction` earns +1 against an `answers.jsonl` row.

    `reference["alias"]` holds the other spellings the grader accepts: a flat
    list for `string`, one list per position for `list`, and one list per row
    (matching the row's name cell) for `table`.
    """
    kind = reference["type"]
    if prediction.get("type") != kind:
        return False
    if kind in ("unanswerable", "abstain"):
        return kind == "unanswerable"  # abstain never earns a point

    predicted, expected = prediction.get("answer"), reference.get("answer")
    alias = reference.get("alias") or []

    if kind in ("number", "percent"):
        return numbers_match(predicted, expected)
    if kind == "boolean":
        return isinstance(predicted, bool) and predicted is expected
    if kind == "date":
        return isinstance(predicted, str) and predicted == expected
    if kind == "string":
        return _name_matches(predicted, expected, alias)
    if kind == "list":
        return _list_matches(predicted, expected, alias)
    if kind == "table":
        return _table_matches(predicted, expected, alias)
    return False


def _name_matches(predicted: object, expected: object, alias: list) -> bool:
    accepted = {normalize_text(expected)} | {normalize_text(a) for a in alias}
    return isinstance(predicted, str) and normalize_text(predicted) in accepted


def _cell_matches(predicted: object, expected: object, alias: list) -> bool:
    """One list item or table cell: number rule for numbers, name rule for text."""
    if _is_number(expected):
        return numbers_match(predicted, expected)
    return _name_matches(predicted, expected, alias)


def _list_matches(predicted: object, expected: list, alias: list) -> bool:
    if not isinstance(predicted, list) or len(predicted) != len(expected):
        return False
    return all(
        _cell_matches(p, e, alias[i] if i < len(alias) else [])
        for i, (p, e) in enumerate(zip(predicted, expected))
    )


def _table_matches(predicted: object, expected: dict, alias: list) -> bool:
    if _validate_table(predicted) is not None:
        return False
    assert isinstance(predicted, dict)
    rows = _align_columns(predicted, expected)
    if rows is None:
        return False
    expected_rows = [
        (row, alias[i] if i < len(alias) else []) for i, row in enumerate(expected["rows"])
    ]
    return _rows_match_as_set(rows, expected_rows)


def _align_columns(predicted: dict, expected: dict) -> list[list] | None:
    """Reorder predicted cells to the reference column order (§9: name, else position)."""
    want = [normalize_text(c) for c in expected["columns"]]
    have = [normalize_text(c) for c in predicted["columns"]]
    if len(have) != len(want):
        return None
    if sorted(have) == sorted(want):
        order = [have.index(name) for name in want]
        return [[row[i] for i in order] for row in predicted["rows"]]
    return [list(row) for row in predicted["rows"]]  # names unknown: trust position


def _rows_match_as_set(predicted: list[list], expected: list[tuple[list, list]]) -> bool:
    """Row order is ignored, so pair rows up; backtrack because aliases can tie."""
    if len(predicted) != len(expected):
        return False
    used = [False] * len(expected)

    def place(index: int) -> bool:
        if index == len(predicted):
            return True
        for j, (row, row_alias) in enumerate(expected):
            if used[j] or not _row_matches(predicted[index], row, row_alias):
                continue
            used[j] = True
            if place(index + 1):
                return True
            used[j] = False
        return False

    return place(0)


def _row_matches(predicted: list, expected: list, row_alias: list) -> bool:
    # §9: the first column is the entity name (aliases apply), the rest are measures.
    return all(
        _cell_matches(p, e, row_alias if i == 0 else [])
        for i, (p, e) in enumerate(zip(predicted, expected))
    )


# --------------------------------------------------------------------------
# aggregate report
# --------------------------------------------------------------------------


@dataclass
class Report:
    """Penalized Accuracy over one split, with per-type and per-shop breakdown."""

    total: int = 0
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    by_type: Counter = field(default_factory=Counter)
    by_business: Counter = field(default_factory=Counter)
    #: (question_id, reference_type) of every question marked wrong.
    mistakes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def penalized_accuracy(self) -> float:
        return (self.correct - self.wrong) / self.total if self.total else 0.0

    def __str__(self) -> str:
        lines = [
            f"N={self.total}  correct={self.correct}  wrong={self.wrong}  "
            f"abstain={self.abstained}",
            f"Penalized Accuracy = {self.penalized_accuracy:.4f}",
            "",
            f"{'reference type':<16}{'n':>6}{'ok':>6}{'wrong':>7}{'abstain':>9}",
        ]
        for kind in ANSWER_TYPES:
            n = self.by_type[(kind, "n")]
            if n:
                lines.append(
                    f"{kind:<16}{n:>6}{self.by_type[(kind, 'ok')]:>6}"
                    f"{self.by_type[(kind, 'wrong')]:>7}{self.by_type[(kind, 'abstain')]:>9}"
                )
        lines += ["", f"{'business':<16}{'n':>6}{'ok':>6}{'wrong':>7}{'abstain':>9}{'PA':>8}"]
        shops = sorted({key[0] for key in self.by_business})
        for shop in shops:
            n = self.by_business[(shop, "n")]
            ok = self.by_business[(shop, "ok")]
            wrong = self.by_business[(shop, "wrong")]
            lines.append(
                f"{shop:<16}{n:>6}{ok:>6}{wrong:>7}"
                f"{self.by_business[(shop, 'abstain')]:>9}{(ok - wrong) / n:>8.3f}"
            )
        return "\n".join(lines)


def score(
    predictions: dict[str, dict],
    references: dict[str, dict],
    business_of: dict[str, str] | None = None,
) -> Report:
    """Score every reference question; a missing prediction counts as abstain (§9)."""
    report = Report(total=len(references))
    business_of = business_of or {}
    for qid, reference in references.items():
        prediction = predictions.get(qid) or {"type": "abstain"}
        kind = reference["type"]
        if prediction.get("type") == "abstain":
            outcome = "abstain"
            report.abstained += 1
        elif is_correct(prediction, reference):
            outcome = "ok"
            report.correct += 1
        else:
            outcome = "wrong"
            report.wrong += 1
            report.mistakes.append((qid, kind))
        report.by_type[(kind, "n")] += 1
        report.by_type[(kind, outcome)] += 1
        shop = business_of.get(qid, "?")
        report.by_business[(shop, "n")] += 1
        report.by_business[(shop, outcome)] += 1
    return report
