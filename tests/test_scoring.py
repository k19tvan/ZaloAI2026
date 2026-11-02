"""Checks the logic a wrong answer hides behind: the format gate, the grader's
value matching, and the two conversions in between (`to_jsonable`, `extract_code`).

Run with `uv run python tests/test_scoring.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import scoring
from utils.agent import extract_code
from utils.sandbox import to_jsonable


def test_format_gate():
    assert scoring.validate({"type": "number", "answer": 4820000}) is None
    assert scoring.validate({"type": "percent", "answer": -12.4}) is None
    assert scoring.validate({"type": "unanswerable"}) is None
    assert scoring.validate({"type": "date", "answer": "2025-09"}) is None
    assert scoring.validate({"type": "list", "answer": ["Cà phê sữa"]}) is None

    # Each of these is scored −1 by the grader, so it must not leave the agent.
    assert scoring.validate({"type": "number", "answer": "4820000"})          # money as text
    assert scoring.validate({"type": "number", "answer": True})               # bool is not a number
    assert scoring.validate({"type": "date", "answer": "15/08/2026"})         # wrong date format
    assert scoring.validate({"type": "boolean", "answer": 1})                 # 1 is not true
    assert scoring.validate({"type": "string", "answer": ["Cà phê sữa"]})     # list where string asked
    assert scoring.validate({"type": "list", "answer": "Cà phê sữa, Bạc xỉu"})
    assert scoring.validate({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x"]]}})
    assert scoring.validate({"type": "revenue", "answer": 1})                 # unknown type


def test_number_and_percent_matching():
    ref = {"type": "number", "answer": 89526400}
    assert scoring.is_correct({"type": "number", "answer": 89526400}, ref)
    assert not scoring.is_correct({"type": "number", "answer": 89526401}, ref)   # money is exact
    assert not scoring.is_correct({"type": "percent", "answer": 89526400}, ref)  # type must match

    pct = {"type": "percent", "answer": 2.08}
    assert scoring.is_correct({"type": "percent", "answer": 2.0800000001}, pct)  # float noise only
    assert not scoring.is_correct({"type": "percent", "answer": 0.0208}, pct)    # ratio, not percent
    assert not scoring.is_correct({"type": "percent", "answer": -2.08}, pct)     # sign must match


def test_string_matching_rules():
    ref = {"type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
    assert scoring.is_correct({"type": "string", "answer": "xi măng hà tiên pcb40  bao 50kg"}, ref)
    assert scoring.is_correct({"type": "string", "answer": "SP0001"}, ref)  # code is accepted
    assert not scoring.is_correct({"type": "string", "answer": "Xi mang Ha Tien PCB40 bao 50kg"}, ref)


def test_list_is_ordered_and_alias_is_positional():
    ref = {
        "type": "list",
        "answer": ["Cà phê sữa (M)", "Trà đào cam sả (M)"],
        "alias": [["SP0003"], ["SP0017"]],
    }
    assert scoring.is_correct({"type": "list", "answer": ["Cà phê sữa (M)", "SP0017"]}, ref)
    assert not scoring.is_correct({"type": "list", "answer": ["SP0017", "SP0003"]}, ref)  # order
    assert not scoring.is_correct({"type": "list", "answer": ["Cà phê sữa (M)"]}, ref)    # length


def test_table_columns_by_name_rows_as_set():
    ref = {
        "type": "table",
        "answer": {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158600]],
        },
        "alias": [["KH0124"], ["KH0233"]],
    }
    swapped_rows = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["KH0233", 158600], ["Đỗ Đức An", 404500]],
    }
    assert scoring.is_correct({"type": "table", "answer": swapped_rows}, ref)

    swapped_columns = {
        "columns": ["so_tien_no", "khach_hang"],
        "rows": [[404500, "Đỗ Đức An"], [158600, "Dương Tâm Bích"]],
    }
    assert scoring.is_correct({"type": "table", "answer": swapped_columns}, ref)

    wrong_amount = {
        "columns": ["khach_hang", "so_tien_no"],
        "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158000]],
    }
    assert not scoring.is_correct({"type": "table", "answer": wrong_amount}, ref)


def test_unanswerable_and_abstain():
    ref = {"type": "unanswerable", "answer": None}
    assert scoring.is_correct({"type": "unanswerable"}, ref)
    assert not scoring.is_correct({"type": "abstain"}, ref)          # abstain never earns +1
    assert not scoring.is_correct({"type": "number", "answer": 0}, ref)


def test_penalized_accuracy():
    references = {
        "a": {"type": "number", "answer": 10},
        "b": {"type": "number", "answer": 20},
        "c": {"type": "number", "answer": 30},
        "d": {"type": "number", "answer": 40},  # no prediction at all
    }
    predictions = {
        "a": {"type": "number", "answer": 10},
        "b": {"type": "number", "answer": 99},
        "c": {"type": "abstain"},
    }
    report = scoring.score(predictions, references)
    assert (report.correct, report.wrong, report.abstained) == (1, 1, 2)
    assert report.penalized_accuracy == 0.0
    assert report.mistakes == [("b", "number")]


def test_jsonable_refuses_lossy_values():
    import numpy as np
    import pandas as pd

    assert to_jsonable({"type": "number", "answer": np.int64(5)}) == {"type": "number", "answer": 5}
    assert to_jsonable([np.float64(1.5), np.bool_(True)]) == [1.5, True]

    for bad in (pd.Timestamp("2026-05-20"), float("nan"), pd.Series([1]), pd.NaT):
        try:
            to_jsonable({"answer": bad})
        except TypeError:
            continue
        raise AssertionError(f"to_jsonable silently accepted {bad!r}")


def test_extract_code_takes_the_last_block():
    reply = "thử lần 1:\n```python\nANSWER = 1\n```\nsửa lại:\n```python\nANSWER = 2\n```"
    assert extract_code(reply) == "ANSWER = 2"
    assert extract_code("không có code") is None


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"ok  {name}")
    print("all checks passed")
