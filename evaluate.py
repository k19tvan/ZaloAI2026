#!/usr/bin/env python
"""Score a run against a split's `answers.jsonl` (Penalized Accuracy).

    uv run python evaluate.py dataset/track_2/train/train runs/records.jsonl

Reads either file produced by `inference.py` — records or submission — since
both carry `id`, `type` and `answer`. Prints the overall Penalized Accuracy
(correct +1, wrong −1, abstain 0, over N) with a breakdown per answer type and
per shop, which is what tells you whether a loss comes from one shop's parsing
or from one rule being applied wrong everywhere.

A question missing from the run counts as an abstain, exactly as the grader
treats a missing submission line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils import jsonl, scoring


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_dir = Path(args.split_dir)

    answers_path = split_dir / "answers.jsonl"
    if not answers_path.exists():
        print(f"{answers_path} không tồn tại (split này không công bố đáp án)", file=sys.stderr)
        return 1

    references = {row["id"]: row for row in jsonl.read(answers_path)}
    predictions = {row["id"]: row for row in jsonl.read(args.run)}
    business_of = {q["id"]: q["business_id"] for q in jsonl.read(split_dir / "questions.jsonl")}

    if args.only_answered:
        # For a partial run: score just what was attempted, so the number is not
        # diluted by thousands of untouched questions.
        references = {qid: ref for qid, ref in references.items() if qid in predictions}
        if not references:
            print("không có câu nào trong run khớp với answers.jsonl", file=sys.stderr)
            return 1

    report = scoring.score(predictions, references, business_of)
    print(report)

    if args.mistakes:
        print(f"\nsai {len(report.mistakes)} câu, in {min(args.mistakes, len(report.mistakes))} câu:")
        for qid, kind in report.mistakes[: args.mistakes]:
            prediction = predictions.get(qid, {})
            print(
                f"\n  {qid}  ref_type={kind}"
                f"\n    câu hỏi: {prediction.get('question', '?')}"
                f"\n    đáp án đúng: {json.dumps(references[qid].get('answer'), ensure_ascii=False)}"
                f"\n    nộp:        {json.dumps(prediction.get('answer'), ensure_ascii=False)}"
            )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir", help="thư mục split có answers.jsonl")
    parser.add_argument("run", help="records.jsonl hoặc submission.jsonl từ inference.py")
    parser.add_argument(
        "--only-answered",
        action="store_true",
        help="chỉ chấm các câu có trong run (dùng khi mới chạy thử một phần)",
    )
    parser.add_argument("--mistakes", type=int, default=0, help="in N câu sai kèm đáp án đúng")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
