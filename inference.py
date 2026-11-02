#!/usr/bin/env python
"""Answer a split's questions and write one resumable record per question.

    uv run python inference.py dataset/track_2/train/train --out runs/train.jsonl

A split directory is any folder holding `questions.jsonl`, `rulebook.md`,
`calendar.csv` and `data/<business_id>/`. Pass `--questions questions_english.jsonl`
to read the translations written by `translate.py` instead of the original
Vietnamese; records then carry whichever wording was actually sent.

Two files are produced:

- `--out` (records): everything about one question — the question itself, the
  pandas snippet the model generated, its final raw reply, every failed attempt
  with the error that killed it, and the answer that was submitted. This is the
  file to read when a question scores −1.
- `--submission` (default: `<out>.submission.jsonl`): the same answers reduced to
  the three fields the challenge accepts (`id`, `type`, `answer`). Rewritten
  wholesale from the records at the end of every run, so it can never drift out
  of sync with them.

Re-running is safe and cheap: questions already present in the records file are
skipped, so an interrupted run resumes instead of paying for them twice. Delete
a record line to force that question to be recomputed.

Record schema (one JSON object per line):

    id, business_id, question, as_of_date   # the question, copied verbatim
    type, answer                            # what was submitted ('answer' absent
                                            #   for unanswerable / abstain)
    code                                    # the snippet that produced it, or the
                                            #   last one tried if all failed
    reply                                   # the model's final raw reply
    attempts                                # [{code, error}, ...] failed cycles
    seconds                                 # wall-clock time for the question
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils import jsonl
from utils.agent import Agent, Config, Solution

log = logging.getLogger("inference")

#: Fields of a record that make up a submission line, in order.
SUBMISSION_FIELDS = ("id", "type", "answer")


def build_record(question: dict, solution: Solution, seconds: float) -> dict:
    """Flatten a solved question into one self-contained JSONL record."""
    final = solution.final
    return {
        "id": question["id"],
        "business_id": question["business_id"],
        "question": question["question"],
        "as_of_date": question["as_of_date"],
        **solution.answer,
        "code": final.code if final else None,
        "reply": final.reply if final else None,
        "attempts": [
            {"code": a.code, "error": a.error} for a in solution.attempts if a.error is not None
        ],
        "seconds": round(seconds, 1),
    }


def submission_line(record: dict) -> dict:
    """The upload payload for one record: id, type, and answer when it has one."""
    return {key: record[key] for key in SUBMISSION_FIELDS if key in record}


def solve_question(agent: Agent, question: dict) -> dict:
    """Solve one question, turning an unexpected crash into an abstain.

    A failure here is infrastructure (endpoint down, unreadable file), not a
    wrong answer: abstaining scores 0 while a guess risks −1. The error text is
    kept in the record so the run can be diagnosed afterwards.
    """
    started = time.monotonic()
    try:
        solution = agent.solve(question)
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
        solution = Solution({"type": "abstain"})
        record = build_record(question, solution, time.monotonic() - started)
        record["attempts"] = [{"code": None, "error": f"{type(exc).__name__}: {exc}"}]
        return record
    return build_record(question, solution, time.monotonic() - started)


def select_questions(args: argparse.Namespace, questions: list[dict], done: set[str]) -> list[dict]:
    """Apply the `--business` / `--qid` / `--limit` filters on top of resume."""
    todo = [q for q in questions if q["id"] not in done]
    if args.business:
        wanted = set(args.business)
        todo = [q for q in todo if q["business_id"] in wanted]
    if args.qid:
        wanted = set(args.qid)
        todo = [q for q in todo if q["id"] in wanted]
    return todo[: args.limit] if args.limit else todo


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    agent = Agent(args.split_dir, Config.from_env(args.env_file))

    split_dir = Path(args.split_dir)
    questions_path = Path(args.questions)
    if not questions_path.is_absolute() and not questions_path.exists():
        questions_path = split_dir / args.questions  # plain filename: look inside the split
    questions = jsonl.read(questions_path)
    out_path = Path(args.out)
    submission_path = Path(args.submission or f"{out_path.with_suffix('')}.submission.jsonl")

    done = {record["id"] for record in jsonl.read(out_path)}
    todo = select_questions(args, questions, done)
    log.info(
        "%d câu cần chạy từ %s (%d đã có trong %s)",
        len(todo),
        questions_path,
        len(done),
        out_path,
    )

    counts = {"answered": 0, "abstain": 0}
    lock = Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(solve_question, agent, q): q for q in todo}
            bar = tqdm(as_completed(futures), total=len(todo), unit="câu")
            with logging_redirect_tqdm():
                for future in bar:
                    record = future.result()
                    with lock:
                        counts["abstain" if record["type"] == "abstain" else "answered"] += 1
                        jsonl.write_line(out_fh, record)
                        bar.set_postfix(**counts)
                        log.info(
                            "%s %s -> %s (%.1fs)",
                            record["id"],
                            record["business_id"],
                            record["type"],
                            record["seconds"],
                        )
                        for attempt in record["attempts"]:
                            log.debug("%s lỗi: %s", record["id"], attempt["error"])
            bar.close()

    jsonl.write_all(submission_path, [submission_line(r) for r in jsonl.read(out_path)])
    log.info("submission: %s", submission_path)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir", help="thư mục split (có questions.jsonl, rulebook.md, data/)")
    parser.add_argument(
        "--questions",
        default="questions.jsonl",
        help="file câu hỏi; tên trần thì tìm trong split_dir (vd questions_english.jsonl)",
    )
    parser.add_argument("--out", default="runs/records.jsonl", help="file record, resume theo file này")
    parser.add_argument("--submission", help="mặc định: <out>.submission.jsonl")
    parser.add_argument("--limit", type=int, help="chỉ chạy N câu đầu còn thiếu")
    parser.add_argument("--business", nargs="+", help="lọc theo business_id")
    parser.add_argument("--qid", nargs="+", help="lọc theo id câu hỏi")
    parser.add_argument("--workers", type=int, default=4, help="số câu chạy song song")
    parser.add_argument("--env-file", help="file .env khác mặc định")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="debug thì in cả lỗi của từng attempt",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
