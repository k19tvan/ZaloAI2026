#!/usr/bin/env python
"""Translate a split's Vietnamese questions into English, one JSONL record each.

    uv run python translate.py ../../../dataset/track_2/train/train
    # writes ../../../dataset/track_2/train/train/questions_english.jsonl

Why: the questions are everyday shorthand ("sp bán dc nhiều nhất tuần rồi là
gì", "bao nhiu tiền"), with dropped diacritics and abbreviations. Translating
once, in a step that does nothing else, makes that the model's only job for
that call; `inference.py` then reads a clean sentence while still being graded
against the Vietnamese data.

The output keeps the original wording, so nothing is lost:

    {"id", "business_id", "as_of_date",          # copied from questions.jsonl
     "question",        # the English translation (what inference.py will read)
     "question_vi",     # the original Vietnamese, for debugging a bad answer
     "error"}           # only present if translation failed; 'question' then
                        #   falls back to the Vietnamese text

That schema is a drop-in for `inference.py --questions questions_english.jsonl`.

Entity names stay untranslated on purpose: product, customer and employee names
are matched character-by-character against the shop's files, so "Xi măng Hà Tiên
PCB40" must survive the trip unchanged. Relative dates ("tuần rồi") are
translated literally, never resolved — the rulebook defines what they mean, and
resolving them here would bake in an interpretation.

Re-running is safe: ids already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from utils import jsonl
from utils.llm import Config, chat

#: Name of the file written next to `questions.jsonl` unless `--out` says otherwise.
DEFAULT_FILENAME = "questions_english.jsonl"

SYSTEM_PROMPT = """\
You translate questions that Vietnamese shop owners type about their own sales records. \
Translate into natural English and output nothing but the translation: no quotes, no notes, \
no explanation, no restating the Vietnamese.

Rules:
- Keep proper nouns exactly as written, in Vietnamese, with their diacritics: product names \
("Xi măng Hà Tiên PCB40 bao 50kg", "Cà phê sữa (M)"), customer and employee names, shop \
category names, codes ("SP0001", "ORD000123"). They are matched against the shop's files \
character by character, so translating or re-spelling them breaks the answer.
- Keep units of measure as written ("thùng", "lốc", "bao 50kg", "ly", "viên", "cây").
- Translate relative time expressions literally, do not resolve them to dates: "tuần rồi" -> \
"last week", "tháng này" -> "this month", "hôm qua" -> "yesterday", "dịp Tết" -> "the Tết \
season", "ca chiều" -> "the afternoon shift".
- The input is shorthand with missing diacritics and abbreviations: "sp" = sản phẩm (product), \
"dc" = được, "nhiu"/"bao nhiu" = bao nhiêu (how much/how many), "kh" = khách hàng (customer), \
"nv" = nhân viên (employee), "dt" = doanh thu (revenue), "hđ" = hoá đơn (invoice), "ncc" = nhà \
cung cấp (supplier), "sl" = số lượng (quantity), "tt" = thanh toán (payment), "t5" = tháng 5 \
(May). Expand them, and keep the question's meaning exactly - do not add, drop or sharpen any \
condition.
- Keep the sentence a question, and keep it as vague as the original if it is vague.
"""

USER_PROMPT = "{question}"


def translate_question(config: Config, question: dict) -> dict:
    """Translate one question, falling back to the original text on any failure.

    A failed translation must never remove a question from the pipeline: the
    record keeps the Vietnamese text in `question` and records why, so the
    question is still answerable and the failure is still visible.
    """
    record = {
        "id": question["id"],
        "business_id": question["business_id"],
        "as_of_date": question["as_of_date"],
        "question": question["question"],
        "question_vi": question["question"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=question["question"].strip())},
    ]
    try:
        reply = chat(config, messages, max_tokens=512)
    except Exception as exc:  # noqa: BLE001 - one bad call must not stop the run
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    english = clean_reply(reply)
    if not english:
        record["error"] = f"empty or unusable reply: {reply[:200]!r}"
        return record
    record["question"] = english
    return record


def clean_reply(reply: str) -> str:
    """Strip the wrapping a chat model adds around a one-line answer.

    Keeps only the first non-empty line, since an obedient reply has exactly
    one and a chatty reply puts the translation first.
    """
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0]
    for opening, closing in ('""', "''", "“”", "``"):
        if text.startswith(opening) and text.endswith(closing) and len(text) > 1:
            text = text[1:-1].strip()
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.from_env(args.env_file)
    split_dir = Path(args.split_dir)

    questions = jsonl.read(split_dir / "questions.jsonl")
    out_path = Path(args.out) if args.out else split_dir / DEFAULT_FILENAME

    done = {record["id"] for record in jsonl.read(out_path)}
    todo = [q for q in questions if q["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} câu cần dịch ({len(done)} đã có trong {out_path})", file=sys.stderr)

    failures = 0
    lock = Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(translate_question, config, q): q for q in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                with lock:
                    failures += "error" in record
                    jsonl.write_line(out_fh, record)
                    if index % args.log_every == 0 or "error" in record:
                        print(
                            f"[{index}/{len(todo)}] {record['id']} "
                            f"-> {record.get('error') or record['question'][:70]}",
                            file=sys.stderr,
                        )

    print(f"xong: {out_path} ({failures} câu dịch lỗi, giữ nguyên tiếng Việt)", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("split_dir", help="thư mục split có questions.jsonl")
    parser.add_argument("--out", help=f"mặc định: <split_dir>/{DEFAULT_FILENAME}")
    parser.add_argument("--limit", type=int, help="chỉ dịch N câu đầu còn thiếu")
    parser.add_argument("--workers", type=int, default=8, help="số câu dịch song song")
    parser.add_argument("--log-every", type=int, default=50, help="in log mỗi N câu")
    parser.add_argument("--env-file", help="file .env khác mặc định")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
