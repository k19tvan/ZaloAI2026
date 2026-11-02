"""Turns one question into a typed answer: prompt the model, run its code, repair.

The model never states a number itself. It writes a pandas snippet, the snippet
runs on the shop's real files, and whatever it computes becomes the answer. A
crash or a bad payload goes back to the model with the error attached, up to
`Config.max_repairs` times; if the loop still has no valid answer the question
is abstained, because the rulebook scores an abstain 0 and a wrong guess −1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import scoring
from .llm import Config, chat
from .sandbox import RUNTIME_API, ExecResult, project_root, run_snippet
from .tables import Business

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """\
You are a data assistant for a small Vietnamese shop. You never answer a question from \
reasoning alone: you write a short Python (pandas) snippet that reads the shop's own files, \
computes the result, and assigns it to `ANSWER`.

Questions, the rulebook and the data are in Vietnamese. Keep every name you report exactly as \
it is spelled in the shop's files, including diacritics.

# Your snippet
- One single ```python block. No prose outside the block.
- These names already exist (import nothing, never open a file by hand-written path):
{runtime_api}
- Finish with `ANSWER = {{"type": "<type>", "answer": <value>}}`.
- If the released data cannot answer the question (the shop does not record that kind of \
information, or the asked period lies entirely outside the shop's data range), assign \
`ANSWER = {{"type": "unanswerable"}}`.
- Do not wrap the snippet in try/except to hide failures: on an error you get the traceback \
back and can fix it. A snippet that crashes beats a snippet that returns a wrong number.

# File-reading traps (check them against the RAW preview before writing code)
- Row 0 is often a report title and row 1 a filter line, so the real header may sit on row 3: \
pick the right `header_row`.
- Two-row headers: column names are split across two consecutive rows and must be joined.
- Fill-down: customer, status, shipping fee and order total are written only on the order's \
first line; later blank cells mean "same as the first line", not missing data (`ffill` within \
the order).
- Dirty cells: `100,000`, `55.000đ`, `55k`, `3 chai` inside a quantity column; dates appear as \
`01/04/2025`, `1 thg 5 2025`, real datetimes, or Excel serial numbers.
- When a row does not carry a year, take it from the sheet or file name (`T04_2025`, \
`Thang 02 2025`).
- Ledgers split across files or sheets (`_1`, `_2`, `_2026`, one file per month): concatenate \
all of them.
- Two identical lines inside the SAME order are an export glitch and count once; identical \
lines in different orders are two real sales and both count.
- Files still contain records dated after AS_OF: filter on AS_OF yourself, never trust the \
file's last date.
- Quantities are written in the unit of their own line: convert to the base unit using the \
conversion table or the README.

# Rulebook (how answers are computed; follow it strictly)
{rulebook}
"""

QUESTION_PROMPT = """\
# Shop `{business_id}`
{profile}

# Question
{question}

Today (AS_OF) = {as_of_date}

Write the Python snippet that computes the answer and assigns `ANSWER`.
"""

REPAIR_PROMPT = """\
The snippet did not produce a usable answer:

{error}

Re-read the table preview, fix the cause, then rewrite the WHOLE snippet in one ```python block.
"""


@dataclass
class Attempt:
    """One generate-and-run cycle, kept so a run can be inspected afterwards."""

    #: Raw model reply.
    reply: str
    #: Snippet extracted from the reply, or None if it contained no code block.
    code: str | None
    #: Why this cycle did not yield a usable answer, or None if it did.
    error: str | None
    #: `ANSWER` as produced by the snippet, even when it failed the format gate.
    answer: object | None = None


@dataclass
class Solution:
    """Final answer for one question plus every attempt that led to it."""

    #: Submission payload, always valid; `{"type": "abstain"}` when nothing worked.
    answer: dict
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.answer.get("type") == "abstain"

    @property
    def final(self) -> Attempt | None:
        """The attempt whose answer was submitted, or the last failure."""
        return self.attempts[-1] if self.attempts else None


class Agent:
    """Answers questions for one split (one rulebook, one calendar)."""

    def __init__(
        self,
        split_dir: str | Path,
        config: Config | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.split_dir = Path(split_dir)
        self.config = config or Config.from_env()
        # Caches (table pickles, executed snippets) stay in the project, never
        # inside the dataset directory.
        default_cache = project_root() / ".cache" / self.split_dir.name
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache
        self.rulebook = (self.split_dir / "rulebook.md").read_text(encoding="utf-8")
        self.calendar = self.split_dir / "calendar.csv"
        self._profiles: dict[str, str] = {}

    # -- prompt ------------------------------------------------------------

    def business(self, business_id: str) -> Business:
        return Business(self.split_dir / "data" / business_id, self.cache_dir / business_id)

    def profile(self, business_id: str) -> str:
        """Cached prompt text for a shop: its README plus a preview per table."""
        if business_id not in self._profiles:
            self._profiles[business_id] = self.business(business_id).profile(
                self.config.preview_rows
            )
        return self._profiles[business_id]

    def build_messages(self, question: dict) -> list[dict]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    runtime_api=RUNTIME_API, rulebook=self.rulebook.strip()
                ),
            },
            {
                "role": "user",
                "content": QUESTION_PROMPT.format(
                    business_id=question["business_id"],
                    profile=self.profile(question["business_id"]),
                    question=question["question"].strip(),
                    as_of_date=question["as_of_date"],
                ),
            },
        ]

    # -- solving -----------------------------------------------------------

    def solve(self, question: dict) -> Solution:
        """Generate, run and (if needed) repair a snippet for one question."""
        messages = self.build_messages(question)
        attempts: list[Attempt] = []

        for cycle in range(self.config.max_repairs + 1):
            reply = chat(self.config, messages)
            code = extract_code(reply)

            if code is None:
                attempt = Attempt(reply, None, "the reply contained no ```python block")
            else:
                result = self._execute(question, code, cycle)
                error = result.error or scoring.validate(result.answer or {})
                attempt = Attempt(reply, code, error, result.answer)
            attempts.append(attempt)

            if attempt.error is None:
                assert isinstance(attempt.answer, dict)
                return Solution(attempt.answer, attempts)

            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": REPAIR_PROMPT.format(error=attempt.error)},
            ]

        return Solution({"type": "abstain"}, attempts)

    def _execute(self, question: dict, code: str, cycle: int) -> ExecResult:
        return run_snippet(
            snippet=code,
            data_dir=self.split_dir / "data" / question["business_id"],
            calendar=self.calendar,
            as_of=question["as_of_date"],
            cache_dir=self.cache_dir / question["business_id"],
            script_path=self.cache_dir / "snippets" / f"{question['id']}_{cycle}.py",
            timeout=self.config.exec_timeout,
        )


def extract_code(reply: str) -> str | None:
    """The last fenced block of a reply, or None if it has none.

    Last, not first: a model that explains itself before rewriting the snippet
    leaves the usable version at the end.
    """
    blocks = _CODE_FENCE_RE.findall(reply)
    return blocks[-1].strip() if blocks else None
