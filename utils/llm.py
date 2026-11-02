"""Settings and the single chat-completion call every entrypoint shares.

One place owns the OpenAI-compatible client, so `agent.py` and `translate.py`
cannot drift apart on base URL, retries or sampling. Clients are cached per
endpoint because they hold a connection pool and are safe to share across
threads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv


@dataclass
class Config:
    """Runtime settings, all overridable from `.env` (see `.env.example`)."""

    base_url: str = "http://localhost:4308/v1"
    api_key: str = "dummy"
    model: str = "default"
    temperature: float = 0.0
    max_tokens: int = 4096
    #: Extra attempts after a snippet failure. 0 disables the repair loop.
    max_repairs: int = 2
    #: Seconds a generated snippet may run before it is killed.
    exec_timeout: int = 180
    #: Raw rows shown per table in the prompt.
    preview_rows: int = 6

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Config":
        load_dotenv(env_file) if env_file else load_dotenv()
        return cls(
            base_url=os.getenv("DATAMATE_BASE_URL", cls.base_url),
            api_key=os.getenv("DATAMATE_API_KEY", cls.api_key),
            model=os.getenv("DATAMATE_MODEL", cls.model),
            temperature=float(os.getenv("DATAMATE_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("DATAMATE_MAX_TOKENS", cls.max_tokens)),
            max_repairs=int(os.getenv("DATAMATE_MAX_REPAIRS", cls.max_repairs)),
            exec_timeout=int(os.getenv("DATAMATE_EXEC_TIMEOUT", cls.exec_timeout)),
            preview_rows=int(os.getenv("DATAMATE_PREVIEW_ROWS", cls.preview_rows)),
        )


_clients: dict[tuple[str, str], object] = {}
_clients_lock = Lock()


def chat(config: Config, messages: list[dict], max_tokens: int | None = None) -> str:
    """One chat completion. Returns the reply text, "" if the model sent none."""
    client = _client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=max_tokens or config.max_tokens,
    )
    return response.choices[0].message.content or ""


def _client(config: Config):
    key = (config.base_url, config.api_key)
    with _clients_lock:
        if key not in _clients:
            from openai import OpenAI

            _clients[key] = OpenAI(base_url=config.base_url, api_key=config.api_key)
        return _clients[key]
