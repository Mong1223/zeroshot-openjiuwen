"""Блок [A]: настройки. Всё берётся из переменных окружения; локально — ещё и из Project/.env.

На проверке жюри передаёт LLM_BASE_URL, LLM_API_KEY и BUDGET. Остальное — наши значения
по умолчанию, их можно переопределить переменными окружения при отладке.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file() or os.getenv("DOTENV_DISABLED"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env, override=False)  # переменные окружения жюри важнее файла


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    budget: float                 # бюджет в токенах: output + input / 10
    reserve: float                # доля бюджета, которую не трогаем никогда
    time_limit: float             # секунд на весь прогон (жюри даёт 1 час; берём с запасом)
    llm_concurrency: int          # одновременных запросов к модели
    task_concurrency: int         # одновременно решаемых задач (регламент: до 5 сессий)
    fast_model: str
    strong_model: str
    trace_dir: Path
    llm_rpm: int = 50             # запросов к модели в минуту (лимит ключа — 60, держим запас)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            budget=_float("BUDGET", 100_000),
            reserve=_float("BUDGET_RESERVE", 0.03),
            time_limit=_float("TIME_LIMIT", 50 * 60),
            llm_concurrency=int(_float("LLM_CONCURRENCY", 2)),
            llm_rpm=int(_float("LLM_RPM", 50)),
            task_concurrency=int(_float("TASK_CONCURRENCY", 4)),
            fast_model=os.getenv("FAST_MODEL", "qwen3.8-27b-noreason"),
            strong_model=os.getenv("STRONG_MODEL", "qwen3.8-27b"),
            trace_dir=Path(os.getenv("TRACE_DIR") or Path(tempfile.gettempdir()) / "zeroshot"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)
