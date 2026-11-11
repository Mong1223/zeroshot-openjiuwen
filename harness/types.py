"""Разъёмы между блоками: что передаётся от читателя к решателям и обратно.

Task   — всё, что мы знаем о задаче до решения (читается бесплатно, без LLM).
Result — что вернул решатель.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Status(str, Enum):
    SOLVED = "solved"          # есть ответ, его можно записывать
    INFEASIBLE = "infeasible"  # задача невыполнима — answer.txt не создаём
    FAILED = "failed"          # не смогли решить (в т.ч. упёрлись в лимит)
    UNSURE = "unsure"          # уровень не уверен — нужен следующий уровень лестницы


@dataclass
class Task:
    task_id: str
    task_type: str             # math | data_analysis | search | coding | (что-то новое)
    instruction: str
    reward: float
    check: str                 # exact_int | exact_str | unittest | ...
    folder: Path               # папка задачи: здесь лежит task.json и пишется answer.txt
    workspace: Path            # folder/workspace — файлы задачи
    files: list[str] = field(default_factory=list)          # пути внутри workspace
    features: dict[str, Any] = field(default_factory=dict)  # бесплатные признаки для оценщика
    expected: str | None = None  # есть только в примерах и наших задачах; в финале — None
    raw: dict[str, Any] = field(default_factory=dict)       # task.json целиком, на всякий случай


@dataclass
class Result:
    status: Status
    answer: str | None = None
    confidence: float = 0.0    # 0..1 — насколько решатель уверен в ответе
    level: int = 0             # ступень лестницы: 0 правила, 1 ИИ-переводчик, 2 ИИ-агент
    tokens: float = 0.0        # потрачено по правилам соревнования: output + input / 10
    note: str = ""             # короткое пояснение для логов
