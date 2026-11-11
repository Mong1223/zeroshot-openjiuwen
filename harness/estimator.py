"""Оценщик цены задачи для платного прохода (в единицах жюри: выход + вход/10).

Жюри (встреча 09.11): задач больше, чем можно решить в бюджет; по их гипотезе реально ~30%.
Поэтому важен ПОРЯДОК: сначала задачи с наибольшим «баллы / ожидаемая цена».

Цена считается по тому, как задача пойдёт по нашей лестнице, и по размеру её файлов.
Числа — средние по замерам 10.11 (data/varied, data/varied5, data/hard, llm_calls.jsonl):
    math    — выражение ~30; при «PROGRAM» программа ещё ~150
    data    — JSON-запрос ~75; несколько таблиц / незнакомый формат — программа ~200
    search  — цепочка полей ~35; база текстом — чтение целиком: вход/10 ≈ размер/30 + ~25
    coding  — исправленный файл: выход ≈ размер кода/3.5, вход/10 ≈ (код + тесты + вывод тестов)/30
    любой тип — с некоторой вероятностью уровень 1 не справится и включится агент (~1000)
"""
from __future__ import annotations

from harness.types import Task

AGENT_COST = 1000          # средняя цена агента openJiuwen (уровень 2)
CHARS_PER_TOKEN = 3.5

# доля задач, где уровень 1 не справляется и включается агент (замер 10.11: 0 из 60 «других» задач;
# закладываем небольшой запас — финальные задачи могут быть сложнее)
P_AGENT = {"math": 0.02, "data_analysis": 0.02, "search": 0.03, "coding": 0.03}
P_AGENT_UNKNOWN = 0.1


def _sizes(task: Task) -> dict[str, int]:
    out = {"code": 0, "tests": 0, "kb": 0, "tables": 0, "other": 0, "n_tables": 0, "n_code": 0}
    for name in task.files:
        try:
            size = (task.workspace / name).stat().st_size
        except OSError:
            continue
        low = name.lower()
        if low.endswith(".py"):
            is_test = low.rsplit("/", 1)[-1].startswith("test")
            out["tests" if is_test else "code"] += size
            out["n_code"] += not is_test
        elif low.startswith("kb/"):
            out["kb"] += size
        elif low.endswith((".csv", ".tsv", ".json")):
            out["tables"] += size
            out["n_tables"] += 1
        else:
            out["other"] += size
    return out


def level1_cost(task: Task) -> float:
    s = _sizes(task)
    question = len(task.instruction) / CHARS_PER_TOKEN / 10
    t = task.task_type
    if t == "math":
        return 25 + question + 0.07 * 150          # выражение; изредка — программа
    if t == "data_analysis":
        return (60 if s["n_tables"] == 1 else 200) + question
    if t == "search":
        if s["kb"] <= 12_000:  # маленькую базу при необходимости читаем целиком
            return 25 + question + 0.3 * (s["kb"] / CHARS_PER_TOKEN / 10)
        return 30 + question + 0.5 * 300  # большая база: цепочка или программа
    if t == "coding":
        if s["n_code"] > 1:  # несколько модулей — уровень 1 не берёт, сразу агент
            return AGENT_COST
        out = s["code"] / 4.5 + 20                  # модель возвращает файл целиком
        inp = (s["code"] + s["tests"] + 1500) / CHARS_PER_TOKEN / 10
        return out + inp + question
    return 120 + question + (s["other"] + s["tables"]) / CHARS_PER_TOKEN / 10 * 0.2


def expected_cost(task: Task) -> float:
    """Ожидаемая цена платного решения задачи (уровень 1 + возможный агент)."""
    p = P_AGENT.get(task.task_type, P_AGENT_UNKNOWN)
    return max(10.0, level1_cost(task) + p * AGENT_COST)
