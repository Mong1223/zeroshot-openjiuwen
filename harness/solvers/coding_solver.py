"""Решатель coding, уровень 0: тесты → перебор подмен → тесты. 0 токенов.

1. Запускаем тесты. Уже проходят — ответ DONE.
2. Жадно ищем исправление: пробуем одиночные подмены (сначала в функциях, которые вызывают
   упавшие тесты, сначала самые «типичные»). Подмену оставляем, только если упавших тестов
   стало меньше и ни один прошедший не сломался. Повторяем, пока есть улучшения.
3. Все тесты прошли и файлы тестов не изменились — DONE. Иначе возвращаем код как был
   и отвечаем UNSURE: дальше пусть работает ИИ.

Жюри проверяет задачу скрытыми тестами, поэтому «рискованные» подмены (±1, замена переменной)
снижают уверенность — планировщик может перепроверить такие решения.
"""
from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path

from harness.mutations import Mutation, apply, mutations
from harness.testrun import TestReport, run_tests
from harness.types import Result, Status, Task

MAX_SECONDS = 45
MAX_RUNS = 300
FIRST_RUN_TIMEOUT = 10  # первый запуск тестов
HANG_RUNS = 15          # если тесты зависают, пробуем совсем немного подмен


def solve(task: Task, max_seconds: float = MAX_SECONDS, max_runs: int = MAX_RUNS) -> Result:
    ws = task.workspace
    tests = sorted(ws.glob("test*.py"))
    targets = sorted(p for p in ws.glob("*.py") if p not in tests)
    if not tests or not targets:
        return Result(Status.UNSURE, note=f"tests={len(tests)} targets={len(targets)}")

    deadline = time.monotonic() + max_seconds
    test_hash = _hash(tests)
    original = {p: p.read_text(encoding="utf-8") for p in targets}
    report = run_tests(ws, timeout=min(FIRST_RUN_TIMEOUT, max_seconds))
    if report.ok:
        return Result(Status.SOLVED, answer="DONE", confidence=1.0, note="tests already pass")
    if report.broken and _tests_broken(tests):
        return Result(Status.INFEASIBLE, confidence=0.7, note="test file itself does not parse")

    # Таймаут одного прогона — от обычной длительности тестов: подмена, которая зациклила код,
    # не должна съедать по 10–20 секунд. Если зависает уже исходный код — подмен пробуем мало.
    run_timeout = 2.0 if report.timed_out else max(2.0, 5 * report.seconds)
    if report.timed_out:
        max_runs = min(max_runs, HANG_RUNS)

    current = dict(original)
    accepted: list[Mutation] = []
    runs = 0

    improved = True
    while improved and not report.ok and time.monotonic() < deadline and runs < max_runs:
        improved = False
        focus = _called_by(tests, report.failed)
        for target in targets:
            try:
                tree = ast.parse(current[target])
            except SyntaxError:
                continue
            candidates = sorted(mutations(tree), key=lambda m: (m.function not in focus, m.priority))
            for mutation in candidates:
                if time.monotonic() >= deadline or runs >= max_runs:
                    break
                try:
                    source = apply(tree, mutation)
                except Exception:
                    continue
                target.write_text(source, encoding="utf-8")
                runs += 1
                new = run_tests(ws, timeout=min(run_timeout, deadline - time.monotonic()))
                if _better(new, report):
                    current[target], report = source, new
                    accepted.append(mutation)
                    improved = True
                    break
                target.write_text(current[target], encoding="utf-8")
            if improved:
                break

    fixes = "; ".join(f"[{m.function}] {m.description}" for m in accepted)
    if report.ok and _hash(tests) == test_hash:
        worst = max((m.priority for m in accepted), default=1)
        confidence = {1: 1.0, 2: 0.85, 3: 0.7}[worst]
        return Result(Status.SOLVED, answer="DONE", confidence=confidence,
                      note=f"{len(accepted)} fix(es), {runs} runs: {fixes}")

    _restore(original)
    return Result(Status.UNSURE, note=f"mutations not enough after {runs} runs; "
                                      f"still failing {sorted(report.failed)}; partial: {fixes}")


def _better(new: TestReport, old: TestReport) -> bool:
    return not new.broken and new.score < old.score and old.passed <= new.passed


def _called_by(tests: list[Path], failed: set[str]) -> set[str]:
    """Имена функций, которые вызывают упавшие тесты: m.total(...) → total."""
    names: set[str] = set()
    for path in tests:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in failed:
                for call in ast.walk(node):
                    if isinstance(call, ast.Call):
                        f = call.func
                        names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def _tests_broken(tests: list[Path]) -> bool:
    for path in tests:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            return True
    return False


def _hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def _restore(original: dict[Path, str]) -> None:
    for path, text in original.items():
        path.write_text(text, encoding="utf-8")
