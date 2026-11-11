"""Решатели по типам задач. Здесь — общий вход для уровня 0 (без ИИ)."""
from __future__ import annotations

import logging
from typing import Callable

from harness.solvers import coding_solver, data_solver, math_solver, search_solver
from harness.types import Result, Status, Task

log = logging.getLogger(__name__)

# Внимание: coding_solver меняет файлы в workspace задачи (это и есть ответ).
# Остальные решатели только читают.
LEVEL0: dict[str, Callable[[Task], Result]] = {
    "math": math_solver.solve,
    "data_analysis": data_solver.solve,
    "search": search_solver.solve,
    "coding": coding_solver.solve,
}


def solve_level0(task: Task) -> Result:
    """Бесплатная попытка. Любая ошибка внутри решателя → UNSURE, а не падение программы."""
    solver = LEVEL0.get(task.task_type)
    if solver is None:
        return Result(Status.UNSURE, note=f"no level-0 solver for {task.task_type!r}")
    try:
        result = solver(task)
    except Exception as e:
        log.warning("level0 %s crashed: %s: %s", task.task_id, type(e).__name__, e)
        return Result(Status.UNSURE, note=f"crash: {type(e).__name__}: {e}")
    result.level = 0
    return result
