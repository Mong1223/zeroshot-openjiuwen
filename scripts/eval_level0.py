"""Проверка бесплатных решателей (уровень 0) на папке задач с известными ответами.

    python scripts\\eval_level0.py                  # data/examples
    python scripts\\eval_level0.py путь\\к\\задачам

Работает на временной копии задач (решатель coding правит файлы) и не тратит токены.
Для coding ответ проверяется по-настоящему: тесты в копии должны проходить.
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.solvers import solve_level0  # noqa: E402
from harness.tasks import normalize_answer, read_tasks  # noqa: E402
from harness.testrun import run_tests  # noqa: E402
from harness.types import Status  # noqa: E402


def verdict(task, result) -> str:
    if task.expected is None:
        return "?"
    if task.expected == "INFEASIBLE":  # так помечаем нерешаемые в наших тестовых задачах
        return "✓" if result.status is Status.INFEASIBLE else "✗"
    if result.status is not Status.SOLVED:
        return "·"  # не взялся — не ошибка, задача уйдёт на следующий уровень
    if task.check == "unittest":
        return "✓" if run_tests(task.workspace).ok else "✗"
    return "✓" if normalize_answer(task, result.answer) == task.expected.strip() else "✗"


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "examples"
    with tempfile.TemporaryDirectory(prefix="eval0-") as tmp:
        copy = Path(tmp) / "tasks"
        shutil.copytree(source, copy)
        tasks = read_tasks(copy)
        counts = {"✓": 0, "✗": 0, "·": 0, "?": 0}
        started = time.perf_counter()
        for task in tasks:
            t0 = time.perf_counter()
            result = solve_level0(task)
            v = verdict(task, result)
            counts[v] += 1
            answer = result.answer if result.status is Status.SOLVED else result.status.value
            print(f"{v} {task.task_id:<22} {answer!s:<12} ожид. {task.expected!s:<11} "
                  f"{time.perf_counter() - t0:5.1f}s  {result.note[:90]}")
    print(f"\nверно {counts['✓']}, ошибок {counts['✗']}, не взялся {counts['·']}, без ответа {counts['?']} "
          f"из {len(tasks)} — потрачено 0 токенов, {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
