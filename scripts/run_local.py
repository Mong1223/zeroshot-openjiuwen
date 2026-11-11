"""Репетиция проверки жюри: запускаем run_agent.py на КОПИИ задач и считаем баллы так же, как жюри.

    python scripts\\run_local.py                          # data/examples
    python scripts\\run_local.py data\\hard --budget 5000
    python scripts\\run_local.py --no-llm                 # только бесплатные решатели

Подсчёт: exact_int — целое число, exact_str — строка, unittest — тесты в копии проходят.
Задача с expected = INFEASIBLE засчитывается, если answer.txt НЕ создан.
Логи и траектории — в runs/<время>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.tasks import read_tasks  # noqa: E402
from harness.testrun import run_tests  # noqa: E402


def tests_pass_with_hidden(task) -> bool:
    """Как у жюри: видимые тесты + скрытые (папка hidden/ рядом с task.json, агент её не видит)."""
    if not run_tests(task.workspace).ok:
        return False
    hidden = task.folder / "hidden"
    if not hidden.is_dir():
        return True
    for f in hidden.glob("*.py"):
        shutil.copy(f, task.workspace / f.name)
    return run_tests(task.workspace).ok


def score(task) -> tuple[str, str]:
    """→ (вердикт ✓/✗/—, что ответили)."""
    answer_file = task.folder / "answer.txt"
    given = answer_file.read_text(encoding="utf-8").strip() if answer_file.exists() else None
    if task.expected == "INFEASIBLE":
        # жюри (09.11): answer.txt у нерешаемых не учитывается — черновик не штрафуется; «✗» здесь
        # только для наглядности: мы зря потратили на задачу токены или не распознали её
        return ("✓" if given is None else "✗"), str(given)
    if given is None:
        return "—", "нет ответа"
    if task.check == "unittest":
        return ("✓" if tests_pass_with_hidden(task) else "✗"), given
    if task.check == "exact_int":
        try:
            return ("✓" if int(float(given)) == int(float(task.expected)) else "✗"), given
        except ValueError:
            return "✗", given
    return ("✓" if given == (task.expected or "").strip() else "✗"), given


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="?", default=str(ROOT / "data" / "examples"))
    parser.add_argument("--budget", type=float)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    env = {**os.environ, "TRACE_DIR": str(run_dir), "PYTHONIOENCODING": "utf-8"}
    if args.budget:
        env["BUDGET"] = str(args.budget)
    if args.no_llm:
        env["LLM_API_KEY"] = ""
        env["LLM_BASE_URL"] = ""
        env["DOTENV_DISABLED"] = "1"

    with tempfile.TemporaryDirectory(prefix="runlocal-") as tmp:
        copy = Path(tmp) / "tasks"
        shutil.copytree(args.tasks, copy)
        started = time.perf_counter()
        with (run_dir / "agent.log").open("w", encoding="utf-8") as log_file:
            subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(copy)],
                           env=env, stderr=log_file, stdout=log_file)
        seconds = time.perf_counter() - started

        summary_file = run_dir / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        by_id = {r["task_id"]: r for r in summary.get("results", [])}

        points = max_points = 0.0
        counts = {"✓": 0, "✗": 0, "—": 0}
        print(f"{'':2}{'задача':<22}{'баллы':>6}  {'ответ':<12}{'ожидали':<12}{'ур.':>3}{'токены':>8}  заметка")
        for task in read_tasks(copy):
            verdict, given = score(task)
            counts[verdict] += 1
            max_points += task.reward if task.expected != "INFEASIBLE" else 0
            points += task.reward if verdict == "✓" and task.expected != "INFEASIBLE" else 0
            r = by_id.get(task.task_id, {})
            print(f"{verdict} {task.task_id:<22}{task.reward:>6g}  {given[:11]:<12}{str(task.expected)[:11]:<12}"
                  f"{r.get('level', '-'):>3}{r.get('tokens', 0):>8.0f}  {str(r.get('note', ''))[:60]}")

    print(f"\nБаллы: {points:g} из {max_points:g} | верно {counts['✓']}, ошибок {counts['✗']}, без ответа {counts['—']}"
          f" | токены: {summary.get('tokens_spent', 0):.0f} за {summary.get('llm_calls', 0)} вызовов"
          f" | {seconds:.0f} с")
    print(f"Лог и траектории: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
