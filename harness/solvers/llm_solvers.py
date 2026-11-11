"""Решатели с ИИ (уровень 1). Вызываются только для задач, которые не решились бесплатно.

Правило экономии: модели показываем минимум текста и просим короткий ответ.

program_solve — для math, data_analysis, search и незнакомых типов: модель пишет короткую
    программу на Python, которая сама читает файлы и печатает ответ; запускаем её мы.
    Модель не считает «в уме» (замер 09.11: так она ошибается) и не читает файлы целиком
    (300 строк CSV ≈ 500 токенов) — ей показываем только заголовки и пару строк.
fix_code_solve — для coding: модели даём упавшие тесты и код, она возвращает исправленный
    файл; проверяем тестами, при неудаче откатываем.
"""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.llm import LLM
from harness.tasks import question_text
from harness.testrun import run_tests
from harness.types import Result, Status, Task

RUN_TIMEOUT = 20
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

PROGRAM_RULES = """Write a short Python 3 program that solves the task.
- The program runs in the directory with the files listed above; read them yourself, do not retype data.
- Do the arithmetic in Python, never by hand.
- Print ONLY the final answer (a number or a short string), nothing else.
- If the task cannot be solved from the given data (missing entity, column or file; contradictory
  conditions), print exactly INFEASIBLE.
Reply with a single ```python code block and nothing else."""

FIX_RULES = """Fix the bugs in {target} so that all tests pass. Do not change the tests.
The fix will also be checked with additional hidden tests: make the code correct for ALL valid inputs
(edge cases such as empty input, zero, negatives, single element), not just for the visible tests.
Never special-case the test values. Keep function names and signatures.
Reply with the COMPLETE corrected {target} in a single ```python code block and nothing else."""


# ---------------------------------------------------------------- program_solve

async def program_solve(task: Task, llm: LLM) -> Result:
    prompt = f"Task: {question_text(task.instruction)}\n\n{_describe_files(task)}\n{PROGRAM_RULES}"
    spent = 0.0
    feedback = ""
    for attempt in range(2):
        reply = await llm.ask(prompt + feedback, task_id=task.task_id, purpose=f"program#{attempt}",
                              max_tokens=700)
        spent += reply.cost
        code = _extract_code(reply.content)
        if code is None:
            feedback = "\n\nYour previous reply had no ```python code block. Reply with code only."
            continue
        ok, output = _run_program(code, task.workspace)
        lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
        if ok and lines:
            answer = lines[-1]
            if answer.upper() == "INFEASIBLE":
                return Result(Status.INFEASIBLE, confidence=0.6, level=1, tokens=spent, note="model: INFEASIBLE")
            if task.check == "exact_int" and not _is_int(answer):
                feedback = f"\n\nYour program printed {answer!r}, but the answer must be a single integer."
                continue
            return Result(Status.SOLVED, answer=answer, confidence=0.75, level=1, tokens=spent,
                          note=f"program printed {answer!r}")
        feedback = (f"\n\nYour previous program:\n```python\n{code}\n```\nfailed with:\n{output[-800:]}\n"
                    "Fix it.")
    return Result(Status.FAILED, level=1, tokens=spent, note="program attempts failed")


def _describe_files(task: Task) -> str:
    """Короткое описание файлов: заголовки и несколько строк, а не содержимое целиком."""
    if not task.files:
        return "There are no files."
    lines = ["Files in the current directory:"]
    kb = [f for f in task.files if f.startswith("kb/")]
    other = [f for f in task.files if not f.startswith("kb/")]
    for name in other:
        path = task.workspace / name
        size = path.stat().st_size
        if name.lower().endswith((".csv", ".tsv")):
            head = path.read_text(encoding="utf-8", errors="replace").splitlines()
            preview = "\n".join(head[:4])
            lines.append(f"- {name} ({len(head) - 1} data rows). First lines:\n{preview}")
        elif name.lower().endswith(".json") and size > 1500:
            lines.append(f"- {name} ({size} bytes). Structure: {_json_shape(path)}")
        elif size <= 1500:
            lines.append(f"- {name}:\n{path.read_text(encoding='utf-8', errors='replace').rstrip()}")
        else:
            head = path.read_text(encoding="utf-8", errors="replace").splitlines()
            lines.append(f"- {name} ({size} bytes, {len(head)} lines). First lines:\n" + "\n".join(head[:8]))
    if kb:
        sample = task.workspace / kb[0]
        lines.append(f"- kb/ — {len(kb)} markdown files: {', '.join(Path(f).name for f in kb)}")
        lines.append(f"  Example ({kb[0]}):\n{sample.read_text(encoding='utf-8', errors='replace').rstrip()}")
    return "\n".join(lines) + "\n"


def _json_shape(path: Path) -> str:
    """Коротко о JSON: «list of 38 objects, first: {...}» — вместо всего файла."""
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return "not valid JSON; first 300 chars: " + path.read_text(encoding="utf-8", errors="replace")[:300]
    if isinstance(data, list):
        first = json.dumps(data[0], ensure_ascii=False)[:400] if data else "-"
        return f"list of {len(data)} items, first item: {first}"
    if isinstance(data, dict):
        keys = list(data)[:20]
        sample = {k: data[k] for k in keys[:3]}
        return f"object with keys {keys}; sample: {json.dumps(sample, ensure_ascii=False)[:400]}"
    return json.dumps(data)[:300]


def _run_program(code: str, workspace: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="zs-prog-") as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(code, encoding="utf-8")
        cwd = workspace if workspace.is_dir() else tmp  # у math-задач папки workspace может не быть
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=cwd, capture_output=True,
                                  text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-1500:]
    return True, proc.stdout


# ---------------------------------------------------------------- fix_code_solve

async def fix_code_solve(task: Task, llm: LLM) -> Result:
    ws = task.workspace
    tests = sorted(ws.glob("test*.py"))
    targets = sorted(p for p in ws.glob("*.py") if p not in tests)
    if len(targets) != 1 or not tests:
        return Result(Status.FAILED, level=1, note=f"targets={len(targets)} tests={len(tests)}")
    target = targets[0]
    original = target.read_text(encoding="utf-8")
    test_hash = _hash(tests)

    report = run_tests(ws)
    if report.ok:
        return Result(Status.SOLVED, answer="DONE", confidence=1.0, level=1, note="tests already pass")

    spent = 0.0
    attempts = [(llm.settings.fast_model, 1500), (llm.settings.strong_model, 4000)]
    for i, (model, max_tokens) in enumerate(attempts):
        code_now = target.read_text(encoding="utf-8") if i else original
        prompt = (f"Task: {question_text(task.instruction)}\n\n"
                  + "".join(f"=== {t.name} ===\n{t.read_text(encoding='utf-8')}\n" for t in tests)
                  + f"=== {target.name} ===\n{code_now}\n"
                  + f"=== test output ===\n{report.output[-1500:]}\n\n"
                  + FIX_RULES.format(target=target.name))
        reply = await llm.ask(prompt, task_id=task.task_id, purpose=f"fix#{i}", model=model,
                              max_tokens=max_tokens)
        spent += reply.cost
        code = _extract_code(reply.content)
        if code is None or not _parses(code):
            continue
        target.write_text(code, encoding="utf-8")
        report = run_tests(ws)
        if report.ok and _hash(tests) == test_hash:
            return Result(Status.SOLVED, answer="DONE", confidence=0.9, level=1, tokens=spent,
                          note=f"fixed by {model} on attempt {i + 1}")
    target.write_text(original, encoding="utf-8")
    return Result(Status.FAILED, level=1, tokens=spent, note=f"still failing {sorted(report.failed)}")


# ---------------------------------------------------------------- общее

def _extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text or "")
    return blocks[-1] if blocks else None


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _is_int(text: str) -> bool:
    try:
        return float(text.replace(",", "")).is_integer()
    except ValueError:
        return False


def _hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()
