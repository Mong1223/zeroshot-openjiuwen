"""Блок [B]: чтение задач и запись ответов. Работает без LLM — 0 токенов.

Главное требование: не падать. Битая задача пропускается и попадает в лог,
остальные читаются дальше.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from harness.types import Task

log = logging.getLogger(__name__)

ANSWER_FILE = "answer.txt"
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
# Служебная концовка, одинаковая во всех задачах («Work only inside… echo 42 > answer.txt»).
# В смысловую часть не входит: её числа (42) — мусор для экстракторов и признаков.
_BOILERPLATE = re.compile(r"\s*Work only inside the current directory\b.*\Z", re.S | re.I)


def question_text(instruction: str) -> str:
    """Смысловая часть задания без служебной концовки."""
    return _BOILERPLATE.sub("", instruction).strip()


def read_tasks(tasks_dir: str | Path) -> list[Task]:
    """Прочитать все задачи из папки. Порядок — по имени папки."""
    root = Path(tasks_dir)
    if not root.is_dir():
        log.error("tasks dir not found: %s", root)
        return []

    tasks: list[Task] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            task = read_task(folder)
        except Exception as e:  # одна битая задача не должна ронять остальные
            log.warning("skip %s: %s: %s", folder.name, type(e).__name__, e)
            continue
        if task is not None:
            tasks.append(task)
    log.info("read %d tasks from %s", len(tasks), root)
    return tasks


def read_task(folder: Path) -> Task | None:
    task_json = folder / "task.json"
    if not task_json.is_file():
        log.warning("skip %s: no task.json", folder.name)
        return None

    raw: dict[str, Any] = json.loads(task_json.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("task.json is not an object")

    workspace = folder / "workspace"
    files = _list_files(workspace)
    instruction = str(raw.get("instruction") or "")
    expected = raw.get("expected")

    return Task(
        task_id=str(raw.get("task_id") or folder.name),
        task_type=str(raw.get("task_type") or "unknown").strip().lower(),
        instruction=instruction,
        reward=_to_float(raw.get("reward")),
        check=str(raw.get("check") or "").strip().lower(),
        folder=folder,
        workspace=workspace,
        files=files,
        features=_features(instruction, workspace, files),
        expected=None if expected is None else str(expected),
        raw=raw,
    )


def answer_path(task: Task) -> Path:
    """answer.txt лежит рядом с task.json (ответ организаторов на встрече 07.11)."""
    return task.folder / ANSWER_FILE


_ANSWER_PREFIX = re.compile(r"^(?:final answer|the answer is|the final answer is|answer|result|output)\s*[:=\-]?\s*",
                            re.I)
_INT_IN_TEXT = re.compile(r"[-−]?\d{1,3}(?:[,\s]\d{3})+(?!\d)|[-−]?\d+(?:\.\d+)?")


def clean_answer(text: str) -> str:
    """Убирает обёртку вокруг ответа: «Final answer: **Osaka**.» → «Osaka».

    Жюри сравнивает дословно, а модели любят приставки, кавычки, точку в конце.
    """
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    text = lines[-1] if lines else ""
    for _ in range(3):  # обёртки бывают вложенными: **"Osaka"**
        before = text
        text = _ANSWER_PREFIX.sub("", text).strip()
        text = text.strip("`*_ ").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1].strip()
        text = text.strip("“”‘’").strip()
        if text.endswith("."):  # «Osaka.» → «Osaka», «487.» → «487»
            text = text[:-1].strip()
        if text == before:
            break
    return text


def normalize_answer(task: Task, answer: str) -> str:
    text = clean_answer(answer)
    if task.check == "unittest":
        return "DONE"
    if task.check == "exact_int":
        numbers = _INT_IN_TEXT.findall(text)
        if len(numbers) == 1:  # «487 crates», «$1,250», «1 250»
            raw = re.sub(r"[,\s]", "", numbers[0]).replace("−", "-")
            try:
                value = float(raw)
                if value.is_integer():
                    return str(int(value))
            except ValueError:
                pass
    return text


def plausible(task: Task, answer: str) -> str | None:
    """Дешёвая проверка ответа перед записью. → причина отказа или None, если ответ правдоподобен.

    Неверный ответ на нерешаемую задачу хуже, чем никакого: лучше не записывать мусор.
    """
    text = normalize_answer(task, answer)
    if task.check == "unittest":
        return None
    if not text:
        return "empty answer"
    if "\n" in text or len(text) > 200:
        return "answer is not a single short line"
    if task.check == "exact_int":
        if not re.fullmatch(r"-?\d+", text):
            return f"not an integer: {text[:40]!r}"
        if len(text.lstrip("-")) > 15:
            return "integer is implausibly large"
    if text.upper() in ("INFEASIBLE", "NONE", "NULL", "N/A", "UNKNOWN", "ERROR", "NAN"):
        return f"placeholder answer {text!r}"
    return None


def write_answer(task: Task, answer: str) -> Path:
    """Атомарная запись: сначала временный файл, потом переименование.

    Так при обрыве посреди записи не останется «половины» ответа.
    Перевод строки в конце — как у `echo 42 > answer.txt` из условия задач.
    """
    path = answer_path(task)
    text = normalize_answer(task, answer) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".answer-", dir=task.folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def remove_answer(task: Task) -> None:
    """Для невыполнимой задачи файла быть не должно."""
    answer_path(task).unlink(missing_ok=True)


# ---------------------------------------------------------------- признаки

def _list_files(workspace: Path) -> list[str]:
    if not workspace.is_dir():
        return []
    return sorted(
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*")
        if p.is_file() and p.name != ".DS_Store"
    )


def _features(instruction: str, workspace: Path, files: list[str]) -> dict[str, Any]:
    """Бесплатные признаки задачи — для оценщика цены и для решателей."""
    question = question_text(instruction)
    feats: dict[str, Any] = {
        "instruction_chars": len(instruction),
        "question_words": len(question.split()),
        "sentences": len(_SENTENCE.findall(question)) or (1 if question else 0),
        "numbers_in_text": len(_NUMBER.findall(question)),
        "has_boilerplate": question != instruction.strip(),
        "n_files": len(files),
        "total_bytes": sum((workspace / f).stat().st_size for f in files),
    }

    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if csv_files:
        rows, columns = _csv_shape(workspace / csv_files[0])
        feats.update(csv_files=len(csv_files), csv_rows=rows, csv_columns=columns)

    kb_files = [f for f in files if f.startswith("kb/") and f.endswith(".md")]
    if kb_files:
        feats["kb_files"] = len(kb_files)

    py_files = [f for f in files if f.endswith(".py")]
    if py_files:
        tests = [f for f in py_files if Path(f).name.startswith("test_")]
        feats.update(
            py_files=len(py_files),
            test_files=len(tests),
            test_functions=sum(_count(workspace / f, "def test_") for f in tests),
            code_lines=sum(_count_lines(workspace / f) for f in py_files if f not in tests),
        )
    return feats


def _csv_shape(path: Path) -> tuple[int, list[str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for row in reader if row)
        return rows, [h.strip() for h in header]
    except (OSError, UnicodeDecodeError, csv.Error):
        return 0, []


def _count(path: Path, needle: str) -> int:
    try:
        return path.read_text(encoding="utf-8", errors="replace").count(needle)
    except OSError:
        return 0


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except OSError:
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
