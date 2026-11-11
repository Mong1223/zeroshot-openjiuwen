import json
import shutil
from pathlib import Path

import pytest

from harness.tasks import (answer_path, normalize_answer, plausible, question_text, read_tasks,
                           remove_answer, write_answer)

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


@pytest.fixture
def examples(tmp_path):
    """Копия примеров: тесты пишут answer.txt и не должны трогать оригиналы."""
    dst = tmp_path / "tasks"
    shutil.copytree(EXAMPLES, dst)
    return dst


def test_reads_all_examples(examples):
    tasks = read_tasks(examples)
    assert [t.task_id for t in tasks] == [
        "coding_0003", "coding_0007", "data_analysis_0002", "data_analysis_0006",
        "math_0000", "math_0004", "search_0001", "search_0005",
    ]
    by_id = {t.task_id: t for t in tasks}
    assert by_id["math_0000"].task_type == "math"
    assert by_id["math_0000"].reward == 5
    assert by_id["math_0000"].check == "exact_int"
    assert by_id["math_0000"].expected == "389"
    assert by_id["search_0005"].expected == "Osaka"


def test_features(examples):
    by_id = {t.task_id: t for t in read_tasks(examples)}
    assert by_id["data_analysis_0002"].features["csv_rows"] == 20
    assert by_id["data_analysis_0006"].features["csv_rows"] == 300
    assert by_id["data_analysis_0002"].features["csv_columns"] == [
        "order_id", "region", "product", "month", "units", "price"]
    assert by_id["search_0001"].features["kb_files"] == 36
    assert by_id["coding_0003"].features["test_functions"] == 6
    assert by_id["coding_0007"].features["test_functions"] == 4
    # 1752, 2, 3, 3 — без «42» из служебной концовки
    assert by_id["math_0000"].features["numbers_in_text"] == 4
    assert by_id["math_0004"].features["numbers_in_text"] == 5  # 1669, 36, 3, 2, 87
    assert all(t.features["has_boilerplate"] for t in by_id.values())


def test_question_text_drops_boilerplate(examples):
    task = {t.task_id: t for t in read_tasks(examples)}["search_0001"]
    assert question_text(task.instruction) == (
        "The directory kb/ contains a small knowledge base of markdown files. "
        "In which year was Irina Smith born? Use the file tools or bash (ls, grep, cat) to find it.")


def test_broken_tasks_are_skipped(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "task.json").write_text(json.dumps({"task_id": "ok", "task_type": "math"}))
    (tmp_path / "bad_json").mkdir()
    (tmp_path / "bad_json" / "task.json").write_text("{not json")
    (tmp_path / "no_json").mkdir()
    (tmp_path / "not_a_dir.txt").write_text("x")

    tasks = read_tasks(tmp_path)
    assert [t.task_id for t in tasks] == ["ok"]
    assert tasks[0].reward == 0.0 and tasks[0].files == []


def test_missing_dir_returns_empty(tmp_path):
    assert read_tasks(tmp_path / "nope") == []


def test_write_answer_next_to_task_json(examples):
    task = {t.task_id: t for t in read_tasks(examples)}["math_0000"]
    path = write_answer(task, " 389.0 ")
    assert path == task.folder / "answer.txt"
    assert path.read_text(encoding="utf-8") == "389\n"
    assert not list(task.folder.glob(".answer-*"))  # временный файл убран

    remove_answer(task)
    assert not answer_path(task).exists()


@pytest.mark.parametrize("check, given, expected", [
    ("exact_int", "1,735", "1735"),
    ("exact_int", "abc", "abc"),
    ("exact_str", "  Osaka\n", "Osaka"),
    ("unittest", "done", "DONE"),
    # обёртки, которые любят модели
    ("exact_str", "Final answer: **Osaka**.", "Osaka"),
    ("exact_str", "'north'", "north"),
    ("exact_str", "calculating...\nThe answer is `Medan`", "Medan"),
    ("exact_int", "487 crates", "487"),
    ("exact_int", "Answer: 1 250.", "1250"),
    ("exact_int", "-12", "-12"),
    ("exact_int", "between 3 and 5", "between 3 and 5"),  # два числа — не угадываем
])
def test_normalize_answer(examples, check, given, expected):
    task = read_tasks(examples)[0]
    task.check = check
    assert normalize_answer(task, given) == expected


@pytest.mark.parametrize("check, given, ok", [
    ("exact_int", "42", True), ("exact_int", "12.5", False), ("exact_int", "INFEASIBLE", False),
    ("exact_str", "Osaka", True), ("exact_str", "", False), ("exact_str", "None", False),
    ("unittest", "anything", True),
])
def test_plausible(examples, check, given, ok):
    task = read_tasks(examples)[0]
    task.check = check
    assert (plausible(task, given) is None) == ok
