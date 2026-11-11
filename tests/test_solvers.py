"""Решатели уровня 0: примеры организаторов + перефразировки + невыполнимые задачи."""
import json
import shutil
from pathlib import Path

import pytest

from harness.solvers import solve_level0
from harness.tasks import read_task, read_tasks
from harness.types import Status

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer "
        "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")

SALES = """order_id,region,product,month,units,price
1,west,widget,5,5,33
2,north,sprocket,6,28,28
3,east,widget,7,10,10
4,east,gizmo,7,4,50
5,west,widget,8,20,2
6,east,widget,3,7,1
"""

KB = {
    "person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Alice Smith\n",
    "person_alice_smith.md": "# Alice Smith\n\nBorn: 2002\nEmployer: Granite AI\nMentor: Olga Kowalski\n",
    "person_olga_kowalski.md": "# Olga Kowalski\n\nBorn: 1965\nEmployer: Ghost Corp\n",
    "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Quito\nFounded: 1985\n",
    "company_granite_ai.md": "# Granite AI\n\nHeadquarters: Lisbon\n",
}


def make_task(tmp_path, task_type, question, files):
    folder = tmp_path / f"{task_type}_x"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        p = folder / "workspace" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (folder / "task.json").write_text(json.dumps({
        "task_id": folder.name, "task_type": task_type, "instruction": question + TAIL,
        "reward": 1, "check": "exact_int"}), encoding="utf-8")
    return read_task(folder)


# ---------------------------------------------------------------- примеры организаторов

@pytest.mark.parametrize("task", [t for t in read_tasks(EXAMPLES) if t.task_type != "coding"],
                         ids=lambda t: t.task_id)
def test_examples_solved_for_free(task):
    result = solve_level0(task)
    assert result.status is Status.SOLVED, result.note
    assert result.answer == task.expected
    assert result.confidence == 1.0 and result.tokens == 0


def test_examples_untouched(tmp_path):
    """Решатели только читают — в папке задачи ничего не появляется."""
    dst = tmp_path / "t"
    shutil.copytree(EXAMPLES, dst)
    before = sorted(p.relative_to(dst) for p in dst.rglob("*"))
    for task in read_tasks(dst):
        solve_level0(task)
    assert sorted(p.relative_to(dst) for p in dst.rglob("*")) == before


# ---------------------------------------------------------------- data_analysis

@pytest.mark.parametrize("question, answer", [
    ("The file sales.csv has columns order_id,region,product,month,units,price (6 rows). "
     "How many units of product 'widget' were sold in months 5 through 8 inclusive?", "35"),
    ("How many units of product 'widget' were sold in month 7?", "10"),
    ("How many units of 'widget' were sold in the 'east' region in months 1 through 12?", "17"),
    ("What is the total revenue of product 'widget' in months 7 to 8?", "140"),
    ("How many orders of product 'widget' are there in months 1 through 12?", "4"),
    ("What is the average units of product 'widget' in months 5 through 8?", "11.6667"),
])
def test_data_paraphrases(tmp_path, question, answer):
    result = solve_level0(make_task(tmp_path, "data_analysis", question, {"sales.csv": SALES}))
    assert result.status is Status.SOLVED, result.note
    assert result.answer == answer


def test_data_unknown_product_is_infeasible(tmp_path):
    q = "How many units of product 'doohickey' were sold in months 1 through 3 inclusive?"
    assert solve_level0(make_task(tmp_path, "data_analysis", q, {"sales.csv": SALES})).status is Status.INFEASIBLE


def test_data_unknown_column_is_unsure(tmp_path):
    q = "How many returns of product 'widget' were there in months 1 through 3 inclusive?"
    assert solve_level0(make_task(tmp_path, "data_analysis", q, {"sales.csv": SALES})).status is Status.UNSURE


# ---------------------------------------------------------------- search

@pytest.mark.parametrize("question, answer", [
    ("In which year was Irina Smith born?", "1983"),
    ("In which city is the headquarters of Irina Smith's employer?", "Quito"),
    ("In which year was Irina Smith's mentor born?", "2002"),
    ("In which year was the employer of Irina Smith founded?", "1985"),
    ("In which city is the headquarters of Irina Smith's mentor's employer?", "Lisbon"),
])
def test_search_paraphrases(tmp_path, question, answer):
    kb = {f"kb/{k}": v for k, v in KB.items()}
    result = solve_level0(make_task(tmp_path, "search", question, kb))
    assert result.status is Status.SOLVED, result.note
    assert result.answer == answer


@pytest.mark.parametrize("question, why", [
    ("In which year was Ivan Ivanov born?", "нет человека"),
])
def test_search_infeasible(tmp_path, question, why):
    kb = {f"kb/{k}": v for k, v in KB.items()}
    result = solve_level0(make_task(tmp_path, "search", question, kb))
    assert result.status is Status.INFEASIBLE, why


@pytest.mark.parametrize("question, why", [
    # у других компаний поле Founded есть, а следующего звена нет как заметки: возможно, мы не так
    # поняли вопрос → не объявляем «нерешаемо» сами, отдаём ИИ (ложное «нерешаемо» стоит баллов)
    ("In which year was the employer of Alice Smith founded?", "нет поля Founded"),
    ("In which city is the headquarters of Olga Kowalski's employer?", "нет компании в базе"),
])
def test_search_doubtful_goes_to_llm(tmp_path, question, why):
    kb = {f"kb/{k}": v for k, v in KB.items()}
    result = solve_level0(make_task(tmp_path, "search", question, kb))
    assert result.status is Status.UNSURE, why


def test_search_entity_mentioned_in_text_is_not_infeasible(tmp_path):
    kb = {"kb/notes.txt": "Nina Novak was born in 1971. She works for Cobalt Works.\n"}
    result = solve_level0(make_task(tmp_path, "search", "In which year was Nina Novak born?", kb))
    assert result.status is Status.UNSURE


def test_data_conditions_and_json(tmp_path):
    from harness.solvers.data_solver import Query, execute, load_table
    (tmp_path / "p.json").write_text('[{"cat": "a", "stock": 3, "date": "2025-03-01"},'
                                     ' {"cat": "a", "stock": 9, "date": "2025-04-01"}]', encoding="utf-8")
    header, rows = load_table(tmp_path / "p.json")
    assert header == ["cat", "stock", "date"]
    assert execute(Query("count", None, {"stock": "<5"}), rows) == 1
    assert execute(Query("sum", "stock", {"date": "startswith:2025-04"}), rows) == 9


def test_data_similar_column_is_not_infeasible(tmp_path):
    csv_text = "date,city,temp_c,rain_mm\n2025-03-01,Oslo,3,5\n"
    result = solve_level0(make_task(tmp_path, "data_analysis",
                                    "Compute the total rainfall in millimetres for Oslo.", {"weather.csv": csv_text}))
    assert result.status is not Status.INFEASIBLE


# ---------------------------------------------------------------- math

@pytest.mark.parametrize("question, answer", [
    ("A warehouse starts with 500 boxes. It sells 20 boxes per day for 4 days. "
     "Then the stock is tripled. How many boxes does it have at the end?", "1260"),
    ("A warehouse starts with 100 boxes. It receives three deliveries of 10 boxes each. "
     "How many boxes does it have at the end?", "130"),
    ("A warehouse starts with 1,000 boxes. Then the stock is split equally into 3 warehouses and it keeps "
     "one share (rounded down). It ships 5 boxes. How many boxes does it have at the end?", "328"),
])
def test_math_paraphrases(tmp_path, question, answer):
    result = solve_level0(make_task(tmp_path, "math", question, {}))
    assert result.status is Status.SOLVED, result.note
    assert result.answer == answer


@pytest.mark.parametrize("question", [
    "A warehouse starts with 100 boxes. It burns 10 boxes. How many boxes does it have at the end?",
    "A warehouse starts with 100 boxes in 2024. How many boxes does it have at the end?",
    "It ships 10 boxes. How many boxes does it have at the end?",
])
def test_math_unknown_is_unsure(tmp_path, question):
    assert solve_level0(make_task(tmp_path, "math", question, {})).status is Status.UNSURE


@pytest.mark.parametrize("question, answer", [
    ("A shop begins with two hundred and fifty apples. It sold forty apples a day for three days. "
     "How many apples does it have left?", "130"),
    ("A depot holds 940 crates. For 6 days, it dispatches 35 crates per day. Then it receives a dozen crates. "
     "How many crates remain?", "742"),
    ("A store has 500 pens. Half of the pens are sold (rounded down). The owner is very happy. "
     "How many pens are left?", "250"),
    ("A farm starts with 90 eggs. It sells a third of its eggs. Then the stock doubles. "
     "How many eggs does it have now?", "120"),
    ("A warehouse starts with 1,000 boxes. 120 boxes are shipped. 3 shipments of 50 boxes arrive. "
     "The stock grows by 10%. How many boxes are in stock?", "1133"),
    ("A warehouse starts with 100 boxes. It gives 7 boxes to each of 4 stores. "
     "How many boxes does it have left?", "72"),
    ("A warehouse starts with 1752 boxes. Then the stock is divided evenly among 5 branches and it keeps "
     "one share. Round down whenever needed. How many boxes does it have at the end?", "350"),
])
def test_math_normalized_phrasings(tmp_path, question, answer):
    result = solve_level0(make_task(tmp_path, "math", question, {}))
    assert result.status is Status.SOLVED, result.note
    assert result.answer == answer


@pytest.mark.parametrize("question", [
    # деление не нацело без указания округления
    "A farm starts with 91 eggs. It sells a third of its eggs. How many eggs does it have now?",
    # спрашивают не про остаток
    "A farm starts with 100 eggs. It sells 10 eggs. How many eggs did it sell in total?",
    # лишнее предложение с числом (как GSM-NoOp) — не угадываем
    "A warehouse starts with 100 boxes. 5 of the boxes are damaged. How many boxes does it have?",
])
def test_math_traps_are_unsure(tmp_path, question):
    assert solve_level0(make_task(tmp_path, "math", question, {})).status is Status.UNSURE


def test_normalize_numbers_and_verbs():
    from harness.normalize import normalize, words_to_digits
    assert words_to_digits("twenty-five boxes and one hundred and two pens") == "25 boxes and 102 pens"
    assert words_to_digits("two dozen eggs, a dozen more") == "24 eggs, 12 more"
    assert words_to_digits("it keeps one share") == "it keeps one share"
    assert normalize("35 crates were sold") == "ships 35 crates"
    assert normalize("It purchased 12 boxes") == "it receives 12 boxes"
    assert normalize("A third of them") == "frac_1_3 of them"


def test_solver_crash_becomes_unsure(tmp_path, monkeypatch):
    from harness import solvers
    monkeypatch.setitem(solvers.LEVEL0, "math", lambda t: 1 / 0)
    result = solve_level0(make_task(tmp_path, "math", "x", {}))
    assert result.status is Status.UNSURE and "ZeroDivisionError" in result.note
