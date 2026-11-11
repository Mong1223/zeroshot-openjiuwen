"""Короткие ответы ИИ: выражение / JSON-запрос / JSON-цепочка. Подставная модель, без токенов."""
import asyncio
import shutil
from pathlib import Path

import pytest

from harness.budget import Budget
from harness.llm import Reply
from harness.solvers.llm_short import safe_eval, short_solve
from harness.tasks import read_tasks
from harness.types import Status

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


class FakeLLM:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0
        self.budget = Budget(100_000, 0.03)

    async def ask(self, prompt, *, task_id, purpose, model=None, max_tokens=600, retries=3):
        self.calls += 1
        answer = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        return Reply(answer, 100, 20, 30.0, "fake", 0.1)


@pytest.fixture
def tasks(tmp_path):
    shutil.copytree(EXAMPLES, tmp_path / "t")
    return {t.task_id: t for t in read_tasks(tmp_path / "t")}


@pytest.mark.parametrize("expr, value", [
    ("(940 - 6*35) - (940 - 6*35)//3", 487), ("`2 * (3 + 4)`", 14), ("max(3, 9) - abs(-2)", 7),
    ("Expression: 10 // 4", 2),
])
def test_safe_eval(expr, value):
    assert safe_eval(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('x')", "x = 5", "open('f').read()", "2 ** 1000"])
def test_safe_eval_rejects(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def run(task, llm):
    return asyncio.run(short_solve(task, llm))


def test_math_expression(tasks):
    r = run(tasks["math_0004"], FakeLLM("1669 - 36*3 + 2*87"))
    assert r.status is Status.SOLVED and r.answer == "1735" and r.level == 1


def test_data_query(tasks):
    llm = FakeLLM('{"op": "sum", "column": "units", "group": null, "filters": {"product": "widget"}, '
                  '"months": [7, 8]}')
    r = run(tasks["data_analysis_0002"], llm)
    assert r.status is Status.SOLVED and r.answer == tasks["data_analysis_0002"].expected


def test_data_query_infeasible_value(tasks):
    llm = FakeLLM('{"op": "sum", "column": "units", "filters": {"product": "doohickey"}, "months": null}')
    assert run(tasks["data_analysis_0002"], llm).status is Status.INFEASIBLE


def test_search_chain_multiline_json(tasks):
    llm = FakeLLM('{\n  "entity": "Sergei Garcia",\n  "chain": ["company", "HQ"]\n}')
    r = run(tasks["search_0005"], llm)
    assert r.status is Status.SOLVED and r.answer == "Osaka"


def make(tmp_path, task_type, question, files):
    from harness.tasks import read_task
    folder = tmp_path / "t1"
    for name, text in files.items():
        (folder / "workspace" / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / "workspace" / name).write_text(text, encoding="utf-8")
    folder.mkdir(exist_ok=True)
    (folder / "task.json").write_text(__import__("json").dumps(
        {"task_id": "t1", "task_type": task_type, "instruction": question, "reward": 1, "check": "exact_int"}))
    return read_task(folder)


def test_json_table_with_product_column_and_condition(tmp_path):
    task = make(tmp_path, "data_analysis", "Total stock value of toys with fewer than 10 items?",
                {"products.json": '[{"cat": "toys", "price": 2, "stock": 5}, {"cat": "toys", "price": 3, "stock": 20},'
                                  ' {"cat": "tools", "price": 9, "stock": 1}]'})
    llm = FakeLLM('{"op": "sum", "column": "price*stock", "filters": {"cat": "toys", "stock": "<10"}, "months": null}')
    r = run(task, llm)
    assert r.status is Status.SOLVED and r.answer == "10" and llm.calls == 1


def test_text_kb_is_read_whole(tmp_path):
    task = make(tmp_path, "search", "In which city is the headquarters of the company Nina Kim works for?",
                {"kb/nina_kim.txt": "Nina Kim was born in 1980. She works for Cobalt Works.",
                 "kb/cobalt_works.txt": "Cobalt Works was founded in 1990. Its headquarters are in Lyon."})
    llm = FakeLLM("Lyon")
    r = run(task, llm)
    assert r.status is Status.SOLVED and r.answer == "Lyon" and llm.calls == 1


def test_program_reply_goes_to_program(tasks):
    llm = FakeLLM("PROGRAM", "```python\nprint(1669 - 36*3 + 2*87)\n```")
    r = run(tasks["math_0004"], llm)
    assert r.answer == "1735" and llm.calls == 2


def test_negative_count_is_rejected(tasks):
    llm = FakeLLM("26 - 4 * 26", "```python\nprint(1735)\n```")
    r = run(tasks["math_0004"], llm)  # «How many boxes…» не может быть отрицательным → программа
    assert r.answer == "1735" and llm.calls == 2


def test_bad_short_answer_falls_back_to_program(tasks):
    llm = FakeLLM("I think it is about 1700", "```python\nprint(1669 - 36*3 + 2*87)\n```")
    r = run(tasks["math_0004"], llm)
    assert r.status is Status.SOLVED and r.answer == "1735" and llm.calls == 2
    assert r.tokens == 60  # обе попытки посчитаны
