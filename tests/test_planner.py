"""Планировщик и бюджет — с подставной моделью, без настоящих вызовов и токенов."""
import asyncio
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from harness.budget import Budget
from harness.config import Settings
from harness.llm import BudgetExhausted, Reply
from harness.planner import Planner
from harness.tasks import read_tasks

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "data" / "examples"


class FakeLLM:
    """Отвечает заранее заданным текстом и честно списывает «токены» с бюджета."""

    def __init__(self, settings, answer=None, error=None):
        self.settings = settings
        self.budget = None
        self.answer, self.error, self.calls = answer, error, 0

    async def ask(self, prompt, *, task_id, purpose, model=None, max_tokens=600, retries=3):
        self.calls += 1
        if self.error:
            raise self.error
        cost = self.budget.charge(len(prompt) // 3, 50)
        return Reply(self.answer, len(prompt) // 3, 50, cost, model or "fake", 0.1)


@pytest.fixture
def settings(tmp_path):
    return Settings(llm_base_url="", llm_api_key="", budget=100_000, reserve=0.03, time_limit=600,
                    llm_concurrency=2, task_concurrency=4, fast_model="fast", strong_model="strong",
                    trace_dir=tmp_path / "trace")


@pytest.fixture
def tasks_dir(tmp_path):
    dst = tmp_path / "tasks"
    shutil.copytree(EXAMPLES, dst)
    return dst


def make_unknown_math(tasks_dir):
    folder = tasks_dir / "math_odd"
    (folder / "workspace").mkdir(parents=True)
    (folder / "task.json").write_text(
        '{"task_id": "math_odd", "task_type": "math", "reward": 3, "check": "exact_int",'
        ' "instruction": "A depot holds 10 crates and burns 3. How many are left?"}', encoding="utf-8")
    return folder


def test_budget_cost_rule():
    b = Budget(1000, reserve_fraction=0.1)
    assert b.charge(input_tokens=100, output_tokens=20) == 30  # 20 + 100/10
    assert b.available == 1000 - 100 - 30
    assert b.can_afford(870) and not b.can_afford(871)


def test_rate_limiter_keeps_requests_per_window():
    import time
    from harness.llm import RateLimiter

    async def burst():
        rl = RateLimiter(per_minute=3, window=0.5)
        t0 = time.monotonic()
        for _ in range(4):
            await rl.acquire()
        return time.monotonic() - t0

    # три запроса проходят сразу, четвёртый ждёт, пока освободится окно
    assert asyncio.run(burst()) >= 0.45


def test_budget_hold_settle_release():
    b = Budget(1000, reserve_fraction=0.0)
    assert b.hold(600) and not b.hold(500)      # вторая бронь не влезает, пока первая висит
    assert b.settle(600, input_tokens=100, output_tokens=40) == 50
    assert b.available == 950 and b.held == 0
    assert b.hold(900)
    b.release(900)
    assert b.available == 950
    b.settle(0, 300, 0, failed=True)            # упавший запрос: жюри списывает вход
    assert b.spent == 80 and b.failed_calls == 1


def test_draft_answer_is_kept_when_llm_fails(settings, tasks_dir):
    from harness.llm import LLMError
    folder = tasks_dir / "math_neg"
    folder.mkdir()
    (folder / "task.json").write_text(
        '{"task_id": "math_neg", "task_type": "math", "reward": 3, "check": "exact_int",'
        ' "instruction": "A shop starts with 10 boxes. It sells 30 boxes. How many boxes are left?"}',
        encoding="utf-8")
    llm = FakeLLM(settings, error=LLMError("provider down"))
    summary = asyncio.run(Planner(settings, llm=llm).run(read_tasks(tasks_dir)))
    # бесплатный ответ неуверенный (отрицательный), ИИ не помог — черновик всё равно остаётся:
    # неверный ответ у жюри = отсутствию ответа, а вдруг верен
    assert (folder / "answer.txt").read_text() == "-20\n"
    assert llm.calls >= 1 and summary["answers_written"] == 9


def test_free_pass_writes_all_answers(settings, tasks_dir):
    summary = asyncio.run(Planner(settings).run(read_tasks(tasks_dir)))
    assert summary["solved"] == 8 and summary["tokens_spent"] == 0
    for t in read_tasks(tasks_dir):
        assert (t.folder / "answer.txt").exists()
    assert (settings.trace_dir / "summary.json").exists()


def test_llm_pass_solves_unknown_task(settings, tasks_dir):
    folder = make_unknown_math(tasks_dir)
    llm = FakeLLM(settings, answer="10 - 3")  # короткий ответ: одно выражение
    summary = asyncio.run(Planner(settings, llm=llm).run(read_tasks(tasks_dir)))
    assert (folder / "answer.txt").read_text() == "7\n"
    assert llm.calls == 1 and summary["tokens_spent"] > 0


def test_short_answer_falls_back_to_program(settings, tasks_dir):
    folder = make_unknown_math(tasks_dir)
    # не выражение → короткий путь не сработал → тот же ответ годится как программа
    llm = FakeLLM(settings, answer="```python\nprint(10 - 3)\n```")
    asyncio.run(Planner(settings, llm=llm).run(read_tasks(tasks_dir)))
    assert (folder / "answer.txt").read_text() == "7\n"
    assert llm.calls == 2


def test_budget_exhausted_does_not_crash(settings, tasks_dir):
    folder = make_unknown_math(tasks_dir)
    llm = FakeLLM(settings, error=BudgetExhausted("proxy stopped"))
    summary = asyncio.run(Planner(settings, llm=llm).run(read_tasks(tasks_dir)))
    assert not (folder / "answer.txt").exists()
    assert summary["solved"] == 8  # бесплатные ответы остались


def test_no_budget_skips_llm(settings, tasks_dir):
    make_unknown_math(tasks_dir)
    llm = FakeLLM(settings, answer="```python\nprint(7)\n```")
    asyncio.run(Planner(replace(settings, budget=30), llm=llm).run(read_tasks(tasks_dir)))
    assert llm.calls == 0


def test_run_agent_cli_survives_bad_dir(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(tmp_path / "nope"),
                           "--unexpected-flag"], capture_output=True, text=True, timeout=120,
                          env={"PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
                               "DOTENV_DISABLED": "1", "TRACE_DIR": str(tmp_path / "t")})
    assert proc.returncode == 0
    assert "0 tasks" in proc.stderr
