"""Блок [H]: планировщик, простая версия (P0).

Проход 1 — бесплатный: все задачи через решатели уровня 0 (без ИИ), параллельно.
Проход 2 — платный: оставшиеся задачи по убыванию «баллы / ожидаемая цена», с ИИ,
           пока есть бюджет и время.
Ответ пишется сразу после каждой задачи: если что-то упадёт позже, сделанное сохранится.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict

from harness.agent import AGENT_CAP, agent_solve
from harness.budget import Budget, Clock
from harness.estimator import expected_cost
from harness.config import Settings
from harness.llm import LLM, BudgetExhausted, LLMError
from harness.solvers import solve_level0
from harness.solvers.llm_short import short_solve
from harness.solvers.llm_solvers import fix_code_solve
from harness.tasks import plausible, write_answer
from harness.types import Result, Status, Task

log = logging.getLogger("harness.planner")

ACCEPT_SOLVED = 0.7       # ниже — не доверяем бесплатному ответу, отдаём ИИ
ACCEPT_INFEASIBLE = 0.6   # ниже — не уверены, что задача нерешаема
TIME_MARGIN = 120         # секунд в запасе: новые платные задачи не начинаем
TASK_TIMEOUT = 300        # секунд на одну платную задачу



class Planner:
    def __init__(self, settings: Settings, llm: LLM | None = None):
        self.settings = settings
        self.budget = Budget(settings.budget, settings.reserve)
        self.clock = Clock(settings.time_limit)
        if llm is not None:  # для тестов: подставная модель
            llm.budget = self.budget
        self.llm = llm or (LLM(settings, self.budget) if settings.has_llm else None)
        # агент openJiuwen работает только с настоящей моделью (в тестах подставная — без агента)
        self.use_agent = isinstance(self.llm, LLM) and not os.getenv("AGENT_DISABLED")
        self.results: dict[str, Result] = {}
        self.written: dict[str, tuple[int, float]] = {}  # task_id → (уровень, уверенность) записанного ответа

    async def run(self, tasks: list[Task]) -> dict:
        await self._pass_free(tasks)
        rest = [t for t in tasks if self._needs_llm(self.results[t.task_id])]
        log.info("free pass: %d/%d done, %d left for LLM, %.0fs",
                 len(tasks) - len(rest), len(tasks), len(rest), self.clock.elapsed)
        if rest and self.llm is None:
            log.warning("no LLM_BASE_URL/LLM_API_KEY — %d tasks stay unsolved", len(rest))
        elif rest:
            await self._pass_llm(rest)
        return self.summary(tasks)

    # ------------------------------------------------------------ проход 1

    async def _pass_free(self, tasks: list[Task]) -> None:
        sem = asyncio.Semaphore(max(1, self.settings.task_concurrency))

        async def one(task: Task) -> None:
            async with sem:
                result = await asyncio.to_thread(solve_level0, task)
            self._record(task, result)

        await asyncio.gather(*(one(t) for t in tasks))

    @staticmethod
    def _needs_llm(result: Result) -> bool:
        if result.status is Status.SOLVED:
            return result.confidence < ACCEPT_SOLVED
        if result.status is Status.INFEASIBLE:
            return result.confidence < ACCEPT_INFEASIBLE
        return True

    # ------------------------------------------------------------ проход 2

    async def _pass_llm(self, tasks: list[Task]) -> None:
        queue = sorted(tasks, key=lambda t: t.reward / expected_cost(t), reverse=True)
        sem = asyncio.Semaphore(max(1, self.settings.task_concurrency))

        async def one(task: Task) -> None:
            async with sem:
                if self.budget.exhausted:
                    return
                if self.clock.left < TIME_MARGIN:
                    log.info("%s: skipped, no time", task.task_id)
                    return
                if not self.budget.can_afford(expected_cost(task)):
                    log.info("%s: skipped, budget %.0f left", task.task_id, self.budget.available)
                    return
                solver = fix_code_solve if task.task_type == "coding" else short_solve
                result = await self._attempt(task, solver, level=1)
                if result is None:
                    return
                # уровень 2: агент openJiuwen — только если уровень 1 не справился и есть время и бюджет
                if (result.status not in (Status.SOLVED, Status.INFEASIBLE) and self.use_agent
                        and not self.budget.exhausted and self.clock.left > TIME_MARGIN + 60
                        and self.budget.can_afford(AGENT_CAP)):
                    first = result
                    result = await self._attempt(task, agent_solve, level=2)
                    if result is None:
                        return
                    result.tokens += first.tokens
                    result.note = f"L1: {first.note[:60]} | L2: {result.note}"[:200]
                self._record(task, result)

        await asyncio.gather(*(one(t) for t in queue))

    async def _attempt(self, task: Task, solver, *, level: int) -> Result | None:
        """Один решатель с таймаутом. None — бюджет исчерпан, задачу дальше не трогаем."""
        try:
            return await asyncio.wait_for(solver(task, self.llm),
                                          timeout=max(5.0, min(TASK_TIMEOUT, self.clock.left - TIME_MARGIN / 2)))
        except BudgetExhausted as e:
            log.warning("budget exhausted: %s", e)
            return None
        except (LLMError, asyncio.TimeoutError) as e:
            return Result(Status.FAILED, level=level, note=f"{type(e).__name__}: {e}"[:200])
        except Exception as e:  # ошибка в нашем коде не должна ронять остальные задачи
            log.exception("%s crashed", task.task_id)
            return Result(Status.FAILED, level=level, note=f"crash {type(e).__name__}: {e}"[:200])

    # ------------------------------------------------------------ общее

    def _record(self, task: Task, result: Result) -> None:
        if result.status is Status.SOLVED and result.answer is not None:
            problem = plausible(task, result.answer)
            if problem:  # мусор не записываем: на нерешаемой задаче он стоил бы баллов
                result = Result(Status.FAILED, level=result.level, tokens=result.tokens,
                                note=f"implausible answer ({problem}); {result.note}"[:200])
        self.results[task.task_id] = result
        # Жюри (встреча 09.11): неверный ответ = отсутствию ответа, штрафа нет; answer.txt у нерешаемых
        # задач не учитывается. Поэтому любой правдоподобный ответ записываем СРАЗУ, даже неуверенный
        # («черновик»): следующий уровень ИИ перезапишет его, если найдёт лучше. «Нерешаемо» и неудача
        # черновик НЕ стирают — он ничего не стоит, а вдруг верен.
        try:
            if result.status is Status.SOLVED and result.answer is not None:
                write_answer(task, result.answer)
                self.written[task.task_id] = (result.level, result.confidence)
        except OSError as e:
            log.error("%s: cannot write answer: %s", task.task_id, e)
        log.info("%-22s L%d %-10s %-10s conf=%.2f tok=%5.0f  %s", task.task_id, result.level,
                 result.status.value, str(result.answer)[:10], result.confidence, result.tokens, result.note[:80])

    def summary(self, tasks: list[Task]) -> dict:
        rows = []
        for t in tasks:
            r = self.results.get(t.task_id, Result(Status.FAILED, note="not attempted"))
            row = asdict(r)
            row["status"] = r.status.value
            row["answer_written"] = t.task_id in self.written
            rows.append({"task_id": t.task_id, "type": t.task_type, "reward": t.reward, **row})
        summary = {
            "tasks": len(tasks),
            "solved": sum(r["status"] == "solved" for r in rows),
            "answers_written": len(self.written),
            "infeasible": sum(r["status"] == "infeasible" for r in rows),
            "tokens_spent": round(self.budget.spent, 1),
            "llm_calls": self.budget.calls,
            "seconds": round(self.clock.elapsed, 1),
            "budget_exhausted": self.budget.exhausted,
            "results": rows,
        }
        try:
            self.settings.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.trace_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
        return summary
