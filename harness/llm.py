"""Блок [D]: единственная «дверь» к моделям — через openJiuwen.

- перед вызовом: хватит ли бюджета на худший случай (max_tokens);
- после вызова: расход → счётчик, запись траектории в jsonl;
- 429 «слишком много одновременных запросов» → пауза и повтор;
- признаки исчерпания бюджета у прокси → флаг budget.exhausted, дальше никто не зовёт модель.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from harness.budget import Budget
from harness.config import Settings

log = logging.getLogger("harness.llm")

_EXHAUSTED_HINTS = ("402", "insufficient", "quota", "budget", "credit", "payment", "token limit")


class LLMError(Exception):
    pass


class BudgetExhausted(LLMError):
    pass


class RateLimiter:
    """Не больше `per_minute` запросов за любые 60 секунд (скользящее окно).

    Лимит ключа NeuralDeep — 60 запросов в минуту на аккаунт (памятка, чат #753); при превышении
    прокси отвечает 429, и мы теряем ~20 с на ожидание. Держимся ниже лимита с запасом.
    """

    def __init__(self, per_minute: int, window: float = 60.0):
        self.per_minute = max(1, per_minute)
        self.window = window
        self._times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < self.window]
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                await asyncio.sleep(self.window - (now - self._times[0]) + 0.05)


@dataclass
class Reply:
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str
    seconds: float


class LLM:
    def __init__(self, settings: Settings, budget: Budget):
        self.settings = settings
        self.budget = budget
        self._sem = asyncio.Semaphore(max(1, settings.llm_concurrency))
        self.rate = RateLimiter(settings.llm_rpm)
        self._models: dict[str, object] = {}
        settings.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace = settings.trace_dir / "llm_calls.jsonl"

    def _model(self, name: str, max_tokens: int):
        key = f"{name}:{max_tokens}"
        if key not in self._models:
            from harness.jiuwen_setup import configure_openjiuwen
            configure_openjiuwen()
            from openjiuwen.core.foundation.llm import init_model
            self._models[key] = init_model("OpenAI", name, self.settings.llm_api_key, self.settings.llm_base_url,
                                           max_tokens=max_tokens, timeout=180, max_retries=1)
        return self._models[key]

    async def ask(self, prompt: str, *, task_id: str, purpose: str, model: str | None = None,
                  max_tokens: int = 600, retries: int = 3) -> Reply:
        model = model or self.settings.fast_model
        est_input = len(prompt) // 3 + 1  # ~3 символа на токен (с запасом: в английском ~4)
        worst_case = max_tokens + est_input / 10
        await self._hold(worst_case)

        try:
            for attempt in range(retries + 1):
                async with self._sem:
                    await self.rate.acquire()
                    started = time.monotonic()
                    try:
                        reply = await self._model(model, max_tokens).invoke([{"role": "user", "content": prompt}])
                    except Exception as e:  # openJiuwen оборачивает ошибки провайдера в свои исключения
                        text = str(e)
                        if "429" not in text:  # запрос ушёл — жюри списывает его входные токены
                            self.budget.settle(0, est_input, 0, failed=True)
                        if "429" in text and attempt < retries:
                            wait = 20 * (attempt + 1) + random.uniform(0, 3)
                            log.warning("%s: 429, wait %.0fs", task_id, wait)
                            await asyncio.sleep(wait)
                            continue
                        if any(h in text.lower() for h in _EXHAUSTED_HINTS):
                            self.budget.exhausted = True
                            raise BudgetExhausted(text[:200]) from e
                        if attempt < retries:
                            await asyncio.sleep(2 + attempt * 3)
                            continue
                        raise LLMError(f"{type(e).__name__}: {text[:300]}") from e
                    seconds = time.monotonic() - started
                break
        except BaseException:
            self.budget.release(worst_case)
            raise

        usage = getattr(reply, "usage_metadata", None)
        content = reply.content or ""
        reasoning = getattr(reply, "reasoning_content", None) or ""
        inp = getattr(usage, "input_tokens", 0) or est_input
        out = getattr(usage, "output_tokens", 0) or (len(content) + len(reasoning)) // 3
        cost = self.budget.settle(worst_case, inp, out)
        self._log(task_id=task_id, purpose=purpose, model=model, input=inp, output=out, cost=cost,
                  seconds=round(seconds, 2), max_tokens=max_tokens, spent=round(self.budget.spent, 1))
        return Reply(content, inp, out, cost, model, seconds)

    async def _hold(self, amount: float, max_wait: float = 120) -> None:
        """Бронь бюджета. Если мешают брони параллельных вызовов — ждём, пока они спишутся.
        Не хватает и ждать нечего — BudgetExhausted (дальше эту задачу не трогаем)."""
        waited = 0.0
        while not self.budget.hold(amount):
            if self.budget.exhausted or self.budget.held <= 0 or waited >= max_wait:
                if self.budget.available < 1:  # совсем ничего не осталось — дальше никто не зовёт модель
                    self.budget.exhausted = True
                raise BudgetExhausted(f"not enough budget for {amount:.0f} (left {self.budget.available:.0f})")
            await asyncio.sleep(0.5)
            waited += 0.5

    def _log(self, **row) -> None:
        try:
            with self._trace.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass


def trace_path(trace_dir: Path) -> Path:
    return trace_dir / "llm_calls.jsonl"
