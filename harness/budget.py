"""Блок [C]: счётчик расходов и часы.

Прокси жюри не сообщает остаток бюджета и молча перестаёт отвечать, когда он кончился.
Поэтому считаем сами: стоимость вызова = output + input / 10 (рассуждения уже входят в output).
"""
from __future__ import annotations

import threading
import time


class Budget:
    """Учёт как у жюри: все отправленные запросы, вход/10 + выход (с рассуждениями).

    Жюри (встреча 09.11): прокси при исчерпании бюджета НЕ обрывает работу — бюджет считают они сами
    и засчитывают то, что решено в его пределах. Значит, остановиться вовремя — наша задача:
    перед каждым вызовом бронируем худший случай (hold), после — списываем факт (settle).
    Бронь нужна из-за параллельных вызовов: иначе два запроса одновременно пройдут проверку
    «хватает ли» и вместе превысят бюджет.
    """

    def __init__(self, total: float, reserve_fraction: float = 0.03):
        self.total = total
        self.reserve = total * reserve_fraction
        self.spent = 0.0
        self.held = 0.0
        self.calls = 0
        self.failed_calls = 0
        self.exhausted = False  # бюджет кончился (или прокси сообщил об этом) — дальше не пробуем
        self._lock = threading.Lock()

    @staticmethod
    def cost(input_tokens: int, output_tokens: int) -> float:
        return output_tokens + input_tokens / 10

    def charge(self, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost(input_tokens, output_tokens)
        with self._lock:
            self.spent += cost
            self.calls += 1
        return cost

    def hold(self, amount: float) -> bool:
        """Забронировать худший случай перед вызовом. False — не хватает, вызывать нельзя."""
        with self._lock:
            if self.exhausted or amount > self.total - self.reserve - self.spent - self.held:
                return False
            self.held += amount
            return True

    def settle(self, held: float, input_tokens: int, output_tokens: int, *, failed: bool = False) -> float:
        """Снять бронь и списать фактическую стоимость (для упавшего запроса — его вход)."""
        cost = self.cost(input_tokens, output_tokens)
        with self._lock:
            self.held = max(0.0, self.held - held)
            self.spent += cost
            self.calls += 1
            self.failed_calls += failed
        return cost

    def release(self, held: float) -> None:
        """Снять бронь без списания (запрос не был отправлен)."""
        with self._lock:
            self.held = max(0.0, self.held - held)

    @property
    def available(self) -> float:
        """Сколько ещё можно потратить, не залезая в резерв (с учётом броней)."""
        return max(0.0, self.total - self.reserve - self.spent - self.held)

    def can_afford(self, estimate: float) -> bool:
        return not self.exhausted and estimate <= self.available


class Clock:
    def __init__(self, limit_seconds: float):
        self.started = time.monotonic()
        self.limit = limit_seconds

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def left(self) -> float:
        return max(0.0, self.limit - self.elapsed)
