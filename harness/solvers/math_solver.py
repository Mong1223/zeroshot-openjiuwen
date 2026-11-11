"""Решатель math, уровень 0: предложения → операции над запасом → вычисление кодом. 0 токенов.

Задачи в примерах собраны из шаблонных предложений:
    "A warehouse starts with 1752 boxes."                          → x = 1752
    "It ships 36 boxes per day for 3 days."                        → x -= 36 * 3
    "It receives 2 deliveries of 87 boxes each."                   → x += 2 * 87
    "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down)." → x //= 3
    "Then the stock is doubled by merging with a sister warehouse." → x *= 2

Перед правилами текст нормализуется (harness/normalize.py): числа словами → цифры,
синонимы глаголов → «ships» / «receives», дроби и проценты → метки frac_P_Q.

Защита от неверного ответа (лучше «не уверен», чем ошибка):
  * каждое предложение с числом должно совпасть с правилом, и каждое число должно быть использовано;
  * вопрос должен спрашивать про итоговый запас, а не «сколько отгрузили всего»;
  * деление без указания округления, если не делится нацело, — UNSURE;
  * предложение без чисел и без действий («The manager is happy.») пропускаем, но снижаем уверенность.
"""
from __future__ import annotations

import re
from typing import Callable

from harness.normalize import normalize
from harness.tasks import question_text
from harness.types import Result, Status, Task

A = r"(?P<a>\d[\d,]*)"
B = r"(?P<b>\d[\d,]*)"
FRAC = r"frac_(?P<p>\d+)_(?P<q>\d+)"
PERIOD = r"(?:day|week|hour|month|year|shift|night|morning|evening|weekday)"
RATE = rf"(?:(?:per|a|an|each|every) {PERIOD}|daily|weekly|hourly|monthly|nightly)"
DUR = rf"(?:for|over|during)(?: the next| the following)? {B} {PERIOD}s?"
CONTAINERS = (r"(?:deliveries|shipments|batches|crates|pallets|loads|truckloads|trucks|orders|packs|"
              r"packages|cartons|bags|lots|consignments)")
SUBJECT = r"(?:(?:it|they|he|she|the \w+|\w+) )?"
UP = r"(?:grows|grew|increases|increased|rises|rose|goes up|went up|climbs|climbed)"
DOWN = r"(?:shrinks|shrank|decreases|decreased|falls|fell|drops|dropped|goes down|went down|declines|declined)"


class NotInteger(ValueError):
    """Деление не нацело, а как округлять — не сказано."""


def _div(a: int, b: int, rnd: str | None) -> int:
    if b == 0:
        raise NotInteger("division by zero")
    if a % b == 0 or rnd == "down":
        return a // b
    if rnd == "up":
        return -(-a // b)
    raise NotInteger(f"{a}/{b} is not whole and rounding is not stated")


def _part(x: int, m: re.Match[str], rnd: str | None) -> int:
    return _div(x * int(m["p"]), int(m["q"]), rnd)


Op = Callable[[int, list[int], re.Match[str], str | None], int]

# (шаблон, операция). Порядок важен: более конкретные шаблоны раньше общих.
# Операция получает: текущий запас, числа из шаблона (a, b), само совпадение, округление.
RULES: list[tuple[re.Pattern[str], Op]] = [rule for rule in [
    # --- начальный запас (только первым действием)
    (rf"\b(?:starts?|started|begins?|began|opens?|opened)(?: the \w+)? with {A}\b", lambda x, n, m, r: n[0]),
    (rf"\bthere (?:are|were|is) {A}\b", lambda x, n, m, r: n[0]),
    (rf"\b(?:has|had|holds|held|contains|contained|stores|stored|owns|owned|keeps) {A} \w+",
     lambda x, n, m, r: n[0]),
    # --- «N в день в течение D дней», в обоих порядках
    (rf"\bships {A} \w+ {RATE} {DUR}", lambda x, n, m, r: x - n[0] * n[1]),
    (rf"\breceives {A} \w+ {RATE} {DUR}", lambda x, n, m, r: x + n[0] * n[1]),
    (rf"\b{DUR},? {SUBJECT}ships {A} \w+(?: {RATE})?", lambda x, n, m, r: x - n[0] * n[1]),
    (rf"\b{DUR},? {SUBJECT}receives {A} \w+(?: {RATE})?", lambda x, n, m, r: x + n[0] * n[1]),
    (rf"\b(?:every|each) {PERIOD} {DUR},? {SUBJECT}ships {A}\b", lambda x, n, m, r: x - n[0] * n[1]),
    (rf"\b(?:every|each) {PERIOD} {DUR},? {SUBJECT}receives {A}\b", lambda x, n, m, r: x + n[0] * n[1]),
    # --- «K поставок по M штук»
    (rf"\breceives {A} {CONTAINERS} of {B} \w+(?: each)?", lambda x, n, m, r: x + n[0] * n[1]),
    (rf"\bships {A} {CONTAINERS} of {B} \w+(?: each)?", lambda x, n, m, r: x - n[0] * n[1]),
    (rf"\bships {A} \w+ (?:to|for) each of (?:the |its )?{B} \w+", lambda x, n, m, r: x - n[0] * n[1]),
    (rf"\breceives {A} \w+ from each of (?:the |its )?{B} \w+", lambda x, n, m, r: x + n[0] * n[1]),
    # --- деление поровну, себе одна доля
    (rf"\b(?:split|divided|shared|distributed|separated) (?:equally |evenly )?(?:into|among|between|across) "
     rf"{A} \w+.*?\bkeeps? (?:1|one|a|a single|only 1|only one|only a) (?:share|part|portion|of (?:the|those) (?:shares|parts|portions))",
     lambda x, n, m, r: _div(x, n[0], r)),
    # --- умножение
    (r"\bdoubled\b", lambda x, n, m, r: x * 2),
    (r"\btripled\b", lambda x, n, m, r: x * 3),
    (r"\bquadrupled\b", lambda x, n, m, r: x * 4),
    (r"\bhalved\b", lambda x, n, m, r: _div(x, 2, r)),
    (rf"\bmultiplied by {A}\b", lambda x, n, m, r: x * n[0]),
    (rf"\b(?:is |gets )?divided by {A}\b", lambda x, n, m, r: _div(x, n[0], r)),
    # --- дроби и проценты: «sells a third of its stock», «grows by 25%», «keeps half of it»
    (rf"\bships {FRAC} of\b", lambda x, n, m, r: x - _part(x, m, r)),
    (rf"\b{FRAC} of (?:the|its|their|all|what|it|them)\b.*?\b(?:is|are|was|were|gets|get|got) ships\b",
     lambda x, n, m, r: x - _part(x, m, r)),
    (rf"\bkeeps (?:only )?{FRAC} of\b", lambda x, n, m, r: _part(x, m, r)),
    (rf"\b{UP} by {FRAC}", lambda x, n, m, r: x + _part(x, m, r)),
    (rf"\b{DOWN} by {FRAC}", lambda x, n, m, r: x - _part(x, m, r)),
    (rf"\breceives {FRAC} more\b", lambda x, n, m, r: x + _part(x, m, r)),
    # --- просто «+N» / «−N»
    (rf"\b{UP} by {A}\b", lambda x, n, m, r: x + n[0]),
    (rf"\b{DOWN} by {A}\b", lambda x, n, m, r: x - n[0]),
    (rf"\bships {A}\b", lambda x, n, m, r: x - n[0]),
    (rf"\breceives {A}\b", lambda x, n, m, r: x + n[0]),
] for rule in [(re.compile(rule[0]), rule[1])]]
START_OPS = {op for _, op in RULES[:3]}  # первая операция обязана задавать начальный запас

# подсказки исполнителю — не действия
_HINT = re.compile(r"^(?:compute it|use |calculate|show your work|write |give |answer |round |rounding )", re.I)
_QUESTION = re.compile(r"^(?:how many|how much|what is|what's|find |determine )", re.I)
# вопрос про итоговый запас (а не «сколько отгрузили всего» и т.п.)
_FINAL = re.compile(r"\b(?:have|has|hold|holds|contain|contains|own|left|remain|remains|remaining|end|final|"
                    r"now|stock|in stock|after)\b")
_OTHER_TARGET = re.compile(r"\b(?:each|per|total|in all|altogether|combined|ships|receives|ship|receive|"
                           r"sell|sold|buy|bought|deliver|delivered|lose|lost|give|gave)\b")
_ROUND_DOWN = re.compile(r"\bround(?:ed|ing)? (?:it |them |that |the \w+ (?:\w+ )?)?down\b|\binteger division\b|\bfloor\b")
_ROUND_UP = re.compile(r"\bround(?:ed|ing)? (?:it |them |that |the \w+ (?:\w+ )?)?up\b|\bceil")
_ROUND_OTHER = re.compile(r"\bnearest\b|\bdecimal|\bfraction")
_ACTION = re.compile(r"\b(?:ships|receives|doubled|tripled|quadrupled|halved|split|divid\w*|shar\w*|multipl\w*|"
                     r"increas\w*|decreas\w*|grow\w*|grew|shrink\w*|shrank|drop\w*|fall\w*|fell|ris\w*|rose|"
                     r"keep\w*|kept|frac_\w+|twice|thrice|times|remain\w*|rest|more|less|fewer|lend\w*|lent|"
                     r"borrow\w*|return\w*|transfer\w*|move\w*|load\w*|unload\w*|break\w*|broke\w*|damag\w*|"
                     r"spoil\w*|steal\w*|stole\w*|stolen|deliver\w*|pack\w*|replac\w*)\b")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?![\w.]\d)")

IRRELEVANT_CONFIDENCE = 0.85  # были пропущенные «лишние» предложения


def solve(task: Task) -> Result:
    program = parse(question_text(task.instruction))
    if isinstance(program, str):
        return Result(Status.UNSURE, note=program)
    steps, skipped = program
    try:
        value, trace = execute(steps)
    except NotInteger as e:
        return Result(Status.UNSURE, note=str(e))
    confidence = 1.0 if value >= 0 else 0.6  # отрицательный запас подозрителен
    if skipped:
        confidence = min(confidence, IRRELEVANT_CONFIDENCE)
        trace += f" (skipped {skipped} sentence(s) without actions)"
    return Result(Status.SOLVED, answer=str(value), confidence=confidence, note=trace)


Step = tuple[Op, list[int], re.Match[str], "str | None", str]


def parse(question: str) -> tuple[list[Step], int] | str:
    """→ (список операций, сколько предложений пропущено) или строка с причиной отказа."""
    program: list[Step] = []
    skipped = 0
    questions = 0
    global_rnd = None  # округление из отдельной подсказки: «Round down whenever needed.»
    for original in _sentences(question):
        s = normalize(original)
        if _QUESTION.match(original):
            questions += 1
            if not _FINAL.search(s) or _OTHER_TARGET.search(s) or _NUMBER.search(s):
                return f"question asks something else: {original[:80]!r}"
            global_rnd = global_rnd or _rounding(s)
            continue
        if _HINT.match(original) and not _NUMBER.search(s):
            global_rnd = global_rnd or _rounding(s)
            continue
        for rx, op in RULES:
            m = rx.search(s)
            if m:
                nums = [int(m[g].replace(",", "")) for g in ("a", "b") if g in rx.groupindex and m[g]]
                if len(_NUMBER.findall(s)) != len(nums):
                    return f"unused numbers in: {original[:80]!r}"
                program.append((op, nums, m, _rounding(s), original))
                break
        else:
            if _NUMBER.search(s) or _ACTION.search(s) or "frac_" in s:
                return f"unknown sentence: {original[:80]!r}"
            if _rounding(s):  # «Always round down.» — подсказка
                global_rnd = global_rnd or _rounding(s)
                continue
            skipped += 1  # «The manager is happy.» — нет ни чисел, ни действий

    if not program:
        return "no operations"
    if questions != 1:
        return f"expected exactly one question, got {questions}"
    if program[0][0] not in START_OPS or any(step[0] in START_OPS for step in program[1:]):
        return "no single starting amount"
    program = [(op, nums, m, rnd or global_rnd, text) for op, nums, m, rnd, text in program]
    return program, skipped


def execute(program: list[Step]) -> tuple[int, str]:
    x = 0
    steps = []
    for op, nums, m, rnd, _ in program:
        x = op(x, nums, m, rnd if rnd != "other" else None)
        steps.append(str(x))
    return x, " → ".join(steps)


def _rounding(s: str) -> str | None:
    if _ROUND_OTHER.search(s) and re.search(r"\bround", s):
        return "other"
    if _ROUND_DOWN.search(s):
        return "down"
    if _ROUND_UP.search(s):
        return "up"
    return None


def _sentences(text: str) -> list[str]:
    # точка внутри "(rounded down)." и т.п. — граница предложения, а вот "python3 -c" — нет
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip().rstrip(".") for p in parts if p.strip()]
