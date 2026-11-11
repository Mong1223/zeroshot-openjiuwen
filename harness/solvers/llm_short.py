"""Короткие ответы ИИ (улучшение №3 из PLAN.md): модель описывает решение, считает наш код.

Вместо целой программы (150–300 выходных токенов) модель присылает одну строку:
    math           → выражение:        (940 - 6*35) - (940 - 6*35)//3
    data_analysis  → JSON-запрос:      {"op": "sum", "column": "units", "filters": {"product": "gizmo"}, ...}
    search         → JSON-цепочка:     {"entity": "Irina Smith", "chain": ["Employer", "Headquarters"]}
Запрос выполняют уже проверенные бесплатные решатели (data_solver.execute, search_solver.follow),
выражение — безопасный вычислитель (только числа и арифметика, без eval).

Если короткий ответ не удалось разобрать или выполнить — переходим к программе (program_solve),
потраченные токены суммируются.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re
from fractions import Fraction

from harness.llm import LLM
from harness.solvers import data_solver, search_solver
from harness.solvers.llm_solvers import program_solve
from harness.tasks import question_text
from harness.types import Result, Status, Task

SHORT_MAX_TOKENS = 160
CONFIDENCE = 0.75
INFEASIBLE_CONFIDENCE = 0.6

MATH_RULES = """Reply with ONE line only: a single Python arithmetic expression whose value is the final answer.
Use only numbers, + - * / // % ** parentheses, min(), max(), abs(). Follow the events in order and repeat
sub-expressions if needed (no variables). Use // where the task says to round down.
If the task cannot be solved from the text (some needed quantity is not given), reply INFEASIBLE.
If the answer needs solving an equation or a search over values, reply PROGRAM."""

DATA_RULES = """Do NOT compute the answer. Reply with ONE line of JSON describing the query:
{"op": "sum|avg|max|min|count|revenue|argmax|argmin", "column": <numeric column, or "count"/"revenue" for argmax/argmin, or null>,
 "group": <column to group by for argmax/argmin, else null>, "filters": {<column>: <exact value>}, "months": [first, last] or null}
revenue means units*price. argmax = which group has the largest total.
"column" may also be a product of two numeric columns, e.g. "price*stock" (op "sum" for a total value).
A filter value is an exact value, or a condition: ">2", ">=10", "<5", "<=2016", "!=0", or "startswith:2025-03"
(e.g. for dates). "months" works only with a column named month; otherwise use a filter.
If the data cannot answer the question (missing column, value or period), reply INFEASIBLE.
If the query language above cannot express the question, reply PROGRAM."""

SEARCH_RULES = """Do NOT answer the question. Reply with ONE line of JSON:
{{"entity": "<name from the question>", "chain": [<fields to follow from the entity to the answer>]}}
for example {{"entity": "Irina Smith", "chain": ["Employer", "Headquarters"]}}.
For "how many people ..." questions reply {{"count": "<field>", "value": "<name>"}}.
Allowed fields: {fields}. Fields whose value is a name lead to that note.
If what is asked is not one of these fields (for example a translator, ISBN, address, phone, email, salary),
reply INFEASIBLE. Never substitute a different, merely related field."""


async def short_solve(task: Task, llm: LLM) -> Result:
    """Короткий ответ для math / data / search; иначе (или при неудаче) — программа."""
    handler = {"math": _math, "data_analysis": _data, "search": _search}.get(task.task_type)
    if handler is None:
        return await program_solve(task, llm)
    result = await handler(task, llm)
    if result.status in (Status.SOLVED, Status.INFEASIBLE):
        return result
    fallback = await program_solve(task, llm)
    fallback.tokens += result.tokens
    fallback.note = f"short failed ({result.note[:60]}); {fallback.note}"
    return fallback


async def _ask(task: Task, llm: LLM, prompt: str, purpose: str) -> tuple[str, float]:
    """→ (весь ответ модели без ```-обёрток, цена)."""
    reply = await llm.ask(prompt, task_id=task.task_id, purpose=purpose, max_tokens=SHORT_MAX_TOKENS)
    return re.sub(r"```(?:json|python|py)?", "", reply.content or "").strip(), reply.cost


def _infeasible(reply: str) -> bool:
    return _last_line(reply).strip(".`\"'*").upper() == "INFEASIBLE"


# ---------------------------------------------------------------- math: выражение

async def _math(task: Task, llm: LLM) -> Result:
    text, spent = await _ask(task, llm, f"Task: {question_text(task.instruction)}\n\n{MATH_RULES}", "expr")
    if _infeasible(text):
        return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=spent,
                      note="model: INFEASIBLE")
    text = _last_line(text)
    try:
        value = safe_eval(text)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        return Result(Status.FAILED, level=1, tokens=spent, note=f"bad expression {text[:60]!r}: {e}")
    if isinstance(value, (float, Fraction)):
        if value != int(value):
            return Result(Status.FAILED, level=1, tokens=spent, note=f"non-integer {float(value):g} from {text[:60]!r}")
        value = int(value)
    if value < 0 and re.search(r"\bhow (?:many|much|old|long|far)\b", task.instruction, re.I):
        # отрицательное количество/возраст — выражение составлено неверно, пусть решает программа
        return Result(Status.FAILED, level=1, tokens=spent, note=f"negative {value} from {text[:60]!r}")
    return Result(Status.SOLVED, answer=str(value), confidence=CONFIDENCE, level=1, tokens=spent,
                  note=f"expr {text[:80]}")


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round, "int": int, "floor": math.floor, "ceil": math.ceil}


def safe_eval(text: str) -> int | float:
    """Вычисляет арифметическое выражение без eval: только числа, операции и min/max/abs/round."""
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(?:python|answer|expression)\s*[:=]?\s*", "", text, flags=re.I)
    if "=" in text.replace("==", ""):  # «x = 5*3» — не выражение
        raise ValueError("assignment, not an expression")
    tree = ast.parse(text.replace("×", "*").replace("−", "-"), mode="eval")

    def ev(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # точные дроби: 1/(1/4 + 1/12) должно дать ровно 3, а не 2.9999999999999996
            return Fraction(node.value) if type(node.value) is int else Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("power too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords and 1 <= len(node.args) <= 8):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"not allowed: {type(node).__name__}")

    return ev(tree)


# ---------------------------------------------------------------- data: JSON-запрос

async def _data(task: Task, llm: LLM) -> Result:
    tables = data_solver.table_files(task)
    if len(tables) != 1:
        return Result(Status.FAILED, level=1, note=f"table files: {len(tables)}")
    header, rows = data_solver.load_table(task.workspace / tables[0])
    if not header:
        return Result(Status.FAILED, level=1, note="unsupported table")
    preview = "\n".join(",".join(r.get(h, "") for h in header) for r in rows[:3])
    prompt = (f"Task: {question_text(task.instruction)}\n\nThe file {tables[0]} has {len(rows)} records with fields: "
              f"{','.join(header)}\nFirst records:\n{preview}\n\n{DATA_RULES}")
    text, spent = await _ask(task, llm, prompt, "query")
    if _infeasible(text):
        return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=spent,
                      note="model: INFEASIBLE")
    query = _to_query(text, header)
    if isinstance(query, str):
        return Result(Status.FAILED, level=1, tokens=spent, note=query)
    asks_name = re.match(r"\s*(?:who|whose)\b", question_text(task.instruction), re.I)
    if query.op not in ("argmax", "argmin") and (asks_name or task.check == "exact_str"):
        # спрашивают имя/строку, а запрос считает число — модель не так поняла вопрос
        return Result(Status.FAILED, level=1, tokens=spent, note=f"numeric query for a text question: {text[:50]}")
    problem = data_solver._infeasible(query, header, rows)
    if problem:
        return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=spent, note=problem)
    value = data_solver.execute(query, rows)
    if value is None:
        return Result(Status.FAILED, level=1, tokens=spent, note=f"cannot compute {text[:60]}")
    value = apply_rounding(value, task.instruction)
    return Result(Status.SOLVED, answer=data_solver._fmt(value), confidence=CONFIDENCE, level=1, tokens=spent,
                  note=f"query {text[:80]}")


def apply_rounding(value, instruction: str):
    """Дробный результат + в задаче сказано, как округлять → округляем так же («rounded down»)."""
    if not isinstance(value, float) or value.is_integer():
        return value
    text = instruction.lower()
    if re.search(r"round(?:ed|ing)? down|floor|whole number|integer part|truncat", text):
        return float(math.floor(value))
    if re.search(r"round(?:ed|ing)? up|ceil", text):
        return float(math.ceil(value))
    if re.search(r"round(?:ed)? to the nearest (?:integer|whole)|nearest integer|rounded\b", text):
        return float(math.floor(value + 0.5))
    return value


def _to_query(text: str, header: list[str]) -> data_solver.Query | str:
    data = _json(text)
    if not isinstance(data, dict):
        return f"bad json {text[:60]!r}"
    cols = {h.lower(): h for h in header}
    op = str(data.get("op", "")).lower()
    if op not in ("sum", "avg", "max", "min", "count", "revenue", "argmax", "argmin"):
        return f"bad op {op!r}"
    column = data.get("column")
    if isinstance(column, str) and "*" in column:  # «price*stock»
        parts = [p.strip().lower() for p in column.split("*")]
        if len(parts) != 2 or any(p not in cols for p in parts):
            return f"bad column {column!r}"
        column = f"{cols[parts[0]]}*{cols[parts[1]]}"
        if op == "revenue":
            op = "sum"
    elif isinstance(column, str) and column.lower() in cols:
        column = cols[column.lower()]
    elif op in ("argmax", "argmin") and column in ("count", "revenue"):
        pass
    elif op in ("count", "revenue"):
        column = None
    else:
        return f"bad column {column!r}"
    group = data.get("group")
    if op in ("argmax", "argmin"):
        if not isinstance(group, str) or group.lower() not in cols:
            return f"bad group {group!r}"
        group = cols[group.lower()]
    else:
        group = None
    filters = {}
    for col, val in (data.get("filters") or {}).items():
        if str(col).lower() not in cols:
            return f"bad filter column {col!r}"
        filters[cols[str(col).lower()]] = str(val)
    months = data.get("months")
    if months is not None:
        try:
            a, b = int(months[0]), int(months[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return f"bad months {months!r}"
        if "month" not in cols:
            return "months but no month column"
        months = (min(a, b), max(a, b))
    return data_solver.Query(op, column, filters, months, exact=False, group=group)


# ---------------------------------------------------------------- search: JSON-цепочка

async def _search(task: Task, llm: LLM) -> Result:
    kb_dir = task.workspace / "kb"
    if not kb_dir.is_dir():
        return Result(Status.FAILED, level=1, note="no kb/")
    kb = search_solver.load_kb(kb_dir)
    fields = sorted({k for rec in kb.values() for k in rec if k != "name" and not k.startswith("_")})
    if not fields:  # база обычным текстом — цепочку полей не пройти; маленькую базу просто читаем
        return await _read_kb(task, llm, kb_dir)
    question = re.split(r"\bUse the file tools\b|\bUse bash\b", question_text(task.instruction))[0]
    prompt = f"Task: {question}\n\n{SEARCH_RULES.format(fields=', '.join(fields))}"
    text, spent = await _ask(task, llm, prompt, "chain")
    if _infeasible(text):
        return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=spent,
                      note="model: INFEASIBLE")
    data = _json(text)
    if not isinstance(data, dict):
        return Result(Status.FAILED, level=1, tokens=spent, note=f"bad json {text[:60]!r}")

    if "count" in data:  # «How many people work for X?»
        field, value = str(data.get("count", "")).title(), str(data.get("value", ""))
        if field not in fields or not value:
            return Result(Status.FAILED, level=1, tokens=spent, note=f"bad count {text[:60]!r}")
        n = sum(search_solver._norm(r.get(field, "")) == search_solver._norm(value) for r in kb.values())
        if n == 0 and search_solver._norm(value) not in kb:
            return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=spent,
                          note=f"{value!r} not in kb")
        return Result(Status.SOLVED, answer=str(n), confidence=CONFIDENCE, level=1, tokens=spent,
                      note=f"count {field}={value}")

    entity, chain = data.get("entity"), data.get("chain")
    if not isinstance(entity, str) or not isinstance(chain, list) or not chain:
        return Result(Status.FAILED, level=1, tokens=spent, note=f"bad chain {text[:60]!r}")
    chain = [_field(str(f), fields) for f in chain]
    if any(f is None for f in chain):  # непонятное поле — пусть разбирается программа
        return Result(Status.FAILED, level=1, tokens=spent, note=f"unknown field in {text[:60]!r}")
    result = search_solver.follow(kb, entity, chain, confidence=CONFIDENCE,
                                  infeasible_confidence=INFEASIBLE_CONFIDENCE, corpus=search_solver.kb_corpus(kb_dir))
    result.level, result.tokens = 1, spent
    return result


READ_LIMIT = 12_000  # символов текста базы (~3000 входных токенов = ~300 в зачёт: вход идёт по 1/10)

READ_RULES = """Answer the question using ONLY the notes above. Follow links between notes when needed
(for example: person → employer → the employer's note). Do not guess, and never answer with a merely
related fact (a city is not an address, an author is not a translator).
Reply with ONE line: just the answer value (a name, city or number, no extra words),
or INFEASIBLE if the notes do not contain the information."""


async def _read_kb(task: Task, llm: LLM, kb_dir) -> Result:
    """Маленькая база обычным текстом: показываем её модели целиком и просим только ответ."""
    files = sorted(p for p in kb_dir.rglob("*") if p.is_file())
    text = "\n".join(f"--- {p.relative_to(kb_dir).as_posix()} ---\n{p.read_text(encoding='utf-8', errors='replace').strip()}"
                     for p in files)
    if len(text) > READ_LIMIT:
        return Result(Status.FAILED, level=1, note=f"kb text too large to read ({len(text)} chars)")
    question = re.split(r"\bUse the file tools\b|\bUse bash\b", question_text(task.instruction))[0]
    reply = await llm.ask(f"Notes from kb/:\n{text}\n\nQuestion: {question}\n\n{READ_RULES}", task_id=task.task_id,
                          purpose="read", max_tokens=60)
    answer = _last_line(reply.content).strip(" .`*\"'")
    if _infeasible(answer):
        return Result(Status.INFEASIBLE, confidence=INFEASIBLE_CONFIDENCE, level=1, tokens=reply.cost,
                      note="read: INFEASIBLE")
    if not answer or len(answer) > 80:
        return Result(Status.FAILED, level=1, tokens=reply.cost, note=f"read: bad answer {answer[:60]!r}")
    return Result(Status.SOLVED, answer=answer, confidence=CONFIDENCE, level=1, tokens=reply.cost,
                  note=f"read {len(files)} notes")


def _field(name: str, fields: list[str]) -> str | None:
    """Имя поля от модели → поле базы, с синонимами: «company» → Employer, «HQ» → Headquarters."""
    low = name.strip().lower()
    for f in fields:
        if f.lower() == low:
            return f
    if low in search_solver.RELATIONS and search_solver.RELATIONS[low] in fields:
        return search_solver.RELATIONS[low]
    for rx, f in search_solver.ATTRIBUTES:
        if rx.search(low) and f in fields:
            return f
    return None


# ---------------------------------------------------------------- общее

def _last_line(text: str) -> str:
    text = re.sub(r"```(?:json|python|py)?", "", text or "")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
