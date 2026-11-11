"""Решатель data_analysis, уровень 0: вопрос по CSV → запрос → подсчёт кодом. 0 токенов.

Анализатор понимает вопросы вида
    "How many units of product 'widget' were sold in months 7 through 8 inclusive?"
и близкие варианты (один месяц, регион, сумма/среднее/максимум, число заказов, выручка).
Всё, что он не узнал уверенно, возвращается как UNSURE — дальше решает следующий уровень.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.tasks import question_text
from harness.types import Result, Status, Task

_QUOTED = re.compile(r"""['"‘’“”]([^'"‘’“”]+)['"‘’“”]""")
_MONTH_RANGE = re.compile(
    r"months?\s+(\d{1,2})\s*(?:through|to|until|till|-|–|and)\s*(?:months?\s+)?(\d{1,2})"
    r"|between\s+months?\s+(\d{1,2})\s+and\s+(?:months?\s+)?(\d{1,2})", re.I)
_COUNT = re.compile(r"\b(?:how many|count(?: the| all)?|number of|total number of) "
                    r"(?:orders|rows|records|transactions|sales|entries)\b")
# «Which region sold the most units…», «In which month were the fewest orders…»
_TOP = re.compile(r"\b(?:which|what) (\w+)\b.*?\b(most|highest|largest|biggest|greatest|least|fewest|lowest|smallest)\b")
_MONTH_ONE = re.compile(r"\bin\s+month\s+(\d{1,2})\b", re.I)
_REGION = re.compile(r"\bin\s+(?:the\s+)?(?:region\s+['\"]?(\w+)['\"]?|['\"]?(\w+)['\"]?\s+region)\b", re.I)
_ROWS_CLAIM = re.compile(r"\((\d+)\s+rows?\)", re.I)
_COMPARISON = re.compile(r"\b(?:fewer|less|more|greater|higher|lower|larger|smaller|bigger) than\b|\bat (?:least|most)\b"
                         r"|\b(?:above|below|exceed\w*|strictly|over \d|under \d|after \d|before \d|since \d)\b")


def _strip_header(q: str) -> str:
    """Убирает описание файла «has columns a,b,c (300 rows)», чтобы оно не мешало разбору вопроса."""
    return re.sub(r"\bcolumns?\s+[\w,]+(?:\s*\(\d+\s+rows?\))?", " ", q)


@dataclass
class Query:
    op: str                       # sum | avg | max | min | count | revenue
    column: str | None            # колонка для sum/avg/max/min
    filters: dict[str, str] = field(default_factory=dict)
    months: tuple[int, int] | None = None
    exact: bool = True            # вопрос совпал с хорошо известным шаблоном
    group: str | None = None      # для argmax/argmin: по какой колонке группировать


def table_files(task: Task) -> list[str]:
    """Файлы-таблицы задачи: CSV/TSV и JSON со списком объектов."""
    return [f for f in task.files if f.lower().endswith((".csv", ".tsv", ".json")) and not f.startswith("kb/")]


def solve(task: Task) -> Result:
    tables = table_files(task)
    if len(tables) != 1:
        return Result(Status.UNSURE, note=f"table files: {len(tables)}")
    header, rows = load_table(task.workspace / tables[0])
    if not header:
        return Result(Status.UNSURE, note="empty or unsupported table")

    question = question_text(task.instruction)
    query = parse(question, header, rows)
    if query is None:
        missing = _unknown_column(question.lower(), header, rows)
        if missing:  # «average discount», а столбца discount нет
            return Result(Status.INFEASIBLE, confidence=0.65, note=f"no column like {missing!r}")
        return Result(Status.UNSURE, note="question not recognized")

    problem = _infeasible(query, header, rows)
    if problem:
        return Result(Status.INFEASIBLE, confidence=0.8, note=problem)

    value = execute(query, rows)
    if value is None:
        return Result(Status.UNSURE, note="cannot compute")

    note = f"{query.op}({query.column}) {query.filters} months={query.months}"
    claimed = _ROWS_CLAIM.search(question)
    if claimed and int(claimed.group(1)) != len(rows):
        note += f"; rows claimed {claimed.group(1)} != actual {len(rows)}"
    return Result(Status.SOLVED, answer=_fmt(value), confidence=1.0 if query.exact else 0.8, note=note)


def parse(question: str, header: list[str], rows: list[dict[str, str]] | None = None) -> Query | None:
    q = question.lower()
    cols = {c.lower(): c for c in header}

    # фильтры: значение в кавычках относим к колонке — по подписи рядом или по самим данным
    filters: dict[str, str] = {}
    for value in _QUOTED.findall(question):
        if value.lower() in cols:  # это имя колонки в кавычках, а не значение
            continue
        column = _column_before(q, value.lower(), cols) or _column_with_value(value, rows or [])
        if column is None:
            column = cols.get("product")  # неизвестное значение без подписи — скорее всего товар
        if column is None:
            return None
        filters[column] = value
    region = _REGION.search(question)
    if (region and "region" in cols and "region" not in filters
            and (region.group(1) or region.group(2)).lower() not in ("which", "what", "each", "every", "any")):
        filters[cols["region"]] = region.group(1) or region.group(2)

    months = None
    m = _MONTH_RANGE.search(question)
    if m:
        a, b = [int(x) for x in m.groups() if x is not None]
        months = (min(a, b), max(a, b))
    elif (m := _MONTH_ONE.search(question)):
        months = (int(m.group(1)), int(m.group(1)))
    if months and "month" not in cols:
        return None

    # «какой регион/товар/месяц больше всего…» — группировка и выбор лучшей группы
    top = _TOP.search(q)
    if top and top.group(1) in cols:
        group = cols[top.group(1)]
        lowest = top.group(2) in ("least", "fewest", "lowest", "smallest")
        if re.search(r"\brevenue\b", q) and "units" in cols and "price" in cols:
            metric = "revenue"
        elif _COUNT.search(q) or re.search(r"\bmost orders\b|\bfewest orders\b", q):
            metric = "count"
        else:
            metric = _numeric_column_mentioned(q, cols, rows or [], exclude={group})
            if metric is None:
                return None
        return Query("argmin" if lowest else "argmax", metric, filters, months, exact=False, group=group)

    # операция и колонка
    if re.search(r"\brevenue\b|\bsales value\b", q):
        if "units" not in cols or "price" not in cols:
            return None
        return Query("revenue", None, filters, months, exact=False)
    if _COUNT.search(q):
        return Query("count", None, filters, months, exact=False)

    # Условия-сравнения правила не разбирают («fewer than 10 in stock», «hired after 2016») —
    # без них ответ был бы неверным, а не просто неточным. Такие вопросы — следующему уровню.
    if _COMPARISON.search(_strip_header(q)):
        return None
    many = re.search(r"how many (\w+)", q)
    if many and many.group(1) not in cols:  # «how many products…» — это подсчёт строк с условием, не сумма
        return None

    op = "sum"
    if re.search(r"\baverage\b|\bmean\b", q):
        op = "avg"
    elif re.search(r"\b(?:maximum|largest|highest|max)\b", q):
        op = "max"
    elif re.search(r"\b(?:minimum|smallest|lowest|min)\b", q):
        op = "min"

    column = None
    m = re.search(r"how many (\w+)", q) or re.search(r"(?:total|sum of|average|mean|maximum|minimum)\s+(?:of\s+)?(\w+)", q)
    if m and m.group(1) in cols:
        column = cols[m.group(1)]
    if column is None:  # «total number of units…», «the highest price of…»
        column = _numeric_column_mentioned(q, cols, rows or [], exclude=set(filters))
    if column is None:
        return None
    exact = op == "sum" and q.startswith("the file") and "how many" in q and months is not None
    return Query(op, column, filters, months, exact=exact)


_CMP = re.compile(r"^\s*(>=|<=|!=|>|<|=)\s*(-?\d+(?:\.\d+)?)\s*$")


def _match(cell: str, cond: str) -> bool:
    """Условие фильтра: точное значение, сравнение («>2», «<=2016», «!=0») или «startswith:2025-03»."""
    cell = cell.strip()
    m = _CMP.match(cond)
    if m:
        x = _num(cell)
        if x is None:
            return False
        op, v = m.group(1), float(m.group(2))
        return {">": x > v, "<": x < v, ">=": x >= v, "<=": x <= v, "!=": x != v, "=": x == v}[op]
    if cond.lower().startswith("startswith:"):
        return cell.lower().startswith(cond[11:].strip().lower())
    return cell.lower() == cond.strip().lower()


def is_condition(cond: str) -> bool:
    return bool(_CMP.match(cond)) or cond.lower().startswith("startswith:")


def execute(query: Query, rows: list[dict[str, str]]) -> float | str | None:
    selected = []
    for row in rows:
        if not all(_match(row.get(col, ""), val) for col, val in query.filters.items()):
            continue
        if query.months:
            month = _num(row.get("month"))
            if month is None or not query.months[0] <= month <= query.months[1]:
                continue
        selected.append(row)

    if query.op in ("argmax", "argmin"):
        totals: dict[str, float] = {}
        for row in selected:
            key = row.get(query.group, "").strip()
            if query.column == "count":
                value = 1.0
            elif query.column == "revenue":
                u, p = _num(row.get("units")), _num(row.get("price"))
                value = None if u is None or p is None else u * p
            else:
                value = _num(row.get(query.column))
            if value is None:
                return None
            totals[key] = totals.get(key, 0.0) + value
        if not totals:
            return None
        best = (max if query.op == "argmax" else min)(totals.values())
        winners = [k for k, v in totals.items() if v == best]
        return winners[0] if len(winners) == 1 else None  # ничья — ответ неоднозначен

    if query.op == "count":
        return len(selected)
    if query.op == "revenue":
        values = [(_num(r.get("units")), _num(r.get("price"))) for r in selected]
        if any(u is None or p is None for u, p in values):
            return None
        return sum(u * p for u, p in values)

    if query.column and "*" in query.column:  # «price*stock» — произведение двух столбцов
        a, b = [c.strip() for c in query.column.split("*", 1)]
        pairs = [(_num(r.get(a)), _num(r.get(b))) for r in selected]
        values = [None if x is None or y is None else x * y for x, y in pairs]
    else:
        values = [_num(r.get(query.column)) for r in selected]
    if any(v is None for v in values):
        return None
    if query.op == "sum":
        return sum(values)
    if not values:
        return None
    return {"avg": sum(values) / len(values), "max": max(values), "min": min(values)}[query.op]


def _infeasible(query: Query, header: list[str], rows: list[dict[str, str]]) -> str | None:
    """Признаки того, что задачу нельзя решить по имеющимся данным."""
    for col, val in query.filters.items():
        if is_condition(val):  # «>2» — условие, а не значение; пустой результат ещё не значит «нерешаемо»
            continue
        present = {r.get(col, "").strip().lower() for r in rows}
        if val.lower() not in present:
            return f"value {val!r} not in column {col!r}"
    if query.months and not (1 <= query.months[0] <= 12 and 1 <= query.months[1] <= 12):
        return f"months out of range {query.months}"
    return None


def _column_before(q: str, value: str, cols: dict[str, str]) -> str | None:
    """Колонка, имя которой стоит перед значением: "product 'widget'" → product."""
    pos = q.find(value)
    before = q[:pos].split()[-2:]
    for word in reversed(before):
        word = word.strip("'\"‘“:,")
        if word in cols:
            return cols[word]
    after = q[pos + len(value):].split()[:2]  # "the 'east' region"
    for word in after:
        word = word.strip("'\"’”:,?.")
        if word in cols:
            return cols[word]
    return None


def _numeric_column_mentioned(q: str, cols: dict[str, str], rows: list[dict[str, str]],
                              exclude: set[str]) -> str | None:
    """Единственная числовая колонка, названная в вопросе («units», «price»). Иначе None."""
    q = re.sub(r"\bcolumns?\s+[\w,]+", " ", q)  # описание файла «has columns a,b,c» — не вопрос
    found = []
    for low, col in cols.items():
        if col in exclude or low in ("order_id", "id", "month"):
            continue
        if not re.search(rf"\b{re.escape(low)}\b|\b{re.escape(low.rstrip('s'))}\b", q):
            continue
        sample = [r.get(col, "") for r in rows[:20]]
        if sample and all(_num(v) is not None for v in sample):
            found.append(col)
    return found[0] if len(found) == 1 else None


# слова, которые означают существующие величины, даже если так столбец не назван
_KNOWN_MEASURES = {"number", "amount", "quantity", "units", "unit", "revenue", "sales", "sale", "orders", "order",
                   "value", "count", "price", "prices", "cost", "total", "sum", "items", "rows", "records"}
_ASKED_MEASURE = re.compile(r"\b(?:average|mean|total|sum of|maximum|minimum|highest|lowest|largest|smallest|"
                            r"median) (?:the )?([a-z_]+)")


def _unknown_column(q: str, header: list[str], rows: list[dict[str, str]]) -> str | None:
    """Спрашивают величину, которой нет ни в столбцах, ни в данных («average discount»)."""
    q = re.sub(r"\bcolumns?\s+[\w,]+", " ", q)
    m = _ASKED_MEASURE.search(q)
    if not m:
        return None
    word = m.group(1)
    if word in _KNOWN_MEASURES or any(word.rstrip("s") in h.lower() for h in header):
        return None
    # «rainfall» ↔ rain_mm, «temperature» ↔ temp_c: общее начало из 4+ букв — это тот же столбец
    tokens = [t for h in header for t in re.split(r"[_\W]+", h.lower()) if len(t) >= 3]
    if any(word[:4] == t[:4] for t in tokens if len(t) >= 4) or any(word.startswith(t) for t in tokens):
        return None
    if any(word == cell.strip().lower() for row in rows[:500] for cell in row.values()):
        return None
    return word


def _column_with_value(value: str, rows: list[dict[str, str]]) -> str | None:
    value = value.lower()
    for row in rows:
        for col, cell in row.items():
            if cell.strip().lower() == value:
                return col
    return None


def load_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """CSV/TSV или JSON-список объектов → (заголовок, строки со строковыми значениями)."""
    if path.suffix.lower() == ".json":
        import json
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            return [], []
        if isinstance(data, dict):  # {"items": [...]} — берём единственный список внутри
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if len(lists) == 1 else None
        if not isinstance(data, list) or not data or not all(isinstance(x, dict) for x in data):
            return [], []
        header = list(dict.fromkeys(k for x in data for k in x))
        rows = [{k: "" if x.get(k) is None else str(x.get(k)) for k in header} for x in data]
        return header, rows
    if path.suffix.lower() == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = [{k.strip(): (v or "") for k, v in row.items() if k} for row in reader]
            return [h.strip() for h in (reader.fieldnames or [])], rows
    return _load(path)


def _load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): (v or "") for k, v in row.items() if k} for row in reader]
        return [h.strip() for h in (reader.fieldnames or [])], rows


def _num(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fmt(value: float | str) -> str:
    if isinstance(value, str):  # argmax/argmin: название группы
        return value
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"
