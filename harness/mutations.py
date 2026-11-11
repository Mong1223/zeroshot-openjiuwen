"""Генератор «подмен» в Python-коде — кандидатов на исправление мелкого бага. Работает по AST.

Примеры подмен: `s -= x` → `s += x`, `x < best` → `x > best`, `% 2 == 1` → `% 2 == 0`,
`sorted(...)` → `reversed(...)`, `xs[:n]` → `xs[:n + 1]`.

У каждой подмены есть приоритет: 1 — замена оператора/функции (так обычно и выглядят баги
в задачах), 2 — ±1 в числах и границах, 3 — замена переменной. Чем ниже приоритет, тем выше
риск «подогнать» код под видимые тесты, а проверяют задачу скрытыми тестами.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Callable, Iterator

BIN_SWAPS: dict[type, list[type]] = {
    ast.Add: [ast.Sub, ast.Mult], ast.Sub: [ast.Add],
    ast.Mult: [ast.Add, ast.FloorDiv, ast.Div, ast.Pow], ast.Div: [ast.FloorDiv, ast.Mult],
    ast.FloorDiv: [ast.Div, ast.Mod, ast.Mult], ast.Mod: [ast.FloorDiv],
    ast.Pow: [ast.Mult],
}
CMP_SWAPS: dict[type, list[type]] = {
    ast.Lt: [ast.Gt, ast.LtE], ast.Gt: [ast.Lt, ast.GtE], ast.LtE: [ast.GtE, ast.Lt],
    ast.GtE: [ast.LtE, ast.Gt], ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq],
    ast.In: [ast.NotIn], ast.NotIn: [ast.In], ast.Is: [ast.IsNot], ast.IsNot: [ast.Is],
}
FUNC_SWAPS = {"sorted": ["reversed"], "reversed": ["sorted"], "min": ["max"], "max": ["min"],
              "any": ["all"], "all": ["any"], "sum": ["len"], "len": ["sum"], "abs": [],
              "floor": ["ceil"], "ceil": ["floor"], "round": ["int"], "int": ["round"]}
UNWRAP = {"sorted", "reversed", "list", "set", "abs", "int", "round", "str"}
METHOD_SWAPS = {"lower": ["upper"], "upper": ["lower"], "append": ["extend"], "extend": ["append"],
                "lstrip": ["strip", "rstrip"], "rstrip": ["strip", "lstrip"], "strip": ["lstrip", "rstrip"],
                "startswith": ["endswith"], "endswith": ["startswith"], "split": ["rsplit"]}


@dataclass
class Mutation:
    priority: int
    function: str | None      # в какой функции подмена (для фокуса на упавших тестах)
    description: str
    apply: Callable[[ast.Module], None]  # меняет копию дерева


def mutations(tree: ast.Module) -> Iterator[Mutation]:
    """Все одиночные подмены в модуле."""
    for path, node, func in _walk(tree):
        yield from _for_node(path, node, func, tree)


def apply(tree: ast.Module, mutation: Mutation) -> str:
    """Исходный код модуля с применённой подменой (оригинальное дерево не трогается)."""
    new = copy.deepcopy(tree)
    mutation.apply(new)
    ast.fix_missing_locations(new)
    return ast.unparse(new)


# ---------------------------------------------------------------- обход

def _walk(tree: ast.AST):
    """(путь от корня, узел, имя функции верхнего уровня)."""
    stack: list[tuple[tuple, ast.AST, str | None]] = [((), tree, None)]
    while stack:
        path, node, func = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and func is None:
            func = node.name
        yield path, node, func
        for name, value in reversed(list(ast.iter_fields(node))):
            if isinstance(value, list):
                for i in reversed(range(len(value))):
                    if isinstance(value[i], ast.AST):
                        stack.append((path + ((name, i),), value[i], func))
            elif isinstance(value, ast.AST):
                stack.append((path + ((name, None),), value, func))


def _get(tree: ast.AST, path: tuple) -> ast.AST:
    node = tree
    for name, i in path:
        node = getattr(node, name) if i is None else getattr(node, name)[i]
    return node


def _replace(tree: ast.AST, path: tuple, new: ast.AST) -> None:
    parent = _get(tree, path[:-1])
    name, i = path[-1]
    if i is None:
        setattr(parent, name, new)
    else:
        getattr(parent, name)[i] = new


def _plus(expr: ast.expr, k: int) -> ast.expr:
    return ast.BinOp(left=copy.deepcopy(expr), op=ast.Add() if k > 0 else ast.Sub(), right=ast.Constant(abs(k)))


# ---------------------------------------------------------------- подмены по типам узлов

def _for_node(path, node, func, tree) -> Iterator[Mutation]:
    src = lambda: ast.unparse(node)  # noqa: E731 — описание считаем лениво

    if isinstance(node, (ast.BinOp, ast.AugAssign)):
        for new_op in BIN_SWAPS.get(type(node.op), []):
            yield Mutation(1, func, f"{type(node.op).__name__}→{new_op.__name__} in {src()}",
                           lambda t, p=path, o=new_op: setattr(_get(t, p), "op", o()))

    elif isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for new_op in CMP_SWAPS.get(type(op), []):
                yield Mutation(1, func, f"{type(op).__name__}→{new_op.__name__} in {src()}",
                               lambda t, p=path, i=i, o=new_op: _get(t, p).ops.__setitem__(i, o()))

    elif isinstance(node, ast.BoolOp):
        new_op = ast.Or if isinstance(node.op, ast.And) else ast.And
        yield Mutation(1, func, f"And/Or swap in {src()}", lambda t, p=path, o=new_op: setattr(_get(t, p), "op", o()))

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        yield Mutation(2, func, f"drop unary in {src()}",
                       lambda t, p=path: _replace(t, p, copy.deepcopy(_get(t, p).operand)))

    elif isinstance(node, ast.Constant) and type(node.value) is bool:
        yield Mutation(1, func, f"{node.value}→{not node.value}",
                       lambda t, p=path: setattr(_get(t, p), "value", not _get(t, p).value))

    elif isinstance(node, ast.Constant) and type(node.value) is int:
        for k in (1, -1):
            yield Mutation(2, func, f"{node.value}→{node.value + k}",
                           lambda t, p=path, k=k: setattr(_get(t, p), "value", _get(t, p).value + k))

    elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 3:
        # частые баги в строках-разделителях: ' ' ↔ '' ↔ ','
        for alt in (" ", "", ",", "-"):
            if alt != node.value:
                yield Mutation(3, func, f"{node.value!r}→{alt!r}",
                               lambda t, p=path, a=alt: setattr(_get(t, p), "value", a))

    elif isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        for alt in FUNC_SWAPS.get(name, []):
            yield Mutation(1, func, f"{name}→{alt}",
                           lambda t, p=path, a=alt: setattr(_get(t, p).func, "id", a))
        if name in UNWRAP and len(node.args) == 1 and not node.keywords:
            yield Mutation(2, func, f"unwrap {name}(...)",
                           lambda t, p=path: _replace(t, p, copy.deepcopy(_get(t, p).args[0])))
        if name == "range":
            for i, arg in enumerate(node.args):
                for k in (1, -1):
                    yield Mutation(2, func, f"range arg {i} {k:+d}",
                                   lambda t, p=path, i=i, k=k: _get(t, p).args.__setitem__(i, _plus(_get(t, p).args[i], k)))
            if len(node.args) == 1:  # range(n) → range(1, n): забыли начало
                yield Mutation(2, func, "range start 1",
                               lambda t, p=path: _get(t, p).args.insert(0, ast.Constant(1)))
            elif (len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                  and node.args[0].value in (0, 1)):  # range(1, n) → range(n)
                yield Mutation(2, func, "range drop start",
                               lambda t, p=path: _get(t, p).args.pop(0))
        if isinstance(node.func, ast.Attribute):
            for alt in METHOD_SWAPS.get(node.func.attr, []):
                yield Mutation(1, func, f".{node.func.attr}→.{alt}",
                               lambda t, p=path, a=alt: setattr(_get(t, p).func, "attr", a))
        if any(kw.arg == "reverse" for kw in node.keywords):
            yield Mutation(1, func, "flip reverse=",
                           lambda t, p=path: [setattr(kw, "value", ast.UnaryOp(ast.Not(), kw.value))
                                              for kw in _get(t, p).keywords if kw.arg == "reverse"])
        elif name == "sorted":
            yield Mutation(1, func, "add reverse=True",
                           lambda t, p=path: _get(t, p).keywords.append(ast.keyword("reverse", ast.Constant(True))))

    elif isinstance(node, (ast.Break, ast.Continue)):  # break ↔ continue (как в mutmut / MutPy)
        new_cls = ast.Continue if isinstance(node, ast.Break) else ast.Break
        yield Mutation(1, func, f"{type(node).__name__}→{new_cls.__name__}",
                       lambda t, p=path, c=new_cls: _replace(t, p, c()))

    elif isinstance(node, (ast.If, ast.While, ast.IfExp)) and not (
            isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)):
        # добавить not в условие (обратное «drop unary» уже есть выше)
        yield Mutation(2, func, f"negate condition {ast.unparse(node.test)[:40]}",
                       lambda t, p=path: setattr(_get(t, p), "test", ast.UnaryOp(ast.Not(), _get(t, p).test)))

    elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and isinstance(node.ctx, ast.Load):
        # лишний срез: s[1:] → s
        # (±1 в границах среза даёт ветка ast.Slice: обход заходит и в сам срез)
        yield Mutation(2, func, f"drop slice in {src()[:40]}",
                       lambda t, p=path: _replace(t, p, copy.deepcopy(_get(t, p).value)))

    elif isinstance(node, ast.Slice):
        for field in ("lower", "upper"):
            if getattr(node, field) is not None and not isinstance(getattr(node, field), ast.Constant):
                for k in (1, -1):
                    yield Mutation(2, func, f"slice {field} {k:+d}",
                                   lambda t, p=path, f=field, k=k: setattr(_get(t, p), f, _plus(getattr(_get(t, p), f), k)))

    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and func is not None:
        # замена переменной на другую из той же функции (return s вместо return x)
        fn = next(n for _, n, f in _walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {a.arg for a in fn.args.args}
        for alt in sorted(names - {node.id}):
            if alt in FUNC_SWAPS or alt in UNWRAP or alt == "range":
                continue
            yield Mutation(3, func, f"{node.id}→{alt}", lambda t, p=path, a=alt: setattr(_get(t, p), "id", a))
