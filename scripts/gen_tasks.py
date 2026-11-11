"""Генератор своих проверочных задач (улучшение №2 из PLAN.md), по образцу GSM-Symbolic.

Создаёт задачи всех четырёх типов в формате жюри, с правильным ответом в поле "expected".
Чтобы проверить решатели на незнакомых формулировках, генератор:
  * перефразирует каждое действие несколькими способами (синонимы, числа словами, другой порядок слов);
  * добавляет «лишние» предложения: без чисел (ответ не меняется) и с числом (ловушка GSM-NoOp);
  * делает нерешаемые задачи (expected = "INFEASIBLE"): нет такого товара, человека, поля, месяца.
Ответ генератор считает сам, независимо от решателей.

    python scripts\\gen_tasks.py                    # 100 задач в data/gen, seed 1
    python scripts\\gen_tasks.py --n 200 --seed 7 --out data/gen7
    python scripts\\eval_level0.py data\\gen          # как справляются бесплатные решатели
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING_TAIL = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
               "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
INFEASIBLE = "INFEASIBLE"

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def in_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + in_words(rest))
    return str(n)


# ====================================================================== math

class MathGen:
    PLACES = ["warehouse", "shop", "depot", "store", "bakery", "farm", "library"]
    UNITS = ["boxes", "crates", "apples", "books", "eggs", "bottles", "pens"]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def num(self, n: int) -> str:
        """Иногда пишем число словами — как могут сделать в финальном наборе."""
        return in_words(n) if n < 100 and self.rng.random() < 0.3 else str(n)

    def make(self, trap: bool = False) -> tuple[str, str]:
        r, place, u = self.rng, self.rng.choice(self.PLACES), self.rng.choice(self.UNITS)
        x = r.randint(200, 2000)
        start = r.choice([
            f"A {place} starts with {x} {u}.",
            f"A {place} has {x} {u} at the start.",
            f"At the beginning, a {place} holds {x} {u}.",
            f"There are {x} {u} in a {place}.",
            f"A {place} begins with {x} {u}.",
        ])
        sentences = [start]
        for _ in range(r.randint(2, 4)):
            text, x = self.step(x, u)
            sentences.append(text)
        if r.random() < 0.25:  # лишнее предложение без чисел: ответ тот же, решатель должен справиться
            sentences.insert(r.randint(1, len(sentences)), r.choice([
                "The manager is very happy about this.", "It is a sunny day.",
                "Everyone works hard.", "The building was painted recently."]))
        if trap:  # лишнее предложение С числом (GSM-NoOp): ответ тот же, но угадывать нельзя
            k = r.randint(2, 9)
            sentences.insert(r.randint(1, len(sentences)), r.choice([
                f"{k} of the {u} are slightly smaller than the others.",
                f"The {place} has {k} employees.",
                f"The {place} is open {k} days a week."]))
        question = r.choice([f"How many {u} does it have at the end?", f"How many {u} are left?",
                             f"How many {u} remain in the {place}?", f"How many {u} does the {place} have now?"])
        sentences.append(question)
        sentences.append("Compute it with the bash tool (python3 -c).")
        return " ".join(sentences), str(x)

    def step(self, x: int, u: str) -> tuple[str, int]:
        r = self.rng
        kind = r.choice(["take_rate", "add_batches", "take", "add", "split", "mult", "frac", "pct", "each_of"])
        if kind == "take_rate":
            b = r.randint(2, 7)
            a = r.randint(5, max(6, x // (b * 2)))
            text = r.choice([f"It ships {self.num(a)} {u} per day for {self.num(b)} days.",
                             f"For {b} days, it sells {a} {u} each day.",
                             f"Every week for {b} weeks, it sends out {a} {u}.",
                             f"It dispatched {a} {u} daily for {b} days."])
            return text, x - a * b
        if kind == "add_batches":
            a, b = r.randint(2, 6), r.randint(10, 120)
            text = r.choice([f"It receives {self.num(a)} deliveries of {b} {u} each.",
                             f"{a} shipments of {b} {u} arrive.",
                             f"Then it buys {a} crates of {b} {u} each."])
            return text, x + a * b
        if kind == "take":
            a = r.randint(1, max(2, x // 3))
            text = r.choice([f"It sells {a} {u}.", f"{a} {u} are given away.", f"It loses {self.num(a)} {u}.",
                             f"Then {a} {u} were donated."])
            return text, x - a
        if kind == "add":
            a = r.randint(1, 300)
            text = r.choice([f"It receives {a} more {u}.", f"It buys {self.num(a)} {u}.",
                             f"{a} new {u} arrive.", f"Then it purchased {a} {u}."])
            return text, x + a
        if kind == "split":
            a = r.randint(2, 5)
            text = r.choice([f"Then the stock is split equally into {a} warehouses and it keeps one share (rounded down).",
                             f"The {u} are divided evenly among {self.num(a)} branches and it keeps one share (rounded down).",
                             f"Then the stock is shared equally between {a} stores and it keeps one share (rounded down)."])
            return text, x // a
        if kind == "mult":
            k, word = r.choice([(2, "doubled"), (3, "tripled"), (2, "doubles")])
            text = r.choice([f"Then the stock is {word} by merging with a sister warehouse.", f"The stock {word}."]
                            if word != "doubles" else ["Then the stock doubles.", "The stock doubles overnight."])
            return text, x * k
        if kind == "frac":
            (p, q), name = r.choice([((1, 2), "half"), ((1, 3), "a third"), ((1, 4), "a quarter")])
            amount = x * p // q
            text = r.choice([f"It sells {name} of its {u} (round down the amount sold).",
                             f"{name.capitalize()} of the {u} are sold (rounded down)."])
            return text, x - amount
        if kind == "pct":
            p = r.choice([10, 20, 25, 50])
            return f"The stock grows by {p}% (rounded down).", x + x * p // 100
        a, b = r.randint(2, 15), r.randint(2, 6)  # each_of
        a = min(a, max(1, x // (b * 2)))
        return r.choice([f"It gives {a} {u} to each of {b} stores.",
                         f"It sends {a} {u} to each of {self.num(b)} shops."]), x - a * b


# ====================================================================== data

class DataGen:
    REGIONS = ["north", "south", "east", "west"]
    PRODUCTS = ["widget", "gadget", "sprocket", "gizmo"]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def table(self) -> list[dict]:
        n = self.rng.choice([20, 60, 150, 300])
        return [{"order_id": i + 1, "region": self.rng.choice(self.REGIONS),
                 "product": self.rng.choice(self.PRODUCTS), "month": self.rng.randint(1, 12),
                 "units": self.rng.randint(1, 50), "price": self.rng.randint(5, 100)} for i in range(n)]

    def make(self, infeasible: bool = False) -> tuple[str, str, str, str]:
        """→ (вопрос, ответ, check, csv-текст)."""
        r, rows = self.rng, self.table()
        p, reg = r.choice(self.PRODUCTS), r.choice(self.REGIONS)
        m1 = r.randint(1, 11)
        m2 = r.randint(m1, 12)
        if infeasible:
            q = r.choice([
                f"How many units of product 'doohickey' were sold in months {m1} through {m2} inclusive?",
                f"How many units of product '{p}' were sold in months 13 through 14 inclusive?",
                f"What is the total revenue for region 'central'?",
                f"What is the average discount for product '{p}'?"])
            return q, INFEASIBLE, "exact_int", self._csv(rows)
        kind = r.choice(["units_months", "units_months", "revenue_region", "count", "max_price", "top_region",
                         "units_region_product"])
        sel = lambda f: [x for x in rows if f(x)]  # noqa: E731
        if kind == "units_months":
            q = r.choice([f"How many units of product '{p}' were sold in months {m1} through {m2} inclusive?",
                          f"What is the total number of units of '{p}' sold from month {m1} to month {m2} (inclusive)?"])
            a = sum(x["units"] for x in sel(lambda x: x["product"] == p and m1 <= x["month"] <= m2))
            return q, str(a), "exact_int", self._csv(rows)
        if kind == "revenue_region":
            q = r.choice([f"What is the total revenue (units times price) for region '{reg}'?",
                          f"Compute the total revenue, i.e. the sum of units*price, of all orders in the '{reg}' region."])
            a = sum(x["units"] * x["price"] for x in sel(lambda x: x["region"] == reg))
            return q, str(a), "exact_int", self._csv(rows)
        if kind == "count":
            q = r.choice([f"How many orders are there for product '{p}' in region '{reg}'?",
                          f"Count the orders of '{p}' placed in the {reg} region."])
            return q, str(len(sel(lambda x: x["product"] == p and x["region"] == reg))), "exact_int", self._csv(rows)
        if kind == "max_price":
            q = f"What is the highest price of product '{p}'?"
            return q, str(max(x["price"] for x in sel(lambda x: x["product"] == p))), "exact_int", self._csv(rows)
        if kind == "top_region":
            q = r.choice([f"Which region sold the most units of product '{p}'?",
                          f"In which region were the most units of '{p}' sold?"])
            totals = {g: sum(x["units"] for x in sel(lambda x: x["product"] == p and x["region"] == g))
                      for g in self.REGIONS}
            best = max(totals.values())
            if list(totals.values()).count(best) > 1:  # ничья — неоднозначный ответ, пересоздаём
                return self.make()
            return q, max(totals, key=totals.get), "exact_str", self._csv(rows)
        q = f"How many units of '{p}' were sold in the {reg} region?"
        a = sum(x["units"] for x in sel(lambda x: x["product"] == p and x["region"] == reg))
        return q, str(a), "exact_int", self._csv(rows)

    @staticmethod
    def _csv(rows: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()


# ====================================================================== search

class SearchGen:
    FIRST = ["Irina", "Sergei", "Alice", "Bob", "Maria", "Ivan", "Chen", "Aisha", "Lucas", "Mei", "Omar", "Nina"]
    LAST = ["Smith", "Garcia", "Petrova", "Kim", "Novak", "Ali", "Rossi", "Tanaka"]
    COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Analytics"]
    CITIES = ["Medan", "Osaka", "Lyon", "Porto", "Tartu", "Quito", "Perth", "Kazan"]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def kb(self) -> tuple[dict, dict, dict[str, str]]:
        r = self.rng
        companies = {c: {"Headquarters": r.choice(self.CITIES), "Founded": r.randint(1950, 2015)}
                     for c in r.sample(self.COMPANIES, 4)}
        names = r.sample([f"{f} {l}" for f in self.FIRST for l in self.LAST], 10)
        people = {}
        for i, n in enumerate(names):
            people[n] = {"Born": r.randint(1950, 2000), "Employer": r.choice(list(companies)),
                         "Mentor": names[(i + r.randint(1, 9)) % len(names)]}
        files = {}
        for c, d in companies.items():
            files[f"kb/company_{c.lower().replace(' ', '_')}.md"] = (
                f"# {c}\n\nHeadquarters: {d['Headquarters']}\nFounded: {d['Founded']}\n")
        for n, d in people.items():
            files[f"kb/person_{n.lower().replace(' ', '_')}.md"] = (
                f"# {n}\n\nBorn: {d['Born']}\nEmployer: {d['Employer']}\nMentor: {d['Mentor']}\n")
        return people, companies, files

    def make(self, infeasible: bool = False) -> tuple[str, str, str, dict[str, str]]:
        r = self.rng
        people, companies, files = self.kb()
        p = r.choice(list(people))
        d = people[p]
        if infeasible:
            ghost = "Zoe Quinn"
            q = r.choice([f"What is the phone number of {p}?", f"In which year was {ghost} born?",
                          f"What is the email address of {p}'s employer?",
                          f"In which city is the headquarters of {ghost}'s employer?"])
            return q, INFEASIBLE, "exact_str", files
        kind = r.choice(["born", "hq", "founded", "mentor_hq", "count", "mentor_born"])
        emp = companies[d["Employer"]]
        if kind == "born":
            q = r.choice([f"In which year was {p} born?", f"What is the birth year of {p}?"])
            return q, str(d["Born"]), "exact_int", files
        if kind == "hq":
            q = r.choice([f"In which city is the headquarters of {p}'s employer?",
                          f"Where is the company that employs {p} headquartered? Give the city."])
            return q, emp["Headquarters"], "exact_str", files
        if kind == "founded":
            q = r.choice([f"In which year was {p}'s employer founded?",
                          f"When was the employer of {p} founded? Give the year."])
            return q, str(emp["Founded"]), "exact_int", files
        if kind == "mentor_hq":
            m = people[d["Mentor"]]
            q = f"In which city is the headquarters of the employer of {p}'s mentor?"
            return q, companies[m["Employer"]]["Headquarters"], "exact_str", files
        if kind == "mentor_born":
            q = r.choice([f"In which year was {p}'s mentor born?", f"What is the birth year of {p}'s mentor?"])
            return q, str(people[d["Mentor"]]["Born"]), "exact_int", files
        c = r.choice(list(companies))
        q = r.choice([f"How many people in the knowledge base work for {c}?",
                      f"How many people are employed by {c}?"])
        return q, str(sum(x["Employer"] == c for x in people.values())), "exact_int", files


# ====================================================================== coding

# (имя функции, правильный код, тесты, список «поломок» (было → стало))
CODING = [
    ("total", "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
     ["self.assertEqual(m.total([1, 2, 3]), 6)", "self.assertEqual(m.total([]), 0)"],
     [("s += x", "s -= x"), ("s = 0", "s = 1")]),
    ("largest", "def largest(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
     ["self.assertEqual(m.largest([3, 9, 2]), 9)", "self.assertEqual(m.largest([-5, -2]), -2)"],
     [("x > best", "x < best")]),
    ("count_even", "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
     ["self.assertEqual(m.count_even([1, 2, 3, 4]), 2)", "self.assertEqual(m.count_even([1, 3]), 0)"],
     [("x % 2 == 0", "x % 2 == 1"), ("x % 2 == 0", "x % 2 != 0")]),
    ("first_n", "def first_n(n):\n    return [i for i in range(1, n + 1)]\n",
     ["self.assertEqual(m.first_n(3), [1, 2, 3])", "self.assertEqual(m.first_n(0), [])"],
     [("range(1, n + 1)", "range(1, n)"), ("range(1, n + 1)", "range(n + 1)")]),
    ("is_palindrome", "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]\n",
     ["self.assertTrue(m.is_palindrome('Abba'))", "self.assertFalse(m.is_palindrome('abc'))"],
     [("s.lower()", "s.upper()[1:]"), ("s == s[::-1]", "s != s[::-1]")]),
    ("both_positive", "def both_positive(a, b):\n    return a > 0 and b > 0\n",
     ["self.assertTrue(m.both_positive(1, 2))", "self.assertFalse(m.both_positive(1, -2))",
      "self.assertFalse(m.both_positive(-1, 2))"],
     [("a > 0 and b > 0", "a > 0 or b > 0")]),
    ("has_vowel", "def has_vowel(word):\n    for ch in word:\n        if ch in 'aeiou':\n            return True\n    return False\n",
     ["self.assertTrue(m.has_vowel('cat'))", "self.assertFalse(m.has_vowel('xyz'))"],
     [("ch in 'aeiou'", "ch not in 'aeiou'"), ("return False", "return True")]),
    ("top_two", "def top_two(xs):\n    return sorted(xs, reverse=True)[:2]\n",
     ["self.assertEqual(m.top_two([4, 1, 7, 3]), [7, 4])", "self.assertEqual(m.top_two([5]), [5])"],
     [("reverse=True", "reverse=False"), ("[:2]", "[:3]")]),
    ("factorial", "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
     ["self.assertEqual(m.factorial(5), 120)", "self.assertEqual(m.factorial(0), 1)"],
     [("r *= i", "r += i"), ("range(2, n + 1)", "range(2, n)")]),
    ("skip_negatives", "def skip_negatives(xs):\n    out = []\n    for x in xs:\n        if x < 0:\n            continue\n        out.append(x)\n    return out\n",
     ["self.assertEqual(m.skip_negatives([1, -2, 3]), [1, 3])", "self.assertEqual(m.skip_negatives([-1]), [])"],
     [("continue", "break")]),
]


def coding_task(rng: random.Random) -> tuple[str, dict[str, str]]:
    picked = rng.sample(CODING, rng.randint(1, 3))
    code, tests, broken = [], [], 0
    for name, src, checks, bugs in picked:
        if broken == 0 or rng.random() < 0.6:
            old, new = rng.choice(bugs)
            src = src.replace(old, new, 1)
            broken += 1
        code.append(src)
        tests.append(f"    def test_{name}(self):\n" + "".join(f"        {c}\n" for c in checks))
    test_file = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                 + "\n".join(tests) + "\n\nif __name__ == '__main__':\n    unittest.main()\n")
    q = rng.choice(["Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in "
                    "mylib.py (do not edit the tests) until all tests pass.",
                    "The unit tests in test_mylib.py are failing. Repair mylib.py so that every test passes; "
                    "the test file must stay unchanged."])
    return q, {"mylib.py": "\n\n".join(code), "test_mylib.py": test_file}


# ====================================================================== сборка

def write_task(out: Path, task_id: str, task_type: str, question: str, expected: str, check: str,
               reward: int, files: dict[str, str]) -> None:
    ws = out / task_id / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (ws / name).parent.mkdir(parents=True, exist_ok=True)
        (ws / name).write_text(text, encoding="utf-8")
    tail = CODING_TAIL if check == "unittest" else TAIL
    (out / task_id / "task.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "instruction": question + tail,
        "reward": reward, "expected": expected, "check": check}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "gen"))
    ap.add_argument("--infeasible", type=float, default=0.2, help="доля нерешаемых (по регламенту 15–25%)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    math_g, data_g, search_g = MathGen(rng), DataGen(rng), SearchGen(rng)
    stats: dict[str, int] = {}
    for i in range(args.n):
        kind = rng.choice(["math", "data_analysis", "search", "coding"])
        infeasible = kind in ("data_analysis", "search") and rng.random() < args.infeasible * 2
        reward = rng.randint(1, 5)
        task_id = f"{kind}_{i:04d}"
        if kind == "math":
            trap = rng.random() < 0.15
            q, a = math_g.make(trap=trap)
            task_id += "_trap" if trap else ""
            write_task(out, task_id, kind, q, a, "exact_int", reward, {})
        elif kind == "data_analysis":
            q, a, check, text = data_g.make(infeasible)
            n_rows = text.count("\n") - 1
            q = (f"The file sales.csv has columns order_id,region,product,month,units,price ({n_rows} rows). "
                 f"{q} Use bash with python3 to compute it.")
            write_task(out, task_id, kind, q, a, check, reward, {"sales.csv": text})
        elif kind == "search":
            q, a, check, files = search_g.make(infeasible)
            q = ("The directory kb/ contains a small knowledge base of markdown files. "
                 f"{q} Use the file tools or bash (ls, grep, cat) to find it.")
            write_task(out, task_id, kind, q, a, check, reward, files)
        else:
            q, files = coding_task(rng)
            write_task(out, task_id, kind, q, "tests_pass", "unittest", reward, files)
        key = kind + (" (нерешаемая)" if infeasible else "")
        stats[key] = stats.get(key, 0) + 1
    print(f"создано {args.n} задач в {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
