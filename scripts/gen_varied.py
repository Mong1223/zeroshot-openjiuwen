"""Генератор «других» задач: формулировки и данные НЕ похожи на 8 примеров жюри.

Организатор 10.11: «формулировки будут другими», однотипность внутри категории не гарантируется.
Этот набор проверяет прежде всего ИИ-уровень: бесплатные правила почти ничего здесь не возьмут.

    math          — покупки и сдача, скорость, возраст, скидки, трубы, делимость, прогрессии, остатки;
                    нерешаемые — не хватает данных
    data_analysis — employees.csv, weather.csv, products.json (не sales.csv!)
    search        — база книг (книга → автор → страна; издатель → город) и база в виде ОБЫЧНОГО ТЕКСТА
    coding        — другие модули и имена файлов, логические баги (не только замена оператора) и
                    СКРЫТЫЕ тесты в папке hidden/ (как у жюри: «тесты будут шире выданных»)

    python scripts\\gen_varied.py                         # 60 задач в data/varied
    python scripts\\gen_varied.py --n 80 --seed 5 --out data/varied5
    python scripts\\run_local.py data\\varied              # репетиция с ИИ
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING_TAIL = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
               "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
INFEASIBLE = "INFEASIBLE"
NAMES = ["Maria", "Peter", "Aisha", "Chen", "Olga", "Lucas", "Nina", "Omar", "Sofia", "Ivan"]


# ====================================================================== math

def math_task(r: random.Random) -> tuple[str, str]:
    kind = r.choice(["shop", "travel", "ages", "discount", "pipes", "divisible", "series", "teams", "printer",
                     "infeasible"])
    who = r.choice(NAMES)
    if kind == "shop":
        a, p1, b, p2 = r.randint(2, 6), r.randint(2, 9), r.randint(2, 8), r.randint(1, 5)
        total = a * p1 + b * p2
        bill = next(x for x in (20, 50, 100) if x > total)
        return (f"{who} buys {a} notebooks for ${p1} each and {b} pens for ${p2} each. "
                f"{who} pays with a ${bill} bill. How much change, in dollars, does {who} receive?", str(bill - total))
    if kind == "travel":
        v, t, d = r.choice([12, 15, 18, 20]), r.randint(2, 5), r.randint(5, 30)
        return (f"A cyclist rides at {v} km per hour for {t} hours, takes a break, and then rides another {d} km. "
                f"What total distance, in km, did the cyclist cover?", str(v * t + d))
    if kind == "ages":
        s, k = r.randint(4, 15), r.choice([3, 4, 5])
        y = (k - 2) * s
        return (f"{who} is {k} times as old as {who}'s daughter. In {y} years {who} will be exactly twice as old "
                f"as the daughter will be then. How old is the daughter now?", str(s))
    if kind == "discount":
        p, d, t = r.choice([100, 200, 300, 400]), r.choice([10, 20, 25, 50]), r.choice([10, 20])
        final = Fraction(p) * (100 - d) / 100 * (100 + t) / 100
        return (f"A jacket costs {p} dollars. During a sale its price is reduced by {d}%, and then a {t}% sales tax "
                f"is added to the reduced price. How much does the customer pay, in dollars?", str(int(final)))
    if kind == "pipes":
        a, b = r.choice([(3, 6), (4, 12), (6, 12), (10, 15), (12, 24), (20, 30)])
        return (f"Pipe A alone can fill a pool in {a} hours, and pipe B alone can fill it in {b} hours. "
                f"How many hours does it take to fill the empty pool with both pipes open?", str(a * b // (a + b)))
    if kind == "divisible":
        n, a, b = r.randint(50, 500), r.choice([3, 4, 6]), r.choice([5, 7, 10])
        return (f"How many integers from 1 to {n} inclusive are divisible by {a} or by {b}?",
                str(sum(1 for i in range(1, n + 1) if i % a == 0 or i % b == 0)))
    if kind == "series":
        s, d, n = r.randint(2, 6), r.randint(1, 3), r.randint(5, 12)
        return (f"{who} runs {s} km on the first day of training and every following day runs {d} km more than "
                f"the day before. How many km does {who} run in total over {n} days?",
                str(sum(s + i * d for i in range(n))))
    if kind == "teams":
        n, k = r.randint(40, 200), r.choice([6, 7, 8, 9])
        return (f"A school has {n} students who are split into teams of exactly {k}. Students who do not fit into "
                f"a full team become reserves. How many reserves are there?", str(n % k))
    if kind == "printer":
        p, h, m = r.randint(10, 40), r.randint(1, 3), r.choice([15, 20, 30, 45])
        return (f"A printer prints {p} pages per minute. How many pages does it print in {h} hours and {m} minutes?",
                str(p * (h * 60 + m)))
    return r.choice([
        "A truck carries some boxes. It delivers 12 boxes to each shop it visits and visits 5 shops. "
        "How many boxes are left in the truck?",
        f"{who} saves some money every week. After a few weeks {who} spends $30 on a game. "
        f"How much money does {who} have now?",
        "A tank is being filled at 8 liters per minute. How many liters will be in the tank when it is full?",
    ]), INFEASIBLE


# ====================================================================== data

DEPTS = ["Sales", "Engineering", "Support", "Finance", "Marketing"]
CITIES = ["Oslo", "Lima", "Hanoi", "Accra", "Perth"]


def _unique_max(d: dict) -> str | None:
    best = max(d.values())
    keys = [k for k, v in d.items() if v == best]
    return keys[0] if len(keys) == 1 else None


def data_task(r: random.Random) -> tuple[str, str, str, dict[str, str]]:
    kind = r.choice(["employees", "weather", "products"])
    if kind == "employees":
        rows = [{"id": i + 1, "name": f"{r.choice(NAMES)} {chr(65 + i % 26)}.", "department": r.choice(DEPTS),
                 "salary": r.randint(30, 120) * 1000 + r.choice([0, 250, 500]), "hire_year": r.randint(2010, 2024)}
                for i in range(r.randint(30, 80))]
        text = _csv(rows)
        dept = r.choice(sorted({x["department"] for x in rows}))
        q = r.choice(["avg", "after", "top_dept", "top_name", "inf_age", "inf_dept"])
        sel = [x for x in rows if x["department"] == dept]
        if q == "avg":
            return (f"The file employees.csv lists the staff of a company. What is the average salary in the "
                    f"{dept} department, rounded down to a whole number?",
                    str(sum(x["salary"] for x in sel) // len(sel)), "exact_int", {"employees.csv": text})
        if q == "after":
            y = r.randint(2013, 2021)
            return (f"Using employees.csv, count the employees who were hired after {y} (not including {y}).",
                    str(sum(x["hire_year"] > y for x in rows)), "exact_int", {"employees.csv": text})
        if q == "top_dept":
            totals = {}
            for x in rows:
                totals[x["department"]] = totals.get(x["department"], 0) + x["salary"]
            best = _unique_max(totals)
            if best:
                return ("Which department in employees.csv has the largest total salary budget? "
                        "Answer with the department name.", best, "exact_str", {"employees.csv": text})
        if q == "top_name":
            best = _unique_max({x["name"]: x["salary"] for x in sel})
            if best and len({x["name"] for x in sel}) == len(sel):
                return (f"In employees.csv, what is the name of the highest-paid employee in {dept}?",
                        best, "exact_str", {"employees.csv": text})
        if q == "inf_age":
            return (f"What is the average age of the employees in the {dept} department (see employees.csv)?",
                    INFEASIBLE, "exact_int", {"employees.csv": text})
        return ("How many employees work in the Legal department according to employees.csv?" if q == "inf_dept"
                else f"How many people work in {dept} according to employees.csv?",
                INFEASIBLE if q == "inf_dept" else str(len(sel)), "exact_int", {"employees.csv": text})

    if kind == "weather":
        rows = []
        for city in r.sample(CITIES, 3):
            for month in (2, 3, 4):
                for day in range(1, 11):
                    rows.append({"date": f"2025-{month:02d}-{day:02d}", "city": city,
                                 "temp_c": r.randint(-5, 30), "rain_mm": r.choice([0, 0, 0, 1, 2, 5, 8, 12, 20])})
        city = r.choice(sorted({x["city"] for x in rows}))
        text = _csv(rows)
        q = r.choice(["rainy", "max_march", "total_rain", "inf_city", "inf_wind"])
        mine = [x for x in rows if x["city"] == city]
        if q == "rainy":
            k = r.choice([2, 5, 8])
            return (f"weather.csv contains daily observations. On how many days was the rainfall in {city} "
                    f"strictly greater than {k} mm?", str(sum(x["rain_mm"] > k for x in mine)), "exact_int",
                    {"weather.csv": text})
        if q == "max_march":
            return (f"What was the highest temperature (in °C) recorded in {city} during March 2025? "
                    f"The data is in weather.csv.",
                    str(max(x["temp_c"] for x in mine if x["date"][5:7] == "03")), "exact_int",
                    {"weather.csv": text})
        if q == "total_rain":
            return (f"Compute the total rainfall in millimetres for {city} over the whole period in weather.csv.",
                    str(sum(x["rain_mm"] for x in mine)), "exact_int", {"weather.csv": text})
        if q == "inf_city":
            return ("What was the highest temperature recorded in Reykjavik in March 2025 according to weather.csv?",
                    INFEASIBLE, "exact_int", {"weather.csv": text})
        return (f"What was the average wind speed in {city} in April 2025 according to weather.csv?",
                INFEASIBLE, "exact_int", {"weather.csv": text})

    cats = ["tools", "toys", "books", "garden"]
    products = [{"sku": f"P{i:03d}", "name": f"{r.choice(['Red', 'Blue', 'Big', 'Mini', 'Smart'])} "
                 f"{r.choice(['Hammer', 'Robot', 'Atlas', 'Shovel', 'Puzzle', 'Lamp'])} {i}",
                 "category": r.choice(cats), "price": r.randint(3, 90), "stock": r.randint(0, 60)}
                for i in range(1, r.randint(15, 40))]
    text = json.dumps(products, indent=1)
    cat = r.choice(sorted({p["category"] for p in products}))
    q = r.choice(["value", "low", "inf"])
    if q == "value":
        return (f"products.json is a list of products. What is the total value of the stock (price multiplied by "
                f"stock, summed) in the '{cat}' category?",
                str(sum(p["price"] * p["stock"] for p in products if p["category"] == cat)), "exact_int",
                {"products.json": text})
    if q == "low":
        k = r.choice([5, 10, 20])
        return (f"How many products in the '{cat}' category of products.json have fewer than {k} items in stock?",
                str(sum(p["category"] == cat and p["stock"] < k for p in products)), "exact_int",
                {"products.json": text})
    p = r.choice(products)
    return (f"Who is the supplier of the product '{p['name']}' in products.json?", INFEASIBLE, "exact_str",
            {"products.json": text})


def _csv(rows: list[dict]) -> str:
    head = list(rows[0])
    return "\n".join([",".join(head)] + [",".join(str(x[h]) for h in head) for x in rows]) + "\n"


# ====================================================================== search

def search_task(r: random.Random) -> tuple[str, str, str, dict[str, str]]:
    return books_task(r) if r.random() < 0.55 else prose_task(r)


def books_task(r: random.Random):
    countries = ["Peru", "Kenya", "Norway", "Vietnam", "Chile", "Ghana"]
    cities = ["Lima", "Nairobi", "Bergen", "Hue", "Valparaiso", "Kumasi"]
    authors = {f"{r.choice(NAMES)} {r.choice(['Ortiz', 'Mwangi', 'Berg', 'Tran', 'Rojas', 'Mensah'])}": {
        "Born": r.randint(1920, 1990), "Country": r.choice(countries)} for _ in range(6)}
    publishers = {p: {"City": r.choice(cities)} for p in ["Quill Press", "Lantern House", "Northwind Books"]}
    words = ["Silent", "River", "Glass", "Winter", "Garden", "Stone", "Echo", "Harbor", "Night", "Paper"]
    books = {}
    while len(books) < 9:
        books[f"The {r.choice(words)} {r.choice(words)}"] = {
            "Author": r.choice(list(authors)), "Published": r.randint(1950, 2020), "Publisher": r.choice(list(publishers))}
    files = {}
    for t, d in books.items():
        files[f"kb/book_{t.lower().replace(' ', '_')}.md"] = (
            f"# {t}\n\nAuthor: {d['Author']}\nPublished: {d['Published']}\nPublisher: {d['Publisher']}\n")
    for a, d in authors.items():
        files[f"kb/author_{a.lower().replace(' ', '_')}.md"] = f"# {a}\n\nBorn: {d['Born']}\nCountry: {d['Country']}\n"
    for p, d in publishers.items():
        files[f"kb/publisher_{p.lower().replace(' ', '_')}.md"] = f"# {p}\n\nCity: {d['City']}\n"
    title = r.choice(list(books))
    b = books[title]
    q = r.choice(["country", "city", "count", "born", "inf"])
    head = "The directory kb/ is a small library catalogue in markdown. "
    if q == "country":
        return head + f"In which country was the author of '{title}' born?", authors[b["Author"]]["Country"], \
            "exact_str", files
    if q == "city":
        return head + f"In which city is the publisher of '{title}' located?", \
            publishers[b["Publisher"]]["City"], "exact_str", files
    if q == "count":
        a = r.choice(list(authors))
        return head + f"How many books in the catalogue were written by {a}?", \
            str(sum(x["Author"] == a for x in books.values())), "exact_int", files
    if q == "born":
        return head + f"What is the birth year of the person who wrote '{title}'?", \
            str(authors[b["Author"]]["Born"]), "exact_int", files
    return head + r.choice([f"What is the ISBN of '{title}'?", f"Who translated '{title}' into English?"]), \
        INFEASIBLE, "exact_str", files


def prose_task(r: random.Random):
    companies = {c: {"city": r.choice(["Osaka", "Lyon", "Porto", "Tartu"]), "year": r.randint(1960, 2015)}
                 for c in ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics"]}
    people = {}
    names = r.sample([f"{f} {l}" for f in NAMES for l in ["Smith", "Kim", "Novak", "Rossi"]], 8)
    for i, n in enumerate(names):
        people[n] = {"born": r.randint(1950, 2000), "emp": r.choice(list(companies)),
                     "mentor": names[(i + r.randint(1, 7)) % 8]}
    files = {}
    for n, d in people.items():
        files[f"kb/{n.lower().replace(' ', '_')}.txt"] = (
            f"{n} was born in {d['born']}. These days {n.split()[0]} works for {d['emp']}. "
            f"{n.split()[0]}'s mentor is {d['mentor']}.\n")
    for c, d in companies.items():
        files[f"kb/{c.lower().replace(' ', '_')}.txt"] = (
            f"{c} was founded in {d['year']}. Its headquarters are in {d['city']}.\n")
    p = r.choice(names)
    d = people[p]
    head = "The folder kb/ contains short text notes about people and companies. "
    q = r.choice(["hq", "mentor_born", "founded", "inf"])
    if q == "hq":
        return head + f"In which city is the headquarters of the company {p} works for?", \
            companies[d["emp"]]["city"], "exact_str", files
    if q == "mentor_born":
        return head + f"In what year was {p}'s mentor born?", str(people[d["mentor"]]["born"]), "exact_int", files
    if q == "founded":
        return head + f"When was {p}'s employer founded? Give the year.", str(companies[d["emp"]]["year"]), \
            "exact_int", files
    return head + f"What is the home address of {p}?", INFEASIBLE, "exact_str", files


# ====================================================================== coding

# (модуль, правильный код, сломанный код, видимые тесты, скрытые тесты)
CODING = [
    ("primes", "def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n"
               "        if n % d == 0:\n            return False\n    return True\n",
     "def is_prime(n):\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n"
     "    return True\n",
     ["self.assertTrue(m.is_prime(7))", "self.assertFalse(m.is_prime(9))", "self.assertFalse(m.is_prime(1))"],
     ["self.assertFalse(m.is_prime(0))", "self.assertFalse(m.is_prime(-3))", "self.assertTrue(m.is_prime(2))",
      "self.assertTrue(m.is_prime(97))", "self.assertFalse(m.is_prime(91))"]),
    ("stats", "def mean(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n",
     "def mean(xs):\n    return sum(xs) / len(xs)\n",
     ["self.assertEqual(m.mean([2, 4]), 3)", "self.assertEqual(m.mean([]), 0)"],
     ["self.assertEqual(m.mean([5]), 5)", "self.assertAlmostEqual(m.mean([1, 2]), 1.5)"]),
    ("search", "def find(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
               "        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n"
               "            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
     "def find(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
     "        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n"
     "            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
     ["self.assertEqual(m.find([1, 3, 5, 7], 7), 3)", "self.assertEqual(m.find([1, 3, 5], 4), -1)"],
     ["self.assertEqual(m.find([2], 2), 0)", "self.assertEqual(m.find([], 1), -1)",
      "self.assertEqual(m.find(list(range(0, 100, 2)), 98), 49)", "self.assertEqual(m.find([1, 3, 5, 7], 1), 0)"]),
    ("textutils", "def word_count(text):\n    return len(text.split())\n",
     "def word_count(text):\n    return len(text.split(' '))\n",
     ["self.assertEqual(m.word_count('a b c'), 3)", "self.assertEqual(m.word_count('a  b'), 2)"],
     ["self.assertEqual(m.word_count(''), 0)", "self.assertEqual(m.word_count('  hello   world  '), 2)",
      "self.assertEqual(m.word_count('one\\ntwo'), 2)"]),
    ("bounds", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
     "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
     ["self.assertEqual(m.clamp(5, 0, 10), 5)", "self.assertEqual(m.clamp(-3, 0, 10), 0)"],
     ["self.assertEqual(m.clamp(15, 0, 10), 10)", "self.assertEqual(m.clamp(0, 0, 0), 0)",
      "self.assertEqual(m.clamp(7, 7, 9), 7)"]),
    ("dedupe", "def unique(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n"
               "            seen.add(x)\n            out.append(x)\n    return out\n",
     "def unique(xs):\n    return list(set(xs))\n",
     ["self.assertEqual(m.unique([3, 1, 3, 2, 1]), [3, 1, 2])"],
     ["self.assertEqual(m.unique([]), [])", "self.assertEqual(m.unique(['b', 'a', 'b']), ['b', 'a'])",
      "self.assertEqual(m.unique([5, 5, 5]), [5])"]),
    ("convert", "def c_to_f(c):\n    return c * 9 / 5 + 32\n",
     "def c_to_f(c):\n    return c * 5 / 9 + 32\n",
     ["self.assertEqual(m.c_to_f(100), 212)", "self.assertEqual(m.c_to_f(0), 32)"],
     ["self.assertEqual(m.c_to_f(-40), -40)", "self.assertAlmostEqual(m.c_to_f(37), 98.6)"]),
    ("median", "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    if n % 2:\n        return s[n // 2]\n"
               "    return (s[n // 2 - 1] + s[n // 2]) / 2\n",
     "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n // 2]\n",
     ["self.assertEqual(m.median([3, 1, 2]), 2)", "self.assertEqual(m.median([4, 1, 3, 2]), 2.5)"],
     ["self.assertEqual(m.median([10, 20]), 15)", "self.assertEqual(m.median([5]), 5)",
      "self.assertEqual(m.median([9, 1, 8, 2, 7, 3]), 5)"]),
    ("fizz", "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n"
             "    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n",
     "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n"
     "    if n % 15 == 0:\n        return 'FizzBuzz'\n    return str(n)\n",
     ["self.assertEqual(m.fizzbuzz(15), 'FizzBuzz')", "self.assertEqual(m.fizzbuzz(9), 'Fizz')"],
     ["self.assertEqual(m.fizzbuzz(30), 'FizzBuzz')", "self.assertEqual(m.fizzbuzz(10), 'Buzz')",
      "self.assertEqual(m.fizzbuzz(7), '7')", "self.assertEqual(m.fizzbuzz(45), 'FizzBuzz')"]),
    ("money", "def split_bill(total, people):\n    share = total // people\n    rest = total - share * people\n"
              "    return [share + 1 if i < rest else share for i in range(people)]\n",
     "def split_bill(total, people):\n    share = total // people\n    return [share for i in range(people)]\n",
     ["self.assertEqual(m.split_bill(10, 3), [4, 3, 3])", "self.assertEqual(sum(m.split_bill(7, 2)), 7)"],
     ["self.assertEqual(m.split_bill(9, 3), [3, 3, 3])", "self.assertEqual(m.split_bill(5, 4), [2, 1, 1, 1])",
      "self.assertEqual(sum(m.split_bill(101, 7)), 101)"]),
]


def coding_task(r: random.Random) -> tuple[str, dict[str, str], dict[str, str]]:
    mod, good, bad, visible, hidden = r.choice(CODING)

    def test_file(checks: list[str], cls: str) -> str:
        body = "".join(f"    def test_{i}(self):\n        {c}\n\n" for i, c in enumerate(checks))
        return f"import unittest\n\nimport {mod} as m\n\n\nclass {cls}(unittest.TestCase):\n{body}\n" \
               f"if __name__ == '__main__':\n    unittest.main()\n"

    q = r.choice([f"The module {mod}.py has a bug. The tests in test_{mod}.py describe the expected behaviour. "
                  f"Fix {mod}.py so that it is correct in general (not only for the given tests); "
                  f"do not modify the tests.",
                  f"Some tests in test_{mod}.py fail. Find and fix the bug in {mod}.py. Your fix will also be "
                  f"checked with additional tests, so make the function correct for all valid inputs.",
                  f"Run `python3 -m unittest` here. Repair {mod}.py until every test passes; keep the test file "
                  f"unchanged."])
    workspace = {f"{mod}.py": bad, f"test_{mod}.py": test_file(visible, "Visible")}
    hidden_files = {f"test_hidden_{mod}.py": test_file(visible + hidden, "Hidden")}
    return q, workspace, hidden_files


# ====================================================================== сборка

def write_task(out: Path, task_id: str, task_type: str, question: str, expected: str, check: str, reward: int,
               files: dict[str, str], hidden: dict[str, str] | None = None) -> None:
    ws = out / task_id / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (ws / name).parent.mkdir(parents=True, exist_ok=True)
        (ws / name).write_text(text, encoding="utf-8")
    for name, text in (hidden or {}).items():  # скрытые тесты — вне workspace, агент их не видит
        (out / task_id / "hidden").mkdir(exist_ok=True)
        (out / task_id / "hidden" / name).write_text(text, encoding="utf-8")
    tail = CODING_TAIL if check == "unittest" else TAIL
    (out / task_id / "task.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "instruction": question + tail,
        "reward": reward, "expected": expected, "check": check}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "data" / "varied"))
    args = ap.parse_args()

    r = random.Random(args.seed)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    stats: dict[str, int] = {}
    for i in range(args.n):
        kind = ["math", "data_analysis", "search", "coding"][i % 4]
        task_id, reward = f"{kind}_v{i:03d}", r.randint(1, 5)
        if kind == "math":
            q, a = math_task(r)
            write_task(out, task_id, kind, q, a, "exact_int", reward, {})
        elif kind == "data_analysis":
            q, a, check, files = data_task(r)
            write_task(out, task_id, kind, q, a, check, reward, files)
        elif kind == "search":
            q, a, check, files = search_task(r)
            write_task(out, task_id, kind, q, a, check, reward, files)
        else:
            q, files, hidden = coding_task(r)
            write_task(out, task_id, kind, q, "tests_pass", "unittest", reward, files, hidden)
            a = "tests_pass"
        key = kind + (" (нерешаемая)" if a == INFEASIBLE else "")
        stats[key] = stats.get(key, 0) + 1
    print(f"создано {args.n} задач в {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
