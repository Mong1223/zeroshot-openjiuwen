"""Создаёт data/hard — задачи, которые бесплатные решатели НЕ берут (незнакомые формулировки,
сложный баг, нерешаемые). Нужны, чтобы проверить платную часть (ИИ) на репетиции.

    python scripts\\make_hard.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "data" / "examples"
OUT = ROOT / "data" / "hard"
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING_TAIL = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
               "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def task(task_id, task_type, question, expected, check, reward, files=None, copy_from=None):
    folder = OUT / task_id
    ws = folder / "workspace"
    if copy_from:
        shutil.copytree(EX / copy_from / "workspace", ws)
    ws.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {}).items():
        (ws / name).parent.mkdir(parents=True, exist_ok=True)
        (ws / name).write_text(text, encoding="utf-8")
    tail = CODING_TAIL if check == "unittest" else TAIL
    (folder / "task.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "instruction": question + tail, "reward": reward,
        "expected": expected, "check": check}, indent=2), encoding="utf-8")


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    task("math_h1", "math",
         "A depot holds 940 crates at dawn. Every hour for 6 hours, 35 crates are loaded onto trucks. "
         "Afterwards a third of what remains is sent to a partner depot (round down the amount sent). "
         "How many crates are left in the depot?", "487", "exact_int", 4)
    task("data_h1", "data_analysis",
         "The file sales.csv has columns order_id,region,product,month,units,price (300 rows). "
         "Which region sold the most units of product 'gizmo' over the whole year? Use bash with python3.",
         None, "exact_str", 3, copy_from="data_analysis_0006")
    task("search_h1", "search",
         "The directory kb/ contains a small knowledge base of markdown files. How many people in the "
         "knowledge base work for Dune Robotics? Use the file tools or bash (ls, grep, cat) to find it.",
         None, "exact_int", 3, copy_from="search_0001")
    task("search_h2_infeasible", "search",
         "The directory kb/ contains a small knowledge base of markdown files. What is the phone number of "
         "Irina Smith? Use the file tools or bash (ls, grep, cat) to find it.",
         "INFEASIBLE", "exact_str", 5, copy_from="search_0001")
    task("coding_h1", "coding",
         "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
         "(do not edit the tests) until all tests pass.", "tests_pass", "unittest", 4, files={
             "mylib.py": "def fib(n):\n    return n * 2\n\n\ndef is_palindrome(s):\n    return s == s[1:]\n",
             "test_mylib.py": ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                               "    def test_fib(self):\n"
                               "        self.assertEqual([m.fib(i) for i in range(8)], [0, 1, 1, 2, 3, 5, 8, 13])\n\n"
                               "    def test_palindrome(self):\n"
                               "        self.assertTrue(m.is_palindrome('abba'))\n"
                               "        self.assertFalse(m.is_palindrome('abc'))\n\n\n"
                               "if __name__ == '__main__':\n    unittest.main()\n")})

    # --- задачи для уровня 2 (агент openJiuwen): уровень 1 на них не рассчитан
    task("coding_h2_multifile", "coding",
         "Run `python3 -m unittest` in the current directory; some tests fail. The bug may be in any module "
         "except the tests. Fix it (do not edit the tests).", "tests_pass", "unittest", 5, files={
             "shop.py": "from pricing import discount\n\n\ndef total(prices, code=None):\n"
                        "    s = sum(prices)\n    return round(s - discount(s, code), 2)\n",
             "pricing.py": "CODES = {'SAVE10': 10, 'HALF': 50}\n\n\ndef discount(amount, code):\n"
                           "    if code not in CODES:\n        return 0\n    return amount * CODES[code] / 10\n",
             "test_shop.py": ("import unittest\n\nfrom shop import total\n\n\nclass T(unittest.TestCase):\n"
                              "    def test_plain(self):\n        self.assertEqual(total([10, 20]), 30)\n\n"
                              "    def test_save10(self):\n        self.assertEqual(total([50, 50], 'SAVE10'), 90)\n\n"
                              "    def test_half(self):\n        self.assertEqual(total([8], 'HALF'), 4)\n\n\n"
                              "if __name__ == '__main__':\n    unittest.main()\n")})
    task("data_h2_join", "data_analysis",
         "There are two files: customers.csv (customer_id,name,city) and orders.csv (order_id,customer_id,amount). "
         "What is the total amount of all orders placed by customers who live in Lima?", "",
         "exact_int", 4, files={
             "customers.csv": "customer_id,name,city\n1,Ana,Lima\n2,Bo,Oslo\n3,Cy,Lima\n4,Di,Quito\n",
             "orders.csv": "order_id,customer_id,amount\n1,1,120\n2,2,75\n3,3,40\n4,1,15\n5,4,300\n6,3,5\n"})
    task("logs_h1", "log_analysis",
         "The file server.log contains one event per line. How many distinct users had at least one ERROR "
         "event?", "", "exact_int", 3, files={
             "server.log": "\n".join([
                 "2025-03-01 10:00:01 INFO user=ana login", "2025-03-01 10:02:11 ERROR user=bo timeout",
                 "2025-03-01 10:03:00 WARN user=ana slow", "2025-03-01 10:04:40 ERROR user=cy db",
                 "2025-03-01 10:05:12 ERROR user=bo retry", "2025-03-01 10:06:00 INFO user=di logout",
                 "2025-03-01 10:07:30 ERROR user=ana disk"]) + "\n"})
    _set_expected("data_h2_join", "180")
    _set_expected("logs_h1", "3")

    # правильные ответы для задач, где их проще посчитать, чем написать руками
    import csv
    rows = list(csv.DictReader((OUT / "data_h1/workspace/sales.csv").open(encoding="utf-8")))
    totals = {}
    for r in rows:
        if r["product"] == "gizmo":
            totals[r["region"]] = totals.get(r["region"], 0) + int(r["units"])
    _set_expected("data_h1", max(totals, key=totals.get))
    kb = (OUT / "search_h1/workspace/kb").glob("person_*.md")
    _set_expected("search_h1", str(sum("Employer: Dune Robotics" in p.read_text() for p in kb)))
    print("created", len(list(OUT.iterdir())), "tasks in", OUT)


def _set_expected(task_id, value):
    path = OUT / task_id / "task.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["expected"] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
