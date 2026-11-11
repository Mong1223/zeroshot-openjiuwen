"""Решатель coding (уровень 0): перебор подмен на разных видах багов."""
import json
import textwrap

import pytest

from harness.solvers import solve_level0
from harness.tasks import read_task
from harness.testrun import run_tests
from harness.types import Status

TESTS = '''
import unittest
import mylib as m

class T(unittest.TestCase):
{body}

if __name__ == '__main__':
    unittest.main()
'''


def make(tmp_path, lib: str, tests: dict[str, str]):
    folder = tmp_path / "coding_x"
    ws = folder / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(textwrap.dedent(lib), encoding="utf-8")
    body = "\n".join(f"    def {name}(self):\n        {assertion}" for name, assertion in tests.items())
    (ws / "test_mylib.py").write_text(TESTS.format(body=body), encoding="utf-8")
    (folder / "task.json").write_text(json.dumps({
        "task_id": "coding_x", "task_type": "coding", "instruction": "Fix mylib.py", "reward": 2,
        "check": "unittest"}), encoding="utf-8")
    return read_task(folder)


@pytest.mark.parametrize("lib, tests", [
    pytest.param("""
        def first_n(xs, n):
            return xs[:n - 1]
        """, {"test_first_n": "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"}, id="off-by-one slice"),
    pytest.param("""
        def total(xs):
            s = 0
            for x in xs:
                s += x
            return x
        """, {"test_total": "self.assertEqual(m.total([1, 2, 3]), 6)",
              "test_total2": "self.assertEqual(m.total([4, 4]), 8)"}, id="wrong variable"),
    pytest.param("""
        def count_upto(n):
            return len(list(range(1, n)))

        def is_adult(age):
            return age > 18 and True
        """, {"test_count": "self.assertEqual(m.count_upto(5), 5)",
              "test_adult": "self.assertTrue(m.is_adult(18))",
              "test_child": "self.assertFalse(m.is_adult(17))"}, id="range bound + comparison"),
    pytest.param("""
        def shout(s):
            return s.lower() + '!'

        def largest(xs):
            return min(xs)

        def desc(xs):
            return sorted(xs)
        """, {"test_shout": "self.assertEqual(m.shout('hi'), 'HI!')",
              "test_largest": "self.assertEqual(m.largest([3, 9, 2]), 9)",
              "test_desc": "self.assertEqual(m.desc([1, 3, 2]), [3, 2, 1])"}, id="three bugs: method, func, reverse"),
])
def test_fixes_bugs(tmp_path, lib, tests):
    task = make(tmp_path, lib, tests)
    test_file = (task.workspace / "test_mylib.py").read_text()
    result = solve_level0(task)
    assert result.status is Status.SOLVED, result.note
    assert result.answer == "DONE"
    assert run_tests(task.workspace).ok
    assert (task.workspace / "test_mylib.py").read_text() == test_file


def test_already_passing(tmp_path):
    task = make(tmp_path, "def f():\n    return 1\n", {"test_f": "self.assertEqual(m.f(), 1)"})
    result = solve_level0(task)
    assert result.status is Status.SOLVED and result.note == "tests already pass"


def test_too_hard_restores_original(tmp_path):
    lib = """
        def fib(n):
            return n * 2
        """
    task = make(tmp_path, lib, {"test_fib": "self.assertEqual([m.fib(i) for i in range(7)], [0, 1, 1, 2, 3, 5, 8])"})
    original = (task.workspace / "mylib.py").read_text()
    result = solve_level0(task)
    assert result.status is Status.UNSURE
    assert (task.workspace / "mylib.py").read_text() == original  # откатили


def test_broken_test_file_is_infeasible(tmp_path):
    task = make(tmp_path, "def f():\n    return 1\n", {"test_f": "self.assertEqual(m.f(), 1)"})
    (task.workspace / "test_mylib.py").write_text("import unittest\nclass T(:\n", encoding="utf-8")
    assert solve_level0(task).status is Status.INFEASIBLE


def test_infinite_loop_does_not_hang(tmp_path):
    lib = """
        def f():
            while True:
                pass
        """
    task = make(tmp_path, lib, {"test_f": "self.assertEqual(m.f(), 1)"})
    from harness.solvers import coding_solver
    result = coding_solver.solve(task, max_seconds=5, max_runs=3)
    assert result.status is Status.UNSURE
