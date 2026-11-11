"""Инструменты агента openJiuwen (уровень 2) и тихая настройка openJiuwen — без вызовов модели."""
import os
import subprocess
import sys
from pathlib import Path

from harness.agent import Workspace

ROOT = Path(__file__).resolve().parent.parent


def make_ws(tmp_path, coding=True):
    (tmp_path / "lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_lib.py").write_text(
        "import unittest\nfrom lib import add\n\n\nclass T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n", encoding="utf-8")
    (tmp_path / "kb").mkdir(exist_ok=True)
    (tmp_path / "kb" / "note.txt").write_text("Nina Kim works for Cobalt Works.\n", encoding="utf-8")
    return Workspace(tmp_path, coding=coding)


def test_paths_cannot_leave_workspace(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.read_file("../secret.txt").startswith("ERROR")
    assert ws.write_file("../evil.py", "x = 1").startswith("ERROR")


def test_read_list_search(tmp_path):
    ws = make_ws(tmp_path)
    assert "kb/note.txt" in ws.list_files()
    assert "Cobalt Works" in ws.read_file("kb/note.txt")
    assert "kb/note.txt:1:" in ws.search_text("cobalt")


def test_tests_are_protected_and_fix_works(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.write_file("test_lib.py", "").startswith("ERROR")
    assert "FAILED" in ws.run_tests()
    ws.write_file("lib.py", "def add(a, b):\n    return a + b\n")
    assert ws.run_tests().startswith("ALL TESTS PASS")


def test_run_python_uses_workspace_and_timeout(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.run_python("print(open('kb/note.txt').read().split()[0])") == "Nina"
    assert "Traceback" in ws.run_python("1/0")


def test_coding_tools_only_for_coding(tmp_path):
    names = [s[0] for s in make_ws(tmp_path, coding=False).specs()]
    assert "write_file" not in names and "run_python" in names
    names = [s[0] for s in make_ws(tmp_path, coding=True).specs()]
    assert {"write_file", "run_tests"} <= set(names)


def test_openjiuwen_writes_no_log_files(tmp_path):
    """Контейнер жюри read-only: openJiuwen не должен создавать ./logs и печатать INFO."""
    code = ("from harness.jiuwen_setup import configure_openjiuwen\nconfigure_openjiuwen()\n"
            "from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig\n"
            "from openjiuwen.core.common.logging import logger\nlogger.info('NOISE')\nprint('ok')\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "NOISE" not in proc.stdout + proc.stderr
    assert not (tmp_path / "logs").exists()
