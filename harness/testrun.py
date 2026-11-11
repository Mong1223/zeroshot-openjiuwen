"""Запуск unittest в папке задачи — в отдельном процессе, с таймаутом.

Отдельный процесс нужен, чтобы зависший или упавший код задачи не уронил харнесс.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# "test_total (test_mylib.T.test_total) ... FAIL"
_LINE = re.compile(r"^(test\w*) \(([\w.]+)\)(?:\n.*?)?\s*\.\.\.\s*(ok|FAIL|ERROR|skipped|expected failure|unexpected success)",
                   re.M)


@dataclass
class TestReport:
    passed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    broken: bool = False      # тесты не запустились вовсе (ошибка импорта, синтаксис, таймаут)
    timed_out: bool = False
    seconds: float = 0.0
    output: str = ""

    @property
    def ok(self) -> bool:
        return not self.broken and not self.failed and bool(self.passed)

    @property
    def score(self) -> int:
        """Чем меньше, тем лучше: число упавших тестов; «сломано» — хуже всего."""
        return 10_000 if self.broken else len(self.failed)


def test_modules(workspace: Path) -> list[str]:
    return sorted(p.stem for p in workspace.glob("test*.py"))


def run_tests(workspace: Path, timeout: float = 20) -> TestReport:
    modules = test_modules(workspace)
    if not modules:
        return TestReport(broken=True, output="no test files")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    started = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *modules], cwd=workspace,
                              capture_output=True, text=True, timeout=max(timeout, 0.5), env=env)
    except subprocess.TimeoutExpired:
        return TestReport(broken=True, timed_out=True, seconds=time.monotonic() - started, output="timeout")
    out = proc.stderr + proc.stdout
    report = TestReport(seconds=time.monotonic() - started, output=out[-4000:])
    for name, full, status in _LINE.findall(out):
        if full.startswith("unittest.loader"):  # модуль не импортировался
            report.broken = True
            continue
        (report.passed if status in ("ok", "skipped", "expected failure") else report.failed).add(name)
    if not report.passed and not report.failed:
        report.broken = True
    return report
