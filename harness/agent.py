"""Уровень 2: ИИ-агент openJiuwen (ReActAgent) — «подумал → сделал → посмотрел», с инструментами.

Зовётся только тогда, когда уровень 1 (короткий ответ / программа / починка кода) не справился:
агент дороже (каждый шаг заново отправляет модели всю историю), зато сам исследует файлы
незнакомого формата, пробует, видит ошибки и исправляется.

Инструменты агента работают ТОЛЬКО внутри папки задачи (workspace):
    list_files, read_file, search_text, run_python      — для всех задач
    write_file, run_tests                                — для coding (тесты менять нельзя)

Учёт токенов: каждый вызов модели внутри агента проходит через наш счётчик (Budget):
проверка «хватит ли» до вызова, списание после, запись в llm_calls.jsonl. У каждой задачи
свой потолок (AGENT_CAP): дошли до него — агент останавливается.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from harness.llm import LLM, BudgetExhausted, LLMError, _EXHAUSTED_HINTS
from harness.tasks import clean_answer, question_text
from harness.testrun import run_tests
from harness.types import Result, Status, Task

log = logging.getLogger("harness.agent")

AGENT_CAP = 2500           # потолок «стоимости» на одну задачу (выход + вход/10)
CODING_CAP = 4000          # coding: агент пишет файлы целиком — выходных токенов больше
MAX_ITERATIONS = 8         # шагов «подумал → инструмент»
MAX_TOKENS = 700           # ответ модели за один шаг (для coding — больше: пишет файл целиком)
CODING_MAX_TOKENS = 2000
TOOL_OUTPUT = 3000         # сколько символов результата инструмента показываем модели
RUN_TIMEOUT = 20

SYSTEM = """You solve ONE task inside a working directory, using the tools.
- Look at the data with the tools; never invent file contents.
- Do arithmetic and data processing with run_python, never in your head.
- Be frugal: few tool calls, short messages. Stop as soon as you know the answer.
- Final reply: ONLY the answer value on the last line (a number or a short string, no extra words),
  or INFEASIBLE if the task cannot be solved from the given data (missing entity, field, column or file)."""

SYSTEM_CODING = """You fix bugs in Python code inside a working directory, using the tools.
- Read the code and the tests, change ONLY the non-test files with write_file (complete file content).
- Check with run_tests. Hidden tests will also be run: make the code correct for all valid inputs,
  never special-case test values. Never modify test files.
- Be frugal: few tool calls. Final reply: DONE when all tests pass, otherwise FAILED."""


class AgentCapReached(Exception):
    pass


# ---------------------------------------------------------------- инструменты

class Workspace:
    """Инструменты агента, привязанные к папке одной задачи."""

    def __init__(self, root: Path, coding: bool):
        self.root = root.resolve()
        self.coding = coding

    def _path(self, path: str) -> Path:
        p = (self.root / str(path).lstrip("/\\")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("path is outside the working directory")
        return p

    @staticmethod
    def _clip(text: str) -> str:
        return text if len(text) <= TOOL_OUTPUT else text[:TOOL_OUTPUT] + f"\n... [{len(text) - TOOL_OUTPUT} more chars]"

    def list_files(self) -> str:
        files = [p.relative_to(self.root).as_posix() for p in sorted(self.root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts]
        head = files[:200]
        return "\n".join(head) + (f"\n... and {len(files) - 200} more" if len(files) > 200 else "") or "(empty)"

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 120) -> str:
        try:
            lines = self._path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, ValueError) as e:
            return f"ERROR: {e}"
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + max(1, min(int(max_lines), 400))]
        note = f"[{path}: lines {start}-{start + len(chunk) - 1} of {len(lines)}]\n"
        return self._clip(note + "\n".join(chunk))

    def search_text(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        hits = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(self.root).as_posix()}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 60:
                        return "\n".join(hits) + "\n... (more matches)"
        return "\n".join(hits) or "no matches"

    def run_python(self, code: str) -> str:
        with tempfile.TemporaryDirectory(prefix="zs-agent-") as tmp:
            script = Path(tmp) / "step.py"
            script.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run([sys.executable, str(script)], cwd=self.root, capture_output=True,
                                      text=True, timeout=RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                return f"ERROR: timeout after {RUN_TIMEOUT}s"
        out = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        return self._clip(out.strip() or "(no output)")

    def write_file(self, path: str, content: str) -> str:
        try:
            p = self._path(path)
        except ValueError as e:
            return f"ERROR: {e}"
        if p.name.startswith("test") or "tests" in p.relative_to(self.root).parts:
            return "ERROR: test files must not be modified"
        if p.suffix != ".py":
            return "ERROR: only .py files can be written"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"written {len(content)} chars to {path}"

    def run_tests(self) -> str:
        report = run_tests(self.root)
        if report.ok:
            status = "ALL TESTS PASS"
        elif report.broken:
            status = "TESTS COULD NOT RUN (import/syntax error or timeout)"
        else:
            status = f"FAILED: {sorted(report.failed)}"
        return self._clip(f"{status}\n{report.output[-2000:]}")

    def specs(self) -> list[tuple[str, str, dict, callable]]:
        """(имя, описание, JSON-схема аргументов, функция)."""
        s = [
            ("list_files", "List all files in the working directory (relative paths).",
             {"type": "object", "properties": {}}, self.list_files),
            ("read_file", "Read a text file (relative path). Returns numbered range of lines.",
             {"type": "object", "properties": {
                 "path": {"type": "string"}, "start_line": {"type": "integer", "default": 1},
                 "max_lines": {"type": "integer", "default": 120}}, "required": ["path"]}, self.read_file),
            ("search_text", "Case-insensitive regex search in all files; returns file:line: text.",
             {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
             self.search_text),
            ("run_python", "Run a Python 3 script in the working directory; returns stdout/stderr. "
                           "Use it for all calculations and data processing.",
             {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, self.run_python),
        ]
        if self.coding:
            s += [
                ("write_file", "Overwrite a non-test .py file with the complete new content.",
                 {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                  "required": ["path", "content"]}, self.write_file),
                ("run_tests", "Run the unit tests in the working directory and report the result.",
                 {"type": "object", "properties": {}}, self.run_tests),
            ]
        return s


# ---------------------------------------------------------------- модель со счётчиком

class _Meter:
    """Оборачивает model.invoke: бюджет, потолок задачи, 429, журнал."""

    def __init__(self, llm: LLM, task_id: str, model_name: str, max_tokens: int, cap: float):
        self.llm, self.task_id, self.model_name = llm, task_id, model_name
        self.max_tokens, self.cap = max_tokens, cap
        self.spent = 0.0
        self.calls = 0

    def wrap(self, model):
        original = model.invoke

        async def invoke(messages, *args, **kwargs):
            # вход = история + описания инструментов (они тоже уходят модели)
            prompt_chars = len(json.dumps([messages, kwargs.get("tools")], ensure_ascii=False, default=str))
            est_input = prompt_chars // 3 + 1
            worst = self.max_tokens + est_input / 10
            typical = 250 + est_input / 10  # обычный шаг: короткий ответ + вход/10
            if self.spent + typical > self.cap:
                raise AgentCapReached(f"task cap {self.cap:.0f} reached (spent {self.spent:.0f})")
            await self.llm._hold(worst)
            try:
                for attempt in range(4):
                    async with self.llm._sem:
                        await self.llm.rate.acquire()
                        started = time.monotonic()
                        try:
                            reply = await original(messages, *args, **kwargs)
                            break
                        except Exception as e:  # noqa: BLE001 — ошибки провайдера приходят обёрнутыми
                            text = str(e)
                            if "429" not in text:  # запрос ушёл — жюри списывает его вход
                                self.spent += self.llm.budget.settle(0, est_input, 0, failed=True)
                            if any(h in text.lower() for h in _EXHAUSTED_HINTS):
                                self.llm.budget.exhausted = True
                                raise BudgetExhausted(text[:200]) from e
                            if attempt == 3:
                                raise LLMError(text[:300]) from e
                    wait = (20 * (attempt + 1) if "429" in text else 2 + attempt * 3) + random.uniform(0, 2)
                    await asyncio.sleep(wait)
            except BaseException:
                self.llm.budget.release(worst)
                raise
            usage = getattr(reply, "usage_metadata", None)
            inp = getattr(usage, "input_tokens", 0) or est_input
            out = getattr(usage, "output_tokens", 0) or len(str(getattr(reply, "content", ""))) // 3
            cost = self.llm.budget.settle(worst, inp, out)
            self.spent += cost
            self.calls += 1
            self.llm._log(task_id=self.task_id, purpose=f"agent#{self.calls}", model=self.model_name, input=inp,
                          output=out, cost=cost, seconds=round(time.monotonic() - started, 2),
                          max_tokens=self.max_tokens, spent=round(self.llm.budget.spent, 1))
            return reply

        model.invoke = invoke
        return model


# ---------------------------------------------------------------- запуск агента

async def agent_solve(task: Task, llm: LLM, *, cap: float | None = None) -> Result:
    from harness.jiuwen_setup import configure_openjiuwen
    configure_openjiuwen()
    from openjiuwen.core.foundation.llm import init_model
    from openjiuwen.core.foundation.tool import ToolCard
    from openjiuwen.core.foundation.tool.function.function import LocalFunction
    from openjiuwen.core.runner import Runner
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

    coding = task.check == "unittest" or task.task_type == "coding"
    ws_dir = task.workspace if task.workspace.is_dir() else Path(tempfile.mkdtemp(prefix="zs-empty-"))
    ws = Workspace(ws_dir, coding)
    snapshot = {p: p.read_bytes() for p in ws_dir.rglob("*.py")} if coding else {}
    test_hash = _hash([p for p in snapshot if p.name.startswith("test")])

    max_tokens = CODING_MAX_TOKENS if coding else MAX_TOKENS
    cap = cap or (CODING_CAP if coding else AGENT_CAP)
    model_name = llm.settings.fast_model
    meter = _Meter(llm, task.task_id, model_name, max_tokens, cap)
    model = meter.wrap(init_model("OpenAI", model_name, llm.settings.llm_api_key, llm.settings.llm_base_url,
                                  max_tokens=max_tokens, timeout=180, max_retries=0))

    uid = f"{re.sub(r'[^A-Za-z0-9_]', '_', task.task_id)}_{uuid.uuid4().hex[:8]}"
    config = ReActAgentConfig(model_name=model_name, max_iterations=MAX_ITERATIONS, on_max_iterations="final_answer",
                              prompt_template=[{"role": "system", "content": SYSTEM_CODING if coding else SYSTEM}])
    agent = ReActAgent(card=AgentCard(id=f"zs_{uid}", name="zeroshot_solver",
                                      description="Solves one benchmark task")).configure(config)
    agent.set_llm(model)
    tool_ids = []
    for name, description, schema, func in ws.specs():
        card = ToolCard(id=f"{uid}_{name}", name=name, description=description, input_params=schema)
        agent.ability_manager.add(card)
        Runner.resource_mgr.add_tool(LocalFunction(card=card, func=_safe(func)))
        tool_ids.append(card.id)

    query = question_text(task.instruction)
    note = ""
    output = ""
    try:
        response = await agent.invoke({"query": query, "conversation_id": uid})
        output = str(response.get("output", "")) if isinstance(response, dict) else str(response)
        note = f"agent {meter.calls} calls, {response.get('termination_reason', '') if isinstance(response, dict) else ''}"
    except (BudgetExhausted,):
        raise
    except AgentCapReached as e:
        note = f"agent stopped: {e}"
    except Exception as e:  # noqa: BLE001 — сбой агента не должен ронять остальные задачи
        cause = e.__cause__ or e.__context__
        if isinstance(cause, BudgetExhausted) or isinstance(e, BudgetExhausted):
            raise BudgetExhausted(str(e)[:200]) from e
        note = f"agent error {type(e).__name__}: {str(e)[:120]}"
    finally:
        try:
            Runner.resource_mgr.remove_tool(tool_ids)
        except Exception:  # noqa: BLE001
            pass

    if coding:
        report = run_tests(ws_dir)
        tests_untouched = _hash([p for p in snapshot if p.name.startswith("test")]) == test_hash
        if report.ok and tests_untouched:
            return Result(Status.SOLVED, answer="DONE", confidence=0.85, level=2, tokens=meter.spent,
                          note=f"{note}; tests pass")
        for p, data in snapshot.items():  # не получилось — возвращаем код как был
            p.write_bytes(data)
        return Result(Status.FAILED, level=2, tokens=meter.spent, note=f"{note}; tests still fail")

    answer = clean_answer(output)
    if answer.upper().strip(".") == "INFEASIBLE":
        return Result(Status.INFEASIBLE, confidence=0.6, level=2, tokens=meter.spent, note=f"{note}; INFEASIBLE")
    if not answer or len(answer) > 100 or answer.upper().startswith(("FAILED", "ERROR")):
        return Result(Status.FAILED, level=2, tokens=meter.spent, note=f"{note}; bad answer {answer[:40]!r}")
    return Result(Status.SOLVED, answer=answer, confidence=0.7, level=2, tokens=meter.spent, note=note)


def _safe(func):
    """Инструмент никогда не бросает исключение в агента — возвращает текст ошибки."""
    def wrapper(**kwargs):
        try:
            return func(**kwargs)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
    return wrapper


def _hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(p.read_bytes())
        except OSError:
            pass
    return h.hexdigest()
