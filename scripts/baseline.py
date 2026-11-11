"""Замер «в лоб»: каждая задача отдаётся модели ОДНИМ вызовом, все файлы — прямо в запросе.

Это не наше решение, а точка отсчёта: сколько стоит и насколько точно решать
без правил и без агента. Задачи берутся из копии — оригиналы не меняются.

    python scripts\\baseline.py                                   # все 4 модели, data/examples
    python scripts\\baseline.py --models qwen3.8-27b-noreason
    python scripts\\baseline.py --tasks путь\\к\\задачам --repeat 3

Результат: таблица в консоли + runs/baseline_<время>.json (папка runs/ не идёт в git).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from harness.tasks import question_text, read_tasks  # noqa: E402
from harness.types import Task  # noqa: E402

MODELS = ["qwen3.8-27b-noreason", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b", "qwen3.6-35b-a3b"]
MAX_TOKENS = 8192
# Тариф «Мероприятие» на деле пускает 3 одновременных запроса на аккаунт (не 16, как в памятке);
# за превышение — 429 и пауза 60 с. Держимся ниже потолка.
CONCURRENCY = 2
RETRY_WAIT = 65
RETRIES = 3

ANSWER_RULES = ("You cannot run tools: all files you need are included below. "
                "Reply with ONLY the final answer on the last line, nothing else.")
CODING_RULES = ("You cannot run tools: all files are included below. Fix the bugs in mylib.py so that "
                "all tests pass. Do not change the tests. Reply with the COMPLETE corrected mylib.py "
                "in a single ```python code block and nothing else.")


def build_prompt(task: Task) -> str:
    files = []
    for name in task.files:
        text = (task.workspace / name).read_text(encoding="utf-8", errors="replace")
        files.append(f"=== {name} ===\n{text.rstrip()}")
    rules = CODING_RULES if task.check == "unittest" else ANSWER_RULES
    return f"Task: {question_text(task.instruction)}\n\n{rules}\n\n" + "\n\n".join(files)


def extract_answer(content: str) -> str:
    lines = [ln.strip().strip("`*").strip() for ln in content.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_code(content: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    return blocks[-1] if blocks else None


def run_tests(workspace: Path) -> bool:
    proc = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=workspace,
                          capture_output=True, text=True, timeout=60)
    return proc.returncode == 0


def is_correct(task: Task, answer: str | None, scratch: Path) -> bool:
    if answer is None:
        return False
    if task.check == "unittest":
        work = scratch / task.task_id
        shutil.copytree(task.workspace, work)
        (work / "mylib.py").write_text(answer, encoding="utf-8")
        return run_tests(work)
    if task.check == "exact_int":
        try:
            return int(float(answer.replace(",", ""))) == int(task.expected)
        except ValueError:
            return False
    return answer.strip() == (task.expected or "").strip()


async def solve(task: Task, model_name: str, env: dict, sem: asyncio.Semaphore, scratch: Path) -> dict:
    from openjiuwen.core.foundation.llm import init_model

    model = init_model("OpenAI", model_name, env["key"], env["base"], max_tokens=MAX_TOKENS, timeout=300)
    row = {"task": task.task_id, "type": task.task_type, "model": model_name}
    async with sem:
        for attempt in range(RETRIES + 1):
            started = time.perf_counter()
            try:
                reply = await model.invoke([{"role": "user", "content": build_prompt(task)}])
                break
            except Exception as e:
                if "429" in str(e) and attempt < RETRIES:
                    await asyncio.sleep(RETRY_WAIT)
                    continue
                row.update(ok=False, error=f"{type(e).__name__}: {e}"[:200],
                           seconds=round(time.perf_counter() - started, 1))
                return row
        row["seconds"] = round(time.perf_counter() - started, 1)
        row["retries"] = attempt

    usage = reply.usage_metadata
    inp, out = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
    content = reply.content or ""
    answer = extract_code(content) if task.check == "unittest" else extract_answer(content)
    row.update(
        input=inp, output=out, cost=round(out + inp / 10, 1),
        reasoning_chars=len(reply.reasoning_content or ""),
        answer=(answer or "")[:80] if task.check != "unittest" else ("<code>" if answer else None),
        ok=is_correct(task, answer, scratch / model_name),
    )
    return row


def print_table(rows: list[dict], models: list[str], tasks: list[Task]) -> None:
    short = {m: m.replace("qwen", "").replace("-noreason", "-nr") for m in models}
    print(f"\n{'задача':<20}" + "".join(f"{short[m]:>22}" for m in models))
    for t in tasks:
        cells = []
        for m in models:
            rs = [r for r in rows if r["task"] == t.task_id and r["model"] == m]
            if not rs:
                cells.append("—")
                continue
            ok = sum(r["ok"] for r in rs)
            cost = sum(r.get("cost", 0) for r in rs) / len(rs)
            mark = "✓" if ok == len(rs) else ("✗" if ok == 0 else f"{ok}/{len(rs)}")
            cells.append(f"{mark} {cost:7.0f} ток.")
        print(f"{t.task_id:<20}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n{'итого':<20}" + "".join(
        f"{sum(r['ok'] for r in rows if r['model'] == m):>4}/{sum(1 for r in rows if r['model'] == m)} "
        f"{sum(r.get('cost', 0) for r in rows if r['model'] == m):9.0f} ток." for m in models))
    for r in rows:
        if "error" in r:
            print(f"  ошибка {r['task']} / {r['model']}: {r['error']}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(ROOT / "data" / "examples"))
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--repeat", type=int, default=1, help="сколько раз повторить каждую пару")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    env = {"base": os.getenv("LLM_BASE_URL", "").strip(), "key": os.getenv("LLM_API_KEY", "").strip()}
    if not env["key"]:
        print("LLM_API_KEY пуст: впишите ключ в Project/.env")
        return 1

    tasks = read_tasks(args.tasks)
    sem = asyncio.Semaphore(CONCURRENCY)
    with tempfile.TemporaryDirectory(prefix="baseline-") as tmp:
        jobs = []
        for i in range(args.repeat):
            for m in args.models:
                for t in tasks:
                    jobs.append(solve(t, m, env, sem, Path(tmp) / f"r{i}"))
        print(f"{len(jobs)} вызовов ({len(tasks)} задач × {len(args.models)} моделей × {args.repeat})…")
        rows = await asyncio.gather(*jobs)

    print_table(rows, args.models, tasks)
    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"baseline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nПодробности: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    logging.disable(logging.INFO)
    sys.exit(asyncio.run(main()))
