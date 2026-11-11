"""Точка входа. Жюри запускает:  python /app/run_agent.py --tasks-dir /data/tasks

Переменные окружения: LLM_BASE_URL, LLM_API_KEY, BUDGET (см. harness/config.py).
Ответы — answer.txt рядом с task.json каждой решённой задачи.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("harness")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def quiet_libraries() -> None:
    """openJiuwen очень многословен — оставляем только наши логи и чужие ошибки."""
    for name in list(logging.root.manager.loggerDict):
        if not name.startswith("harness"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.root.setLevel(logging.CRITICAL)


async def main(tasks_dir: str) -> dict:
    from harness.config import Settings
    from harness.planner import Planner
    from harness.tasks import read_tasks

    settings = Settings.from_env()
    if settings.has_llm:
        # до первого импорта openJiuwen: без файлов ./logs (контейнер read-only) и без шума в stdout
        from harness.jiuwen_setup import configure_openjiuwen
        configure_openjiuwen()
        import openjiuwen.core.foundation.llm  # noqa: F401
        import openjiuwen.core.single_agent  # noqa: F401
    quiet_libraries()

    tasks = read_tasks(tasks_dir)
    log.info("%d tasks, budget %.0f, LLM %s", len(tasks), settings.budget, "on" if settings.has_llm else "OFF")
    summary = await Planner(settings).run(tasks)
    log.info("done: solved %d, infeasible %d of %d; tokens %.0f in %d calls; %.0fs",
             summary["solved"], summary["infeasible"], summary["tasks"], summary["tokens_spent"],
             summary["llm_calls"], summary["seconds"])
    return summary


def cli() -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default=os.getenv("TASKS", "/data/tasks"))
    args, unknown = parser.parse_known_args()  # лишние аргументы жюри не должны нас ронять
    if unknown:
        log.warning("ignored args: %s", unknown)
    try:
        asyncio.run(main(args.tasks_dir))
    except BaseException as e:  # что бы ни случилось — уже записанные ответы остаются
        log.exception("fatal: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
