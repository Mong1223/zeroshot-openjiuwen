"""Настройка openJiuwen перед использованием.

По умолчанию openJiuwen:
  * пишет журналы в ФАЙЛЫ ./logs/... в текущей папке — в контейнере жюри файловая система
    только для чтения (--read-only), такая запись сломалась бы;
  * печатает каждое событие (вместе с полным текстом запросов к модели) в stdout.
Поэтому до первого использования переключаем его журналы: без файлов, без консоли, только ошибки.
"""
from __future__ import annotations

import logging
import tempfile

_done = False


def configure_openjiuwen() -> None:
    global _done
    if _done:
        return
    _done = True
    try:
        from openjiuwen.core.common.logging import reset_lazy_loggers
        from openjiuwen.core.common.logging.log_config import log_config
        from openjiuwen.core.common.logging.manager import LogManager
    except ImportError:
        return
    quiet = {
        "backend": "default",
        "level": "ERROR",
        "log_path": tempfile.gettempdir(),  # на случай, если что-то всё же захочет файл: /tmp доступен
        "output": [],
        "interface_output": [],
        "performance_output": [],
        "propagate": False,
        "loggers": {},
    }
    try:
        log_config.load_from_dict(quiet)  # ждёт содержимое секции logging, без обёртки
        LogManager.reset()
        reset_lazy_loggers()
    except Exception:  # noqa: BLE001 — настройка журналов не должна мешать работе
        pass
    for name in list(logging.root.manager.loggerDict):
        if not name.startswith("harness"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
