"""Проверка доступа к моделям через openJiuwen.

Запуск (из папки Project):
    python scripts\\check_api.py
    ... scripts\\check_api.py qwen3.8-27b-noreason     # только одна модель

Каждой модели задаётся один короткий вопрос. Скрипт печатает ответ, токены
и стоимость по правилам соревнования: выходные + входные / 10.
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

MODELS = [
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b-noreason",
    "qwen3.8-27b",
    "qwen3.6-35b-a3b",
]
QUESTION = "Сколько будет 17*23? Ответь только числом."


def cost(input_tokens: int, output_tokens: int) -> float:
    return output_tokens + input_tokens / 10


async def check(model_name: str, api_base: str, api_key: str) -> float:
    from openjiuwen.core.foundation.llm import init_model

    model = init_model("OpenAI", model_name, api_key, api_base, max_tokens=2048, timeout=120)
    started = time.perf_counter()
    try:
        reply = await model.invoke([{"role": "user", "content": QUESTION}])
    except Exception as e:  # покажем ошибку и пойдём к следующей модели
        print(f"  ✗ {model_name}: {type(e).__name__}: {e}")
        return 0.0
    elapsed = time.perf_counter() - started

    usage = reply.usage_metadata
    inp = usage.input_tokens if usage else 0
    out = usage.output_tokens if usage else 0
    reasoning = len(reply.reasoning_content or "")
    answer = (reply.content or "").strip().replace("\n", " ")[:60]
    print(f"  ✓ {model_name}: ответ={answer!r} | in={inp} out={out} "
          f"reasoning_chars={reasoning} | cost={cost(inp, out):.1f} | {elapsed:.1f}s")
    return cost(inp, out)


async def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_base = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        print("LLM_API_KEY пуст: впишите ключ в Project/.env")
        return 1
    print(f"API: {api_base}  ключ: …{api_key[-4:]}")

    models = sys.argv[1:] or MODELS
    total = 0.0
    for name in models:
        total += await check(name, api_base, api_key)
    print(f"Итого потрачено: {total:.1f} токенов (по правилам соревнования)")
    return 0


if __name__ == "__main__":
    logging.disable(logging.INFO)  # openJiuwen очень многословен на INFO
    sys.exit(asyncio.run(main()))
