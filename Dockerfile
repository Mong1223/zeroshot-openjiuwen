# ZeroShot — агент для openJiuwen Challenge 2026.
# Жюри запускает: python /app/run_agent.py --tasks-dir /data/tasks
# с переменными LLM_BASE_URL, LLM_API_KEY, BUDGET; контейнер --read-only, /tmp — tmpfs, HOME=/tmp.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # во время работы интернета нет: никаких попыток что-то скачать
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    # всё временное — только в /tmp (корень контейнера только для чтения)
    TMPDIR=/tmp \
    TRACE_DIR=/tmp/zeroshot \
    MPLCONFIGDIR=/tmp/matplotlib \
    DOTENV_DISABLED=1

WORKDIR /app

COPY requirements.txt constraints.txt ./
# uv качает пакеты параллельно: сборка в разы быстрее pip (лимит жюри — 1000 с); версии те же — из constraints.txt
RUN pip install uv==0.5.11 \
 && uv pip install --system --no-cache -r requirements.txt -c constraints.txt \
 && pip uninstall -y uv \
 # словари tiktoken — заранее, внутрь образа (наш прогон к ним не обращается, но на всякий случай)
 && python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base', 'o200k_base')]"

COPY run_agent.py ./
COPY harness ./harness

# заранее компилируем .pyc: в read-only контейнере их уже нельзя будет записать
RUN python -m compileall -q /app /usr/local/lib/python3.12/site-packages >/dev/null 2>&1 || true \
 && python -c "import harness.planner, harness.agent; print('import ok')"

ENTRYPOINT []
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
