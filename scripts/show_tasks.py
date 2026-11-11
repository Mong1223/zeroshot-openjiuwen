"""Сводка по папке задач — что видит программа до решения (0 токенов).

    python scripts\\show_tasks.py                  # примеры из data/examples
    python scripts\\show_tasks.py путь\\к\\задачам
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.tasks import read_tasks  # noqa: E402

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "examples"


def describe(features: dict) -> str:
    parts = [f"{features['question_words']} слов", f"{features['numbers_in_text']} чисел"]
    if "csv_rows" in features:
        parts.append(f"CSV {features['csv_rows']} строк")
    if "kb_files" in features:
        parts.append(f"kb {features['kb_files']} заметок")
    if "test_functions" in features:
        parts.append(f"{features['test_functions']} тестов, {features['code_lines']} строк кода")
    return ", ".join(parts)


def main() -> None:
    tasks = read_tasks(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    print(f"{'задача':<20} {'тип':<14} {'баллы':>5}  {'проверка':<10} {'ответ':<11} признаки")
    for t in tasks:
        print(f"{t.task_id:<20} {t.task_type:<14} {t.reward:>5g}  {t.check:<10} "
              f"{(t.expected or '—'):<11} {describe(t.features)}")
    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
    print(f"\nВсего {len(tasks)} задач, {sum(t.reward for t in tasks):g} баллов: "
          + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))


if __name__ == "__main__":
    main()
