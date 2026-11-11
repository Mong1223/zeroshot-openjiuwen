"""Решатель search, уровень 0: база заметок kb/*.md → граф → проход по цепочке связей. 0 токенов.

Заметка выглядит так:
    # Irina Smith
    Born: 1983
    Employer: Dune Robotics
    Mentor: Alice Smith

Вопрос разбирается в «сущность + цепочка полей»:
    "In which year was Irina Smith born?"                       → Irina Smith → Born
    "In which city is the headquarters of Sergei Garcia's employer?" → Sergei Garcia → Employer → Headquarters
Сущности нет в базе или нет нужного поля — задача невыполнима.
"""
from __future__ import annotations

import re
from pathlib import Path

from harness.tasks import question_text
from harness.types import Result, Status, Task

# слово в вопросе → поле заметки
RELATIONS = {
    "employer": "Employer", "company": "Employer", "work": "Employer", "works": "Employer",
    "mentor": "Mentor",
}
ATTRIBUTES = [  # (признак в вопросе, поле) — что спрашивают в конце цепочки
    (re.compile(r"\bborn\b|\bbirth\b"), "Born"),
    (re.compile(r"\bheadquarter|\bhq\b|\bbased\b"), "Headquarters"),
    (re.compile(r"\bfounded\b|\bestablished\b"), "Founded"),
]
_STOP = {"in", "which", "what", "who", "where", "when", "the", "use", "a", "an", "of", "is", "was"}
_NAME = re.compile(r"[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*")


def load_kb(folder: Path) -> dict[str, dict[str, str]]:
    """{нормализованное имя: {"name": ..., поле: значение}}"""
    kb: dict[str, dict[str, str]] = {}
    for path in sorted(folder.rglob("*.md")):
        record: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("#") and "name" not in record:
                record["name"] = line.lstrip("#").strip()
            elif ":" in line:
                key, _, value = line.partition(":")
                record[key.strip().title()] = value.strip()
        if "name" in record:
            key = _norm(record["name"])
            if key in kb:  # два файла с одним именем — неоднозначность
                kb[key]["_duplicate"] = "1"
            else:
                kb[key] = record
    return kb


def solve(task: Task) -> Result:
    kb_dir = task.workspace / "kb"
    if not kb_dir.is_dir():
        return Result(Status.UNSURE, note="no kb/")
    kb = load_kb(kb_dir)
    question = question_text(task.instruction)
    # служебное «Use the file tools or bash…» мешает разбору — отрезаем
    question = re.split(r"\bUse the file tools\b|\bUse bash\b", question)[0]

    if not any(len(rec) > 1 for rec in kb.values()):  # база не в формате «Поле: значение» (например, текст)
        return Result(Status.UNSURE, note="kb is not in 'Field: value' format")

    counted = count_query(question, kb)
    if counted is not None:
        return counted

    parsed = parse(question, kb)
    if parsed is None:
        missing = unknown_attribute(question, kb)
        if missing:
            return Result(Status.INFEASIBLE, confidence=0.7, note=f"kb has no field like {missing!r}")
        return Result(Status.UNSURE, note="question not recognized")
    entity, chain, exact = parsed
    return follow(kb, entity, chain, confidence=1.0 if exact else 0.8, infeasible_confidence=0.8,
                  corpus=kb_corpus(kb_dir))


def kb_corpus(folder: Path) -> str:
    """Весь текст базы одной строкой (нижний регистр) — проверить, упоминается ли имя хоть где-то."""
    parts = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".md", ".txt", ".json", ".csv", ".yaml", ".yml"):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return _norm(" ".join(parts))


def follow(kb: dict[str, dict[str, str]], entity: str, chain: list[str], *, confidence: float,
           infeasible_confidence: float, corpus: str = "") -> Result:
    """Проход по цепочке полей от сущности: Irina Smith → Employer → Headquarters.

    «Нерешаемо» говорим только при надёжных признаках: имени нет НИГДЕ в тексте базы,
    а нужного поля нет ни в одной заметке. Иначе — UNSURE: скорее всего, мы не так поняли
    вопрос или формат базы, и задачу должен посмотреть следующий уровень.
    """
    if not chain:
        return Result(Status.UNSURE, note="empty chain")
    if _norm(entity) not in kb:
        if corpus and _norm(entity) in corpus:
            return Result(Status.UNSURE, note=f"{entity!r} is mentioned in kb, but not as a note title")
        return Result(Status.INFEASIBLE, confidence=infeasible_confidence, note=f"entity {entity!r} not in kb")
    record = kb[_norm(entity)]
    known_fields = {k for rec in kb.values() for k in rec}
    path = [record["name"]]
    for i, field in enumerate(chain):
        if record.get("_duplicate"):
            return Result(Status.UNSURE, note=f"ambiguous {record['name']!r}")
        value = record.get(field)
        if value is None:
            if field in known_fields:  # у других заметок поле есть — скорее всего, цепочка понята неверно
                return Result(Status.UNSURE, note=f"{record['name']!r} has no {field} (others do)")
            return Result(Status.INFEASIBLE, confidence=infeasible_confidence,
                          note=f"{record['name']!r} has no {field}")
        path.append(f"{field}={value}")
        if i == len(chain) - 1:
            return Result(Status.SOLVED, answer=value, confidence=confidence, note=" → ".join(path))
        nxt = kb.get(_norm(value))
        if nxt is None:  # следующего звена нет как заметки — данные неполные или вопрос понят не так
            return Result(Status.UNSURE, note=f"{value!r} not in kb as a note")
        record = nxt
    return Result(Status.UNSURE, note="empty chain")


def parse(question: str, kb: dict[str, dict[str, str]]) -> tuple[str, list[str], bool] | None:
    """→ (сущность, цепочка полей, совпал ли точный шаблон) или None."""
    q = question.lower()

    m = re.search(r"in which year was (.+?) born\?", question, re.I)
    if m and "'s" not in m.group(1):
        return m.group(1).strip(), ["Born"], True
    m = re.search(r"in which city is the headquarters of (.+?)'s employer\?", question, re.I)
    if m and "'s" not in m.group(1):
        return m.group(1).strip(), ["Employer", "Headquarters"], True

    # общий случай: находим имя, затем связи до и после него
    entity = _find_entity(question, kb)
    if entity is None:
        return None
    start = question.find(entity)
    before, after = q[:start], q[start + len(entity):]

    chain: list[str] = []
    # после имени — притяжательная цепочка: "X's mentor's employer"
    for word in re.findall(r"'s\s+(\w+)", after):
        if word in RELATIONS:
            chain.append(RELATIONS[word])
    # до имени — "the employer of the mentor of X": применяем в обратном порядке
    for word in reversed(re.findall(r"(\w+)\s+of\s+(?:the\s+)?", before)):
        if word in RELATIONS:
            chain.append(RELATIONS[word])
    # «the company that employs X», «the company X works for» — это работодатель X
    if not chain and (re.search(r"\b(?:company|firm|organi[sz]ation|employer)\s+(?:that|which)\s+employs\s*$", before)
                      or re.search(r"^\s*(?:works?|worked)\s+(?:for|at)\b", after)):
        chain.append("Employer")

    attribute = next((f for rx, f in ATTRIBUTES if rx.search(q)), None)
    if attribute:
        chain.append(attribute)
    elif re.search(r"\bwho is\b|\bwhich company\b|\bwhere does .* work\b", q) and chain:
        pass  # спрашивают саму связь — последняя связь и есть ответ
    else:
        return None
    if not chain or not _chain_valid(chain):
        return None
    return entity, chain, False


_COUNT_PEOPLE = re.compile(
    r"how many (?:people|persons|employees|researchers|individuals)(?: in the knowledge base)? "
    r"(?:(?:currently )?work (?:for|at)|are employed (?:by|at)|have (?:the )?employer) (.+?)\?", re.I)
_COUNT_MENTEES = re.compile(
    r"how many (?:people|persons|individuals)(?: in the knowledge base)? "
    r"(?:are mentored by|have (?:the )?mentor|have .+? as (?:their|a) mentor) (.+?)\?", re.I)


def count_query(question: str, kb: dict[str, dict[str, str]]) -> Result | None:
    """«How many people work for Dune Robotics?» → число заметок с Employer = Dune Robotics."""
    for rx, field in ((_COUNT_PEOPLE, "Employer"), (_COUNT_MENTEES, "Mentor")):
        m = rx.search(question)
        if not m:
            continue
        target = _norm(m.group(1).strip(" .'\""))
        if target not in kb:
            return Result(Status.INFEASIBLE, confidence=0.7, note=f"{m.group(1)!r} not in kb")
        n = sum(_norm(rec.get(field, "")) == target for rec in kb.values())
        return Result(Status.SOLVED, answer=str(n), confidence=0.9, note=f"count {field}={m.group(1)}: {n}")
    return None


# что точно не хранится в заметках, если такого поля нет ни в одной из них
_ASKED = re.compile(r"\bwhat (?:is|was|are) (?:the )?([a-z][a-z ]{2,30}?) (?:of|for)\b"
                    r"|'s ([a-z][a-z ]{2,30}?)\?", re.I)


def unknown_attribute(question: str, kb: dict[str, dict[str, str]]) -> str | None:
    """Спрашивают поле, которого нет в базе («phone number», «email address») → его название."""
    m = _ASKED.search(question)
    if not m:
        return None
    asked = (m.group(1) or m.group(2)).lower().strip()
    if any(word in RELATIONS for word in asked.split()) or any(rx.search(asked) for rx, _ in ATTRIBUTES):
        return None
    fields = {k.lower() for rec in kb.values() for k in rec if not k.startswith("_")}
    if any(word in fields for word in asked.split()):
        return None
    return asked


def _find_entity(question: str, kb: dict[str, dict[str, str]]) -> str | None:
    """Имя из базы, а если его нет — самая правдоподобная «Имя Фамилия» из текста."""
    known = [rec["name"] for rec in kb.values() if rec["name"] in question]
    if known:
        return max(known, key=len)
    for candidate in _NAME.findall(question):
        words = candidate.split()
        while words and words[0].lower() in _STOP:
            words = words[1:]
        if len(words) >= 2:
            return " ".join(words)
    return None


def _chain_valid(chain: list[str]) -> bool:
    """Person: Born, Employer→Company, Mentor→Person. Company: Headquarters, Founded."""
    kind = "person"
    for field in chain:
        if kind == "person" and field == "Mentor":
            kind = "person"
        elif kind == "person" and field == "Employer":
            kind = "company"
        elif kind == "person" and field == "Born":
            kind = "value"
        elif kind == "company" and field in ("Headquarters", "Founded"):
            kind = "value"
        else:
            return False
    return True


def _norm(name: str) -> str:
    return " ".join(name.lower().split())
