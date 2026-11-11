"""Нормализация текста перед правилами (улучшение №1 из PLAN.md).

Цель — чтобы одно правило покрывало много формулировок:
    "twenty-five"  → "25",   "a dozen" → "12",   "two dozen" → "24"
    "sold", "sells", "dispatched", "gives away" → "ships"      (убыль)
    "bought", "purchases", "collected"          → "receives"   (прибыль)
    "35 crates are sold"                        → "ships 35 crates"  (страдательный залог)
    "doubles" → "doubled",  "halves" → "halved"

Работает только со строками, без внешних библиотек: в коробке жюри нет интернета,
а лишняя зависимость — лишний риск при сборке.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ числа словами

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

_WORD = r"(?:" + "|".join(list(_UNITS) + list(_TENS) + list(_SCALES)) + r")"
# последовательность числительных: "one hundred and twenty-five", "two thousand three hundred"
_NUMBER_WORDS = re.compile(rf"\b{_WORD}(?:(?:\s+and\s+|\s+|-){_WORD})*\b", re.I)
# «one» — не число задачи в оборотах «keeps one share», «one of», «one more», «one by one»
_ONE_IS_NOT_A_NUMBER = re.compile(r"\s+(?:share|shares|part|portion|of|another|more|by|at a time)\b", re.I)


def _words_value(words: list[str]) -> int | None:
    total, current = 0, 0
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
        else:
            return None
    return total + current


def words_to_digits(text: str) -> str:
    """Числительные словами → цифры. «a dozen» → 12, «two dozen» → 24, «a hundred» → 100."""

    def repl(m: re.Match[str]) -> str:
        phrase = m.group(0)
        words = [w for w in re.split(r"[\s-]+", phrase.lower()) if w != "and"]
        if words == ["one"] and _ONE_IS_NOT_A_NUMBER.match(text, m.end()):
            return phrase
        value = _words_value(words)
        return phrase if value is None else str(value)

    text = re.sub(r"\ba (hundred|thousand)\b", lambda m: str(_SCALES[m.group(1).lower()]), text, flags=re.I)
    text = _NUMBER_WORDS.sub(repl, text)
    text = re.sub(r"\b(\d+) dozens?\b", lambda m: str(int(m.group(1)) * 12), text, flags=re.I)
    text = re.sub(r"\ba dozen\b", "12", text, flags=re.I)
    return text


# ------------------------------------------------------------------ глаголы

# убыль запаса: все формы → "ships"; прибыль → "receives"
_TAKE_FORMS = {
    "ship": "ships|shipped|shipping|ship",
    "sell": "sells|sold|selling|sell",
    "send out": "sends out|sent out|sending out|send out",
    "send away": "sends away|sent away|send away|sends|sent",
    "dispatch": "dispatches|dispatched|dispatch",
    "lose": "loses|lost|lose",
    "give away": "gives away|gave away|given away|give away|gives|gave|given",
    "remove": "removes|removed|remove",
    "use": "uses|used|use up|uses up|used up",
    "donate": "donates|donated|donate",
    "discard": "discards|discarded|discard",
    "throw away": "throws away|threw away|thrown away|throw away|throws out|threw out|thrown out",
    "spend": "spends|spent|spend",
    "consume": "consumes|consumed|consume",
    "eat": "eats|ate|eaten|eat",
    "withdraw": "withdraws|withdrew|withdrawn|withdraw",
    "take out": "takes out|took out|taken out|take out",
    "sell off": "sells off|sold off",
}
_ADD_FORMS = {
    "receive": "receives|received|receive",
    "buy": "buys|bought|buy",
    "purchase": "purchases|purchased|purchase",
    "get": "gets|got|gotten",
    "add": "adds|added|add",
    "gain": "gains|gained|gain",
    "produce": "produces|produced|produce",
    "acquire": "acquires|acquired|acquire",
    "collect": "collects|collected|collect",
    "manufacture": "manufactures|manufactured|manufacture",
    "make": "makes|made",
    "harvest": "harvests|harvested|harvest",
    "earn": "earns|earned|earn",
    "bake": "bakes|baked|bake",
    "restock": "restocks with|restocked with|restocks|restocked",
    "order in": "orders in|ordered in",
}


def _alternation(forms: dict[str, str]) -> str:
    variants = sorted({v for alts in forms.values() for v in alts.split("|")}, key=len, reverse=True)
    return "|".join(re.escape(v) for v in variants)


TAKE_VERBS = _alternation(_TAKE_FORMS)
ADD_VERBS = _alternation(_ADD_FORMS)
_BE = r"(?:are|is|were|was|get|gets|got|have been|has been|will be)"

# количество: «35 crates», «3 shipments of 50 boxes», «12 new boxes»
_QTY = r"(\d[\d,]*(?: \w+ of \d[\d,]*)? (?:new |more |extra |additional )?\w+)"
# "35 crates are sold (to customers)" → "ships 35 crates"
_PASSIVE_TAKE = re.compile(rf"\b{_QTY} {_BE} (?:{TAKE_VERBS})\b", re.I)
_PASSIVE_ADD = re.compile(rf"\b{_QTY} {_BE} (?:{ADD_VERBS}|delivered|added to (?:it|the stock))\b", re.I)
# "3 shipments of 50 boxes arrive" → "receives 3 shipments of 50 boxes"
_ARRIVE = re.compile(rf"\b{_QTY} (?:arrive|arrives|arrived|come in|came in)\b", re.I)
_ACTIVE_TAKE = re.compile(rf"\b(?:{TAKE_VERBS})\b", re.I)
_ACTIVE_ADD = re.compile(rf"\b(?:{ADD_VERBS})\b", re.I)

_MULTIPLY = [
    (re.compile(r"\b(?:doubles|doubled|double)\b", re.I), "doubled"),
    (re.compile(r"\b(?:triples|tripled|triple)\b", re.I), "tripled"),
    (re.compile(r"\b(?:quadruples|quadrupled|quadruple)\b", re.I), "quadrupled"),
    (re.compile(r"\b(?:halves|halved|is cut in half|gets cut in half)\b", re.I), "halved"),
]


def canonical_verbs(text: str) -> str:
    """Синонимы и формы глаголов → один глагол. Регистр текста приводится к нижнему."""
    text = text.lower()
    for rx, word in _MULTIPLY:  # раньше глаголов: «gets doubled» не должно стать «receives doubled»
        text = rx.sub(word, text)
    text = _PASSIVE_TAKE.sub(r"ships \1", text)
    text = _PASSIVE_ADD.sub(r"receives \1", text)
    text = _ARRIVE.sub(r"receives \1", text)
    text = _ACTIVE_TAKE.sub("ships", text)
    text = _ACTIVE_ADD.sub("receives", text)
    return text


# ------------------------------------------------------------------ дроби и проценты

# Дробь превращается в метку frac_P_Q: так её цифры не путаются с числами задачи.
_FRACTIONS = [
    (r"\b(?:a|1) half\b|\bhalf\b(?! of a\b)", "frac_1_2"),
    (r"\b(?:a|1) third\b", "frac_1_3"), (r"\b2 thirds\b", "frac_2_3"),
    (r"\b(?:a|1) quarter\b", "frac_1_4"), (r"\b3 quarters\b", "frac_3_4"),
    (r"\b(?:a|1) fifth\b", "frac_1_5"), (r"\b(?:a|1) tenth\b", "frac_1_10"),
]
_PERCENT = re.compile(r"\b(\d+) ?(?:%|percent\b)", re.I)


def fractions_to_tokens(text: str) -> str:
    """«a third» → frac_1_3, «25%» → frac_25_100."""
    text = _PERCENT.sub(lambda m: f"frac_{m.group(1)}_100", text)
    for pattern, token in _FRACTIONS:
        text = re.sub(pattern, token, text, flags=re.I)
    return text


def normalize(text: str) -> str:
    """Полная нормализация для решателя math: числа словами → цифры, дроби → метки, глаголы → канон."""
    return canonical_verbs(fractions_to_tokens(words_to_digits(text)))
