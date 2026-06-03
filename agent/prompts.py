"""Prompts and formatting helpers for Stage 4 agent.

LLM context (spec v9): query, name, address, rubric, reviews, pricelist.
Parquet columns only — COL_COMBINED_TEXT is not stored (see utils/config.py).
"""

from __future__ import annotations

from typing import Any

from utils.config import (
    COL_ADDRESS,
    COL_NAME,
    COL_PRICELIST,
    COL_QUERY,
    COL_REVIEWS,
    COL_RUBRIC,
)


def _field(row: dict[str, Any], col: str) -> str:
    """Non-empty string from a parquet column, or empty string."""
    val = row.get(col)
    if val is None:
        return ""
    return str(val).strip()


def format_org_context(row: dict[str, Any]) -> str:
    """Контекст для LLM — все поля включая COL_PRICELIST (spec v9 §4.2)."""
    return (
        f"Запрос: {_field(row, COL_QUERY)}\n\n"
        f"Название: {_field(row, COL_NAME)}\n"
        f"Адрес: {_field(row, COL_ADDRESS)}\n"
        f"Рубрика: {_field(row, COL_RUBRIC)}\n"
        f"Отзывы: {_field(row, COL_REVIEWS)}\n"
        f"Прайслист: {_field(row, COL_PRICELIST)}"
    )


def format_search_decision_vars(row: dict[str, Any]) -> dict[str, str]:
    """Variables for the search-decision prompt."""
    name_full = _field(row, COL_NAME)
    name_short = name_full.split(";")[0].strip() if name_full else ""

    return {
        "query": _field(row, COL_QUERY),
        "name": name_full,
        "name_short": name_short,
        "address": _field(row, COL_ADDRESS),
        "rubric": _field(row, COL_RUBRIC),
        "reviews": _field(row, COL_REVIEWS),
        "pricelist": _field(row, COL_PRICELIST),
    }


SEARCH_DECISION_PROMPT = """Определи, нужно ли искать дополнительную информацию \
об организации, чтобы достоверно определить её релевантность запросу.

### Когда нужен поиск:
1. В запросе есть уточнения по режиму работы, конкретным услугам, ценам.
2. Рубрика слишком общая, а запрос специфичный.
3. Неясно, предоставляет ли организация нужную услугу.
4. В запросе есть временные или географические ограничения.

### Когда поиск не нужен:
1. Рубрика явно соответствует или не соответствует запросу.
2. Отзывы или прайслист подтверждают ключевые требования из запроса.
3. Запрос общий и информации достаточно.

### Примеры:
Запрос: Шиномонтаж 24
Название: Шиномонтаж
Адрес: Республика Калмыкия, Элиста, улица В.И. Ленина, 7
Рубрика: Шиномонтаж
Отзывы: хвалят высокое качество работы.
Прайслист:
Ответ: SEARCH: Шиномонтаж Элиста, улица В.И. Ленина, 7 круглосуточно

Запрос: где можно дешево поесть в санкт-петербурге на невском проспекте
Название: Столовая 100; Stolovaya 100; Столовая
Адрес: Санкт-Петербург, Лиговский проспект, 50Ф
Рубрика: Столовая
Отзывы: Организация занимается питанием, предоставляя широкий выбор блюд по доступным ценам.
Прайслист: Столовая «100» предлагает разнообразные блюда и напитки, включая супы, каши, мясные блюда, салаты и морсы.
Ответ: NO_SEARCH_NEEDED

### Оцени:
Запрос: {query}
Название: {name}
Адрес: {address}
Рубрика: {rubric}
Отзывы: {reviews}
Прайслист: {pricelist}

Если нужен поиск — используй в запросе "{name_short}" (первое название) + {address}
+ {query}.

Формат ответа (строго одна строка):
SEARCH: <поисковый запрос>
или
NO_SEARCH_NEEDED"""


CLASSIFICATION_PROMPT = """Ты — эксперт по оценке релевантности организаций для Яндекс.Карт запросу пользователя.

Определи: организация РЕЛЕВАНТНА (1) или НЕРЕЛЕВАНТНА (0) для данного запроса.

Правила:
1. Сверяй адрес и рубрику с запросом; несовпадение города/района/станции метро → 0.
2. Если в запросе указан номер отделения/филиала/организации — он должен совпадать с названием → иначе 0.
3. Используй отзывы, прайслист и дополнительную информацию из интернета, если она есть.

{org_context}
{search_context}

Отвечай ТОЛЬКО цифрой: 0 или 1. Никаких пояснений."""
