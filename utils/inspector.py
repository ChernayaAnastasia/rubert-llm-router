"""
Интерактивный HTML-просмотр предсказаний Stage 4 (hybrid_val_preds.parquet).

Основные функции:
- `load_hybrid_for_inspection` — загрузка hybrid + agent + val_merged (адрес, search_query)
- `llm_helped_mask` / `search_helped_mask` — маски «агент / поиск исправили ошибку BERT»
- `filter_hybrid_examples` — отбор примеров по типу кейса
- `inspect_row` / `inspect_examples` — HTML в Jupyter
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Literal

import pandas as pd
from IPython.display import HTML, display

from utils.config import (
    AGENT_LOW_CONF_PREDS_PATH,
    COL_ADDRESS,
    COL_BERT_PRED,
    COL_FINAL_PRED,
    COL_ID,
    COL_NAME,
    COL_PRICELIST,
    COL_QUERY,
    COL_REVIEWS,
    COL_ROUTED_TO,
    COL_RUBRIC,
    COL_SEARCH_USED,
    HYBRID_VAL_PREDS_PATH,
    TARGET,
    VAL_MERGED_PREDS_PATH,
)

HYBRID_VAL_PREDS_PATH = Path(HYBRID_VAL_PREDS_PATH)
AGENT_PREDS_PATH = Path(AGENT_LOW_CONF_PREDS_PATH)

HybridFilter = Literal[
    "llm_helped",
    "search_helped",
    "llm_hurt",
    "search_used",
    "agent_wrong",
    "all",
]


def _esc(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return html.escape(str(value))


def _badge(text: str, *, ok: bool | None = None) -> str:
    if ok is True:
        color, bg = "#1b5e20", "#e8f5e9"
    elif ok is False:
        color, bg = "#b71c1c", "#ffebee"
    else:
        color, bg = "#424242", "#eeeeee"
    return (
        f'<span style="display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;'
        f'border-radius:6px;background:{bg};color:{color};font-size:12px;">'
        f"{html.escape(text)}</span>"
    )


def llm_helped_mask(df: pd.DataFrame) -> pd.Series:
    """LLM-маршрут: гибрид верен, BERT ошибся."""
    return (
        (df[COL_ROUTED_TO] == "llm")
        & (df[COL_FINAL_PRED] == df[TARGET])
        & (df[COL_BERT_PRED] != df[TARGET])
    )


def search_helped_mask(df: pd.DataFrame) -> pd.Series:
    """Tavily использован: гибрид верен, BERT ошибся."""
    return (
        df[COL_SEARCH_USED].fillna(False).astype(bool)
        & (df[COL_FINAL_PRED] == df[TARGET])
        & (df[COL_BERT_PRED] != df[TARGET])
    )


def llm_hurt_mask(df: pd.DataFrame) -> pd.Series:
    """LLM-маршрут: BERT был верен, агент испортил."""
    return (
        (df[COL_ROUTED_TO] == "llm")
        & (df[COL_FINAL_PRED] != df[TARGET])
        & (df[COL_BERT_PRED] == df[TARGET])
    )


def _apply_filter(df: pd.DataFrame, kind: HybridFilter) -> pd.DataFrame:
    if kind == "all":
        return df
    if kind == "llm_helped":
        return df[llm_helped_mask(df)]
    if kind == "search_helped":
        return df[search_helped_mask(df)]
    if kind == "llm_hurt":
        return df[llm_hurt_mask(df)]
    if kind == "search_used":
        return df[df[COL_SEARCH_USED].fillna(False).astype(bool)]
    if kind == "agent_wrong":
        return df[(df[COL_ROUTED_TO] == "llm") & (df[COL_FINAL_PRED] != df[TARGET])]
    raise ValueError(f"Unknown filter kind: {kind}")


def load_hybrid_for_inspection(
    *,
    hybrid_path: Path | str = HYBRID_VAL_PREDS_PATH,
    agent_path: Path | str = AGENT_PREDS_PATH,
    val_merged_path: Path | str = VAL_MERGED_PREDS_PATH,
) -> pd.DataFrame:
    """Загружает hybrid_val_preds и обогащает agent/search и текстовыми полями."""
    hybrid = pd.read_parquet(hybrid_path)

    if Path(agent_path).exists():
        agent_cols = [
            COL_ID,
            "search_query",
            "prompt_tokens",
            "completion_tokens",
            "latency_sec",
        ]
        agent = pd.read_parquet(agent_path)
        use_cols = [c for c in agent_cols if c in agent.columns]
        hybrid = hybrid.merge(agent[use_cols], on=COL_ID, how="left")

    if Path(val_merged_path).exists():
        extra_cols = [COL_ID, COL_ADDRESS, COL_REVIEWS, COL_PRICELIST]
        val = pd.read_parquet(val_merged_path)
        use_cols = [c for c in extra_cols if c in val.columns]
        if len(use_cols) > 1:
            hybrid = hybrid.merge(val[use_cols], on=COL_ID, how="left")

    return hybrid


def filter_hybrid_examples(
    df: pd.DataFrame,
    kind: HybridFilter = "llm_helped",
    *,
    n: int | None = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Отбирает примеры по типу кейса (llm_helped, search_helped, …)."""
    subset = _apply_filter(df, kind)
    if n is None or n >= len(subset):
        return subset.copy()
    return subset.sample(n=int(n), random_state=random_state).copy()


def hybrid_filter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по основным типам кейсов."""
    agent = df[df[COL_ROUTED_TO] == "llm"]
    rows = [
        ("all", len(df)),
        ("llm_helped", int(llm_helped_mask(df).sum())),
        ("search_helped", int(search_helped_mask(df).sum())),
        ("llm_hurt", int(llm_hurt_mask(df).sum())),
        ("search_used", int(df[COL_SEARCH_USED].fillna(False).astype(bool).sum())),
        ("agent_wrong", int(((agent[COL_FINAL_PRED] != agent[TARGET])).sum()) if len(agent) else 0),
    ]
    return pd.DataFrame(rows, columns=["filter", "count"]).set_index("filter")


def inspect_row_html(row: pd.Series, idx, *, label_col: str = TARGET) -> str:
    """Формирует HTML для одной строки hybrid_val_preds."""
    true_label = row.get(label_col)
    bert_pred = row.get(COL_BERT_PRED)
    final_pred = row.get(COL_FINAL_PRED)
    bert_ok = bert_pred == true_label if pd.notna(bert_pred) and pd.notna(true_label) else None
    hybrid_ok = final_pred == true_label if pd.notna(final_pred) and pd.notna(true_label) else None
    llm_fixed = bool(llm_helped_mask(pd.DataFrame([row])).iloc[0]) if COL_ROUTED_TO in row else False
    search_fixed = bool(search_helped_mask(pd.DataFrame([row])).iloc[0]) if COL_SEARCH_USED in row else False

    bert_proba = row.get("bert_proba1")
    bert_max = row.get("bert_max_proba")
    proba_str = "—"
    if pd.notna(bert_proba):
        proba_str = f"p(relevant)={float(bert_proba):.3f}"
        if pd.notna(bert_max):
            proba_str += f", max_conf={float(bert_max):.3f}"

    tags = [
        _badge(f"label={true_label}"),
        _badge(f"BERT={bert_pred}", ok=bert_ok),
        _badge(f"hybrid={final_pred}", ok=hybrid_ok),
        _badge(f"route={row.get(COL_ROUTED_TO, '—')}"),
    ]
    if row.get(COL_SEARCH_USED):
        tags.append(_badge("search_used"))
    if llm_fixed:
        tags.append(_badge("LLM помог", ok=True))
    if search_fixed:
        tags.append(_badge("поиск помог", ok=True))

    search_block = ""
    if row.get("search_query") and pd.notna(row["search_query"]):
        search_block = f"""
  <p><strong>Поисковый запрос (Tavily):</strong><br>{_esc(row['search_query'])}</p>"""

    meta_block = ""
    meta_parts = []
    if "latency_sec" in row and pd.notna(row["latency_sec"]):
        meta_parts.append(f"latency={float(row['latency_sec']):.2f}s")
    if "prompt_tokens" in row and pd.notna(row["prompt_tokens"]):
        meta_parts.append(f"tokens in/out={int(row['prompt_tokens'])}/{int(row.get('completion_tokens') or 0)}")
    if meta_parts:
        meta_block = f"<p><strong>Agent:</strong> {', '.join(meta_parts)}</p>"

    tfidf_block = ""
    if "tfidf_pred" in row and pd.notna(row["tfidf_pred"]):
        tfidf_proba = row.get("tfidf_proba1")
        tfidf_str = f"TF-IDF={int(row['tfidf_pred'])}"
        if pd.notna(tfidf_proba):
            tfidf_str += f" (p={float(tfidf_proba):.3f})"
        tfidf_block = f"<p><strong>Baseline:</strong> {tfidf_str}</p>"

    return f"""
<div style="border:1px solid #ccc;padding:16px;border-radius:10px;
            font-family:sans-serif;background:#f9f9f9;margin-bottom:20px">
  <h2 style="margin-top:0">Index: {_esc(idx)} · {_esc(row.get(COL_ID, '—'))}</h2>
  <p>{''.join(tags)}</p>
  <p><strong>Запрос:</strong><br>{_esc(row.get(COL_QUERY))}</p>
  <p><strong>Название:</strong><br>{_esc(row.get(COL_NAME))}</p>
  <p><strong>Адрес:</strong><br>{_esc(row.get(COL_ADDRESS))}</p>
  <p><strong>Рубрика:</strong><br>{_esc(row.get(COL_RUBRIC))}</p>
  <p><strong>BERT:</strong> pred={_esc(bert_pred)} · {proba_str}</p>
  {tfidf_block}
  {search_block}
  {meta_block}
  <p><strong>Отзывы:</strong><br>{_esc(row.get(COL_REVIEWS))}</p>
  <p><strong>Pricelist (не видит агент):</strong><br>{_esc(row.get(COL_PRICELIST))}</p>
</div>
"""


def inspect_row(
    df: pd.DataFrame,
    idx,
    *,
    label_col: str = TARGET,
) -> None:
    """Показывает HTML-блок для одной строки."""
    row = df.loc[idx]
    display(HTML(inspect_row_html(row, idx, label_col=label_col)))


def inspect_examples(
    df: pd.DataFrame,
    kind: HybridFilter = "llm_helped",
    *,
    n: int = 5,
    random_state: int = 42,
    show_summary: bool = True,
) -> pd.DataFrame:
    """
    Показывает n примеров выбранного типа и возвращает отфильтрованный DataFrame.

    kind:
        llm_helped   — агент исправил ошибку BERT
        search_helped — Tavily + агент исправили ошибку BERT
        llm_hurt     — BERT был верен, агент ошибся
        search_used  — любые примеры с поиском
        agent_wrong  — все ошибки агента
        all          — без фильтра (sample)
    """
    if show_summary:
        display(hybrid_filter_summary(df))

    subset = filter_hybrid_examples(df, kind, n=n, random_state=random_state)
    print(f"{kind}: показано {len(subset)} / {len(_apply_filter(df, kind))}")

    for idx, row in subset.iterrows():
        display(HTML(inspect_row_html(row, idx)))

    return subset
