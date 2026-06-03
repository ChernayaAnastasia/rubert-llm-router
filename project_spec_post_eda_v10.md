# ТЗ: Этапы проекта после EDA (v10)
## Yandex Maps Relevance — cross-encoder + temperature scaling + error analysis + гибрид (BERT + LLM)

> **Исходная точка:** EDA завершена, артефакты в `PROCESSED_DATA_DIR`:
> `train_baseline.parquet`, `val_baseline.parquet`, `eval_baseline.parquet`, `rel_minus_baseline.parquet`
>
> **Схема колонок в parquet (`KEEP_COLS` в `utils/config.py`):**
> `COL_ID`, `COL_QUERY`, `COL_NAME`, `COL_ADDRESS`, `COL_RUBRIC`, `TARGET`, `COL_RELEVANCE`, `COL_REVIEWS`, `COL_PRICELIST`
>
> `COL_COMBINED_TEXT` **не хранится** — пересобирается на лету для TF-IDF (`build_combined_text()`).
> `COL_ORG_TEXT` **не хранится** — пересобирается для cross-encoder (`make_org_text()` / `attach_org_text()`).

---

## Выводы EDA (`notebooks/eda.ipynb`, `reports/eda_reports/`)

Источник: Stage 0. Полная сводка — `table3_eda_summary.csv`.

### Датасет

- **35 089** строк после удаления 5 шумных дубликатов (`query` + `permalink`); классы `relevance`: 0 — 41%, 0.1 — 13%, 1 — 46%.
- Пропуски: отзывы ~4%, прайс-лист ~41%, оба поля ~4%. Связь «пропуск ↔ класс» статистически слабая (Cramér’s V < 0.1) — строки **не отбрасываем**; в parquet плейсхолдеры `No reviews.` / `No pricelist.`.

### Решения, влияющие на Stages 1–5

| Вопрос | Решение | Обоснование (кратко) |
|--------|---------|----------------------|
| Сколько классов? | Бинарная задача **{0, 1}** | Exp A/B: 0.1 не отделяется линейной моделью (F1≈random, равномерная путаница) |
| Класс 0.1 | **Исключить** из train/val/eval (B3) | Exp C: лучший binary val acc при полном исключении 0.1 (~0.639 vs B1/B2 ~0.633) |
| OOD | `rel_minus_baseline.parquet` (4 703) | Sanity / распределение proba в Stage 3 |
| Сплиты | 70/15/15, стратификация по `label`, **row-level** | train 21 270 / val 4 558 / eval 4 558 (eval LOCKED); баланс ~48/52 |
| Метрики | Accuracy (primary), macro-F1 (secondary) | Зафиксировано в EDA и во всех отчётах |
| Routing к агенту | **Не** по пропускам полей | §6 EDA: acc «оба поля пусты» ≥ acc с полным контекстом (Δ≈−0.02); порог — Stage 3 по `bert_max_proba` |
| Текст Stage 1 | `combined_text` на лету | median ~2k символов → длинный cross-encoder, `truncation='only_second'` |

### Ограничения, зафиксированные в EDA

- Сплит по строкам, не по query (один запрос — несколько организаций).
- Стратегии B1/B2 (перенос 0.1 в 0 или 1) для transformer **не** пересматривались — только TF-IDF; при необходимости — отдельный эксперимент.

---

## Константы и пути (`utils/config.py`)

| Группа | Константы |
|--------|-----------|
| Модель | `BERT_MODEL_NAME = "deepvk/RuModernBERT-base"`, `BERT_MAX_LENGTH = 1024` |
| Каталоги | `BERT_DIR`, `BERT_BEST_CHECKPOINT_DIR`, `BERT_CHECKPOINTS_DIR` |
| Калибровка | `BERT_CALIBRATION_PATH` → `models/bert/calibration.json` |
| Предсказания | `BERT_VAL_PREDS_PATH`, `BERT_OOD_PREDS_PATH`, `BERT_EVAL_PREDS_PATH`, `VAL_MERGED_PREDS_PATH`, `AGENT_LOW_CONF_PREDS_PATH`, `HYBRID_VAL_PREDS_PATH`, `AGENT_EVAL_PREDS_PATH` |
| Отчёты | `STAGE1_REPORTS_DIR` … `STAGE4_REPORTS_DIR`, `FINAL_EVAL_DIR`, `STAGE3_COMPARISON_PATH` |
| LLM-агент (конфиг) | `AGENT_LLM_MODEL`, `VSEGPT_BASE_URL`, `SEARCH_CACHE_DIR`, `CONFIDENCE_THRESHOLD_DEFAULT = 0.75` |

Опционально: `config_local.py` с `PROJECT_ROOT` для нестандартного расположения проекта.

Этапы проекта можно выполнять в Jupyter-ноутбуках (notebooks/stage*.ipynb) — так выполнялся данный проект — или через python scripts/run_stageN.py, которые вызывают модули utils/stage*_*.py. Логика и артефакты одинаковые; скрипты не обязательны, если этап уже пройден в ноутбуке.

### Терминология (Stages 4–5)

| Термин | Что это |
|--------|---------|
| **Гибридная система** | Полный пайплайн релевантности: уверенные примеры → предсказание BERT; low-confidence → **LLM-агент** (опционально Tavily). Метрики «hybrid» в отчётах — про эту систему целиком. |
| **LLM-агент** | Компонент в каталоге `agent/` (LangGraph + VseGPT): обрабатывает **только** low-confidence после routing. |
| Имена `agent_*`, `hybrid_*` в путях | Исторические имена файлов в коде: `agent_*` — чаще выход LLM-агента или его цикл; `hybrid_*` — итог гибридной системы на всём split. |

---

## Структура пакета (фактическая)

```
project_root/
├── data/
│   ├── raw/                          # DATA_PATH (jsonl)
│   └── processed/                    # PROCESSED_DATA_DIR
│       ├── train_baseline.parquet
│       ├── val_baseline.parquet
│       ├── eval_baseline.parquet     # LOCKED до Stage 5
│       └── rel_minus_baseline.parquet
├── models/
│   ├── bert/                         # BERT_DIR
│   │   ├── best_checkpoint/
│   │   ├── checkpoints/
│   │   ├── training_args.json
│   │   └── calibration.json
│   └── tfidf_baseline/               # TFIDF_BASELINE_DIR (зарезервировано)
├── predictions/
│   ├── bert_val_preds.parquet        # Stage 2
│   ├── bert_ood_preds.parquet        # Stage 2 (OOD)
│   ├── val_merged_preds.parquet      # Stage 3 → вход Stage 4
│   ├── agent_low_conf_preds.parquet  # Stage 4 — только low-conf (LLM-агент)
│   ├── hybrid_val_preds.parquet      # Stage 4 — весь val (гибридная система)
│   ├── bert_eval_preds.parquet       # Stage 5A — только BERT на eval
│   └── agent_eval_preds.parquet      # Stage 5B — весь eval (гибридная система)
├── reports/
│   ├── eda_reports/
│   ├── stage1_baseline/metrics.json
│   ├── stage2_bert/                  # metrics.json, reliability_*.png
│   ├── stage3_error_analysis/        # comparison.json, fig_*, bert_errors_sample.csv
│   ├── stage4_agent/                 # отчёт Stage 4 (метрики гибрида на val)
│   │   └── agent_metrics.json
│   └── final_eval/                   # Stage 5B: final_metrics.json, error_matrix, fig
├── notebooks/                        # основной способ запуска этапов (см. § выше)
│   ├── eda.ipynb                     # Stage 0 — EDA, baseline parquet
│   ├── stage1_baseline.ipynb         # Stage 1 — TF-IDF референс
│   ├── stage2_bert_finetune.ipynb    # Stage 2 — RuModernBERT + calibration (Colab/GPU)
│   ├── stage3_error_analysis.ipynb   # Stage 3 — порог, taxonomy, val_merged_preds
│   ├── stage4_agent.ipynb            # Stage 4 — гибридная система на val
│   ├── stage5a_bert_inference.ipynb  # Stage 5A — BERT на eval (Colab/GPU)
│   └── stage5b_agent_loop.ipynb      # Stage 5B — гибридная система на eval
├── utils/
│   ├── config.py
│   ├── data_loader.py
│   ├── calibration.py
│   ├── metrics.py
│   ├── predict.py
│   ├── bert_routing.py
│   ├── agent_import.py
│   ├── langchain_compat.py
│   ├── stage1_baseline.py
│   ├── stage2_bert.py
│   ├── stage3_error_analysis.py
│   ├── stage4_agent.py
│   ├── stage5_bert.py
│   ├── stage5_agent.py
│   └── stage5_eval.py
├── scripts/
│   ├── run_stage1.py … run_stage5.py
├── agent/                            # код LLM-агента (LangGraph), не вся гибридная система
│   ├── graph.py, nodes.py, state.py, llm.py, search.py, prompts.py
│   └── search_cache/
├── environment.yml
├── known_project_limits.txt
└── .env                              # ENV_PATH
```

---

## Подготовка текста — `utils/data_loader.py`

### `build_combined_text(row)` — Stage 1 (TF-IDF)

Одна строка: `Query: … Address: … Name: … Rubric: … Reviews: … Pricelist: …`
Используй `attach_combined_text(df)` после загрузки parquet.

### `make_org_text(df)` — Stage 2, 4, 5 (cross-encoder sequence B)

Без запроса; поля через ` | `:

`COL_NAME | COL_ADDRESS | COL_RUBRIC | COL_REVIEWS | COL_PRICELIST`

Прайслист **включён** и в BERT, и в LLM — для запросов с указанием конкретной услуги\товара.

```python
from utils.data_loader import attach_org_text
# или: df[COL_ORG_TEXT] = make_org_text(df)
val_df = attach_org_text(pd.read_parquet(...))
```

### Токенизация cross-encoder

`[CLS] COL_QUERY [SEP] COL_ORG_TEXT [SEP]`, `truncation='only_second'` — запрос не обрезается.

- **Обучение / Trainer.predict:** `padding='max_length'`, `max_length=BERT_MAX_LENGTH`
- **Inference (`utils/predict.py`):** dynamic padding по батчу, `max_length` из `training_args.json` или `BERT_MAX_LENGTH`

---

## Stage 1 — TF-IDF референс

**Модуль:** `utils/stage1_baseline.py`  
**Запуск:** `python scripts/run_stage1.py`  
**Время:** ~20 мин (CPU)  
**Цель:** нижний bound; **не** участвует в гибридном пайплайне (Stages 4–5).

### Поведение

1. Загрузить `train_baseline.parquet`, `val_baseline.parquet`
2. `attach_combined_text()`
3. `Pipeline(TfidfVectorizer(max_features=50_000, ngram_range=(1,2), sublinear_tf=True, min_df=3) + LogisticRegression(...))`
4. Сохранить только `reports/stage1_baseline/metrics.json` (модель и preds на диск **не** пишутся)

**Ожидаемо:** val accuracy ~0.63–0.65, macro-F1 ~0.62–0.64.

---

## Stage 2 — Fine-tune RuModernBERT + Temperature Scaling

**Модуль:** `utils/stage2_bert.py`  
**Запуск:** `python scripts/run_stage2.py` [опции]  
**Среда:** GPU рекомендуется (Colab); ноутбук `notebooks/stage2_bert_finetune.ipynb` — опционально.

### CLI (`scripts/run_stage2.py`)

| Флаг | Назначение |
|------|------------|
| `--epochs`, `--batch-size`, `--lr` | Гиперпараметры |
| `--no-fp16` | Отключить mixed precision |
| `--skip-train` | Только inference (нужен `best_checkpoint`) |
| `--skip-calibration` | Сырые softmax в parquet (не для production) |
| `--early-stopping-patience` | По умолчанию 1; `0` или `--no-early-stopping` — все эпохи |
| `--resume-checkpoint`, `--no-auto-resume` | Продолжение обучения |

### Пайплайн `run_stage2()`

1. `load_stage2_splits()` → train / val / OOD (`rel_minus_baseline.parquet`) + `COL_ORG_TEXT`
2. `validate_train_data()` — метки `{0,1}`, train > 20k строк
3. `log_token_length_stats()` — p50/p90/p95, доля обрезки при `BERT_MAX_LENGTH`
4. Fine-tune: `num_train_epochs=3`, `batch_size=16`, `lr=2e-5`, `eval_strategy='epoch'`, `metric_for_best_model='accuracy'`, early stopping (опционально)
5. `save_best_checkpoint()` → `models/bert/best_checkpoint/`, `training_args.json`
6. Temperature scaling на val logits → `calibration.json`, reliability diagrams
7. Parquet: `bert_val_preds.parquet`, `bert_ood_preds.parquet`
8. `reports/stage2_bert/metrics.json`

### Колонки в `bert_val_preds.parquet`

Все `KEEP_COLS` + `org_text` +:

| Колонка | Описание |
|---------|----------|
| `bert_pred` | argmax логитов (не меняется от T) |
| `bert_proba1` | **калиброванная** P(class=1) после temperature scaling |
| `bert_correct` | `bert_pred == label` |

**Ожидаемо:** val accuracy ~0.74–0.80, macro-F1 ~0.73–0.79.

### Калибровка (`utils/calibration.py`)

- `fit_temperature` (LBFGS), `apply_temperature`, ECE, reliability plots
- `argmax(logits / T) == argmax(logits)` при T > 0

---

## Stage 3 — Error Analysis

**Модуль:** `utils/stage3_error_analysis.py`  
**Запуск:** `python scripts/run_stage3.py` [`--threshold 0.75`]  
**Время:** ~2–3 ч (ручная taxonomy в ноутбуке)

### Результаты Stage 3

1. **`CONFIDENCE_THRESHOLD`** — первый порог с coverage ≥ 70% и accuracy на уверенных ≥ overall + 5 pp; иначе `CONFIDENCE_THRESHOLD_DEFAULT`
2. **Taxonomy ошибок** — для решения «нужен ли Tavily»; см. ниже § «Ручная разметка ошибок»
3. **`comparison.json`** — порог, T, taxonomy, `searchable_share`, `agent_architecture`, метрики TF-IDF/BERT, `low_confidence`, `ood`

### Ручная разметка ошибок (taxonomy)

Размечаем **не все** ошибки BERT на val, а репрезентативную выборку (~**96** строк в `bert_errors_sample.csv`).

**Почему 96:** цель — оценить **доли типов ошибок** (нужен ли поиск и т.д.), а не пересмотреть каждый промах. Размер задаётся формулой Кохрана (`min_sample_size_for_taxonomy()` в ноутбуке Stage 3): при доле \(p \approx 0{,}30\), доверии **95%** и допуске **±10 pp** получается \(n \approx 96\) — достаточно для решения: гибрид **с Tavily** или **только LLM** (`agent_architecture` в comparison.json).

**Как отбираем строки** (из `errors = bert_val[~bert_correct]`):

1. **Приоритет:** ошибки с высокой уверенностью модели (`bert_max_proba >= CONFIDENCE_THRESHOLD_DEFAULT`, 0.75) — модель уверена, но неправа; для taxonomy они информативнее случайных low-conf.
2. **Стратификация:** по каждому классу `TARGET` (0/1) добираем до `n // 2` примеров: сначала high-conf по классу, остаток — случайно из оставшихся ошибок этого класса.
3. Перемешивание → CSV для ручной разметки категорий (`requires_search`, `hard_semantic`, `fact_verification`, `label_noise`, `other`).

Порог **0.75** в отборе sample — ориентир «уверенной ошибки»; финальный `CONFIDENCE_THRESHOLD` для routing может отличаться (§3.2). После разметки счётчики передаются в `comparison.json` / `run_stage3(error_taxonomy=...)`.

### Артефакты

| Файл | Содержимое |
|------|------------|
| `predictions/val_merged_preds.parquet` | **Val + BERT preds + `bert_max_proba`** (вход Stage 4) |
| `reports/stage3_error_analysis/comparison.json` | Порог, `agent_architecture` (гибрид с/без Tavily), сравнение моделей |
| `fig_accuracy_coverage.png` | Accuracy vs coverage, coverage vs threshold |
| `bert_errors_sample.csv` | ~96 строк для ручной taxonomy (см. § выше) |

### `val_merged_preds.parquet`

Это **обогащённый validation**:

- Исходные колонки parquet + `org_text`
- `bert_pred`, `bert_proba1` (калиброванные), `bert_correct`
- **`bert_max_proba`** = `max(bert_proba1, 1 - bert_proba1)` — для routing

Создаётся в `run_stage3()` из `bert_val_preds` через `enrich_bert_predictions()`.

### Уверенность

`bert_proba1` — temperature-scaled. Порог применяется к **`bert_max_proba`**, не к сырому `bert_proba1`.

### OOD

Sanity check на `bert_ood_preds.parquet` (медиана proba ~0.5, распределение pred).

---

## Stage 4 — Гибридная система на val (BERT routing + LLM-агент)

**Модули:** `utils/stage4_agent.py` (оркестрация гибрида), `agent/` (LLM-агент)  
**Запуск:** `python scripts/run_stage4.py` [`--sample 50`] [`--sleep 0.2`] [`--model ...`]

### Архитектура (LangGraph)

```
Вход (строка val_merged / dict)
        │
        ▼
[bert_route]  max(bert_proba1, 1-p) vs CONFIDENCE_THRESHOLD (из comparison.json)
        │
    ┌───┴────────────────┐
  ≥ threshold          < threshold
    │                      │
    ▼                      ▼
  END                  [decide_search]  LLM
final_pred=bert_pred         │
routed_to=bert        SEARCH: … / NO_SEARCH_NEEDED
                         │              │
                         ▼              │
                    [search] Tavily     │
                         └──────┬───────┘
                                ▼
                          [classify] → 0/1
                                ▼
                               END
                         routed_to=llm
```

**Контекст LLM** (`agent/prompts.format_org_context`):  
`COL_QUERY`, `COL_NAME`, `COL_ADDRESS`, `COL_RUBRIC`, `COL_REVIEWS`, `COL_PRICELIST`.

### Зависимости

```bash
# environment.yml или pip:
# openai, tavily-python, python-dotenv, langgraph==0.4.8, langchain-core
```

### `.env` (ENV_PATH)

```bash
VSEGPT_API_KEY=...          # или OPENAI_API_KEY (fallback в agent/llm.py)
TAVILY_API_KEY=tvly-...
AGENT_LLM_MODEL=deepseek/deepseek-v4-flash-alt
AGENT_USE_CACHE=true
```

Импорт агента: `utils/agent_import.py` (проверка полноты `agent/`, `langchain_compat`).

### Вход Stage 4

Предпочтительно `val_merged_preds.parquet` (Stage 3).  
Fallback: `bert_val_preds.parquet` + `enrich_bert_predictions()`.

Порог: `utils/bert_routing.load_confidence_threshold()` → `comparison.json`.

### Выход Stage 4

| Файл | Описание |
|------|----------|
| `agent_low_conf_preds.parquet` | Только low-conf: прогон **LLM-агента** (final_pred, routed_to, search_used, tokens, latency) |
| `hybrid_val_preds.parquet` | Весь val: **гибридная система** (high-conf → BERT, low-conf → LLM-агент) |
| `reports/stage4_agent/agent_metrics.json` | Метрики **гибрида** vs BERT-only, доля поиска, стоимость, latency |

### Метрики гибридной системы на val

High-confidence: `final_pred = bert_pred` (маршрут BERT, без LLM).  
Low-confidence: `final_pred` от **LLM-агента**.  
Accuracy и macro-F1 на объединении (`utils/metrics.eval_core`).

---

## Stage 5 — Финальная оценка на eval (LOCKED)

**Не открывать `eval_baseline.parquet` до Stage 5.**

Разбит на две части (можно на разных машинах):

### Stage 5A — BERT inference (`utils/stage5_bert.py`)

```bash
python scripts/run_stage5.py --bert-only
```

1. `load_eval_df()` — `eval_baseline.parquet` + `COL_ORG_TEXT`
2. `predict_bert()` — калиброванные proba, T из `calibration.json`
3. → `predictions/bert_eval_preds.parquet` (`bert_pred`, `bert_proba1`, `bert_max_proba`, …)

GPU/Colab удобен для 5A; **без** langchain.

### Stage 5B — Гибридная система на eval (`utils/stage5_agent.py`)

```bash
python scripts/run_stage5.py --agent-only
```

1. Читает `bert_eval_preds.parquet`
2. Low-confidence → `run_agent()` (LLM-агент); high-confidence → `final_pred = bert_pred` (BERT)
3. → `agent_eval_preds.parquet` (весь eval, гибридная система; имя файла историческое)
4. `reports/final_eval/final_metrics.json`, `error_matrix.json`, `fig_hybrid_vs_bert.png` (гибрид vs BERT-only)

### Полный прогон (отладка, одна машина)

```bash
python scripts/run_stage5.py [--sample N] [--sleep 0.1]
```

`utils/stage5_eval.run_stage5()` = 5A + 5B.

Порог и T — те же, что на val (зафиксированы в Stage 3 / Stage 2).

---

## `utils/metrics.py`

| Функция | Назначение |
|---------|------------|
| `eval_core` | accuracy + macro-F1 |
| `eval_subset` | метрики на подвыборке (пустая → NaN) |
| `eval_binary` | + classification_report + сохранение JSON |
| `error_matrix` | сравнение BERT-only vs **гибридной системы** на eval (Stage 5B) |

---

## `utils/bert_routing.py`

- `load_confidence_threshold()` — из `STAGE3_COMPARISON_PATH`
- `max_confidence_series()` — векторизованный `max(p, 1-p)`

---

## `utils/predict.py`

Единая точка inference для Stage 5A и внешних скриптов:  
`predict_bert(queries, org_texts)` → `{pred, proba1}` с temperature scaling.

---

## known_project_limits.txt

См. файл в корне проекта. Кратко:

| ID | Суть |
|----|------|
| LIMIT-01 | Temperature scaling на val (нет отдельного calibration split) |
| LIMIT-02 | Порог уверенности подобран на val |
| LIMIT-03 | Только temperature scaling (не Platt/isotonic на val) |
| LIMIT-04 | `BERT_MAX_LENGTH=1024`; при OOM — меньше batch / gradient accumulation |
| LIMIT-05 | LLM недетерминирован; кэш Tavily и LLM снижают разброс |

---

## Чеклист по дням

| День | Этап | Команда / артефакт |
|------|------|-------------------|
| 1 | Данные + `make_org_text` в коде | parquet в `data/processed/` |
| 2 | Stage 1 | `python scripts/run_stage1.py` |
| 3 | Stage 2 | `python scripts/run_stage2.py` → checkpoint, calibration, bert_*_preds |
| 4 | Stage 3 | `python scripts/run_stage3.py` → comparison.json, val_merged_preds |
| 5 | Taxonomy | Разметить ~96 строк (`bert_errors_sample.csv`, § «Ручная разметка ошибок») |
| 6 | Stage 4 (гибрид) | `python scripts/run_stage4.py` |
| 7 | Stage 5A | `python scripts/run_stage5.py --bert-only` |
| 8 | Stage 5B (гибрид) | `python scripts/run_stage5.py --agent-only` |
| 9 | Отчёт | Собрать метрики из `reports/` и `final_eval/` |

---

## Ключевые правила

1. **`EVAL_BASELINE_PATH` не открывать до Stage 5.**
2. **`make_org_text()` / `attach_org_text()` одинаково** в Stage 2, 4, 5A; контекст BERT и агента согласован (включая pricelist).
3. **`truncation='only_second'`** — запрос не обрезается.
4. **T подбирается на val один раз** → `BERT_CALIBRATION_PATH`; применяется к val, OOD, eval.
5. **Все `bert_proba1` в parquet — калиброванные** (кроме явного `--skip-calibration`).
6. **Порог фиксируется в Stage 3** (`comparison.json`); не менять после просмотра eval.
7. **Routing по `bert_max_proba`**, порог из `comparison.json`.
8. **После каждого этапа сохранять артефакты** (особенно в Colab).
9. **Primary: accuracy; secondary: macro-F1** на всех этапах.
10. **Решение про поиск** — по taxonomy в Stage 3 (`searchable_share` > 30% → LLM + Tavily).

---

## История версий (кратко)

| Версия | Суть |
|--------|------|
| v4 | Cross-encoder вместо single-string BERT| v4 | Cross-encoder вместо single-string BERT |
| v5 | Temperature scaling |
| v6 | `utils/calibration.py` |
| v7 | Parquet без combined_text; reviews + pricelist в KEEP_COLS |
| v8 | Pricelist в org_text и агенте |
| v9 | RuModernBERT-base; черновик с `RUBERT_*`, MAX_LENGTH=512, Colab-first |
| **v10** | **Синхронизация с кодом:** `BERT_*`, `models/bert/`, MAX_LENGTH=1024, scripts+utils, val_merged/hybrid preds, Stage 5A/5B |
