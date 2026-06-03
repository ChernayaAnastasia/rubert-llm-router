# Yandex Maps — Binary Relevance Classification (ModernBERT + LLM Hybrid)

Бинарная классификация релевантности организаций широким запросам пользователей Яндекс Карт.  
Гибридная система: cross-encoder **RuModernBERT-base** обрабатывает уверенные случаи, LLM-агент (LangGraph + VseGPT) — неуверенные.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-green)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

### Основные результаты

**Eval (Stage 5, n = 4 558)**:
 
`BERT-only` accuracy: **0.771** macro-F1: **0.771**
 
`Hybrid` accuracy: **0.782** macro-F1: **0.781**
 
LLM-агент обработал 21.3% примеров (threshold = 0.68) и улучшил accuracy на low-confidence подмножестве с 0.573 до 0.618.


## Содержание

1. [Постановка задачи](#1-постановка-задачи)
2. [Архитектура системы](#2-архитектура-системы)
3. [Результаты](#3-результаты)
4. [Установка](#4-установка)
5. [Быстрый старт](#5-быстрый-старт)
6. [Данные](#6-данные)
7. [Этапы пайплайна](#7-этапы-пайплайна)
8. [Конфигурация](#8-конфигурация)
9. [Структура проекта](#9-структура-проекта)
10. [Известные ограничения](#10-известные-ограничения)
11. [Воспроизводимость](#11-воспроизводимость)
12. [Терминология](#12-терминология)


## 1. Постановка задачи

**Задача:** по паре (поисковый запрос, карточка организации) предсказать бинарную метку релевантности — релевантна (1) или нерелевантна (0).

**Данные:** 35 089 строк после удаления 5 шумных дубликатов по `query + permalink`. Исходные классы `relevance`: 0 — 41%, 0.1 — 13%, 1 — 46%. Исходные [данные]( https://disk.yandex.ru/d/6d5hFHvpAZjQdw) предоставлены компанией Яндекс в учебных целях и являются результатами асессорской разметки релевантности.

**Почему бинарная задача и почему класс 0.1 исключён.** Эксперименты A/B на TF-IDF (`notebooks/eda.ipynb`) показали, что класс 0.1 не отделяется линейной моделью: F1 ≈ random, равномерная путаница с обоими соседними классами. Эксперимент C подтвердил, что полное исключение 0.1 даёт лучший binary val accuracy (~0.639 против ~0.633 для стратегий переноса 0.1 → 0 или 0.1 → 1). Все дальнейшие этапы работают с метками `{0, 1}`.

**Сплиты:** 70 / 15 / 15, стратификация по метке, row-level:

|Сплит|Строк|Примечание|
|-|-|-|
|train|21 270||
|val|4 558|Используется для обучения, калибровки, подбора порога|
|eval|4 558|**LOCKED** — не открывать до Stage 5|
|OOD (`rel_minus`)|4 703|Sanity check распределения вероятностей|

**Метрики:** Accuracy (primary), macro-F1 (secondary) — зафиксированы на всех этапах.


## 2. Архитектура системы

Гибридная система состоит из двух компонентов:

* **BERT (cross-encoder)** — `deepvk/RuModernBERT-base`, дообученный как классификатор пар `[query, org_text]`. Обрабатывает все примеры и выдаёт калиброванные вероятности.
* **LLM-агент** (`agent/`, LangGraph + VseGPT) — подключается только для low-confidence примеров, при необходимости использует веб-поиск Tavily.

```
Входная пара (query, org_text)
            │
            ▼
     [BERT cross-encoder]
     bert_max_proba = max(p, 1-p)
            │
     ┌──────┴──────────────┐
  ≥ threshold           < threshold
  (высокая уверенность)  (низкая уверенность)
     │                       │
     ▼                       ▼
  final_pred = bert_pred  [LLM-агент]
  routed_to  = bert          │
                        [decide_search]
                         LLM решает:
                         нужен поиск?
                        ┌──────┴──────┐
                      Да             Нет
                        │              │
                        ▼              │
                   [Tavily search]     │
                        └──────┬───────┘
                               ▼
                          [classify] → 0 / 1
                          routed_to = llm
```

Порог `CONFIDENCE_THRESHOLD` подбирается в Stage 3 и фиксируется в `reports/stage3_error_analysis/comparison.json`. Решение о подключении Tavily принимается по результатам ручной taxonomy ошибок BERT: если `searchable_share` > 30% — используется LLM + Tavily, иначе — только LLM.

> **Важно:** `agent/` содержит только код LLM-агента (граф LangGraph, узлы, промпты). Оркестрация всей гибридной системы (routing BERT + вызов агента + сборка итогового parquet) реализована в `utils/stage4_agent.py` и `utils/stage5_agent.py`. Файлы `agent_*` в `predictions/` — выход LLM-агента или его цикл; `hybrid_*` — итог гибридной системы на всём сплите.

## 3. Результаты

Финальные метрики — `reports/final/final_metrics.json` и `reports/final_eval/error_matrix.json`.

### Основные метрики на eval (n = 4 558)

|Модель|Accuracy|Macro-F1|Примечание|
|-|-|-|-|
|TF-IDF + LR (Stage 1)|~0.63–0.65|~0.62–0.64|Lower bound, CPU|
|**BERT-only** (Stage 2)|**0.771**|**0.771**|RuModernBERT-base, threshold = 0.68|
|**Hybrid BERT + LLM** (Stage 5)|**0.782**|**0.781**|+1.1 pp accuracy vs BERT-only|

> TF-IDF не участвует в гибридном пайплайне и служит исключительно нижней границей.

### Routing и LLM-агент на eval

|Метрика|Значение|
|-|-|
|Confidence threshold|0.68|
|Доля low-confidence примеров|21.3% (971 / 4 558)|
|Accuracy агента на low-conf|**0.618**|
|Accuracy BERT на том же low-conf (baseline)|0.573|
|Прирост на low-conf|**+4.5 pp**|
|Доля примеров с Tavily-поиском|13.5% от low-conf|
|Parse errors (LLM)|0.7%|
|Общее время прогона агента|~159 мин|

### Error matrix: Hybrid vs BERT-only (n = 4 527)

|Категория|Примеров|Доля|
|-|-|-|
|Оба правы|3 326|73.5%|
|Только гибрид прав|**214**|**4.7%**|
|Только BERT прав|171|3.8%|
|Оба ошиблись|816|18.0%|

Гибрид исправляет на 43 примера больше, чем ломает (214 − 171). Оставшийся потенциал — в категории «оба ошиблись» (18%), которая требует улучшения самой модели или расширения контекста.

![Eval metrics и error matrix](reports/final_eval/fig_hybrid_vs_bert.png)


## 4. Установка

**Требования:** Python 3.10+, conda (рекомендуется), GPU для Stage 2 и Stage 5A (Google Colab или локально).

```bash
git clone <repo-url>
cd <repo>
conda env create -f environment.yml
conda activate <env-name>
```

Или через pip (ключевые зависимости):

```bash
pip install transformers torch openai tavily-python python-dotenv \\
            "langgraph==0.4.8" langchain-core pandas pyarrow scikit-learn
```

Файл `.env` в корне проекта (скопируйте из `.env.example`):

```bash
VSEGPT_API_KEY=...            # или OPENAI_API_KEY (fallback в agent/llm.py)
TAVILY_API_KEY=tvly-...       # нужен только если agent_architecture = with_tavily
AGENT_LLM_MODEL=deepseek/deepseek-v4-flash-alt
AGENT_USE_CACHE=true
```

> **GPU / Colab.** Stage 2 (fine-tune) и Stage 5A (BERT inference на eval) рекомендуется запускать на GPU. Ноутбуки `notebooks/stage2_bert_finetune.ipynb` и `notebooks/stage5a_bert_inference.ipynb` адаптированы для Colab. После прогона обязательно скачивайте артефакты (`models/bert/best_checkpoint/`, `predictions/bert_eval_preds.parquet`).

Нестандартное расположение проекта: создайте `utils/config_local.py` с переменной `PROJECT_ROOT`.

## 5. Быстрый старт

Этапы можно выполнять через скрипты или напрямую в ноутбуках `notebooks/stage*.ipynb` — логика и артефакты идентичны.

```bash
# Stage 1 — TF-IDF референс (~20 мин, CPU)
python scripts/run_stage1.py

# Stage 2 — fine-tune BERT + calibration (GPU/Colab)
python scripts/run_stage2.py --epochs 3 --batch-size 16 --lr 2e-5

# Stage 3 — error analysis, подбор порога (~2–3 ч с ручной разметкой)
python scripts/run_stage3.py --threshold 0.75

# Stage 4 — гибридная система на val
python scripts/run_stage4.py

# Stage 5 — финальная оценка на eval (LOCKED до этого момента)
python scripts/run_stage5.py --bert-only   # 5A: BERT inference (GPU)
python scripts/run_stage5.py --agent-only  # 5B: гибрид на eval
```

> **`eval_baseline.parquet` не открывать и не использовать до Stage 5.** Все решения (порог, температура) фиксируются на val.

## 6. Данные

### Исходные данные

Сырые данные в формате JSONL находятся в `data/raw/` (`DATA_PATH` в `utils/config.py`). EDA выполнена в `notebooks/eda.ipynb`, итоговые артефакты — в `data/processed/`.

### Схема parquet (`KEEP_COLS`)

|Колонка|Описание|
|-|-|
|`COL_ID`|Уникальный идентификатор объекта (организации) в системе Яндекс.Карт|
|`COL_QUERY`|Широкий (рубричный) запрос пользователя|
|`COL_NAME`|Название организации|
|`COL_ADDRESS`|Адрес|
|`COL_RUBRIC`|Рубрика|
|`TARGET`|Бинарная метка {0, 1}|
|`COL_RELEVANCE`|Исходная метка {0, 0.1, 1} (только для справки)|
|`COL_REVIEWS`|Отзывы (плейсхолдер `No reviews.` при отсутствии)|
|`COL_PRICELIST`|Прайс-лист (плейсхолдер `No pricelist.` при отсутствии, ~41% строк)|

Пропуски в `COL_REVIEWS` (~4%) и `COL_PRICELIST` (~41%) не отбрасываются: связь «пропуск ↔ класс» статистически слабая (Cramér's V < 0.1). Routing к LLM-агенту также не основан на наличии/отсутствии полей — accuracy при «оба поля пусты» не хуже полного контекста (Δ ≈ −0.02 по EDA).

### Формирование текстов на лету (`utils/data_loader.py`)

`COL_COMBINED_TEXT` и `COL_ORG_TEXT` **не хранятся** в parquet — пересобираются при загрузке:

```python
from utils.data_loader import attach_combined_text, attach_org_text

# Stage 1 (TF-IDF): "Query: … Address: … Name: … Rubric: … Reviews: … Pricelist: …"
df = attach_combined_text(pd.read_parquet("data/processed/train_baseline.parquet"))

# Stage 2, 4, 5 (cross-encoder, sequence B): "Name | Address | Rubric | Reviews | Pricelist"
df = attach_org_text(pd.read_parquet("data/processed/val_baseline.parquet"))
```

Прайс-лист включён в `org_text` и в контекст LLM — он важен для запросов с указанием конкретной услуги или товара. Токенизация cross-encoder: `[CLS] query [SEP] org_text [SEP]`, `truncation='only_second'` — запрос не обрезается.


## 7. Этапы пайплайна

### Stage 0 — EDA

**Ноутбук:** `notebooks/eda.ipynb`  
**Артефакты:** `data/processed/*.parquet`, `reports/eda_reports/`, `reports/eda_reports/table3_eda_summary.csv`

Разведочный анализ, удаление дубликатов, обоснование бинарной постановки, формирование сплитов, фиксация метрик и ключевых решений проекта.

### Stage 1 — TF-IDF референс

**Модуль:** `utils/stage1\_baseline.py` | **Ноутбук:** `notebooks/stage1\_baseline.ipynb`  
**Время:** \~20 мин (CPU)

```bash
python scripts/run\_stage1.py
```

TF-IDF (50k фичей, ngram 1–2, sublinear TF) + LogisticRegression на `combined\_text`. Служит нижней границей; в гибридный пайплайн не входит. Модель и предсказания на диск не пишутся.

**Выход:** `reports/stage1\_baseline/metrics.json` — val accuracy \~0.63–0.65, macro-F1 \~0.62–0.64.

\---

### Stage 2 — Fine-tune RuModernBERT + Temperature Scaling

**Модуль:** `utils/stage2\_bert.py` | **Ноутбук:** `notebooks/stage2\_bert\_finetune.ipynb` (Colab/GPU)

```bash
python scripts/run\_stage2.py \[--epochs 3] \[--batch-size 16] \[--lr 2e-5]
                             \[--no-fp16] \[--skip-train] \[--skip-calibration]
                             \[--early-stopping-patience 1] \[--no-early-stopping]
                             \[--resume-checkpoint PATH] \[--no-auto-resume]
```

Пайплайн:

1. Загрузка train / val / OOD, формирование `org\_text`
2. Проверка меток `{0,1}`, train > 20k строк, статистика длин токенов (p50/p90/p95)
3. Fine-tune: 3 эпохи, batch 16, lr 2e-5, `eval\_strategy='epoch'`, best checkpoint по accuracy
4. Temperature scaling на val logits → `models/bert/calibration.json`, reliability diagrams
5. Сохранение `bert\_val\_preds.parquet`, `bert\_ood\_preds.parquet`

Ключевые артефакты:

|Файл|Содержимое|
|-|-|
|`models/bert/best\_checkpoint/`|Веса модели|
|`models/bert/calibration.json`|Температура T, NLL и ECE до/после scaling|
|`predictions/bert\_val\_preds.parquet`|`bert\_pred`, `bert\_proba1` (калиброванные), `bert\_correct`|
|`predictions/bert\_ood\_preds.parquet`|То же для OOD-сплита|
|`reports/stage2\_bert/metrics.json`|Val accuracy \~0.74–0.80, macro-F1 \~0.73–0.79|
|`reports/stage2\_bert/reliability\_\*.png`|Reliability diagrams до/после calibration|

> Все `bert\_proba1` в parquet — калиброванные (если не передан `--skip-calibration`). Температура подбирается один раз и применяется к val, OOD и eval.

\---

### Stage 3 — Error Analysis и порог routing

**Модуль:** `utils/stage3\_error\_analysis.py` | **Ноутбук:** `notebooks/stage3\_error\_analysis.ipynb`  
**Время:** \~2–3 ч (включая ручную разметку)

```bash
python scripts/run\_stage3.py \[--threshold 0.75]
```

Этап решает два вопроса: (1) при каком пороге `bert\_max\_proba` передавать пример LLM-агенту; (2) нужен ли Tavily или достаточно только LLM.

**Подбор порога.** Первый порог с coverage ≥ 70% и accuracy на уверенных ≥ overall + 5 pp. Если такой не найден — используется `CONFIDENCE\_THRESHOLD\_DEFAULT = 0.75`. Routing всегда по `bert\_max\_proba = max(bert\_proba1, 1 − bert\_proba1)`, не по сырому `bert\_proba1`.

**Ручная taxonomy ошибок.** Из ошибок BERT на val отбирается \~96 строк (`bert\_errors\_sample.csv`): приоритет — high-confidence ошибки (`bert\_max\_proba ≥ 0.75`), стратификация по классу TARGET. Размер выборки по формуле Кохрана: p ≈ 0.30, доверие 95%, допуск ±10 pp. Категории разметки: `requires\_search`, `hard\_semantic`, `fact\_verification`, `label\_noise`, `other`. Если `searchable\_share` > 30% — `agent\_architecture = with\_tavily`, иначе `= llm\_only`.

Артефакты:

|Файл|Содержимое|
|-|-|
|`predictions/val\_merged\_preds.parquet`|Val + `bert\_pred`, `bert\_proba1`, `bert\_correct`, `bert\_max\_proba` → вход Stage 4|
|`reports/stage3\_error\_analysis/comparison.json`|Порог, T, taxonomy, `searchable\_share`, `agent\_architecture`, сравнение моделей|
|`reports/stage3\_error\_analysis/fig\_accuracy\_coverage.png`|Accuracy vs coverage, coverage vs threshold|
|`reports/stage3\_error\_analysis/bert\_errors\_sample.csv`|\~96 строк для ручной разметки|

\---

### Stage 4 — Гибридная система на val

**Модули:** `utils/stage4\_agent.py` (оркестрация) + `agent/` (LLM-агент) | **Ноутбук:** `notebooks/stage4\_agent.ipynb`

```bash
python scripts/run\_stage4.py \[--sample 50] \[--sleep 0.2] \[--model MODEL\_NAME]
```

Вход: `predictions/val\_merged\_preds.parquet` (предпочтительно) или `bert\_val\_preds.parquet` + `enrich\_bert\_predictions()`. Порог читается из `comparison.json` через `utils/bert\_routing.load\_confidence\_threshold()`.

Граф LangGraph (`agent/graph.py`): `bert\_route` → при высокой уверенности `END` (pred = bert\_pred), при низкой — `decide\_search` → опционально `\[search]` Tavily → `\[classify]` LLM → `END`. Контекст LLM: query, name, address, rubric, reviews, pricelist.

Артефакты:

|Файл|Содержимое|
|-|-|
|`predictions/agent\_low\_conf\_preds.parquet`|Только low-conf: final\_pred, routed\_to, search\_used, tokens, latency|
|`predictions/hybrid\_val\_preds.parquet`|Весь val: гибридная система|
|`reports/stage4\_agent/agent\_metrics.json`|Accuracy / F1 гибрида vs BERT-only, доля поиска, стоимость, latency|

\---

### Stage 5 — Финальная оценка на eval

> ⚠️ \*\*Открывать `eval\_baseline.parquet` только здесь.\*\* Порог и температура не меняются — они зафиксированы в Stages 2 и 3.

Этап разбит на две части, которые можно выполнять на разных машинах.

**Stage 5A — BERT inference** (`utils/stage5\_bert.py`, GPU/Colab):

```bash
python scripts/run\_stage5.py --bert-only
```

Читает `eval\_baseline.parquet`, строит `org\_text`, применяет модель с калиброванной температурой → `predictions/bert\_eval\_preds.parquet`.

**Stage 5B — гибридная система на eval** (`utils/stage5\_agent.py`):

```bash
python scripts/run\_stage5.py --agent-only
```

Читает `bert\_eval\_preds.parquet`, прогоняет low-confidence через LLM-агент → `predictions/agent\_eval\_preds.parquet` (исторически — выход гибридной системы на всём eval).

Полный прогон на одной машине (отладка):

```bash
python scripts/run\_stage5.py \[--sample N] \[--sleep 0.1]
```

Финальные артефакты:

|Файл|Содержимое|
|-|-|
|`predictions/agent\_eval\_preds.parquet`|Весь eval: гибридная система|
|`reports/final\_eval/final\_metrics.json`|Accuracy / F1: BERT-only и гибрид на eval|
|`reports/final\_eval/error\_matrix.json`|Сравнение BERT-only vs гибрид|
|`reports/final\_eval/fig\_hybrid\_vs\_bert.png`|Визуализация сравнения|

\---

## 8\. Конфигурация

Все константы и пути — в `utils/config.py`. Ключевые:

|Константа|Значение|Назначение|
|-|-|-|
|`BERT\_MODEL\_NAME`|`deepvk/RuModernBERT-base`|Base model|
|`BERT\_MAX\_LENGTH`|`1024`|Максимум токенов (поддерживается до 8192)|
|`CONFIDENCE\_THRESHOLD\_DEFAULT`|`0.75`|Fallback-порог (фактически использован 0.68, см. `comparison.json`)|
|`AGENT\_LLM\_MODEL`|из `.env`|Модель для LLM-агента|
|`AGENT\_USE\_CACHE`|`true`|Кэш ответов LLM и Tavily|
|`BERT\_CALIBRATION\_PATH`|`models/bert/calibration.json`|Температура T|
|`STAGE3\_COMPARISON\_PATH`|`reports/stage3\_error\_analysis/comparison.json`|Порог и architecture|

Для нестандартного расположения проекта создайте `utils/config\_local.py`:

```python
PROJECT\_ROOT = "/path/to/your/project"
```

\---

## 9\. Структура проекта

```
project\_root/
├── data/
│   ├── raw/                          # Исходные данные (JSONL)
│   └── processed/
│       ├── train\_baseline.parquet
│       ├── val\_baseline.parquet
│       ├── eval\_baseline.parquet     # LOCKED до Stage 5
│       └── rel\_minus\_baseline.parquet  # OOD (4 703 строки)
├── models/
│   └── bert/
│       ├── best\_checkpoint/          # Веса дообученной модели
│       ├── checkpoints/              # Промежуточные чекпоинты
│       ├── training\_args.json
│       └── calibration.json          # Температура T, ECE
├── predictions/                      # Parquet с предсказаниями по этапам
├── reports/
│   ├── eda\_reports/
│   ├── stage1\_baseline/
│   ├── stage2\_bert/
│   ├── stage3\_error\_analysis/
│   ├── stage4\_agent/
│   └── final\_eval/
├── notebooks/                        # Основной способ запуска
│   ├── eda.ipynb
│   ├── stage1\_baseline.ipynb
│   ├── stage2\_bert\_finetune.ipynb    # Colab/GPU
│   ├── stage3\_error\_analysis.ipynb
│   ├── stage4\_agent.ipynb
│   ├── stage5a\_bert\_inference.ipynb  # Colab/GPU
│   └── stage5b\_agent\_loop.ipynb
├── utils/
│   ├── config.py                     # Все константы и пути
│   ├── data\_loader.py                # build\_combined\_text, make\_org\_text
│   ├── calibration.py                # fit\_temperature, apply\_temperature, ECE
│   ├── metrics.py                    # eval\_core, eval\_binary, error\_matrix
│   ├── predict.py                    # predict\_bert() — единая точка inference
│   ├── bert\_routing.py               # load\_confidence\_threshold, max\_confidence\_series
│   ├── agent\_import.py               # Проверка окружения агента
│   ├── langchain\_compat.py
│   └── stage1\_baseline.py … stage5\_eval.py
├── scripts/
│   └── run\_stage1.py … run\_stage5.py
├── agent/                            # Только LLM-агент (LangGraph)
│   ├── graph.py                      # Граф LangGraph
│   ├── nodes.py                      # Узлы: bert\_route, decide\_search, search, classify
│   ├── state.py
│   ├── llm.py                        # VseGPT / OpenAI client
│   ├── search.py                     # Tavily wrapper
│   ├── prompts.py                    # format\_org\_context
│   └── search\_cache/
├── environment.yml
├── known\_project\_limits.txt
└── .env
```

Ключевые модули `utils/`:

* `data\_loader.py` — единственное место, где собираются `combined\_text` и `org\_text`; одинаковая логика на всех этапах.
* `predict.py` — единая точка BERT inference для Stage 5A и внешних скриптов: `predict\_bert(queries, org\_texts)` → `{pred, proba1}` с temperature scaling.
* `bert\_routing.py` — `load\_confidence\_threshold()` читает порог из `comparison.json`; `max\_confidence\_series()` — векторизованный `max(p, 1−p)`.
* `calibration.py` — `fit\_temperature` (LBFGS), `apply\_temperature`, ECE, reliability plots.
* `metrics.py` — `eval\_core` (accuracy + macro-F1), `eval\_binary` (+ classification\_report + JSON), `error\_matrix` (BERT-only vs гибрид).

\---

## 10\. Известные ограничения

Подробнее — [`known\_project\_limits.txt`](known_project_limits.txt).

**LIMIT-01 — Temperature scaling подобран на val.**  
Нет отдельного calibration split: val используется и для обучения модели, и для подбора температуры. Следствие: небольшой оптимистичный bias T на eval. Реализация: `utils/calibration.py` → `fit\_temperature` (LBFGS), результат в `BERT\_CALIBRATION\_PATH`.

**LIMIT-02 — `CONFIDENCE\_THRESHOLD` подобран на val.**  
Нет отдельного held-out set для подбора порога. Порог немного оптимистичен; реальную картину показывает Stage 5. Загрузка: `utils/bert\_routing.load\_confidence\_threshold()` → `comparison.json`.

**LIMIT-03 — Только temperature scaling.**  
Platt scaling и isotonic regression не применялись: оба метода рискуют переобучиться на val, который уже задействован для обучения модели. ECE после scaling сохранена в `calibration.json`.

**LIMIT-04 — `BERT\_MAX\_LENGTH = 1024`.**  
Обучение: `padding='max\_length'`; inference: dynamic padding до значения из `training\_args.json`. RuModernBERT поддерживает до 8 192 токенов. При OOM: уменьшить `batch\_size` и/или добавить `gradient\_accumulation\_steps`.

**LIMIT-05 — LLM-агент недетерминирован.**  
Результаты могут незначительно различаться между прогонами. Кэш Tavily (`SEARCH\_CACHE\_DIR`) и `AGENT\_USE\_CACHE=true` снижают вариативность при повторных запусках.

\---

## 11\. Воспроизводимость

* **BERT:** зафиксируйте `seed` в `training\_args` (Stage 2). Лучший чекпоинт сохраняется в `models/bert/best\_checkpoint/`.
* **Калибровка и порог:** однократно подбираются на val и фиксируются в `calibration.json` и `comparison.json`. Не пересчитываются на этапах 4–5.
* **LLM-агент:** включить `AGENT\_USE\_CACHE=true` — повторные запросы к Tavily и LLM возвращают кэшированные ответы из `agent/search\_cache/`.
* **Сплиты:** train/val/eval формируются в `notebooks/eda.ipynb` и далее не меняются.

\---

## 12\. Терминология

|Термин|Определение|
|-|-|
|**Гибридная система**|Полный пайплайн: уверенные примеры → BERT, low-confidence → LLM-агент (опционально Tavily). Метрики «hybrid» в отчётах — про эту систему целиком.|
|**LLM-агент**|Компонент в `agent/` (LangGraph + VseGPT). Обрабатывает только low-confidence примеры после routing.|
|`bert\_max\_proba`|`max(bert\_proba1, 1 − bert\_proba1)` — уверенность модели, по которой происходит routing.|
|`agent\_\*` в именах файлов|Исторически: выход LLM-агента или его цикл (в т.ч. `agent\_eval\_preds.parquet` = весь eval гибридной системы).|
|`hybrid\_\*` в именах файлов|Итог гибридной системы на всём сплите val.|
|`org\_text`|Текст организации без запроса: `Name \| Address \| Rubric \| Reviews \| Pricelist`. Sequence B в cross-encoder.|
|`combined\_text`|Полный текст для TF-IDF: `Query: … Address: … Name: … Rubric: … Reviews: … Pricelist: …`.|
|eval LOCKED|`eval\_baseline.parquet` не используется до Stage 5 — для честной финальной оценки.|



