import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from config_local import PROJECT_ROOT as _root
    BASE_DIR = _root
except ImportError:
    # Fallback: config.py лежит в utils/, поднимаемся на уровень выше
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Пути к данным, env-файлу
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.jsonl")
ENV_PATH = os.path.join(BASE_DIR, ".env")

RANDOM_STATE = 42

# Исходные колонки
COL_ID = 'permalink'
COL_QUERY = 'text'
COL_NAME = 'name'
COL_ADDRESS = 'address'
COL_RUBRIC = 'normalized_main_rubric_name_ru'
COL_RELEVANCE = 'relevance'
COL_REVIEWS = 'reviews_summarized'
COL_PRICELIST = 'prices_summarized'

# Производные признаки      
COL_COMBINED_TEXT    = 'combined_text'
COL_ORG_TEXT         = 'org_text'   # sequence B for cross-encoder (Stage 2+)

LABEL_NAMES_BINARY   = {0: 'irrelevant', 1: 'relevant'}
LABEL_MAP_BINARY     = {0.0: 0,
                        0.1: 0,   # artefact of Exp C, soft negative → 0
                        1.0: 1}

TARGET = 'label'   # целевая переменная в baseline artefacts — int {0,1}
                   # COL_RELEVANCE для оригинальных меток {0.0, 0.1, 1.0}

TARGET_NAMES_REPORT = [v.capitalize() for v in LABEL_NAMES_BINARY.values()]

# Колонки предсказаний (схема parquet между этапами)
COL_TFIDF_PRED = 'tfidf_pred'
COL_TFIDF_PROBA1 = 'tfidf_proba1'
COL_TFIDF_CORRECT = 'tfidf_correct'
COL_TFIDF_MAX_PROBA = 'tfidf_max_proba'
# Stage 2+ cross-encoder (поля bert_* в parquet)
COL_BERT_PRED = 'bert_pred'
COL_BERT_PROBA1 = 'bert_proba1'
COL_BERT_CORRECT = 'bert_correct'
COL_BERT_MAX_PROBA = 'bert_max_proba'
COL_FINAL_PRED = 'final_pred'
COL_ROUTED_TO = 'routed_to'
COL_SEARCH_USED = 'search_used'

# --- Директории ---
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR    = os.path.join(BASE_DIR, "predictions")
AGENT_DIR	   = os.path.join(BASE_DIR, "agent")
REPORTS_DIR        = os.path.join(BASE_DIR, "reports")
EDA_REPORTS_DIR    = os.path.join(REPORTS_DIR, "eda_reports")
STAGE1_REPORTS_DIR = os.path.join(REPORTS_DIR, "stage1_baseline")
STAGE2_REPORTS_DIR = os.path.join(REPORTS_DIR, "stage2_bert")
STAGE3_REPORTS_DIR = os.path.join(REPORTS_DIR, "stage3_error_analysis")
STAGE4_REPORTS_DIR = os.path.join(REPORTS_DIR, "stage4_agent")

TFIDF_BASELINE_DIR = os.path.join(MODELS_DIR, "tfidf_baseline")

# Stage 2 — RuModernBERT cross-encoder (pipeline id: bert; не cointegrated/rubert-tiny2)
BERT_MODEL_NAME = "deepvk/RuModernBERT-base"
BERT_MAX_LENGTH = 1024
BERT_DIR = os.path.join(MODELS_DIR, "bert")
BERT_CHECKPOINTS_DIR = os.path.join(BERT_DIR, "checkpoints")
BERT_BEST_CHECKPOINT_DIR = os.path.join(BERT_DIR, "best_checkpoint")
BERT_TRAINING_ARGS_PATH = os.path.join(BERT_DIR, "training_args.json")
BERT_CALIBRATION_PATH = os.path.join(BERT_DIR, "calibration.json")
STAGE2_RELIABILITY_BEFORE_PATH = os.path.join(
    STAGE2_REPORTS_DIR, "reliability_before_scaling.png"
)
STAGE2_RELIABILITY_AFTER_PATH = os.path.join(
    STAGE2_REPORTS_DIR, "reliability_after_scaling.png"
)
STAGE2_RELIABILITY_DIAGRAM_PATH = STAGE2_RELIABILITY_AFTER_PATH

# Предсказания BERT (parquet)
BERT_VAL_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "bert_val_preds.parquet")
BERT_OOD_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "bert_ood_preds.parquet")
BERT_EVAL_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "bert_eval_preds.parquet")
VAL_MERGED_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "val_merged_preds.parquet")
AGENT_LOW_CONF_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "agent_low_conf_preds.parquet")
HYBRID_VAL_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "hybrid_val_preds.parquet")
AGENT_EVAL_PREDS_PATH = os.path.join(PREDICTIONS_DIR, "agent_eval_preds.parquet")
EVAL_BASELINE_PATH = os.path.join(PROCESSED_DATA_DIR, "eval_baseline.parquet")
CONFIDENCE_THRESHOLD_DEFAULT = 0.75

STAGE3_COMPARISON_PATH = os.path.join(STAGE3_REPORTS_DIR, "comparison.json")
STAGE3_ACCURACY_COVERAGE_FIG_PATH = os.path.join(
    STAGE3_REPORTS_DIR, "fig_accuracy_coverage.png"
)
STAGE3_BERT_ERRORS_SAMPLE_PATH = os.path.join(
    STAGE3_REPORTS_DIR, "bert_errors_sample.csv"
)
STAGE4_REPORT_PATH = os.path.join(STAGE4_REPORTS_DIR, "agent_metrics.json")
FINAL_EVAL_DIR = os.path.join(REPORTS_DIR, "final_eval")
FINAL_METRICS_PATH = os.path.join(FINAL_EVAL_DIR, "final_metrics.json")
ERROR_MATRIX_PATH = os.path.join(FINAL_EVAL_DIR, "error_matrix.json")
ERROR_MATRIX_FIG_PATH = os.path.join(FINAL_EVAL_DIR, "fig_hybrid_vs_bert.png")

# --- Stage 0 EDA ---
KEEP_COLS = [COL_ID, COL_QUERY, COL_NAME, COL_ADDRESS, COL_RUBRIC, TARGET, COL_RELEVANCE, COL_REVIEWS, COL_PRICELIST]

# --- Stage 4 agent ---
SEARCH_CACHE_DIR = os.path.join(AGENT_DIR, "search_cache")
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "deepseek/deepseek-v4-flash-alt")
VSEGPT_BASE_URL = "https://api.vsegpt.ru/v1"
AGENT_USE_CACHE = os.getenv("AGENT_USE_CACHE", "true").lower() == "true"

# Создание необходимых директорий
def create_directories():
    """
    Создает необходимые директории
    """
    dirs_to_create = [
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        MODELS_DIR,
        TFIDF_BASELINE_DIR,
        BERT_DIR,
        PREDICTIONS_DIR,
        AGENT_DIR,
        REPORTS_DIR,
        EDA_REPORTS_DIR,
        STAGE1_REPORTS_DIR,
        STAGE2_REPORTS_DIR,
        STAGE3_REPORTS_DIR,
        STAGE4_REPORTS_DIR,
        FINAL_EVAL_DIR,
        SEARCH_CACHE_DIR,
    ]
    
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Не удалось создать директорию {dir_path}: {e}")