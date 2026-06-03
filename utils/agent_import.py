"""Import project agent.graph.run_agent with a clear error if agent/ is incomplete."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_DIR = _PROJECT_ROOT / "agent"
_REQUIRED_AGENT_FILES: List[str] = [
    "__init__.py",
    "graph.py",
    "nodes.py",
    "state.py",
    "llm.py",
    "search.py",
    "prompts.py",
]


def _missing_agent_files() -> List[str]:
    return [name for name in _REQUIRED_AGENT_FILES if not (_AGENT_DIR / name).is_file()]


def ensure_project_agent_on_path() -> Path:
    """Put project root first on sys.path and evict a foreign ``agent`` package."""
    root = str(_PROJECT_ROOT)
    if sys.path[:1] != [root]:
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    for key in list(sys.modules):
        if key == "agent" or key.startswith("agent."):
            del sys.modules[key]
    return _PROJECT_ROOT


def import_run_agent() -> Callable[..., dict]:
    """Return ``run_agent`` from ``agent.graph``, or raise with sync instructions."""
    missing = _missing_agent_files()
    if missing:
        raise ModuleNotFoundError(
            "No module named 'agent.graph': папка agent/ на проекте неполная. "
            f"Отсутствуют: {missing}. "
            f"Ожидаемый каталог: {_AGENT_DIR}. "
            "Скопируйте все файлы agent/ из репозитория на Google Drive (Colab) "
            "и перезапустите ячейку с путями."
        )
    ensure_project_agent_on_path()
    from utils.langchain_compat import apply_langchain_compat

    apply_langchain_compat()
    from agent.graph import run_agent

    return run_agent
