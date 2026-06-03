"""Optional shims for langchain / langgraph version mismatches (Stage 4–5B only)."""


def apply_langchain_compat() -> None:
    """Apply compatibility patches before importing ``agent.graph``."""
    return
