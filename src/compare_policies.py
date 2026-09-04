"""Build the main policy comparison table."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .evaluate import summarize_policy


def build_comparison(policy_rows: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = [summarize_policy(name, rs) for name, rs in policy_rows.items()]
    return pd.DataFrame(rows)
