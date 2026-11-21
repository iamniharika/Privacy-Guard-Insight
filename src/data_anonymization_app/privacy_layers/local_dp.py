from __future__ import annotations

import math
import secrets
from collections.abc import Sequence
from typing import Any

import pandas as pd


def randomized_response_truth_probability(epsilon: float, category_count: int) -> float:
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number.")
    if category_count < 2:
        raise ValueError("Randomized response requires at least two categories.")
    return 1.0 / (1.0 + (category_count - 1) * math.exp(-epsilon))


def _is_missing(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        raise ValueError("Randomized response supports scalar categorical values only.")
    missing = pd.isna(value)
    return bool(missing)


def _same_category(left: Any, right: Any) -> bool:
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def apply_randomized_response(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply per-column generalized randomized response using OS-backed randomness."""
    selected_columns = list(dict.fromkeys(columns))
    if not selected_columns:
        raise ValueError("Select at least one categorical column.")
    missing_columns = [column for column in selected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns: {', '.join(missing_columns)}")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number.")

    rng = secrets.SystemRandom()
    protected = dataframe.copy(deep=True)
    column_results: dict[str, dict[str, Any]] = {}

    for column in selected_columns:
        values = dataframe[column].tolist()
        categories = list(pd.unique(dataframe[column]))
        category_count = len(categories)
        truth_probability = randomized_response_truth_probability(epsilon, category_count)
        randomized_values: list[Any] = []

        for value in values:
            original_index = next(
                index
                for index, category in enumerate(categories)
                if _same_category(category, value)
            )
            if rng.random() < truth_probability:
                randomized_values.append(value)
                continue

            alternatives = [
                category
                for index, category in enumerate(categories)
                if index != original_index
            ]
            randomized_values.append(rng.choice(alternatives))

        protected[column] = pd.Series(
            randomized_values,
            index=dataframe.index,
            dtype=object,
        )
        column_results[column] = {
            "category_count": category_count,
            "truth_probability": truth_probability,
        }

    return protected, {
        "mechanism": "generalized randomized response",
        "epsilon_per_column": float(epsilon),
        "composed_epsilon_upper_bound": float(epsilon * len(selected_columns)),
        "columns": column_results,
        "row_count": int(len(dataframe)),
    }
