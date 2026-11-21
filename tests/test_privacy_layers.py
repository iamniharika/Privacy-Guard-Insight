from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data_anonymization_app.privacy_layers.local_dp import (
    apply_randomized_response,
    randomized_response_truth_probability,
)
from data_anonymization_app.privacy_layers.model_dp import (
    train_private_adversarial_model,
)


def test_randomized_response_applies_per_column_local_privacy() -> None:
    dataframe = pd.DataFrame(
        {
            "consent_flag": [True, False, True, False],
            "unchanged": ["a", "b", "c", "d"],
        }
    )

    protected, report = apply_randomized_response(
        dataframe,
        ["consent_flag"],
        epsilon=1.0,
    )

    assert set(protected["consent_flag"]) <= {True, False}
    assert protected["unchanged"].equals(dataframe["unchanged"])
    assert report["epsilon_per_column"] == 1.0
    assert report["composed_epsilon_upper_bound"] == 1.0
    assert report["columns"]["consent_flag"]["truth_probability"] == pytest.approx(
        math.e / (math.e + 1)
    )


def test_randomized_response_counts_missing_values_as_a_category() -> None:
    probability = randomized_response_truth_probability(epsilon=0.0, category_count=3)
    protected, report = apply_randomized_response(
        pd.DataFrame({"flag": ["yes", "no", None]}),
        ["flag"],
        epsilon=0.0,
    )

    assert probability == pytest.approx(1 / 3)
    assert report["columns"]["flag"]["category_count"] == 3
    assert set(protected["flag"]) <= {"yes", "no", None}


def test_randomized_response_rejects_non_categorical_or_unknown_columns() -> None:
    dataframe = pd.DataFrame({"constant": ["only"]})

    with pytest.raises(ValueError, match="at least two categories"):
        apply_randomized_response(dataframe, ["constant"], epsilon=1.0)
    with pytest.raises(ValueError, match="Unknown columns"):
        apply_randomized_response(dataframe, ["missing"], epsilon=1.0)


def test_private_adversarial_training_reports_accounted_budget() -> None:
    rng = np.random.default_rng(42)
    row_count = 100
    feature = rng.normal(size=row_count)
    dataframe = pd.DataFrame(
        {
            "feature": feature,
            "outcome": (feature > 0).astype(int),
            "sensitive": (rng.random(row_count) < 0.5).astype(int),
        }
    )

    result = train_private_adversarial_model(
        dataframe,
        target_column="outcome",
        sensitive_column="sensitive",
        feature_columns=["feature"],
        target_epsilon=3.0,
        delta=1e-5,
        epochs=1,
        batch_size=16,
        hidden_units=4,
        random_state=5,
    )

    assert result["achieved_epsilon"] <= result["target_epsilon"]
    assert result["delta"] == 1e-5
    assert result["training_steps"] > 0
    assert 0.0 <= result["utility_accuracy"] <= 1.0
    assert 0.0 <= result["sensitive_adversary_accuracy"] <= 1.0
    assert result["validation_metrics_are_not_private"] is True
    assert result["accountant"].startswith("Poisson-sampled Gaussian RDP")


def test_private_adversarial_training_rejects_invalid_class_and_budget_inputs() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": np.arange(50, dtype=float),
            "outcome": [1] * 50,
            "sensitive": [0, 1] * 25,
        }
    )

    with pytest.raises(ValueError, match="exactly two classes"):
        train_private_adversarial_model(
            dataframe,
            target_column="outcome",
            sensitive_column="sensitive",
            feature_columns=["feature"],
        )

    dataframe["outcome"] = [0, 1] * 25
    with pytest.raises(ValueError, match="delta must be smaller"):
        train_private_adversarial_model(
            dataframe,
            target_column="outcome",
            sensitive_column="sensitive",
            feature_columns=["feature"],
            delta=0.1,
        )
