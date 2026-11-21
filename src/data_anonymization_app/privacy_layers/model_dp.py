from __future__ import annotations

import math
import secrets
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


_RDP_ORDERS = tuple(range(2, 65))
_MAX_FEATURES = 100


def _log_binomial(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _subsampled_gaussian_rdp(order: int, sample_rate: float, noise_multiplier: float) -> float:
    if sample_rate == 1.0:
        return order / (2.0 * noise_multiplier**2)

    log_q = math.log(sample_rate)
    log_one_minus_q = math.log1p(-sample_rate)
    terms = [
        _log_binomial(order, index)
        + (order - index) * log_one_minus_q
        + index * log_q
        + (index * index - index) / (2.0 * noise_multiplier**2)
        for index in range(order + 1)
    ]
    max_term = max(terms)
    log_a = max_term + math.log(sum(math.exp(term - max_term) for term in terms))
    return log_a / (order - 1)


def _epsilon_for_noise(
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    return min(
        steps * _subsampled_gaussian_rdp(order, sample_rate, noise_multiplier)
        + math.log(1.0 / delta) / (order - 1)
        for order in _RDP_ORDERS
    )


def _calibrate_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> tuple[float, float]:
    low = 0.0
    high = 1.0
    while _epsilon_for_noise(sample_rate, steps, high, delta) > target_epsilon:
        high *= 2.0
        if high > 1_000_000:
            raise ValueError("Unable to calibrate noise for the requested privacy budget.")

    for _ in range(64):
        middle = (low + high) / 2.0
        if _epsilon_for_noise(sample_rate, steps, middle, delta) > target_epsilon:
            low = middle
        else:
            high = middle

    achieved_epsilon = _epsilon_for_noise(sample_rate, steps, high, delta)
    return high, achieved_epsilon


def _binary_encode(series: pd.Series, column_name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"'{column_name}' contains missing labels; resolve them before model training.")
    categories = list(pd.unique(series))
    if len(categories) != 2:
        raise ValueError(f"'{column_name}' must contain exactly two classes.")
    categories.sort(key=lambda value: str(value))
    return (series.to_numpy() == categories[1]).astype(np.float64)


def _secure_gaussian_noise(
    shape: tuple[int, ...],
    standard_deviation: float,
    rng: secrets.SystemRandom,
) -> np.ndarray:
    count = math.prod(shape)
    values = [rng.gauss(0.0, standard_deviation) for _ in range(count)]
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities >= 0.5) == labels))


def train_private_adversarial_model(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    sensitive_column: str,
    feature_columns: Sequence[str],
    target_epsilon: float = 3.0,
    delta: float = 1e-5,
    epochs: int = 10,
    batch_size: int = 32,
    clip_norm: float = 1.0,
    adversary_strength: float = 1.0,
    learning_rate: float = 0.03,
    hidden_units: int = 16,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Train a binary utility/adversary network with clipped, Gaussian-noised DP-SGD.

    The accountant uses Poisson sampling and integer-order RDP for the Gaussian
    mechanism. The reported privacy budget applies to model parameters on the
    training split, conditional on the supplied feature schema and preprocessing.
    """
    columns = list(dict.fromkeys(feature_columns))
    if target_column == sensitive_column:
        raise ValueError("Choose different utility-target and sensitive columns.")
    required_columns = [target_column, sensitive_column, *columns]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns: {', '.join(missing_columns)}")
    if not columns:
        raise ValueError("Select at least one numeric feature.")
    if len(columns) > _MAX_FEATURES:
        raise ValueError(f"Select no more than {_MAX_FEATURES} numeric features.")
    if not math.isfinite(target_epsilon) or target_epsilon <= 0:
        raise ValueError("target_epsilon must be a finite positive number.")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError("epochs must be an integer from 1 to 100.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be a finite positive number.")
    if not math.isfinite(adversary_strength) or adversary_strength <= 0:
        raise ValueError("adversary_strength must be a finite positive number.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    if isinstance(hidden_units, bool) or not isinstance(hidden_units, int) or not 4 <= hidden_units <= 64:
        raise ValueError("hidden_units must be an integer from 4 to 64.")

    utility_labels = _binary_encode(dataframe[target_column], target_column)
    sensitive_labels = _binary_encode(dataframe[sensitive_column], sensitive_column)
    feature_frame = dataframe[columns].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = feature_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Numeric features must be finite after missing-value handling.")
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(row_norms, 1.0)

    row_count, feature_count = features.shape
    if row_count < 40:
        raise ValueError("At least 40 rows are required for a train/validation split.")

    split_rng = np.random.default_rng(random_state)
    row_order = split_rng.permutation(row_count)
    train_count = int(row_count * 0.8)
    train_indices = row_order[:train_count]
    validation_indices = row_order[train_count:]
    train_utility = utility_labels[train_indices]
    train_sensitive = sensitive_labels[train_indices]
    if np.unique(train_utility).size != 2 or np.unique(train_sensitive).size != 2:
        raise ValueError("Both classes of the target and sensitive field must occur in the training split.")

    if delta >= 1.0 / train_count:
        raise ValueError("delta must be smaller than 1 / training rows.")
    if batch_size > train_count:
        raise ValueError(f"batch_size must not exceed the {train_count} training rows.")

    sample_rate = batch_size / train_count
    steps_per_epoch = math.ceil(train_count / batch_size)
    steps = epochs * steps_per_epoch
    noise_multiplier, achieved_epsilon = _calibrate_noise_multiplier(
        target_epsilon,
        sample_rate,
        steps,
        delta,
    )

    model_rng = np.random.default_rng(random_state)
    weight_scale = 1.0 / math.sqrt(max(feature_count, hidden_units))
    parameters = [
        model_rng.normal(0.0, weight_scale, size=(feature_count, hidden_units)),
        np.zeros(hidden_units, dtype=np.float64),
        model_rng.normal(0.0, weight_scale, size=hidden_units),
        np.zeros(1, dtype=np.float64),
        model_rng.normal(0.0, weight_scale, size=hidden_units),
        np.zeros(1, dtype=np.float64),
    ]
    secure_rng = secrets.SystemRandom()
    expected_batch_size = sample_rate * train_count

    for _ in range(steps):
        sampled_indices = [
            index
            for index in range(train_count)
            if secure_rng.random() < sample_rate
        ]
        batch_features = features[train_indices[sampled_indices]]
        batch_utility = train_utility[sampled_indices, None]
        batch_sensitive = train_sensitive[sampled_indices, None]

        if sampled_indices:
            weights, hidden_bias, utility_weights, utility_bias, sensitive_weights, sensitive_bias = parameters
            hidden = np.tanh(batch_features @ weights + hidden_bias)
            utility_prediction = _sigmoid(hidden @ utility_weights[:, None] + utility_bias)
            sensitive_prediction = _sigmoid(hidden @ sensitive_weights[:, None] + sensitive_bias)
            utility_error = utility_prediction - batch_utility
            sensitive_error = sensitive_prediction - batch_sensitive

            reversed_hidden_error = (
                utility_error * utility_weights[None, :]
                - adversary_strength * sensitive_error * sensitive_weights[None, :]
            )
            hidden_error = reversed_hidden_error * (1.0 - hidden**2)
            per_example_gradients = [
                batch_features[:, :, None] * hidden_error[:, None, :],
                hidden_error,
                hidden * utility_error,
                utility_error,
                hidden * sensitive_error,
                sensitive_error,
            ]
            squared_norms = np.zeros(len(sampled_indices), dtype=np.float64)
            for gradient in per_example_gradients:
                squared_norms += np.sum(
                    gradient.reshape(len(sampled_indices), -1) ** 2,
                    axis=1,
                )
            clip_factors = np.minimum(
                1.0,
                clip_norm / np.maximum(np.sqrt(squared_norms), 1e-12),
            )
            clipped_sums = [
                np.sum(
                    gradient
                    * clip_factors.reshape(
                        (len(sampled_indices),) + (1,) * (gradient.ndim - 1)
                    ),
                    axis=0,
                )
                for gradient in per_example_gradients
            ]
        else:
            clipped_sums = [np.zeros_like(parameter) for parameter in parameters]

        noise_std = noise_multiplier * clip_norm
        for index, parameter in enumerate(parameters):
            noisy_average = (
                clipped_sums[index]
                + _secure_gaussian_noise(parameter.shape, noise_std, secure_rng)
            ) / expected_batch_size
            parameter -= learning_rate * noisy_average

    weights, hidden_bias, utility_weights, utility_bias, sensitive_weights, sensitive_bias = parameters
    validation_hidden = np.tanh(features[validation_indices] @ weights + hidden_bias)
    utility_probabilities = _sigmoid(
        validation_hidden @ utility_weights[:, None] + utility_bias
    ).ravel()
    sensitive_probabilities = _sigmoid(
        validation_hidden @ sensitive_weights[:, None] + sensitive_bias
    ).ravel()
    validation_utility = utility_labels[validation_indices]
    validation_sensitive = sensitive_labels[validation_indices]

    return {
        "target_epsilon": float(target_epsilon),
        "achieved_epsilon": float(achieved_epsilon),
        "delta": float(delta),
        "noise_multiplier": float(noise_multiplier),
        "clip_norm": float(clip_norm),
        "sample_rate": float(sample_rate),
        "training_steps": int(steps),
        "train_rows": int(train_count),
        "validation_rows": int(len(validation_indices)),
        "feature_columns": columns,
        "utility_accuracy": _accuracy(utility_probabilities, validation_utility),
        "sensitive_adversary_accuracy": _accuracy(sensitive_probabilities, validation_sensitive),
        "sensitive_majority_baseline": float(
            max(np.mean(validation_sensitive), 1.0 - np.mean(validation_sensitive))
        ),
        "accountant": "Poisson-sampled Gaussian RDP, integer orders 2-64",
        "validation_metrics_are_not_private": True,
    }
