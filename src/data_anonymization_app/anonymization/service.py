from __future__ import annotations

from math import floor
from pathlib import Path
import re
from typing import Any, Final

import pandas as pd
from pandas.api.types import is_numeric_dtype

from data_anonymization_app.config.settings import get_settings
from data_anonymization_app.storage.mapping_vault import (
    get_or_create_token,
    load_mapping_vault,
    save_mapping_vault,
)

_ACTION_BY_CLASSIFICATION: Final[dict[str, str]] = {
    "direct_identifier": "drop",
    "sensitive_attribute": "add_noise",
    "quasi_identifier": "generalize",
    "non_sensitive": "keep",
    "technical_identifier": "drop",
}
_GENERALIZE_LEVELS: Final[tuple[str, ...]] = (
    "generalize_light",
    "generalize_medium",
    "generalize_strict",
)
_ACTION_ALIASES: Final[dict[str, str]] = {
    "generalize": "generalize_light",
    "generalize / reduce precision": "generalize_light",
    "generalize_reduce_precision": "generalize_light",
    "reduce precision": "generalize_light",
}
_SUPPORTED_ACTIONS: Final[set[str]] = {
    "drop",
    "keep",
    "generalize",
    "generalize_light",
    "generalize_medium",
    "generalize_strict",
    "add_noise",
    "pseudonymize",
    "mask",
    "unsupported_generalize",
}
_INTERNAL_LINKAGE_USES: Final[set[str]] = {
    "internal analysis",
    "dashboard analytics",
    "ml training",
    "research",
}


def _normalize_action_key(action: Any) -> str:
    normalized = str(action or "").strip().lower()
    return _ACTION_ALIASES.get(normalized, normalized)


def _is_generalize_action(action: Any) -> bool:
    return _normalize_action_key(action) in set(_GENERALIZE_LEVELS)


def plan_anonymization(
    classification_summary: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Create deterministic anonymization actions from classification output."""
    required_columns = {"column_name", "classification"}
    missing_columns = required_columns.difference(classification_summary.columns)
    if missing_columns:
        missing_joined = ", ".join(sorted(missing_columns))
        raise ValueError(f"Classification summary missing required columns: {missing_joined}")

    context_data = context or {}
    intended_use = str(context_data.get("intended_use", "")).strip().lower()
    need_record_linkage = bool(context_data.get("need_record_linkage", False))
    dataset_domain = str(context_data.get("dataset_domain", "")).strip().lower()
    dataset_size_rows_raw = context_data.get("dataset_size_rows")
    dataset_size_rows: int | None = None
    if dataset_size_rows_raw is not None and not pd.isna(dataset_size_rows_raw):
        dataset_size_rows = int(dataset_size_rows_raw)

    plan_rows: list[dict[str, str]] = []
    for row in classification_summary.itertuples(index=False):
        functional_role = getattr(row, "functional_role", None)
        column_name = str(getattr(row, "column_name"))
        classification = str(getattr(row, "classification"))
        override_action_value = getattr(row, "override_action", None)

        if classification not in _ACTION_BY_CLASSIFICATION:
            raise ValueError(f"Unsupported classification: {classification}")

        recommended_action = _ACTION_BY_CLASSIFICATION[classification]
        planning_reason_code = "DEFAULT_CLASSIFICATION_MAPPING"
        planning_reason = "default mapping by classification"

        # Role-aware handling for analytical fields.
        if functional_role == "analytical":
            if classification == "quasi_identifier":
                if any(
                    keyword in column_name.lower()
                    for keyword in (
                        "job_title",
                        "occupation",
                        "education",
                        "education_level",
                        "department",
                    )
                ):
                    recommended_action = "keep"
                    planning_reason_code = "LOW_RISK_ANALYTICAL_KEEP"
                    planning_reason = "role-aware: low-risk categorical analytical feature"
                elif any(
                    keyword in column_name.lower()
                    for keyword in (
                        "sex",
                        "gender",
                    )
                ) and intended_use in {"ml training", "internal analysis", "dashboard analytics"}:
                    recommended_action = "keep"
                    planning_reason_code = "LOW_RISK_DEMOGRAPHIC_KEEP"
                    planning_reason = "role-aware: low-risk demographic feature"
                else:
                    recommended_action = "generalize"
                    planning_reason_code = "ANALYTICAL_QUASI_GENERALIZE"
                    planning_reason = "role-aware: analytical quasi-identifier generalized"
            elif classification == "sensitive_attribute":
                recommended_action = "add_noise"
                planning_reason_code = "SENSITIVE_ANALYTICAL_NOISE"
                planning_reason = "role-aware: sensitive analytical feature"

        # Role-aware handling for identity/linkage fields with backward-compatible fallback.
        if functional_role in {"identity", "linkage"} or (
            functional_role is None and classification == "direct_identifier"
        ):
            if intended_use == "external sharing":
                recommended_action = "drop"
                planning_reason_code = "EXTERNAL_IDENTITY_REMOVE"
                planning_reason = "policy: identity/linkage field dropped for external sharing"
            elif need_record_linkage and (
                intended_use in _INTERNAL_LINKAGE_USES or intended_use == ""
            ):
                recommended_action = "pseudonymize"
                planning_reason_code = "LINKAGE_REQUIRED_PSEUDONYMIZE"
                planning_reason = (
                    "policy: identity/linkage field pseudonymized because record linkage is required"
                )

        # Use-case-specific stricter policy for external sharing.
        if intended_use == "external sharing":
            external_policy_actions = {
                "technical_identifier": "drop",
                "quasi_identifier": "generalize",
            }
            if classification in external_policy_actions:
                recommended_action = external_policy_actions[classification]
                if classification == "technical_identifier":
                    planning_reason_code = "EXTERNAL_TECHNICAL_REMOVE"
                elif classification == "quasi_identifier":
                    planning_reason_code = "EXTERNAL_QUASI_GENERALIZE"
                planning_reason = f"policy: stricter action for external sharing ({classification})"

        # Sensitive attributes remain policy-driven, but now coexist with role-awareness.
        if classification == "sensitive_attribute":
            if intended_use == "external sharing":
                recommended_action = "generalize"
                planning_reason_code = "EXTERNAL_SENSITIVE_GENERALIZE"
                planning_reason = "policy: sensitive attribute generalized for external sharing"
            elif intended_use == "research":
                recommended_action = "generalize"
                planning_reason_code = "RESEARCH_SENSITIVE_GENERALIZE"
                planning_reason = "policy: sensitive attribute generalized for research"
            elif intended_use == "ml training":
                recommended_action = "add_noise"
                planning_reason_code = "ML_SENSITIVE_NOISE"
                planning_reason = "policy: sensitive attribute noised for ML training"
            elif intended_use == "dashboard analytics":
                recommended_action = "add_noise"
                planning_reason_code = "DASHBOARD_SENSITIVE_NOISE"
                planning_reason = "policy: sensitive attribute noised for dashboard analytics"

            if (
                dataset_size_rows is not None
                and dataset_size_rows < 1000
                and intended_use != "ml training"
                and recommended_action == "add_noise"
            ):
                recommended_action = "generalize"
                planning_reason_code = "SMALL_DATASET_SENSITIVE_GENERALIZE"
                planning_reason = "policy: small dataset favors generalization for sensitive attribute"

        if dataset_domain == "healthcare" and classification == "sensitive_attribute":
            planning_reason = f"{planning_reason}; domain policy: stricter handling for healthcare"

        if dataset_domain == "finance" and classification in {"direct_identifier", "technical_identifier"}:
            planning_reason = f"{planning_reason}; domain policy: stricter handling for finance"

        # Final safety guard: analytical fields should not end up pseudonymized.
        if functional_role == "analytical" and recommended_action == "pseudonymize":
            recommended_action = "generalize"
            planning_reason_code = "ANALYTICAL_SAFETY_NO_PSEUDONYMIZATION"
            planning_reason = "safety: analytical fields should not be pseudonymized"

        final_action = recommended_action
        if (
            override_action_value is not None
            and not pd.isna(override_action_value)
            and str(override_action_value).strip() != ""
        ):
            candidate_action = _normalize_action_key(override_action_value)
            if candidate_action not in _SUPPORTED_ACTIONS:
                raise ValueError(f"Unsupported action: {candidate_action}")
            final_action = candidate_action
            planning_reason_code = "USER_OVERRIDE"
            planning_reason = f"{planning_reason}; override_action provided by user"

        plan_rows.append(
        {
            "column_name": column_name,
            "classification": classification,
            "functional_role": functional_role,
            "recommended_action": recommended_action,
            "final_action": final_action,
            "planning_reason_code": planning_reason_code,
            "planning_reason": planning_reason,
            "policy_context": intended_use,
            "record_linkage_required": need_record_linkage,
        }
    )

    plan_dataframe = pd.DataFrame(
        plan_rows,
        columns=[
            "column_name",
            "classification",
            "functional_role",
            "recommended_action",
            "final_action",
            "planning_reason_code",
            "planning_reason",
            "policy_context",
            "record_linkage_required",
        ],
    )
    return plan_dataframe, plan_rows

def _bucket_numeric_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    bucket_start = int(floor(float(value) / 10.0) * 10)
    bucket_end = bucket_start + 9
    return f"{bucket_start}-{bucket_end}"


def _generalize_numeric_value(value: Any, level: str) -> str | None:
    if pd.isna(value):
        return None

    normalized_level = _normalize_action_key(level)
    bucket_sizes = {
        "generalize_light": 10,
        "generalize_medium": 25,
        "generalize_strict": 50,
    }
    bucket_size = bucket_sizes.get(normalized_level, 10)
    bucket_start = int(floor(float(value) / float(bucket_size)) * bucket_size)
    bucket_end = bucket_start + bucket_size - 1
    return f"{bucket_start}-{bucket_end}"


def _generalize_datetime_series(parsed_dates: pd.Series, level: str) -> tuple[pd.Series, str]:
    normalized_level = _normalize_action_key(level)

    if normalized_level == "generalize_medium":
        return parsed_dates.dt.strftime("%Y"), "date-like values generalized to year"

    if normalized_level == "generalize_strict":
        decades = parsed_dates.dt.year.map(
            lambda year: None if pd.isna(year) else f"{int(floor(year / 10.0) * 10)}s"
        )
        return decades, "date-like values generalized to decade"

    return parsed_dates.dt.strftime("%Y-%m"), "date-like values generalized to month"


def _generalize_categorical_series(series: pd.Series, level: str) -> tuple[pd.Series, str]:
    normalized_level = _normalize_action_key(level)
    top_value_limits = {
        "generalize_light": 10,
        "generalize_medium": 5,
        "generalize_strict": 2,
    }
    top_limit = top_value_limits.get(normalized_level, 10)
    value_counts = series.value_counts(dropna=True)
    top_values = set(value_counts.head(top_limit).index)

    transformed = series.apply(
        lambda value: value if pd.isna(value) or value in top_values else "other"
    )

    notes_by_level = {
        "generalize_light": "categorical generalized lightly (rare values grouped to other)",
        "generalize_medium": "categorical generalized to broader groups (top values + other)",
        "generalize_strict": "categorical generalized strongly (major groups only)",
    }
    return transformed, notes_by_level.get(normalized_level, "categorical generalized")


def _add_deterministic_noise(series: pd.Series) -> pd.Series:
    transformed_values: list[Any] = []
    for idx, value in enumerate(series.tolist()):
        if pd.isna(value):
            transformed_values.append(None)
            continue
        noise = ((idx % 5) - 2) * 0.1
        transformed_values.append(float(value) + noise)
    return pd.Series(transformed_values, index=series.index)


def _pseudonymize_series(
    series: pd.Series,
    column_name: str,
    vault_data: dict[str, dict[str, dict[str, str]]],
) -> pd.Series:
    transformed_values: list[Any] = []
    for value in series.tolist():
        if pd.isna(value):
            transformed_values.append(None)
            continue

        value_key = str(value)
        token = get_or_create_token(
            vault_data=vault_data,
            column_name=column_name,
            original_value=value_key,
        )
        transformed_values.append(token)

    return pd.Series(transformed_values, index=series.index)


def _mask_string_value(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text_value = str(value)
    if text_value == "":
        return ""

    email_match = re.fullmatch(r"([^@\s]+)@([^@\s]+)", text_value)
    if email_match:
        local_part, domain_part = email_match.groups()
        if len(local_part) == 1:
            return f"{local_part}***@{domain_part}"
        return f"{local_part[0]}***@{domain_part}"

    compact_value = re.sub(r"[\s().+-]", "", text_value)
    if compact_value.isdigit():
        trailing_length = min(4, max(2, len(compact_value) // 2))
        return f"***{compact_value[-trailing_length:]}"

    if len(text_value) == 1:
        return "*"
    return f"{text_value[0]}{'*' * (len(text_value) - 1)}"


def _mask_series(series: pd.Series) -> pd.Series:
    return series.map(_mask_string_value)


def anonymize_dataset(
    dataframe: pd.DataFrame,
    transformation_plan: pd.DataFrame,
    vault_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply a deterministic v1 anonymization plan to a dataframe copy."""
    required_columns = {"column_name", "classification"}
    missing_columns = required_columns.difference(transformation_plan.columns)
    if missing_columns:
        missing_joined = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transformation plan missing required columns: {missing_joined}")
    if "final_action" not in transformation_plan.columns and "recommended_action" not in transformation_plan.columns:
        raise ValueError(
            "Transformation plan must include 'final_action' or 'recommended_action'."
        )

    transformed_df = dataframe.copy(deep=True)
    transformation_log: list[dict[str, Any]] = []
    effective_vault_path = Path(vault_path) if vault_path is not None else Path(get_settings().mapping_vault_path)
    vault_data: dict[str, dict[str, dict[str, str]]] | None = None
    vault_modified = False

    for row in transformation_plan.itertuples(index=False):
        column_name = str(getattr(row, "column_name"))
        classification = str(getattr(row, "classification"))
        final_action_raw = getattr(row, "final_action", None)
        recommended_action_raw = getattr(row, "recommended_action", None)
        if final_action_raw is not None and str(final_action_raw).strip() != "":
            action_requested = _normalize_action_key(final_action_raw)
        elif recommended_action_raw is not None and str(recommended_action_raw).strip() != "":
            action_requested = _normalize_action_key(recommended_action_raw)
        else:
            raise ValueError(
                f"Missing action for column '{column_name}': expected final_action or recommended_action."
            )

        if action_requested not in _SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action: {action_requested}")
        if column_name not in transformed_df.columns:
            raise ValueError(f"Column not found in dataframe: {column_name}")

        action_applied = action_requested
        notes = "applied"

        if action_requested == "drop":
            transformed_df = transformed_df.drop(columns=[column_name])
        elif action_requested == "keep":
            pass
        elif _is_generalize_action(action_requested):
            series = transformed_df[column_name]

            if classification == "technical_identifier":
                action_applied = "keep"
                notes = f"{action_requested} requested, but technical_identifier columns are not generalized"

            elif is_numeric_dtype(series):
                transformed_df[column_name] = series.map(
                    lambda value: _generalize_numeric_value(value, action_requested)
                )
                notes_by_level = {
                    "generalize_light": "numeric generalized lightly to narrow buckets",
                    "generalize_medium": "numeric generalized to wider buckets",
                    "generalize_strict": "numeric generalized strongly to wide buckets",
                }
                notes = notes_by_level.get(action_requested, "numeric generalized")

            else:
                parsed_dates = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
                parse_ratio = parsed_dates.notna().mean() if len(series) > 0 else 0.0

                if parse_ratio >= 0.8:
                    transformed_series, notes = _generalize_datetime_series(
                        parsed_dates,
                        action_requested,
                    )
                    transformed_df[column_name] = transformed_series
                    transformed_df.loc[parsed_dates.isna(), column_name] = None

                elif series.dtype == "object":
                    transformed_df[column_name], notes = _generalize_categorical_series(
                        series,
                        action_requested,
                    )

                else:
                    action_applied = "unsupported_generalize"
                    notes = f"{action_requested} requested, but generalization is not implemented for this column type"
        elif action_requested == "add_noise":
            if is_numeric_dtype(transformed_df[column_name]):
                transformed_df[column_name] = _add_deterministic_noise(transformed_df[column_name])
            else:
                action_applied = "keep"
                notes = "add_noise skipped: non-numeric column"
        elif action_requested == "pseudonymize":
            if vault_data is None:
                vault_data = load_mapping_vault(effective_vault_path)
            transformed_df[column_name] = _pseudonymize_series(
                transformed_df[column_name],
                column_name=column_name,
                vault_data=vault_data,
            )
            notes = "persistent vault mapping applied"
            vault_modified = True
        elif action_requested == "mask":
            transformed_df[column_name] = _mask_series(transformed_df[column_name])
            notes = "display-oriented masking applied"

        transformation_log.append(
            {
                "column_name": column_name,
                "classification": classification,
                "action_requested": action_requested,
                "action_applied": action_applied,
                "notes": notes,
            }
        )

    if vault_modified and vault_data is not None:
        save_mapping_vault(effective_vault_path, vault_data)

    return transformed_df, transformation_log
