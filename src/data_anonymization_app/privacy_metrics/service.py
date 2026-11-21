from __future__ import annotations

from numbers import Integral
from typing import Any

import pandas as pd

_INTERNAL_INTENDED_USES = {
    "internal analysis",
    "dashboard analytics",
    "ml training",
    "research",
}
_GENERALIZE_ACTIONS = {
    "generalize",
    "generalize_light",
    "generalize_medium",
    "generalize_strict",
}


def compute_k_anonymity(
    dataframe: pd.DataFrame,
    quasi_identifier_columns: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Measure equivalence-class sizes for the available quasi-identifiers."""
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")

    columns = list(dict.fromkeys(
        column for column in quasi_identifier_columns if column in dataframe.columns
    ))
    result: dict[str, Any] = {
        "k": int(k),
        "columns": columns,
        "applicable": bool(columns),
        "available": False,
        "row_count": int(len(dataframe)),
        "equivalence_class_count": 0,
        "minimum_class_size": None,
        "records_below_k": None,
        "records_below_k_percent": None,
        "satisfies_k": None,
    }

    if not columns or dataframe.empty:
        return result

    class_sizes = dataframe.groupby(
        columns,
        dropna=False,
        observed=True,
        sort=False,
    ).size()
    records_below_k = int(class_sizes[class_sizes < k].sum())

    result.update(
        {
            "available": True,
            "equivalence_class_count": int(len(class_sizes)),
            "minimum_class_size": int(class_sizes.min()),
            "records_below_k": records_below_k,
            "records_below_k_percent": records_below_k / len(dataframe) * 100.0,
            "satisfies_k": records_below_k == 0,
        }
    )
    return result


def _normalize_action_key(action: Any) -> str:
    normalized = str(action or "").strip().lower()
    if normalized in {"generalize", "generalize / reduce precision", "generalize_reduce_precision"}:
        return "generalize_light"
    return normalized


def _resolve_action(plan_row: dict[str, Any]) -> str:
    final_action = plan_row.get("final_action")
    if final_action is not None and not pd.isna(final_action) and str(final_action).strip() != "":
        return _normalize_action_key(final_action)

    recommended_action = plan_row.get("recommended_action")
    if (
        recommended_action is not None
        and not pd.isna(recommended_action)
        and str(recommended_action).strip() != ""
    ):
        return _normalize_action_key(recommended_action)

    return ""


def _resolve_applied_action(log_row: dict[str, Any]) -> str:
    action_applied = log_row.get("action_applied")
    if action_applied is not None and not pd.isna(action_applied) and str(action_applied).strip() != "":
        return _normalize_action_key(action_applied)

    action_requested = log_row.get("action_requested")
    if action_requested is not None and not pd.isna(action_requested) and str(action_requested).strip() != "":
        return _normalize_action_key(action_requested)

    return ""


def _is_stronger_protection_requested(action_requested: str) -> bool:
    return _normalize_action_key(action_requested) in {
        "drop",
        "pseudonymize",
        "generalize_light",
        "generalize_medium",
        "generalize_strict",
        "add_noise",
        "mask",
    }


def _is_failed_protection_attempt(action_requested: str, action_applied: str) -> bool:
    requested = _normalize_action_key(action_requested)
    applied = _normalize_action_key(action_applied)

    if not _is_stronger_protection_requested(requested):
        return False

    if applied == requested:
        return False

    if applied in {"unsupported_generalize", "unsupported_add_noise"}:
        return True

    if applied == "keep" and requested != "keep":
        return True

    return False


def _is_acceptable_technical_identifier_action(
    plan_row: dict[str, Any],
    action_applied: str,
) -> bool:
    applied = str(action_applied).strip()
    if applied == "drop":
        return True

    if applied != "pseudonymize":
        return False

    functional_role = str(plan_row.get("functional_role", "") or "").strip().lower()
    intended_use = str(plan_row.get("policy_context", "") or "").strip().lower()
    linkage_required = bool(plan_row.get("record_linkage_required", False))

    return (
        functional_role == "linkage"
        and linkage_required
        and intended_use in _INTERNAL_INTENDED_USES
    )


def _missing_contextual_technical_validation_metadata(plan_row: dict[str, Any]) -> list[str]:
    missing_fields: list[str] = []
    for field_name in ("functional_role", "policy_context", "record_linkage_required"):
        if field_name not in plan_row:
            missing_fields.append(field_name)
            continue

        value = plan_row.get(field_name)
        if value is None or (not isinstance(value, bool) and pd.isna(value)):
            missing_fields.append(field_name)
            continue

        if isinstance(value, str) and value.strip() == "":
            missing_fields.append(field_name)

    return missing_fields



def validate_anonymization_output(
    classification_summary: pd.DataFrame,
    reviewed_transformation_plan: pd.DataFrame,
    anonymized_dataframe: pd.DataFrame,
    transformation_log: list[dict[str, Any]] | pd.DataFrame | None = None,
    uniqueness_threshold: float = 0.8,
    original_dataframe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Run deterministic privacy validation checks on anonymization output.

    Validation is conservative for contextual technical-identifier checks.
    If the reviewed transformation plan omits ``functional_role``, ``policy_context``,
    or ``record_linkage_required``, pseudonymized technical identifiers are treated as
    not contextually validated and remain residual technical-identifier findings.
    """
    classification_required = {"column_name", "classification"}
    missing_classification = classification_required.difference(classification_summary.columns)
    if missing_classification:
        missing_joined = ", ".join(sorted(missing_classification))
        raise ValueError(f"Classification summary missing required columns: {missing_joined}")

    plan_required = {"column_name"}
    missing_plan = plan_required.difference(reviewed_transformation_plan.columns)
    if missing_plan:
        missing_joined = ", ".join(sorted(missing_plan))
        raise ValueError(f"Transformation plan missing required columns: {missing_joined}")

    if "final_action" not in reviewed_transformation_plan.columns and "recommended_action" not in reviewed_transformation_plan.columns:
        raise ValueError("Transformation plan must include 'final_action' or 'recommended_action'.")

    classification_by_column = {
        str(row["column_name"]): str(row["classification"])
        for row in classification_summary.to_dict(orient="records")
    }
    plan_rows_by_column = {
        str(row.get("column_name")): row
        for row in reviewed_transformation_plan.to_dict(orient="records")
    }

    plan_action_by_column = {
        str(row.get("column_name")): _resolve_action(row)
        for row in reviewed_transformation_plan.to_dict(orient="records")
    }

    requested_action_by_column = plan_action_by_column.copy()

    if transformation_log is None:
        action_by_column = plan_action_by_column
        transformation_log_df = pd.DataFrame()
    else:
        if isinstance(transformation_log, pd.DataFrame):
            transformation_log_df = transformation_log.copy()
        else:
            transformation_log_df = pd.DataFrame(transformation_log)

        if transformation_log_df.empty or "column_name" not in transformation_log_df.columns:
            action_by_column = plan_action_by_column
        else:
            action_by_column = {
                str(row.get("column_name")): _resolve_applied_action(row)
                for row in transformation_log_df.to_dict(orient="records")
            }

            if "action_requested" in transformation_log_df.columns:
                requested_action_by_column = {
                    str(row.get("column_name")): _normalize_action_key(row.get("action_requested", ""))
                    for row in transformation_log_df.to_dict(orient="records")
                }

            for column_name, planned_action in plan_action_by_column.items():
                action_by_column.setdefault(column_name, planned_action)
                requested_action_by_column.setdefault(column_name, planned_action)

    direct_identifier_columns = [
        column_name
        for column_name, classification in classification_by_column.items()
        if classification == "direct_identifier"
    ]
    residual_direct_identifier_columns = [
        column_name
        for column_name in direct_identifier_columns
        if action_by_column.get(column_name) not in {"drop", "pseudonymize"}
    ]

    technical_identifier_columns = [
        column_name
        for column_name, classification in classification_by_column.items()
        if classification == "technical_identifier"
    ]
    residual_technical_identifier_columns = [
        column_name
        for column_name in technical_identifier_columns
        if not _is_acceptable_technical_identifier_action(
            plan_rows_by_column.get(column_name, {}),
            action_by_column.get(column_name, ""),
        )
    ]
    contextual_technical_validation_incomplete_columns = sorted(
        column_name
        for column_name in technical_identifier_columns
        if str(action_by_column.get(column_name, "")).strip() == "pseudonymize"
        and _missing_contextual_technical_validation_metadata(
            plan_rows_by_column.get(column_name, {})
        )
    )

    total_rows = int(len(anonymized_dataframe))
    high_uniqueness_columns: list[str] = []
    for column_name in anonymized_dataframe.columns:
        unique_count = int(anonymized_dataframe[column_name].nunique(dropna=True))
        uniqueness_ratio = (unique_count / total_rows) if total_rows > 0 else 0.0
        if uniqueness_ratio > uniqueness_threshold:
            high_uniqueness_columns.append(str(column_name))

    quasi_identifier_columns = [
        column_name
        for column_name, classification in classification_by_column.items()
        if classification == "quasi_identifier"
    ]

    quasi_identifier_columns_remaining = [
        column_name
        for column_name in quasi_identifier_columns
        if column_name in anonymized_dataframe.columns
    ]
    k_anonymity = compute_k_anonymity(
        anonymized_dataframe,
        quasi_identifier_columns_remaining,
    )
    k_anonymity_before = (
        compute_k_anonymity(
            original_dataframe,
            quasi_identifier_columns_remaining,
        )
        if original_dataframe is not None
        else None
    )
    k_anonymity_comparison_available = bool(
        k_anonymity_before
        and k_anonymity_before["available"]
        and k_anonymity["available"]
    )

    exact_actions = {"keep", "mask"}
    generalized_actions = _GENERALIZE_ACTIONS.union({"add_noise", "pseudonymize"})

    quasi_identifier_exact_remaining = [
        column_name
        for column_name in quasi_identifier_columns_remaining
        if action_by_column.get(column_name) in exact_actions
    ]

    quasi_identifier_generalized_remaining = [
        column_name
        for column_name in quasi_identifier_columns_remaining
        if action_by_column.get(column_name) in generalized_actions
    ]

    failed_protection_attempt_details: list[dict[str, Any]] = []
    failed_protection_attempt_columns: list[str] = []

    for column_name, requested_action in requested_action_by_column.items():
        applied_action = action_by_column.get(column_name, requested_action)

        if _is_failed_protection_attempt(requested_action, applied_action):
            failed_protection_attempt_columns.append(str(column_name))
            failed_protection_attempt_details.append(
                {
                    "column_name": str(column_name),
                    "classification": classification_by_column.get(str(column_name), ""),
                    "action_requested": str(requested_action),
                    "action_applied": str(applied_action),
                }
            )

    validation_warnings: list[str] = []
    validation_warning_items: list[dict[str, Any]] = []
    if residual_direct_identifier_columns:
        validation_warnings.append(
            "Residual direct identifiers not dropped or pseudonymized: "
            + ", ".join(sorted(residual_direct_identifier_columns))
        )
        validation_warning_items.append(
            {
                "kind": "direct_identifiers",
                "columns": sorted(residual_direct_identifier_columns),
                "risk_category": "direct_identifier",
                "risk_level": "high",
            }
        )
    if residual_technical_identifier_columns:
        validation_warnings.append(
            "Residual technical identifiers not safely protected for current context: "
            + ", ".join(sorted(residual_technical_identifier_columns))
        )
        validation_warning_items.append(
            {
                "kind": "technical_identifiers",
                "columns": sorted(residual_technical_identifier_columns),
                "risk_category": "technical",
                "risk_level": "moderate",
            }
        )
    if contextual_technical_validation_incomplete_columns:
        validation_warnings.append(
            "Contextual technical-identifier validation used strict fallback because required metadata was missing for: "
            + ", ".join(contextual_technical_validation_incomplete_columns)
        )
        validation_warning_items.append(
            {
                "kind": "technical_context_missing",
                "columns": contextual_technical_validation_incomplete_columns,
                "risk_category": "technical",
                "risk_level": "low",
            }
        )
    if high_uniqueness_columns:
        validation_warnings.append(
            "High-uniqueness columns remain after anonymization: "
            + ", ".join(sorted(high_uniqueness_columns))
        )
        validation_warning_items.append(
            {
                "kind": "high_uniqueness",
                "columns": sorted(high_uniqueness_columns),
                "risk_category": "uniqueness",
                "risk_level": "moderate",
            }
        )
    if quasi_identifier_exact_remaining:
        validation_warnings.append(
            "Exact quasi-identifiers remain (higher re-identification risk): "
            + ", ".join(sorted(quasi_identifier_exact_remaining))
        )
        validation_warning_items.append(
            {
                "kind": "exact_indirect_identifiers",
                "columns": sorted(quasi_identifier_exact_remaining),
                "risk_category": "quasi_identifier",
                "risk_level": "high",
            }
        )
    if quasi_identifier_generalized_remaining:
        validation_warnings.append(
            "Generalized quasi-identifiers still present: "
            + ", ".join(sorted(quasi_identifier_generalized_remaining))
        )
        validation_warning_items.append(
            {
                "kind": "generalized_indirect_identifiers",
                "columns": sorted(quasi_identifier_generalized_remaining),
                "risk_category": "quasi_identifier",
                "risk_level": "moderate",
            }
        )
    if k_anonymity["available"] and not k_anonymity["satisfies_k"]:
        validation_warnings.append(
            f"{k_anonymity['records_below_k']} records "
            f"({k_anonymity['records_below_k_percent']:.2f}%) belong to "
            f"quasi-identifier groups smaller than k={k_anonymity['k']}."
        )
        validation_warning_items.append(
            {
                "kind": "k_anonymity",
                "columns": k_anonymity["columns"],
                "risk_category": "quasi_identifier",
                "risk_level": "moderate",
            }
        )
    if failed_protection_attempt_columns:
        validation_warnings.append(
            "Requested protection could not be applied for: "
            + ", ".join(sorted(failed_protection_attempt_columns))
        )

    summary = {
        "residual_direct_identifier_count": len(residual_direct_identifier_columns),
        "residual_direct_identifier_columns": sorted(residual_direct_identifier_columns),
        "residual_technical_identifier_count": len(residual_technical_identifier_columns),
        "residual_technical_identifier_columns": sorted(residual_technical_identifier_columns),
        "contextual_technical_validation_incomplete_columns": contextual_technical_validation_incomplete_columns,
        "contextual_technical_validation_incomplete_count": len(
            contextual_technical_validation_incomplete_columns
        ),
        "high_uniqueness_columns_after_anonymization": sorted(high_uniqueness_columns),
        "high_uniqueness_columns_after_anonymization_count": len(high_uniqueness_columns),
        "quasi_identifier_columns_remaining": sorted(quasi_identifier_columns_remaining),
        "quasi_identifier_columns_remaining_count": len(quasi_identifier_columns_remaining),
        "quasi_identifier_exact_remaining_columns": sorted(quasi_identifier_exact_remaining),
        "quasi_identifier_exact_remaining_count": len(quasi_identifier_exact_remaining),
        "quasi_identifier_generalized_remaining_columns": sorted(quasi_identifier_generalized_remaining),
        "quasi_identifier_generalized_remaining_count": len(quasi_identifier_generalized_remaining),
        "k_anonymity": k_anonymity,
        "k_anonymity_before": k_anonymity_before,
        "k_anonymity_comparison_available": k_anonymity_comparison_available,
        "failed_protection_attempt_columns": sorted(failed_protection_attempt_columns),
        "failed_protection_attempt_count": len(failed_protection_attempt_columns),
        "failed_protection_attempt_details": failed_protection_attempt_details,
        "validation_warnings": validation_warnings,
        "validation_warning_items": validation_warning_items,
    }

    validation_rows: list[dict[str, Any]] = [
        {
            "check_name": "residual_direct_identifier_count",
            "value": summary["residual_direct_identifier_count"],
            "columns": summary["residual_direct_identifier_columns"],
        },
        {
            "check_name": "residual_technical_identifier_count",
            "value": summary["residual_technical_identifier_count"],
            "columns": summary["residual_technical_identifier_columns"],
        },
        {
            "check_name": "contextual_technical_validation_incomplete",
            "value": summary["contextual_technical_validation_incomplete_count"],
            "columns": summary["contextual_technical_validation_incomplete_columns"],
        },
        {
            "check_name": "high_uniqueness_columns_after_anonymization",
            "value": summary["high_uniqueness_columns_after_anonymization_count"],
            "columns": summary["high_uniqueness_columns_after_anonymization"],
        },
        {
            "check_name": "quasi_identifier_columns_remaining",
            "value": summary["quasi_identifier_columns_remaining_count"],
            "columns": summary["quasi_identifier_columns_remaining"],
        },
        {
            "check_name": "quasi_identifier_exact_remaining",
            "value": summary["quasi_identifier_exact_remaining_count"],
            "columns": summary["quasi_identifier_exact_remaining_columns"],
        },
        {
            "check_name": "quasi_identifier_generalized_remaining",
            "value": summary["quasi_identifier_generalized_remaining_count"],
            "columns": summary["quasi_identifier_generalized_remaining_columns"],
        },
        {
            "check_name": "failed_protection_attempts",
            "value": summary["failed_protection_attempt_count"],
            "columns": summary["failed_protection_attempt_columns"],
        },
        {
            "check_name": "validation_warnings",
            "value": len(summary["validation_warnings"]),
            "columns": summary["validation_warnings"],
        },
    ]
    if k_anonymity["available"]:
        validation_rows.append(
            {
                "check_name": "records_in_equivalence_classes_below_k",
                "value": k_anonymity["records_below_k"],
                "columns": k_anonymity["columns"],
            }
        )

    validation_dataframe = pd.DataFrame(
        validation_rows,
        columns=["check_name", "value", "columns"],
    )
    return validation_dataframe, validation_rows, summary



def compute_privacy_metrics() -> str:
    """Placeholder for privacy metric calculations.

    TODO: Add k-anonymity / l-diversity style metric hooks.
    """
    return "[privacy_metrics] TODO: implement privacy metrics."
