from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


_ACTION_LABELS = {
    "generalize": "Generalize / reduce precision",
    "generalize_light": "Generalize / reduce precision",
    "generalize_medium": "Generalize more",
    "generalize_strict": "Generalize strongly",
}


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    return {}


def _display_action(value: Any) -> Any:
    normalized = str(value or "").strip()
    return _ACTION_LABELS.get(normalized, value)


def _dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No data_"

    columns = [str(column) for column in dataframe.columns.tolist()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in dataframe.to_dict(orient="records"):
        row_values = [str(_safe_scalar(record.get(column))) for column in columns]
        rows.append("| " + " | ".join(row_values) + " |")
    return "\n".join([header, separator, *rows])


def _normalize_policy_context(planning_context: dict[str, Any] | None) -> dict[str, Any]:
    context = planning_context if planning_context is not None else {}
    return {
        "intended_use": _safe_scalar(context.get("intended_use")),
        "need_record_linkage": bool(context.get("need_record_linkage", False)),
        "dataset_domain": _safe_scalar(context.get("dataset_domain")),
        "dataset_size_rows": _safe_scalar(context.get("dataset_size_rows")),
    }


def _build_report_json(
    *,
    report_header: dict[str, Any],
    dataset_overview: dict[str, Any],
    policy_context: dict[str, Any],
    profiling_summary: dict[str, Any],
    presidio_summary: dict[str, Any],
    classification_summary: dict[str, Any],
    anonymization_plan: dict[str, Any],
    transformation_log: dict[str, Any],
    privacy_validation_summary: dict[str, Any],
    validation_warnings: list[str],
    output_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_header": report_header,
        "dataset_overview": dataset_overview,
        "policy_context": policy_context,
        "profiling_summary": profiling_summary,
        "presidio_summary": presidio_summary,
        "classification_summary": classification_summary,
        "anonymization_plan": anonymization_plan,
        "transformation_log": transformation_log,
        "privacy_validation_summary": privacy_validation_summary,
        "validation_warnings": validation_warnings,
        "output_summary": output_summary,
    }


def generate_anonymization_report(
    *,
    metadata: Any = None,
    planning_context: dict[str, Any] | None = None,
    profiling_df: pd.DataFrame | None = None,
    presidio_summary_df: pd.DataFrame | None = None,
    classification_df: pd.DataFrame | None = None,
    plan_df: pd.DataFrame | None = None,
    transformation_log: list[dict[str, Any]] | None = None,
    validation_df: pd.DataFrame | None = None,
    validation_summary: dict[str, Any] | None = None,
    anonymized_dataframe: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata_dict = _extract_metadata(metadata)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()

    profiling_df = profiling_df if profiling_df is not None else pd.DataFrame()
    presidio_summary_df = presidio_summary_df if presidio_summary_df is not None else pd.DataFrame()
    classification_df = classification_df if classification_df is not None else pd.DataFrame()
    plan_df = plan_df if plan_df is not None else pd.DataFrame()
    validation_df = validation_df if validation_df is not None else pd.DataFrame()
    validation_summary = validation_summary if validation_summary is not None else {}
    transformation_log = transformation_log if transformation_log is not None else []

    profile_table_df = profiling_df.reindex(columns=["column_name", "dtype", "null_percent", "unique_percent"]).fillna("")
    classification_table_df = classification_df.reindex(columns=["column_name", "classification", "reason"]).fillna("")
    plan_table_df = plan_df.reindex(
        columns=["column_name", "classification", "recommended_action", "final_action", "planning_reason"]
    ).fillna("")
    transformation_log_df = pd.DataFrame(transformation_log).reindex(
        columns=["column_name", "classification", "action_requested", "action_applied", "notes"]
    ).fillna("")
    for column_name in ("recommended_action", "final_action"):
        if column_name in plan_table_df.columns:
            plan_table_df[column_name] = plan_table_df[column_name].map(_display_action)
    for column_name in ("action_requested", "action_applied"):
        if column_name in transformation_log_df.columns:
            transformation_log_df[column_name] = transformation_log_df[column_name].map(_display_action)

    presidio_hits_df = presidio_summary_df
    if "presidio_hit_count" in presidio_hits_df.columns:
        presidio_hits_df = presidio_hits_df[presidio_hits_df["presidio_hit_count"] > 0]
    presidio_hits_df = presidio_hits_df.reindex(
        columns=["column_name", "presidio_entities_detected", "presidio_hit_count"]
    ).fillna("")

    policy_context = _normalize_policy_context(planning_context)

    dataset_rows = metadata_dict.get("rows")
    dataset_columns = metadata_dict.get("columns")
    column_names = metadata_dict.get("column_names", [])
    has_column_names = column_names is not None and len(list(column_names)) > 0
    if (dataset_rows is None or dataset_columns is None or not has_column_names) and anonymized_dataframe is not None:
        dataset_rows = dataset_rows if dataset_rows is not None else int(len(anonymized_dataframe))
        dataset_columns = dataset_columns if dataset_columns is not None else int(anonymized_dataframe.shape[1])
        if not has_column_names:
            column_names = [str(column) for column in anonymized_dataframe.columns.tolist()]

    dataset_overview = {
        "file_type": _safe_scalar(metadata_dict.get("file_type")),
        "row_count": _safe_scalar(dataset_rows),
        "column_count": _safe_scalar(dataset_columns),
        "column_names_preview": [str(column) for column in list(column_names)[:10]],
        "report_generated_at": timestamp,
    }

    profiling_summary = {
        "total_columns": int(len(profile_table_df)),
        "table": profile_table_df.to_dict(orient="records"),
    }
    presidio_summary = {
        "columns_with_hits_count": int(len(presidio_hits_df)),
        "table": presidio_hits_df.to_dict(orient="records"),
    }
    classification_summary = {
        "table": classification_table_df.to_dict(orient="records"),
    }
    anonymization_plan = {
        "table": plan_table_df.to_dict(orient="records"),
    }
    transformation_log_summary = {
        "table": transformation_log_df.to_dict(orient="records"),
    }

    privacy_validation_summary = {
        "residual_direct_identifier_count": int(validation_summary.get("residual_direct_identifier_count", 0)),
        "residual_technical_identifier_count": int(validation_summary.get("residual_technical_identifier_count", 0)),
        "high_uniqueness_columns_after_anonymization": list(
            validation_summary.get("high_uniqueness_columns_after_anonymization", [])
        ),
        "quasi_identifier_columns_remaining": list(validation_summary.get("quasi_identifier_columns_remaining", [])),
        "k_anonymity": dict(validation_summary.get("k_anonymity") or {}),
        "k_anonymity_before": (
            dict(validation_summary["k_anonymity_before"])
            if validation_summary.get("k_anonymity_before")
            else None
        ),
        "k_anonymity_comparison_available": bool(
            validation_summary.get("k_anonymity_comparison_available", False)
        ),
        "validation_table": validation_df.to_dict(orient="records"),
    }

    warnings = list(validation_summary.get("validation_warnings", []))

    action_counts: dict[str, int] = {}
    if not transformation_log_df.empty and "action_applied" in transformation_log_df.columns:
        for action, count in transformation_log_df["action_applied"].fillna("").astype(str).value_counts().items():
            if action:
                action_counts[str(_display_action(action))] = int(count)
    elif not plan_df.empty and "final_action" in plan_df.columns:
        for action, count in plan_df["final_action"].fillna("").astype(str).value_counts().items():
            if action:
                action_counts[str(_display_action(action))] = int(count)

    persistent_vault_used = any(
        "persistent vault mapping applied" in str(entry.get("notes", "")) for entry in transformation_log
    )
    output_summary = {
        "anonymized_output_column_count": int(anonymized_dataframe.shape[1]) if anonymized_dataframe is not None else None,
        "action_counts": action_counts,
        "persistent_pseudonymization_vault_used": persistent_vault_used,
    }

    report_json = _build_report_json(
        report_header={"title": "Anonymization Audit Report", "generated_at": timestamp},
        dataset_overview=dataset_overview,
        policy_context=policy_context,
        profiling_summary=profiling_summary,
        presidio_summary=presidio_summary,
        classification_summary=classification_summary,
        anonymization_plan=anonymization_plan,
        transformation_log=transformation_log_summary,
        privacy_validation_summary=privacy_validation_summary,
        validation_warnings=warnings,
        output_summary=output_summary,
    )

    markdown_sections = [
        "# Anonymization Audit Report",
        "",
        "## Report Header",
        f"- Generated at: `{timestamp}`",
        "",
        "## Dataset Overview",
        f"- File type: `{dataset_overview['file_type']}`",
        f"- Row count: `{dataset_overview['row_count']}`",
        f"- Column count: `{dataset_overview['column_count']}`",
        f"- Column preview: `{dataset_overview['column_names_preview']}`",
        "",
        "## Policy Context",
        f"- Intended use: `{policy_context['intended_use']}`",
        f"- Need record linkage: `{policy_context['need_record_linkage']}`",
        f"- Dataset domain: `{policy_context['dataset_domain']}`",
        f"- Dataset size rows: `{policy_context['dataset_size_rows']}`",
        "",
        "## Profiling Summary",
        f"- Total columns profiled: `{profiling_summary['total_columns']}`",
        _dataframe_to_markdown_table(profile_table_df),
        "",
        "## Presidio Summary",
        f"- Columns with Presidio hits: `{presidio_summary['columns_with_hits_count']}`",
        _dataframe_to_markdown_table(presidio_hits_df),
        "",
        "## Classification Summary",
        _dataframe_to_markdown_table(classification_table_df),
        "",
        "## Anonymization Plan",
        _dataframe_to_markdown_table(plan_table_df),
        "",
        "## Transformation Log",
        _dataframe_to_markdown_table(transformation_log_df),
        "",
        "## Privacy Validation Summary",
        f"- residual_direct_identifier_count: `{privacy_validation_summary['residual_direct_identifier_count']}`",
        f"- residual_technical_identifier_count: `{privacy_validation_summary['residual_technical_identifier_count']}`",
        f"- high_uniqueness_columns_after_anonymization: `{privacy_validation_summary['high_uniqueness_columns_after_anonymization']}`",
        f"- quasi_identifier_columns_remaining: `{privacy_validation_summary['quasi_identifier_columns_remaining']}`",
        f"- k_anonymity: `{privacy_validation_summary['k_anonymity']}`",
        f"- k_anonymity_before: `{privacy_validation_summary['k_anonymity_before']}`",
        "",
        "## Validation Warnings",
    ]

    if warnings is not None and len(warnings) > 0:
        markdown_sections.extend([f"- {warning}" for warning in warnings])
    else:
        markdown_sections.append("- None detected.")

    markdown_sections.extend(
        [
            "",
            "## Output Summary",
            f"- Anonymized output column count: `{output_summary['anonymized_output_column_count']}`",
            f"- Action counts: `{output_summary['action_counts']}`",
            f"- Persistent pseudonymization vault used: `{output_summary['persistent_pseudonymization_vault_used']}`",
        ]
    )

    report_markdown = "\n".join(markdown_sections).strip() + "\n"
    return report_markdown, report_json
