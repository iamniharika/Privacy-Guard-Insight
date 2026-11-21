from __future__ import annotations

import numpy as np
import pandas as pd

from data_anonymization_app.reports.report_generator import generate_anonymization_report


def _build_base_inputs() -> dict[str, object]:
    return {
        "metadata": {
            "file_type": "csv",
            "rows": 3,
            "columns": 2,
            "column_names": ["email", "age"],
        },
        "planning_context": {
            "intended_use": "research",
            "need_record_linkage": False,
            "dataset_domain": "healthcare",
            "dataset_size_rows": 3,
        },
        "profiling_df": pd.DataFrame(
            [
                {"column_name": "email", "dtype": "object", "null_percent": 0.0, "unique_percent": 100.0},
                {"column_name": "age", "dtype": "int64", "null_percent": 0.0, "unique_percent": 66.7},
            ]
        ),
        "presidio_summary_df": pd.DataFrame(
            [
                {
                    "column_name": "email",
                    "presidio_entities_detected": ["EMAIL_ADDRESS"],
                    "presidio_hit_count": 2,
                    "presidio_used": True,
                }
            ]
        ),
        "classification_df": pd.DataFrame(
            [
                {"column_name": "email", "classification": "direct_identifier", "reason": "name-heuristic"},
                {"column_name": "age", "classification": "quasi_identifier", "reason": "name-heuristic"},
            ]
        ),
        "plan_df": pd.DataFrame(
            [
                {
                    "column_name": "email",
                    "classification": "direct_identifier",
                    "recommended_action": "drop",
                    "final_action": "drop",
                    "planning_reason": "default mapping",
                },
                {
                    "column_name": "age",
                    "classification": "quasi_identifier",
                    "recommended_action": "generalize",
                    "final_action": "generalize",
                    "planning_reason": "default mapping",
                },
            ]
        ),
        "transformation_log": [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "action_requested": "drop",
                "action_applied": "drop",
                "notes": "applied",
            }
        ],
        "validation_df": pd.DataFrame(
            [
                {"check_name": "residual_direct_identifier_count", "value": 0, "columns": []},
            ]
        ),
        "validation_summary": {
            "residual_direct_identifier_count": 0,
            "residual_technical_identifier_count": 0,
            "high_uniqueness_columns_after_anonymization": [],
            "quasi_identifier_columns_remaining": ["age"],
            "k_anonymity": {
                "k": 5,
                "columns": ["age"],
                "applicable": True,
                "available": True,
                "row_count": 3,
                "equivalence_class_count": 2,
                "minimum_class_size": 1,
                "records_below_k": 3,
                "records_below_k_percent": 100.0,
                "satisfies_k": False,
            },
            "validation_warnings": ["Quasi-identifier columns still present after anonymization: age"],
        },
        "anonymized_dataframe": pd.DataFrame({"age": ["20-29", "30-39", "20-29"]}),
    }


def test_generate_anonymization_report_markdown_and_json_basic() -> None:
    inputs = _build_base_inputs()
    markdown, report_json = generate_anonymization_report(**inputs, generated_at="2026-03-12T00:00:00Z")

    assert markdown.strip() != ""
    assert "## Dataset Overview" in markdown
    assert "## Privacy Validation Summary" in markdown
    assert "Quasi-identifier columns still present after anonymization: age" in markdown
    assert "k_anonymity:" in markdown
    assert report_json["privacy_validation_summary"]["k_anonymity"]["k"] == 5

    expected_sections = {
        "report_header",
        "dataset_overview",
        "policy_context",
        "profiling_summary",
        "presidio_summary",
        "classification_summary",
        "anonymization_plan",
        "transformation_log",
        "privacy_validation_summary",
        "validation_warnings",
        "output_summary",
    }
    assert expected_sections.issubset(report_json.keys())


def test_generate_anonymization_report_handles_empty_presidio_and_no_warnings() -> None:
    inputs = _build_base_inputs()
    inputs["presidio_summary_df"] = pd.DataFrame(
        columns=["column_name", "presidio_entities_detected", "presidio_hit_count", "presidio_used"]
    )
    inputs["validation_summary"] = {
        "residual_direct_identifier_count": 0,
        "residual_technical_identifier_count": 0,
        "high_uniqueness_columns_after_anonymization": [],
        "quasi_identifier_columns_remaining": [],
        "validation_warnings": [],
    }

    markdown, report_json = generate_anonymization_report(**inputs, generated_at="2026-03-12T00:00:00Z")

    assert report_json["presidio_summary"]["columns_with_hits_count"] == 0
    assert "- None detected." in markdown


def test_generate_anonymization_report_is_deterministic_with_fixed_inputs() -> None:
    inputs = _build_base_inputs()
    markdown_1, json_1 = generate_anonymization_report(**inputs, generated_at="2026-03-12T00:00:00Z")
    markdown_2, json_2 = generate_anonymization_report(**inputs, generated_at="2026-03-12T00:00:00Z")

    assert markdown_1 == markdown_2
    assert json_1 == json_2


def test_generate_anonymization_report_handles_array_like_inputs_without_truth_ambiguity() -> None:
    metadata = {
        "file_type": "csv",
        "rows": 2,
        "columns": 1,
        "column_names": np.array(["email"]),
    }
    validation_summary = {
        "residual_direct_identifier_count": 0,
        "residual_technical_identifier_count": 0,
        "high_uniqueness_columns_after_anonymization": np.array([]),
        "quasi_identifier_columns_remaining": np.array([]),
        "validation_warnings": np.array([]),
    }

    markdown, report_json = generate_anonymization_report(
        metadata=metadata,
        profiling_df=pd.DataFrame(),
        presidio_summary_df=pd.DataFrame(),
        classification_df=pd.DataFrame(),
        plan_df=pd.DataFrame(),
        transformation_log=[],
        validation_df=pd.DataFrame(),
        validation_summary=validation_summary,
        anonymized_dataframe=pd.DataFrame({"email": ["token_1", "token_2"]}),
        generated_at="2026-03-12T00:00:00Z",
    )

    assert markdown.strip() != ""
    assert report_json["dataset_overview"]["column_names_preview"] == ["email"]
    assert report_json["validation_warnings"] == []
