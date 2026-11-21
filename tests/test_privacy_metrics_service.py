from __future__ import annotations

import pandas as pd
import pytest

from data_anonymization_app.privacy_metrics.service import (
    compute_k_anonymity,
    validate_anonymization_output,
)


def test_compute_k_anonymity_reports_records_in_small_equivalence_classes() -> None:
    dataframe = pd.DataFrame({"age_band": ["20-29"] * 5 + ["30-39"] * 2})

    result = compute_k_anonymity(dataframe, ["age_band"], k=5)

    assert result["available"] is True
    assert result["equivalence_class_count"] == 2
    assert result["minimum_class_size"] == 2
    assert result["records_below_k"] == 2
    assert result["records_below_k_percent"] == pytest.approx(2 / 7 * 100)
    assert result["satisfies_k"] is False


def test_compute_k_anonymity_handles_missing_columns_and_invalid_threshold() -> None:
    dataframe = pd.DataFrame({"age_band": ["20-29"]})

    result = compute_k_anonymity(dataframe, ["missing_column"])

    assert result["applicable"] is False
    assert result["available"] is False
    assert result["satisfies_k"] is None
    with pytest.raises(ValueError, match="positive integer"):
        compute_k_anonymity(dataframe, ["age_band"], k=0)


def test_validate_anonymization_output_compares_k_risk_before_and_after() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "age_band", "classification": "quasi_identifier"}]
    )
    reviewed_plan = pd.DataFrame(
        [{"column_name": "age_band", "final_action": "generalize"}]
    )
    original_dataframe = pd.DataFrame(
        {"age_band": ["A"] * 5 + ["B"] * 2}
    )
    anonymized_dataframe = pd.DataFrame(
        {"age_band": ["A"] * 7}
    )

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_dataframe,
        original_dataframe=original_dataframe,
    )

    assert summary["k_anonymity_comparison_available"] is True
    assert summary["k_anonymity_before"]["records_below_k"] == 2
    assert summary["k_anonymity"]["records_below_k"] == 0


def test_validate_anonymization_output_computes_v1_checks_and_summary() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "email", "classification": "direct_identifier"},
            {"column_name": "row_id", "classification": "technical_identifier"},
            {"column_name": "birth_date", "classification": "quasi_identifier"},
            {"column_name": "salary", "classification": "sensitive_attribute"},
        ]
    )
    reviewed_plan = pd.DataFrame(
        [
            {"column_name": "email", "final_action": "keep"},
            {"column_name": "row_id", "final_action": "generalize"},
            {"column_name": "birth_date", "final_action": "generalize"},
            {"column_name": "salary", "final_action": "add_noise"},
        ]
    )
    anonymized_df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
            "row_id": [1, 2, 3, 4],
            "birth_date": ["1980-01-01", "1980-01-01", "1990-01-01", "1990-01-01"],
            "salary": [1000.0, 1001.0, 1002.0, 1003.0],
        }
    )

    validation_df, validation_json, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
    )

    assert validation_df.to_dict(orient="records") == validation_json
    assert list(validation_df.columns) == ["check_name", "value", "columns"]

    assert summary["residual_direct_identifier_count"] == 1
    assert summary["residual_direct_identifier_columns"] == ["email"]
    assert summary["residual_technical_identifier_count"] == 1
    assert summary["residual_technical_identifier_columns"] == ["row_id"]
    assert summary["quasi_identifier_columns_remaining_count"] == 1
    assert summary["quasi_identifier_columns_remaining"] == ["birth_date"]
    assert summary["k_anonymity"]["records_below_k"] == 4
    assert summary["k_anonymity"]["satisfies_k"] is False
    assert "records_in_equivalence_classes_below_k" in validation_df["check_name"].tolist()
    assert "email" in summary["high_uniqueness_columns_after_anonymization"]
    assert "row_id" in summary["high_uniqueness_columns_after_anonymization"]
    assert len(summary["validation_warnings"]) >= 1


def test_validate_anonymization_output_exposes_string_and_structured_warnings() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "email", "classification": "direct_identifier"},
            {"column_name": "birth_date", "classification": "quasi_identifier"},
        ]
    )
    reviewed_plan = pd.DataFrame(
        [
            {"column_name": "email", "final_action": "keep"},
            {"column_name": "birth_date", "final_action": "keep"},
        ]
    )
    anonymized_df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@x.com"],
            "birth_date": ["1980-01-01", "1990-01-01"],
        }
    )

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
    )

    assert "validation_warnings" in summary
    assert "validation_warning_items" in summary
    assert summary["validation_warnings"]
    assert summary["validation_warning_items"]
    assert {
        "kind": "direct_identifiers",
        "columns": ["email"],
        "risk_category": "direct_identifier",
        "risk_level": "high",
    } in summary["validation_warning_items"]


def test_validate_anonymization_output_uses_recommended_action_fallback() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "customer_id", "classification": "direct_identifier"}]
    )
    reviewed_plan = pd.DataFrame(
        [{"column_name": "customer_id", "recommended_action": "pseudonymize"}]
    )
    anonymized_df = pd.DataFrame({"customer_id": ["token_000001", "token_000002"]})

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
    )

    assert summary["residual_direct_identifier_count"] == 0
    assert summary["residual_direct_identifier_columns"] == []


def test_validate_anonymization_output_respects_linkage_pseudonymized_technical_identifier() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "record_id", "classification": "technical_identifier"}]
    )
    reviewed_plan = pd.DataFrame(
        [
            {
                "column_name": "record_id",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
                "functional_role": "linkage",
                "policy_context": "internal analysis",
                "record_linkage_required": True,
            }
        ]
    )
    anonymized_df = pd.DataFrame({"record_id": ["token_000001", "token_000002"]})
    transformation_log = [
        {
            "column_name": "record_id",
            "classification": "technical_identifier",
            "action_requested": "pseudonymize",
            "action_applied": "pseudonymize",
            "notes": "persistent vault mapping applied",
        }
    ]

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
        transformation_log=transformation_log,
    )

    assert summary["residual_technical_identifier_count"] == 0
    assert summary["residual_technical_identifier_columns"] == []
    assert summary["failed_protection_attempt_count"] == 0


def test_validate_anonymization_output_flags_non_linkage_pseudonymized_technical_identifier() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "record_id", "classification": "technical_identifier"}]
    )
    reviewed_plan = pd.DataFrame(
        [
            {
                "column_name": "record_id",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
                "functional_role": "administrative",
                "policy_context": "internal analysis",
                "record_linkage_required": True,
            }
        ]
    )
    anonymized_df = pd.DataFrame({"record_id": ["token_000001", "token_000002"]})
    transformation_log = [
        {
            "column_name": "record_id",
            "classification": "technical_identifier",
            "action_requested": "pseudonymize",
            "action_applied": "pseudonymize",
            "notes": "persistent vault mapping applied",
        }
    ]

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
        transformation_log=transformation_log,
    )

    assert summary["residual_technical_identifier_count"] == 1
    assert summary["residual_technical_identifier_columns"] == ["record_id"]
    assert (
        "Residual technical identifiers not safely protected for current context: record_id"
        in summary["validation_warnings"]
    )
    assert summary["contextual_technical_validation_incomplete_count"] == 0


def test_validate_anonymization_output_uses_strict_fallback_when_technical_context_missing() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "record_id", "classification": "technical_identifier"}]
    )
    reviewed_plan = pd.DataFrame(
        [
            {
                "column_name": "record_id",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            }
        ]
    )
    anonymized_df = pd.DataFrame({"record_id": ["token_000001", "token_000002"]})
    transformation_log = [
        {
            "column_name": "record_id",
            "classification": "technical_identifier",
            "action_requested": "pseudonymize",
            "action_applied": "pseudonymize",
            "notes": "persistent vault mapping applied",
        }
    ]

    validation_df, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
        transformation_log=transformation_log,
    )

    assert summary["residual_technical_identifier_count"] == 1
    assert summary["contextual_technical_validation_incomplete_count"] == 1
    assert summary["contextual_technical_validation_incomplete_columns"] == ["record_id"]
    assert (
        "Contextual technical-identifier validation used strict fallback because required metadata was missing for: record_id"
        in summary["validation_warnings"]
    )
    assert "contextual_technical_validation_incomplete" in validation_df["check_name"].tolist()


def test_validate_anonymization_output_flags_execution_fallback_in_summary() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "salary_band", "classification": "sensitive_attribute"}]
    )
    reviewed_plan = pd.DataFrame(
        [
            {
                "column_name": "salary_band",
                "recommended_action": "add_noise",
                "final_action": "add_noise",
            }
        ]
    )
    anonymized_df = pd.DataFrame({"salary_band": ["low", "high"]})
    transformation_log = [
        {
            "column_name": "salary_band",
            "classification": "sensitive_attribute",
            "action_requested": "add_noise",
            "action_applied": "keep",
            "notes": "add_noise skipped: non-numeric column",
        }
    ]

    _, _, summary = validate_anonymization_output(
        classification_summary=classification_summary,
        reviewed_transformation_plan=reviewed_plan,
        anonymized_dataframe=anonymized_df,
        transformation_log=transformation_log,
    )

    assert summary["failed_protection_attempt_count"] == 1
    assert summary["failed_protection_attempt_columns"] == ["salary_band"]
    assert summary["failed_protection_attempt_details"] == [
        {
            "column_name": "salary_band",
            "classification": "sensitive_attribute",
            "action_requested": "add_noise",
            "action_applied": "keep",
        }
    ]
    assert "Requested protection could not be applied for: salary_band" in summary["validation_warnings"]


def test_validate_anonymization_output_validates_required_fields() -> None:
    classification_summary = pd.DataFrame([{"column_name": "x"}])
    reviewed_plan = pd.DataFrame([{"column_name": "x", "final_action": "keep"}])
    anonymized_df = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match="Classification summary missing required columns"):
        validate_anonymization_output(
            classification_summary=classification_summary,
            reviewed_transformation_plan=reviewed_plan,
            anonymized_dataframe=anonymized_df,
        )

    with pytest.raises(ValueError, match="Transformation plan missing required columns"):
        validate_anonymization_output(
            classification_summary=pd.DataFrame([{"column_name": "x", "classification": "non_sensitive"}]),
            reviewed_transformation_plan=pd.DataFrame([{"final_action": "keep"}]),
            anonymized_dataframe=anonymized_df,
        )

    with pytest.raises(ValueError, match="must include 'final_action' or 'recommended_action'"):
        validate_anonymization_output(
            classification_summary=pd.DataFrame([{"column_name": "x", "classification": "non_sensitive"}]),
            reviewed_transformation_plan=pd.DataFrame([{"column_name": "x"}]),
            anonymized_dataframe=anonymized_df,
        )
