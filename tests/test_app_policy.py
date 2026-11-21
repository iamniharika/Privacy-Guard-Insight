from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
_APP_SPEC = importlib.util.spec_from_file_location("app_module_under_test", _APP_PATH)
assert _APP_SPEC is not None
assert _APP_SPEC.loader is not None
_APP_MODULE = importlib.util.module_from_spec(_APP_SPEC)
_APP_SPEC.loader.exec_module(_APP_MODULE)

_build_external_mask_warning = _APP_MODULE._build_external_mask_warning
_build_review_table = _APP_MODULE._build_review_table
_build_outputs_readiness_summary = _APP_MODULE._build_outputs_readiness_summary
_build_policy_alignment_preview = _APP_MODULE._build_policy_alignment_preview
_build_policy_risk_items = _APP_MODULE._build_policy_risk_items
_protection_strength_rank = _APP_MODULE._protection_strength_rank
_build_step6_risk_summary = _APP_MODULE._build_step6_risk_summary
_build_step6_remaining_risk_groups = _APP_MODULE._build_step6_remaining_risk_groups
_build_step7_next_steps = _APP_MODULE._build_step7_next_steps
_build_step7_use_decision_matrix = _APP_MODULE._build_step7_use_decision_matrix


def test_build_external_mask_warning_flags_external_mask_on_direct_identifier() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "final_action": "mask",
            },
            {
                "column_name": "age",
                "classification": "quasi_identifier",
                "final_action": "generalize",
            },
        ]
    )

    warning_text = _build_external_mask_warning(reviewed_df, "external sharing")

    assert warning_text is not None
    assert "Masking is not recommended for external sharing" in warning_text
    assert "email" in warning_text


def test_build_external_mask_warning_ignores_non_external_or_non_mask_cases() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "final_action": "pseudonymize",
            }
        ]
    )

    assert _build_external_mask_warning(reviewed_df, "external sharing") is None
    assert _build_external_mask_warning(reviewed_df, "internal analysis") is None


def test_build_review_table_preserves_existing_final_action() -> None:
    plan_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
                "functional_role": "linkage",
                "policy_context": "internal analysis",
                "record_linkage_required": True,
            }
        ]
    )
    classification_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "reason": "policy-driven direct identifier handling",
            }
        ]
    )

    review_df = _build_review_table(plan_df, classification_df)

    assert review_df.iloc[0]["recommended_action"] == "pseudonymize"
    assert review_df.iloc[0]["final_action"] == "pseudonymize"
    assert review_df.iloc[0]["functional_role"] == "linkage"
    assert review_df.iloc[0]["policy_context"] == "internal analysis"
    assert bool(review_df.iloc[0]["record_linkage_required"]) is True


def test_policy_alignment_does_not_degrade_for_stronger_override() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "birth_date",
                "recommended_action": "generalize",
                "final_action": "drop",
            }
        ]
    )

    preview_counts, alignment_status = _build_policy_alignment_preview(reviewed_df)

    assert preview_counts["Actions changed by user"] == 1
    assert preview_counts["Higher-risk overrides"] == 0
    assert alignment_status == "Low"


def test_policy_alignment_flags_weaker_override() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "recommended_action": "drop",
                "final_action": "mask",
            }
        ]
    )

    preview_counts, alignment_status = _build_policy_alignment_preview(reviewed_df)

    assert preview_counts["Higher-risk overrides"] == 1
    assert alignment_status == "High"


def test_policy_alignment_counts_display_only_selection_and_marks_moderate() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "recommended_action": "keep",
                "final_action": "mask",
            }
        ]
    )

    preview_counts, alignment_status = _build_policy_alignment_preview(reviewed_df)

    assert preview_counts["Actions changed by user"] == 1
    assert preview_counts["Higher-risk overrides"] == 0
    assert preview_counts["Display-only protections selected"] == 1
    assert alignment_status == "Moderate"


def test_policy_alignment_changed_equal_strength_is_not_higher_risk() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "salary",
                "recommended_action": "generalize_light",
                "final_action": "generalize_medium",
            }
        ]
    )

    preview_counts, alignment_status = _build_policy_alignment_preview(reviewed_df)

    assert preview_counts["Actions changed by user"] == 1
    assert preview_counts["Higher-risk overrides"] == 0
    assert preview_counts["Display-only protections selected"] == 0
    assert alignment_status == "Low"


def test_build_policy_alignment_preview_keeps_existing_shape() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "birth_date",
                "recommended_action": "generalize",
                "final_action": "drop",
            }
        ]
    )

    preview_counts, alignment_status = _build_policy_alignment_preview(reviewed_df)

    assert isinstance(preview_counts, dict)
    assert set(preview_counts) == {
        "Actions changed by user",
        "Higher-risk overrides",
        "Display-only protections selected",
    }
    assert alignment_status in {"High", "Moderate", "Low"}


def test_build_policy_risk_items_returns_structured_policy_taxonomy() -> None:
    reviewed_df = pd.DataFrame(
        [
            {
                "column_name": "email",
                "recommended_action": "drop",
                "final_action": "mask",
            },
            {
                "column_name": "birth_date",
                "recommended_action": "generalize",
                "final_action": "drop",
            },
        ]
    )

    items = _build_policy_risk_items(reviewed_df)

    assert {
        "kind": "weaker_than_recommended",
        "columns": ["email"],
        "risk_category": "policy_alignment",
        "risk_level": "high",
    } in items
    assert {
        "kind": "display_only_selected",
        "columns": ["email"],
        "risk_category": "policy_alignment",
        "risk_level": "moderate",
    } in items
    assert {
        "kind": "changed_but_not_weaker",
        "columns": ["birth_date"],
        "risk_category": "policy_alignment",
        "risk_level": "low",
    } in items


def test_step6_risk_summary_prioritizes_failed_execution_over_quasi_text() -> None:
    risk_summary = _build_step6_risk_summary(
        {
            "failed_protection_attempt_count": 1,
            "failed_protection_attempt_columns": ["salary_band"],
            "quasi_identifier_generalized_remaining_count": 1,
            "quasi_identifier_generalized_remaining_columns": ["age_band"],
        }
    )

    assert risk_summary["level"] == "Moderate"
    assert risk_summary["execution_status"] == "Review needed"
    assert "could not be applied as requested" in risk_summary["text"]


def test_outputs_readiness_summary_keeps_high_uniqueness_as_review_not_blocker() -> None:
    outputs_summary = _build_outputs_readiness_summary(
        planning_context={"intended_use": "internal analysis"},
        validation_summary={
            "high_uniqueness_columns_after_anonymization_count": 1,
            "residual_direct_identifier_count": 0,
            "residual_technical_identifier_count": 0,
            "failed_protection_attempt_count": 0,
            "quasi_identifier_exact_remaining_count": 0,
            "quasi_identifier_generalized_remaining_count": 0,
        },
        reviewed_df=pd.DataFrame(),
    )

    assert outputs_summary["hero_class"] == "outputs-hero-safe"
    assert "review warnings" in outputs_summary["title"].lower()


def test_step7_next_steps_exact_quasi_points_back_to_existing_selector() -> None:
    next_steps = _build_step7_next_steps(
        {"quasi_identifier_exact_remaining_columns": ["birth_date"]}
    )

    assert "Final Protection Decisions" in next_steps[0]
    assert "increase the generalization level" in next_steps[0].lower()


def test_k_anonymity_violation_is_carried_into_risk_guidance() -> None:
    validation_summary = {
        "k_anonymity": {
            "k": 5,
            "records_below_k": 2,
            "records_below_k_percent": 20.0,
            "available": True,
            "applicable": True,
            "row_count": 10,
        }
    }

    risk_summary = _build_step6_risk_summary(validation_summary)
    matrix = _build_step7_use_decision_matrix({}, validation_summary)
    next_steps = _build_step7_next_steps(validation_summary)

    assert risk_summary["level"] == "Moderate"
    assert "smaller than k=5" in risk_summary["text"]
    assert matrix.loc[matrix["Use case"] == "External sharing", "Status"].item() == "⚠️ Not recommended"
    assert "2 records" in next_steps[0]


def test_generalization_strength_ranking_orders_strict_medium_light() -> None:
    assert _protection_strength_rank("generalize_strict") > _protection_strength_rank("generalize_medium")
    assert _protection_strength_rank("generalize_medium") > _protection_strength_rank("generalize_light")
    assert _protection_strength_rank("generalize_light") > _protection_strength_rank("add_noise")
    assert _protection_strength_rank("generalize") == _protection_strength_rank("generalize_light")


def test_step6_risk_summary_points_to_increasing_generalization_level() -> None:
    risk_summary = _build_step6_risk_summary(
        {"quasi_identifier_exact_remaining_count": 1}
    )

    assert "increase protection" in risk_summary["next_step"].lower()
    assert "further generalization" in risk_summary["next_step"].lower()


def test_step6_remaining_risk_groups_exact_only() -> None:
    groups = _build_step6_remaining_risk_groups(
        {
            "validation_warning_items": [
                {
                    "kind": "exact_indirect_identifiers",
                    "columns": ["birth_date"],
                    "risk_category": "quasi_identifier",
                    "risk_level": "high",
                }
            ]
        }
    )

    assert [group["kind"] for group in groups] == ["exact_indirect_identifiers"]
    assert groups[0]["columns"] == ["birth_date"]


def test_step6_remaining_risk_groups_generalized_only() -> None:
    groups = _build_step6_remaining_risk_groups(
        {
            "validation_warning_items": [
                {
                    "kind": "generalized_indirect_identifiers",
                    "columns": ["age_band"],
                    "risk_category": "quasi_identifier",
                    "risk_level": "moderate",
                }
            ]
        }
    )

    assert [group["kind"] for group in groups] == ["generalized_indirect_identifiers"]
    assert groups[0]["columns"] == ["age_band"]


def test_step6_remaining_risk_groups_high_uniqueness_only() -> None:
    groups = _build_step6_remaining_risk_groups(
        {
            "validation_warning_items": [
                {
                    "kind": "high_uniqueness",
                    "columns": ["postcode_hash"],
                    "risk_category": "uniqueness",
                    "risk_level": "moderate",
                }
            ]
        }
    )

    assert [group["kind"] for group in groups] == ["high_uniqueness"]
    assert groups[0]["columns"] == ["postcode_hash"]


def test_step6_remaining_risk_groups_keep_exact_generalized_and_uniqueness_separate() -> None:
    groups = _build_step6_remaining_risk_groups(
        {
            "validation_warning_items": [
                {
                    "kind": "exact_indirect_identifiers",
                    "columns": ["birth_date"],
                    "risk_category": "quasi_identifier",
                    "risk_level": "high",
                },
                {
                    "kind": "generalized_indirect_identifiers",
                    "columns": ["age_band"],
                    "risk_category": "quasi_identifier",
                    "risk_level": "moderate",
                },
                {
                    "kind": "high_uniqueness",
                    "columns": ["rare_code"],
                    "risk_category": "uniqueness",
                    "risk_level": "moderate",
                },
            ]
        }
    )

    assert [group["kind"] for group in groups] == [
        "exact_indirect_identifiers",
        "generalized_indirect_identifiers",
        "high_uniqueness",
    ]
    assert groups[0]["columns"] == ["birth_date"]
    assert groups[1]["columns"] == ["age_band"]
    assert groups[2]["columns"] == ["rare_code"]


def test_step6_remaining_risk_groups_do_not_mix_pseudonymized_direct_identifiers_into_exact_group() -> None:
    groups = _build_step6_remaining_risk_groups(
        {
            "validation_warning_items": [
                {
                    "kind": "direct_identifiers",
                    "columns": ["customer_id"],
                    "risk_category": "direct_identifier",
                    "risk_level": "high",
                },
                {
                    "kind": "exact_indirect_identifiers",
                    "columns": ["birth_date"],
                    "risk_category": "quasi_identifier",
                    "risk_level": "high",
                },
            ]
        }
    )

    assert [group["kind"] for group in groups] == ["exact_indirect_identifiers"]
    assert groups[0]["columns"] == ["birth_date"]
