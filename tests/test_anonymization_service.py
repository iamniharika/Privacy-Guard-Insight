from __future__ import annotations

import pandas as pd
import pytest

from data_anonymization_app.anonymization.service import (
    anonymize_dataset,
    plan_anonymization,
)


def test_plan_anonymization_maps_all_classes_to_actions() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "email", "classification": "direct_identifier"},
            {"column_name": "salary", "classification": "sensitive_attribute"},
            {"column_name": "age", "classification": "quasi_identifier"},
            {"column_name": "city", "classification": "non_sensitive"},
            {"column_name": "row_id", "classification": "technical_identifier"},
        ]
    )

    plan_df, plan_json = plan_anonymization(classification_summary)

    assert isinstance(plan_df, pd.DataFrame)
    assert isinstance(plan_json, list)
    assert plan_df.to_dict(orient="records") == plan_json
    assert list(plan_df.columns) == [
        "column_name",
        "classification",
        "functional_role",
        "recommended_action",
        "final_action",
        "planning_reason_code",
        "planning_reason",
        "policy_context",
        "record_linkage_required",
    ]

    by_name = {row["column_name"]: row["recommended_action"] for row in plan_json}
    assert by_name["email"] == "drop"
    assert by_name["salary"] == "add_noise"
    assert by_name["age"] == "generalize"
    assert by_name["city"] == "keep"
    assert by_name["row_id"] == "drop"
    assert all(row["recommended_action"] == row["final_action"] for row in plan_json)
    assert all(row["planning_reason"] == "default mapping by classification" for row in plan_json)


def test_plan_anonymization_default_behavior_when_context_absent() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "customer_id", "classification": "direct_identifier"},
            {"column_name": "salary", "classification": "sensitive_attribute"},
        ]
    )

    _, baseline_json = plan_anonymization(classification_summary)
    _, explicit_none_context_json = plan_anonymization(classification_summary, context=None)

    assert explicit_none_context_json == baseline_json
    by_name = {row["column_name"]: row for row in baseline_json}
    assert by_name["customer_id"]["recommended_action"] == "drop"
    assert by_name["salary"]["recommended_action"] == "add_noise"


def test_plan_anonymization_validates_required_fields() -> None:
    classification_summary = pd.DataFrame([{"column_name": "email"}])

    with pytest.raises(ValueError, match="missing required columns"):
        plan_anonymization(classification_summary)


def test_plan_anonymization_rejects_unknown_classification() -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "custom", "classification": "custom_type"}]
    )

    with pytest.raises(ValueError, match="Unsupported classification"):
        plan_anonymization(classification_summary)


def test_plan_anonymization_applies_override_action_when_present() -> None:
    classification_summary = pd.DataFrame(
        [
            {
                "column_name": "salary",
                "classification": "sensitive_attribute",
                "override_action": "keep",
            },
            {
                "column_name": "age",
                "classification": "quasi_identifier",
                "override_action": "generalize",
            },
            {
                "column_name": "handle",
                "classification": "direct_identifier",
                "override_action": "pseudonymize",
            },
        ]
    )

    _, plan_json = plan_anonymization(classification_summary)
    by_name = {row["column_name"]: row for row in plan_json}

    assert by_name["salary"]["recommended_action"] == "add_noise"
    assert by_name["salary"]["final_action"] == "keep"
    assert "override_action provided by user" in by_name["salary"]["planning_reason"]
    assert by_name["age"]["final_action"] == "generalize_light"
    assert by_name["handle"]["final_action"] == "pseudonymize"


def test_plan_anonymization_rejects_unsupported_override_action() -> None:
    classification_summary = pd.DataFrame(
        [
            {
                "column_name": "age",
                "classification": "quasi_identifier",
                "override_action": "encrypt",
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported action"):
        plan_anonymization(classification_summary)


@pytest.mark.parametrize(
    "intended_use",
    ["internal analysis", "dashboard analytics"],
)
def test_plan_anonymization_policy_need_record_linkage_prefers_pseudonymize(
    intended_use: str,
) -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "customer_id", "classification": "direct_identifier"}]
    )

    _, plan_json = plan_anonymization(
        classification_summary,
        context={"need_record_linkage": True, "intended_use": intended_use},
    )

    assert plan_json[0]["recommended_action"] == "pseudonymize"
    assert plan_json[0]["final_action"] == "pseudonymize"
    assert (
        plan_json[0]["planning_reason"]
        == "policy: identity/linkage field pseudonymized because record linkage is required"
    )


def test_plan_anonymization_policy_external_sharing_is_stricter() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "customer_id", "classification": "direct_identifier"},
            {"column_name": "record_id", "classification": "technical_identifier"},
            {"column_name": "birth_date", "classification": "quasi_identifier"},
        ]
    )

    _, plan_json = plan_anonymization(
        classification_summary,
        context={"intended_use": "external sharing", "need_record_linkage": True},
    )
    by_name = {row["column_name"]: row for row in plan_json}

    assert by_name["customer_id"]["recommended_action"] == "drop"
    assert by_name["record_id"]["recommended_action"] == "drop"
    assert by_name["birth_date"]["recommended_action"] == "generalize"
    assert by_name["customer_id"]["planning_reason"] == "policy: identity/linkage field dropped for external sharing"


def test_plan_anonymization_linkage_role_allows_technical_identifier_pseudonymization() -> None:
    classification_summary = pd.DataFrame(
        [
            {
                "column_name": "record_id",
                "classification": "technical_identifier",
                "functional_role": "linkage",
            }
        ]
    )

    _, plan_json = plan_anonymization(
        classification_summary,
        context={"need_record_linkage": True, "intended_use": "internal analysis"},
    )

    assert plan_json[0]["recommended_action"] == "pseudonymize"
    assert plan_json[0]["final_action"] == "pseudonymize"
    assert plan_json[0]["record_linkage_required"] is True


def test_plan_anonymization_policy_domain_reason_annotations() -> None:
    healthcare_summary = pd.DataFrame(
        [{"column_name": "diagnosis", "classification": "sensitive_attribute"}]
    )
    _, healthcare_plan_json = plan_anonymization(
        healthcare_summary,
        context={"dataset_domain": "healthcare"},
    )
    assert "healthcare" in healthcare_plan_json[0]["planning_reason"]

    finance_summary = pd.DataFrame(
        [
            {"column_name": "customer_id", "classification": "direct_identifier"},
            {"column_name": "internal_id", "classification": "technical_identifier"},
        ]
    )
    _, finance_plan_json = plan_anonymization(
        finance_summary,
        context={"dataset_domain": "finance"},
    )
    assert all("finance" in row["planning_reason"] for row in finance_plan_json)


@pytest.mark.parametrize(
    "context,expected_action,expected_reason_fragment",
    [
        (
            {"intended_use": "external sharing"},
            "generalize",
            "generalized for external sharing",
        ),
        (
            {"intended_use": "research"},
            "generalize",
            "generalized for research",
        ),
        (
            {"intended_use": "ML training"},
            "add_noise",
            "noised for ML training",
        ),
        (
            {"intended_use": "dashboard analytics"},
            "add_noise",
            "noised for dashboard analytics",
        ),
        (
            {"dataset_size_rows": 999},
            "generalize",
            "small dataset favors generalization",
        ),
        (
            {"intended_use": "dashboard analytics", "dataset_size_rows": 999},
            "generalize",
            "small dataset favors generalization",
        ),
        (
            {"intended_use": "ML training", "dataset_size_rows": 999},
            "add_noise",
            "noised for ML training",
        ),
    ],
)
def test_plan_anonymization_sensitive_attribute_context_rules(
    context: dict[str, object],
    expected_action: str,
    expected_reason_fragment: str,
) -> None:
    classification_summary = pd.DataFrame(
        [{"column_name": "salary", "classification": "sensitive_attribute"}]
    )

    _, plan_json = plan_anonymization(classification_summary, context=context)

    assert plan_json[0]["recommended_action"] == expected_action
    assert plan_json[0]["final_action"] == expected_action
    assert expected_reason_fragment in plan_json[0]["planning_reason"]


def test_anonymize_dataset_applies_v1_actions_deterministically() -> None:
    source_df = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "age": [23, 37, None, 44],
            "salary": [1000.0, 1010.0, None, 980.0],
            "city": ["Paris", "Berlin", "Rome", "Madrid"],
        }
    )
    original_df = source_df.copy(deep=True)

    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "customer_id",
                "classification": "direct_identifier",
                "recommended_action": "drop",
                "final_action": "drop",
            },
            {
                "column_name": "age",
                "classification": "quasi_identifier",
                "recommended_action": "generalize",
                "final_action": "generalize",
            },
            {
                "column_name": "salary",
                "classification": "sensitive_attribute",
                "recommended_action": "add_noise",
                "final_action": "add_noise",
            },
            {
                "column_name": "city",
                "classification": "non_sensitive",
                "recommended_action": "keep",
                "final_action": "keep",
            },
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    pd.testing.assert_frame_equal(source_df, original_df)

    assert list(transformed_df.columns) == ["age", "salary", "city"]
    assert transformed_df["age"].tolist() == ["20-29", "30-39", None, "40-49"]
    salary_values = transformed_df["salary"].tolist()
    assert salary_values[0] == pytest.approx(999.8)
    assert salary_values[1] == pytest.approx(1009.9)
    assert pd.isna(salary_values[2])
    assert salary_values[3] == pytest.approx(980.1)
    assert transformed_df["city"].tolist() == ["Paris", "Berlin", "Rome", "Madrid"]

    assert isinstance(transformation_log, list)
    assert len(transformation_log) == 4
    assert transformation_log[0]["column_name"] == "customer_id"
    assert transformation_log[0]["action_requested"] == "drop"
    assert transformation_log[0]["action_applied"] == "drop"


def test_anonymize_dataset_non_numeric_generalize_and_add_noise_noop() -> None:
    source_df = pd.DataFrame({"group": ["A", "B"], "notes": ["x", "y"]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "group",
                "classification": "quasi_identifier",
                "recommended_action": "generalize",
                "final_action": "generalize",
            },
            {
                "column_name": "notes",
                "classification": "sensitive_attribute",
                "recommended_action": "add_noise",
                "final_action": "add_noise",
            },
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    pd.testing.assert_frame_equal(transformed_df, source_df)
    assert transformation_log[0]["action_requested"] == "generalize_light"
    assert transformation_log[0]["action_applied"] == "generalize_light"
    assert transformation_log[0]["notes"] == "categorical generalized lightly (rare values grouped to other)"
    assert transformation_log[1]["action_requested"] == "add_noise"
    assert transformation_log[1]["action_applied"] == "keep"
    assert transformation_log[1]["notes"] == "add_noise skipped: non-numeric column"


def test_anonymize_dataset_uses_final_action_over_recommendation_in_log() -> None:
    source_df = pd.DataFrame({"customer_id": [101, 102]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "customer_id",
                "classification": "direct_identifier",
                "recommended_action": "drop",
                "final_action": "pseudonymize",
            }
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    assert transformed_df["customer_id"].tolist() != source_df["customer_id"].tolist()
    assert transformation_log[0]["action_requested"] == "pseudonymize"
    assert transformation_log[0]["action_applied"] == "pseudonymize"


def test_anonymize_dataset_validates_required_plan_fields() -> None:
    source_df = pd.DataFrame({"x": [1, 2]})
    invalid_plan = pd.DataFrame([{"column_name": "x"}])

    with pytest.raises(ValueError, match="missing required columns"):
        anonymize_dataset(source_df, invalid_plan)


def test_anonymize_dataset_rejects_unsupported_actions() -> None:
    source_df = pd.DataFrame({"x": [1, 2]})
    invalid_plan = pd.DataFrame(
        [
            {
                "column_name": "x",
                "classification": "non_sensitive",
                "recommended_action": "encrypt",
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported action"):
        anonymize_dataset(source_df, invalid_plan)


def test_anonymize_dataset_accepts_mask_as_supported_action() -> None:
    source_df = pd.DataFrame({"name": ["Alice"]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "name",
                "classification": "direct_identifier",
                "recommended_action": "mask",
                "final_action": "mask",
            }
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    assert transformed_df["name"].tolist() == ["A****"]
    assert transformation_log[0]["action_requested"] == "mask"
    assert transformation_log[0]["action_applied"] == "mask"
    assert transformation_log[0]["notes"] == "display-oriented masking applied"


def test_anonymize_dataset_mask_preserves_nulls_and_masks_deterministically(tmp_path) -> None:
    source_df = pd.DataFrame(
        {
            "email": ["john.smith@gmail.com", None],
            "phone": ["5551237890", None],
            "name": ["Alice", None],
        }
    )
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "email",
                "classification": "direct_identifier",
                "recommended_action": "mask",
                "final_action": "mask",
            },
            {
                "column_name": "phone",
                "classification": "direct_identifier",
                "recommended_action": "mask",
                "final_action": "mask",
            },
            {
                "column_name": "name",
                "classification": "direct_identifier",
                "recommended_action": "mask",
                "final_action": "mask",
            },
        ]
    )
    vault_path = tmp_path / "unused-vault.json"

    transformed_df, transformation_log = anonymize_dataset(
        source_df,
        transformation_plan,
        vault_path=vault_path,
    )

    assert transformed_df["email"].tolist() == ["j***@gmail.com", None]
    assert transformed_df["phone"].tolist() == ["***7890", None]
    assert transformed_df["name"].tolist() == ["A****", None]
    assert all(entry["notes"] == "display-oriented masking applied" for entry in transformation_log)
    assert not vault_path.exists()


def test_anonymize_dataset_prefers_final_action_over_recommended_action() -> None:
    source_df = pd.DataFrame({"value": [11, 15]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "value",
                "classification": "quasi_identifier",
                "recommended_action": "drop",
                "final_action": "generalize",
            }
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    assert transformed_df["value"].tolist() == ["10-19", "10-19"]
    assert transformation_log[0]["action_requested"] == "generalize_light"
    assert transformation_log[0]["action_applied"] == "generalize_light"


def test_anonymize_dataset_technical_identifier_never_generalized() -> None:
    source_df = pd.DataFrame({"row_id": [101, 102]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "row_id",
                "classification": "technical_identifier",
                "recommended_action": "generalize",
                "final_action": "generalize",
            }
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    pd.testing.assert_frame_equal(transformed_df, source_df)
    assert transformation_log[0]["action_requested"] == "generalize_light"
    assert transformation_log[0]["action_applied"] == "keep"
    assert "technical_identifier columns are not generalized" in transformation_log[0]["notes"]


def test_anonymize_dataset_requires_at_least_one_action_column() -> None:
    source_df = pd.DataFrame({"x": [1, 2]})
    invalid_plan = pd.DataFrame([{"column_name": "x", "classification": "non_sensitive"}])

    with pytest.raises(ValueError, match="must include 'final_action' or 'recommended_action'"):
        anonymize_dataset(source_df, invalid_plan)


def test_anonymize_dataset_legacy_generalize_maps_to_light() -> None:
    source_df = pd.DataFrame({"birth_date": ["2024-01-15", "2024-02-20"]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "birth_date",
                "classification": "quasi_identifier",
                "recommended_action": "generalize",
                "final_action": "generalize",
            }
        ]
    )

    transformed_df, transformation_log = anonymize_dataset(source_df, transformation_plan)

    assert transformed_df["birth_date"].tolist() == ["2024-01", "2024-02"]
    assert transformation_log[0]["action_requested"] == "generalize_light"
    assert transformation_log[0]["action_applied"] == "generalize_light"


def test_anonymize_dataset_generalization_levels_produce_different_outputs() -> None:
    source_df = pd.DataFrame(
        {
            "birth_date": ["2024-01-15", "2017-07-03", "1994-09-12"],
            "age": [23, 37, 64],
            "city": ["Paris", "Berlin", "Rome"],
        }
    )

    def _run(level: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        transformation_plan = pd.DataFrame(
            [
                {
                    "column_name": "birth_date",
                    "classification": "quasi_identifier",
                    "recommended_action": level,
                    "final_action": level,
                },
                {
                    "column_name": "age",
                    "classification": "quasi_identifier",
                    "recommended_action": level,
                    "final_action": level,
                },
                {
                    "column_name": "city",
                    "classification": "quasi_identifier",
                    "recommended_action": level,
                    "final_action": level,
                },
            ]
        )
        return anonymize_dataset(source_df, transformation_plan)

    light_df, light_log = _run("generalize_light")
    medium_df, medium_log = _run("generalize_medium")
    strict_df, strict_log = _run("generalize_strict")

    assert light_df["birth_date"].tolist() == ["2024-01", "2017-07", "1994-09"]
    assert medium_df["birth_date"].tolist() == ["2024", "2017", "1994"]
    assert strict_df["birth_date"].tolist() == ["2020s", "2010s", "1990s"]

    assert light_df["age"].tolist() == ["20-29", "30-39", "60-69"]
    assert medium_df["age"].tolist() == ["0-24", "25-49", "50-74"]
    assert strict_df["age"].tolist() == ["0-49", "0-49", "50-99"]

    assert light_df["city"].tolist() == ["Paris", "Berlin", "Rome"]
    assert medium_df["city"].tolist() == ["Paris", "Berlin", "Rome"]
    assert strict_df["city"].tolist() == ["Paris", "Berlin", "other"]

    assert [entry["action_applied"] for entry in light_log] == ["generalize_light"] * 3
    assert [entry["action_applied"] for entry in medium_log] == ["generalize_medium"] * 3
    assert [entry["action_applied"] for entry in strict_log] == ["generalize_strict"] * 3


def test_anonymize_dataset_generalize_fallback_records_requested_vs_applied() -> None:
    source_df = pd.DataFrame({"row_id": [101, 102]})
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "row_id",
                "classification": "technical_identifier",
                "recommended_action": "generalize_strict",
                "final_action": "generalize_strict",
            }
        ]
    )

    _, transformation_log = anonymize_dataset(source_df, transformation_plan)

    assert transformation_log[0]["action_requested"] == "generalize_strict"
    assert transformation_log[0]["action_applied"] == "keep"
    assert "generalize_strict requested" in transformation_log[0]["notes"]


def test_anonymize_dataset_pseudonymize_is_deterministic_and_stringifies_values(tmp_path) -> None:
    source_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", None],
            "customer_number": [1001, 1002, 1001, None],
        }
    )
    transformation_plan = pd.DataFrame(
        [
            {
                "column_name": "name",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            },
            {
                "column_name": "customer_number",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            },
        ]
    )
    vault_path = tmp_path / "mapping_vault.json"

    transformed_df, transformation_log = anonymize_dataset(
        source_df,
        transformation_plan,
        vault_path=vault_path,
    )

    assert transformed_df["name"].tolist() == ["token_000001", "token_000002", "token_000001", None]
    assert transformed_df["customer_number"].tolist() == [
        "token_000001",
        "token_000002",
        "token_000001",
        None,
    ]
    assert transformation_log[0]["action_requested"] == "pseudonymize"
    assert transformation_log[0]["action_applied"] == "pseudonymize"
    assert transformation_log[0]["notes"] == "persistent vault mapping applied"
    assert transformation_log[1]["action_requested"] == "pseudonymize"
    assert transformation_log[1]["action_applied"] == "pseudonymize"
    assert transformation_log[1]["notes"] == "persistent vault mapping applied"


def test_anonymize_dataset_pseudonymize_reuses_tokens_across_runs(tmp_path) -> None:
    vault_path = tmp_path / "vault.json"
    plan = pd.DataFrame(
        [
            {
                "column_name": "name",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            }
        ]
    )

    run1_df = pd.DataFrame({"name": ["Alice", "Bob"]})
    transformed_run1, _ = anonymize_dataset(run1_df, plan, vault_path=vault_path)
    run2_df = pd.DataFrame({"name": ["Bob", "Alice", "Carol"]})
    transformed_run2, _ = anonymize_dataset(run2_df, plan, vault_path=vault_path)

    assert vault_path.exists()
    assert transformed_run1["name"].tolist() == ["token_000001", "token_000002"]
    assert transformed_run2["name"].tolist() == ["token_000002", "token_000001", "token_000003"]


def test_anonymize_dataset_pseudonymize_column_scoped_tokens(tmp_path) -> None:
    source_df = pd.DataFrame({"col_a": ["same"], "col_b": ["same"]})
    plan = pd.DataFrame(
        [
            {
                "column_name": "col_a",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            },
            {
                "column_name": "col_b",
                "classification": "direct_identifier",
                "recommended_action": "pseudonymize",
                "final_action": "pseudonymize",
            },
        ]
    )

    transformed_df, _ = anonymize_dataset(source_df, plan, vault_path=tmp_path / "vault.json")
    assert transformed_df["col_a"].tolist() == ["token_000001"]
    assert transformed_df["col_b"].tolist() == ["token_000001"]
