from __future__ import annotations

import pandas as pd

from data_anonymization_app.anonymization.service import plan_anonymization
from data_anonymization_app.profiling.service import classify_columns


def test_realistic_column_classification_and_reasons() -> None:
    profiling_summary = pd.DataFrame(
        [
            {"column_name": "first_name", "dtype": "object", "unique_percent": 40.0, "sample_values": ["Alice"]},
            {"column_name": "last_name", "dtype": "object", "unique_percent": 45.0, "sample_values": ["Smith"]},
            {"column_name": "full_name", "dtype": "object", "unique_percent": 60.0, "sample_values": ["Alice Smith"]},
            {"column_name": "website", "dtype": "object", "unique_percent": 70.0, "sample_values": ["https://example.com"]},
            {"column_name": "company_name", "dtype": "object", "unique_percent": 30.0, "sample_values": ["Acme Inc"]},
            {
                "column_name": "uuid",
                "dtype": "object",
                "unique_percent": 100.0,
                "sample_values": ["550e8400-e29b-41d4-a716-446655440000"],
            },
            {"column_name": "index", "dtype": "int64", "unique_percent": 100.0, "sample_values": [1, 2, 3]},
            {"column_name": "internal_id", "dtype": "int64", "unique_percent": 100.0, "sample_values": [101, 102]},
            {"column_name": "customer_id", "dtype": "int64", "unique_percent": 100.0, "sample_values": [501, 502]},
            {"column_name": "birth_date", "dtype": "object", "unique_percent": 90.0, "sample_values": ["1990-01-01"]},
            {"column_name": "salary", "dtype": "float64", "unique_percent": 90.0, "sample_values": [5000.0]},
        ]
    )

    classification_df, classification_json = classify_columns(profiling_summary)
    assert classification_df.to_dict(orient="records") == classification_json

    by_name = {row["column_name"]: row for row in classification_json}
    assert by_name["first_name"]["classification"] == "direct_identifier"
    assert by_name["last_name"]["classification"] == "direct_identifier"
    assert by_name["full_name"]["classification"] == "direct_identifier"
    assert by_name["website"]["classification"] == "non_sensitive"
    assert by_name["company_name"]["classification"] == "non_sensitive"
    assert by_name["uuid"]["classification"] == "technical_identifier"
    assert by_name["index"]["classification"] == "technical_identifier"
    assert by_name["internal_id"]["classification"] == "technical_identifier"
    assert by_name["customer_id"]["classification"] == "direct_identifier"
    assert by_name["birth_date"]["classification"] == "quasi_identifier"
    assert by_name["salary"]["classification"] == "sensitive_attribute"

    assert by_name["first_name"]["reason"].startswith("name-heuristic")
    assert by_name["website"]["reason"] == "name-heuristic: generic website/url/domain column"
    assert by_name["company_name"]["reason"] == "name-heuristic: company/business column"
    assert by_name["uuid"]["reason"].startswith("name-heuristic")
    assert by_name["birth_date"]["reason"].startswith("name-heuristic")
    assert by_name["salary"]["reason"].startswith("name-heuristic")


def test_planning_output_and_override_handling_for_realistic_columns() -> None:
    classification_summary = pd.DataFrame(
        [
            {"column_name": "first_name", "classification": "direct_identifier"},
            {"column_name": "website", "classification": "non_sensitive"},
            {"column_name": "company_name", "classification": "non_sensitive"},
            {"column_name": "uuid", "classification": "technical_identifier"},
            {"column_name": "birth_date", "classification": "quasi_identifier"},
            {"column_name": "salary", "classification": "sensitive_attribute", "override_action": "keep"},
        ]
    )

    plan_df, plan_json = plan_anonymization(classification_summary)
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

    by_name = {row["column_name"]: row for row in plan_json}
    assert by_name["first_name"]["recommended_action"] == "drop"
    assert by_name["website"]["recommended_action"] == "keep"
    assert by_name["company_name"]["recommended_action"] == "keep"
    assert by_name["uuid"]["recommended_action"] == "drop"
    assert by_name["birth_date"]["recommended_action"] == "generalize"
    assert by_name["salary"]["recommended_action"] == "add_noise"

    assert by_name["salary"]["final_action"] == "keep"
    assert "override_action provided by user" in by_name["salary"]["planning_reason"]
    assert by_name["uuid"]["final_action"] == "drop"
    assert by_name["uuid"]["planning_reason"] == "default mapping by classification"
