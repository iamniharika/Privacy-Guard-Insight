from __future__ import annotations

import pandas as pd
import pytest

from data_anonymization_app.profiling.service import (
    DatasetIngestionError,
    classify_columns,
    load_dataset,
    profile_dataset,
)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["alice", "bob", "carol"],
        }
    )


@pytest.mark.parametrize(
    "extension,file_type",
    [
        ("csv", "csv"),
        ("xlsx", "xlsx"),
        ("parquet", "parquet"),
    ],
)
def test_load_dataset_supported_formats(tmp_path, extension: str, file_type: str) -> None:
    source_df = _sample_dataframe()
    file_path = tmp_path / f"input.{extension}"

    if extension == "csv":
        source_df.to_csv(file_path, index=False)
    elif extension == "xlsx":
        pytest.importorskip("openpyxl")
        source_df.to_excel(file_path, index=False)
    elif extension == "parquet":
        source_df.to_parquet(file_path, index=False)

    loaded_df, metadata = load_dataset(file_path)

    pd.testing.assert_frame_equal(loaded_df, source_df)
    assert metadata.rows == 3
    assert metadata.columns == 2
    assert metadata.column_names == ["id", "name"]
    assert metadata.file_type == file_type


def test_load_dataset_unsupported_format(tmp_path) -> None:
    file_path = tmp_path / "input.txt"
    file_path.write_text("not tabular")

    with pytest.raises(DatasetIngestionError, match="Unsupported file format"):
        load_dataset(file_path)


def test_load_dataset_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(DatasetIngestionError, match="File not found"):
        load_dataset(missing_path)


def test_profile_dataset_returns_dataframe_and_json_summary() -> None:
    source_df = pd.DataFrame(
        {
            "age": [20, 30, None, 30],
            "city": ["Paris", None, "Paris", "Berlin"],
            "active": [True, False, True, True],
        }
    )

    summary_df, summary_json = profile_dataset(source_df)

    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(summary_json, list)
    assert len(summary_json) == 3
    assert summary_df.to_dict(orient="records") == summary_json

    age_row = next(row for row in summary_json if row["column_name"] == "age")
    assert age_row["null_count"] == 1
    assert age_row["null_percent"] == pytest.approx(25.0)
    assert age_row["unique_count"] == 2
    assert age_row["unique_percent"] == pytest.approx(50.0)
    assert age_row["sample_values"] == [20.0, 30.0, 30.0]
    assert age_row["min_value"] == 20.0
    assert age_row["max_value"] == 30.0

    city_row = next(row for row in summary_json if row["column_name"] == "city")
    assert city_row["null_count"] == 1
    assert city_row["null_percent"] == pytest.approx(25.0)
    assert city_row["unique_count"] == 2
    assert city_row["unique_percent"] == pytest.approx(50.0)
    assert city_row["sample_values"] == ["Paris", "Paris", "Berlin"]
    assert city_row["min_value"] is None
    assert city_row["max_value"] is None


def test_profile_dataset_limits_sample_values_and_handles_empty_dataframe() -> None:
    source_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6]})

    summary_df, summary_json = profile_dataset(source_df)
    id_row = summary_json[0]
    assert id_row["sample_values"] == [1, 2, 3, 4, 5]
    assert id_row["min_value"] == 1
    assert id_row["max_value"] == 6
    assert id_row["null_percent"] == pytest.approx(0.0)
    assert id_row["unique_percent"] == pytest.approx(100.0)
    assert list(summary_df.columns) == [
        "column_name",
        "dtype",
        "null_count",
        "null_percent",
        "unique_count",
        "unique_percent",
        "sample_values",
        "min_value",
        "max_value",
    ]

    empty_df = pd.DataFrame({"value": pd.Series([], dtype="float64")})
    _, empty_summary_json = profile_dataset(empty_df)
    empty_row = empty_summary_json[0]
    assert empty_row["null_count"] == 0
    assert empty_row["null_percent"] == pytest.approx(0.0)
    assert empty_row["unique_count"] == 0
    assert empty_row["unique_percent"] == pytest.approx(0.0)
    assert empty_row["sample_values"] == []
    assert empty_row["min_value"] is None
    assert empty_row["max_value"] is None


def test_classify_columns_returns_dataframe_and_json_summary() -> None:
    profiling_summary = pd.DataFrame(
        [
            {"column_name": "record_id", "dtype": "int64", "unique_percent": 100.0},
            {"column_name": "customer_id", "dtype": "int64", "unique_percent": 100.0},
            {"column_name": "salary", "dtype": "float64", "unique_percent": 75.0},
            {"column_name": "age_group", "dtype": "int64", "unique_percent": 20.0},
            {"column_name": "transaction_amount", "dtype": "float64", "unique_percent": 95.0},
            {"column_name": "city", "dtype": "object", "unique_percent": 10.0},
        ]
    )

    classification_df, classification_json = classify_columns(profiling_summary)

    assert isinstance(classification_df, pd.DataFrame)
    assert isinstance(classification_json, list)
    assert classification_df.to_dict(orient="records") == classification_json
    assert list(classification_df.columns) == ["column_name", "classification", "reason"]

    by_name = {row["column_name"]: row["classification"] for row in classification_json}
    assert by_name["record_id"] == "technical_identifier"
    assert by_name["customer_id"] == "direct_identifier"
    assert by_name["salary"] == "sensitive_attribute"
    assert by_name["age_group"] == "quasi_identifier"
    assert by_name["transaction_amount"] == "quasi_identifier"
    assert by_name["city"] == "quasi_identifier"
    assert all("reason" in row for row in classification_json)


def test_classify_columns_validates_required_fields() -> None:
    profiling_summary = pd.DataFrame([{"column_name": "email"}])

    with pytest.raises(ValueError, match="missing required columns"):
        classify_columns(profiling_summary)


def test_classify_columns_applies_pattern_and_override_rules() -> None:
    profiling_summary = pd.DataFrame(
        [
            {
                "column_name": "contact_value",
                "dtype": "object",
                "unique_percent": 10.0,
                "sample_values": ["alice@example.com", "bob@example.com"],
            },
            {
                "column_name": "support_phone_raw",
                "dtype": "object",
                "unique_percent": 5.0,
                "sample_values": ["+1 (202) 555-0100"],
            },
            {
                "column_name": "session_token",
                "dtype": "object",
                "unique_percent": 100.0,
                "sample_values": ["550e8400-e29b-41d4-a716-446655440000"],
            },
            {
                "column_name": "website",
                "dtype": "object",
                "unique_percent": 60.0,
                "sample_values": ["https://example.com"],
            },
            {
                "column_name": "company_name",
                "dtype": "object",
                "unique_percent": 70.0,
                "sample_values": ["Acme Corp"],
            },
            {
                "column_name": "user_website",
                "dtype": "object",
                "unique_percent": 50.0,
                "sample_values": ["https://portfolio.example"],
            },
        ]
    )

    _, classification_json = classify_columns(profiling_summary)
    by_name = {row["column_name"]: row for row in classification_json}

    assert by_name["contact_value"]["classification"] == "direct_identifier"
    assert by_name["support_phone_raw"]["classification"] == "direct_identifier"
    assert by_name["session_token"]["classification"] == "technical_identifier"
    assert by_name["website"]["classification"] == "non_sensitive"
    assert by_name["company_name"]["classification"] == "non_sensitive"
    assert by_name["user_website"]["classification"] == "direct_identifier"


def test_classify_columns_upgrades_with_presidio_signals() -> None:
    profiling_summary = pd.DataFrame(
        [
            {"column_name": "free_text", "dtype": "object", "unique_percent": 5.0},
            {"column_name": "bio", "dtype": "object", "unique_percent": 20.0},
            {"column_name": "website", "dtype": "object", "unique_percent": 40.0},
        ]
    )
    presidio_summary = pd.DataFrame(
        [
            {
                "column_name": "free_text",
                "presidio_entities_detected": ["EMAIL_ADDRESS"],
                "presidio_hit_count": 2,
                "presidio_used": True,
            },
            {
                "column_name": "bio",
                "presidio_entities_detected": ["PERSON"],
                "presidio_hit_count": 1,
                "presidio_used": True,
            },
            {
                "column_name": "website",
                "presidio_entities_detected": ["URL"],
                "presidio_hit_count": 3,
                "presidio_used": True,
            },
        ]
    )

    _, classification_json = classify_columns(profiling_summary, presidio_summary=presidio_summary)
    by_name = {row["column_name"]: row for row in classification_json}

    assert by_name["free_text"]["classification"] == "direct_identifier"
    assert by_name["free_text"]["reason"] == "presidio: email/phone entity detected"
    assert by_name["bio"]["classification"] == "quasi_identifier"
    assert by_name["bio"]["reason"] == "presidio: person/location entity detected"
    assert by_name["website"]["classification"] == "non_sensitive"
    assert by_name["website"]["reason"] == "presidio: url-only entities detected"


def test_classify_columns_presidio_cannot_downgrade_higher_risk() -> None:
    profiling_summary = pd.DataFrame(
        [
            {"column_name": "salary", "dtype": "float64", "unique_percent": 90.0},
            {"column_name": "record_id", "dtype": "int64", "unique_percent": 100.0},
        ]
    )
    presidio_summary = pd.DataFrame(
        [
            {
                "column_name": "salary",
                "presidio_entities_detected": ["URL"],
                "presidio_hit_count": 2,
                "presidio_used": True,
            },
            {
                "column_name": "record_id",
                "presidio_entities_detected": ["LOCATION"],
                "presidio_hit_count": 1,
                "presidio_used": True,
            },
        ]
    )

    _, classification_json = classify_columns(profiling_summary, presidio_summary=presidio_summary)
    by_name = {row["column_name"]: row for row in classification_json}

    assert by_name["salary"]["classification"] == "sensitive_attribute"
    assert by_name["salary"]["reason"] == "name-heuristic: sensitive attribute keyword"
    assert by_name["record_id"]["classification"] == "technical_identifier"
    assert by_name["record_id"]["reason"] == "name-heuristic: technical identifier keyword"


def test_classify_columns_without_presidio_results_is_stable() -> None:
    profiling_summary = pd.DataFrame(
        [
            {"column_name": "contact_email", "dtype": "object", "unique_percent": 50.0},
            {"column_name": "city", "dtype": "object", "unique_percent": 10.0},
        ]
    )

    _, baseline_json = classify_columns(profiling_summary)
    _, absent_presidio_json = classify_columns(profiling_summary, presidio_summary=pd.DataFrame())

    assert absent_presidio_json == baseline_json
