from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from data_anonymization_app.profiling import service as profiling_service


def test_detect_text_pii_with_presidio_graceful_without_dependency(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {
            "email_text": ["alice@example.com", None],
            "city_text": ["Warsaw", "Berlin"],
            "age": [30, 40],
        }
    )

    monkeypatch.setattr(profiling_service, "_load_presidio_analyzer", lambda: (None, False))

    result_df, result_json = profiling_service.detect_text_pii_with_presidio(dataframe)

    assert result_df.to_dict(orient="records") == result_json
    assert set(result_df["column_name"].tolist()) == {"email_text", "city_text"}
    assert all(row["presidio_entities_detected"] == [] for row in result_json)
    assert all(row["presidio_hit_count"] == 0 for row in result_json)
    assert all(row["presidio_used"] is False for row in result_json)


def test_detect_text_pii_with_presidio_with_mocked_analyzer(monkeypatch) -> None:
    class FakeAnalyzer:
        def analyze(self, text: str, entities: list[str], language: str) -> list[SimpleNamespace]:
            hits: list[SimpleNamespace] = []
            if "@" in text:
                hits.append(SimpleNamespace(entity_type="EMAIL_ADDRESS"))
            if "http://" in text or "https://" in text:
                hits.append(SimpleNamespace(entity_type="URL"))
            if "Warsaw" in text:
                hits.append(SimpleNamespace(entity_type="LOCATION"))
            if "Alice" in text:
                hits.append(SimpleNamespace(entity_type="PERSON"))
            if "+1" in text:
                hits.append(SimpleNamespace(entity_type="PHONE_NUMBER"))
            return hits

    dataframe = pd.DataFrame(
        {
            "contact": ["alice@example.com", "+1 202 555 0100"],
            "about": ["Alice from Warsaw", "https://example.com"],
            "score": [1, 2],
        }
    )

    monkeypatch.setattr(profiling_service, "_load_presidio_analyzer", lambda: (FakeAnalyzer(), True))

    result_df, result_json = profiling_service.detect_text_pii_with_presidio(dataframe)

    assert result_df.to_dict(orient="records") == result_json
    assert set(result_df["column_name"].tolist()) == {"contact", "about"}

    by_name = {row["column_name"]: row for row in result_json}
    assert by_name["contact"]["presidio_used"] is True
    assert by_name["contact"]["presidio_hit_count"] == 2
    assert by_name["contact"]["presidio_entities_detected"] == ["EMAIL_ADDRESS", "PHONE_NUMBER"]

    assert by_name["about"]["presidio_used"] is True
    assert by_name["about"]["presidio_hit_count"] == 3
    assert by_name["about"]["presidio_entities_detected"] == ["LOCATION", "PERSON", "URL"]


def test_detect_text_pii_with_presidio_real_dependency_if_available() -> None:
    pytest.importorskip("presidio_analyzer")
    dataframe = pd.DataFrame(
        {
            "notes": ["Contact alice@example.com"],
            "amount": [10],
        }
    )

    result_df, result_json = profiling_service.detect_text_pii_with_presidio(dataframe)
    assert result_df.to_dict(orient="records") == result_json
    assert list(result_df.columns) == [
        "column_name",
        "presidio_entities_detected",
        "presidio_hit_count",
        "presidio_used",
    ]

