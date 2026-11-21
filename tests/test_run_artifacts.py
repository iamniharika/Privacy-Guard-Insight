from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re

import pandas as pd

from data_anonymization_app.reports.run_artifacts import _generate_run_id, save_run_artifacts


def test_generate_run_id_format_is_valid() -> None:
    run_id = _generate_run_id(
        now=datetime(2026, 3, 12, 20, 15, 30, tzinfo=timezone.utc),
        random_suffix="ab12cd",
    )
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-z]{6}$", run_id) is not None
    assert run_id == "2026-03-12_201530_ab12cd"


def test_save_run_artifacts_creates_expected_files_and_metadata(tmp_path) -> None:
    run_id, run_dir, artifact_paths = save_run_artifacts(
        metadata={"file_type": "csv", "rows": 2, "columns": 2},
        planning_context={"intended_use": "research"},
        anonymized_dataframe=pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        transformation_log=[{"column_name": "a", "action_applied": "keep"}],
        validation_summary={"validation_warnings": []},
        report_markdown="# Report\n",
        report_json={"report_header": {"title": "Anonymization Audit Report"}},
        base_runs_dir=tmp_path / "runs",
        run_id="2026-03-12_201530_ab12cd",
        generated_at="2026-03-12T20:15:30Z",
    )

    assert run_id == "2026-03-12_201530_ab12cd"
    assert run_dir == tmp_path / "runs" / run_id
    assert run_dir.exists()

    for path_str in artifact_paths.values():
        assert Path(path_str).exists()

    metadata_path = Path(artifact_paths["metadata_json"])
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["run_id"] == run_id
    assert metadata_payload["generated_at"] == "2026-03-12T20:15:30Z"
    assert metadata_payload["dataset_file_type"] == "csv"
    assert metadata_payload["row_count"] == 2
    assert metadata_payload["column_count"] == 2
    assert "planning_context" in metadata_payload
    assert "artifacts" in metadata_payload


def test_save_run_artifacts_handles_empty_optional_inputs(tmp_path) -> None:
    run_id, run_dir, artifact_paths = save_run_artifacts(
        base_runs_dir=tmp_path / "runs",
        run_id="2026-03-12_000000_abc123",
        generated_at="2026-03-12T00:00:00Z",
    )

    assert run_id == "2026-03-12_000000_abc123"
    assert run_dir.exists()
    assert Path(artifact_paths["anonymized_dataset_csv"]).exists()
    assert Path(artifact_paths["transformation_log_json"]).exists()
    assert Path(artifact_paths["validation_summary_json"]).exists()
    assert Path(artifact_paths["anonymization_report_md"]).exists()
    assert Path(artifact_paths["anonymization_report_json"]).exists()
    assert Path(artifact_paths["metadata_json"]).exists()


def test_save_run_artifacts_is_deterministic_with_fixed_inputs(tmp_path) -> None:
    args = {
        "metadata": {"file_type": "csv", "rows": 1, "columns": 1},
        "planning_context": {"intended_use": "research", "need_record_linkage": False},
        "anonymized_dataframe": pd.DataFrame({"x": ["token_000001"]}),
        "transformation_log": [{"column_name": "x", "action_applied": "pseudonymize"}],
        "validation_summary": {"validation_warnings": []},
        "report_markdown": "# Fixed\n",
        "report_json": {"fixed": True},
        "base_runs_dir": tmp_path / "runs",
        "run_id": "2026-03-12_111111_aa11bb",
        "generated_at": "2026-03-12T11:11:11Z",
    }

    _, run_dir_1, artifact_paths_1 = save_run_artifacts(**args)
    files_snapshot_1 = {
        key: Path(path).read_text(encoding="utf-8")
        for key, path in artifact_paths_1.items()
        if path.endswith(".json") or path.endswith(".md") or path.endswith(".csv")
    }

    _, run_dir_2, artifact_paths_2 = save_run_artifacts(**args)
    files_snapshot_2 = {
        key: Path(path).read_text(encoding="utf-8")
        for key, path in artifact_paths_2.items()
        if path.endswith(".json") or path.endswith(".md") or path.endswith(".csv")
    }

    assert run_dir_1 == run_dir_2
    assert files_snapshot_1 == files_snapshot_2
