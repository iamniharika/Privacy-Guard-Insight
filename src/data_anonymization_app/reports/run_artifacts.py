from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import secrets
from typing import Any

import pandas as pd


def _generate_run_id(
    *,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H%M%S")
    suffix = random_suffix or secrets.token_hex(3)
    return f"{timestamp}_{suffix}"


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def _serialize_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    return {}


def save_run_artifacts(
    *,
    metadata: Any = None,
    planning_context: dict[str, Any] | None = None,
    anonymized_dataframe: pd.DataFrame | None = None,
    transformation_log: list[dict[str, Any]] | None = None,
    validation_summary: dict[str, Any] | None = None,
    report_markdown: str = "",
    report_json: dict[str, Any] | None = None,
    base_runs_dir: str | Path = "runs",
    run_id: str | None = None,
    generated_at: str | None = None,
) -> tuple[str, Path, dict[str, str]]:
    """Persist anonymization run artifacts under a unique local run directory."""
    metadata_dict = _serialize_metadata(metadata)
    context = planning_context if planning_context is not None else {}
    output_df = anonymized_dataframe if anonymized_dataframe is not None else pd.DataFrame()
    log_data = transformation_log if transformation_log is not None else []
    summary_data = validation_summary if validation_summary is not None else {}
    report_data = report_json if report_json is not None else {}

    effective_run_id = run_id or _generate_run_id()
    base_dir = Path(base_runs_dir)
    run_dir = base_dir / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "anonymized_dataset_csv": str(run_dir / "anonymized_dataset.csv"),
        "transformation_log_json": str(run_dir / "transformation_log.json"),
        "validation_summary_json": str(run_dir / "validation_summary.json"),
        "anonymization_report_md": str(run_dir / "anonymization_report.md"),
        "anonymization_report_json": str(run_dir / "anonymization_report.json"),
        "metadata_json": str(run_dir / "metadata.json"),
    }

    output_df.to_csv(artifact_paths["anonymized_dataset_csv"], index=False, encoding="utf-8")
    _safe_json_dump(Path(artifact_paths["transformation_log_json"]), log_data)
    _safe_json_dump(Path(artifact_paths["validation_summary_json"]), summary_data)

    report_md_path = Path(artifact_paths["anonymization_report_md"])
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(report_markdown, encoding="utf-8")
    _safe_json_dump(Path(artifact_paths["anonymization_report_json"]), report_data)

    row_count = metadata_dict.get("rows")
    column_count = metadata_dict.get("columns")
    if row_count is None:
        row_count = int(len(output_df))
    if column_count is None:
        column_count = int(output_df.shape[1])

    metadata_payload = {
        "run_id": effective_run_id,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "dataset_file_type": metadata_dict.get("file_type"),
        "row_count": row_count,
        "column_count": column_count,
        "planning_context": context,
        "artifacts": {name: Path(path).name for name, path in artifact_paths.items()},
    }
    _safe_json_dump(Path(artifact_paths["metadata_json"]), metadata_payload)

    return effective_run_id, run_dir, artifact_paths

