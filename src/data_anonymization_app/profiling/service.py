from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class DatasetMetadata:
    rows: int
    columns: int
    column_names: list[str]
    file_type: str


class DatasetIngestionError(Exception):
    """Raised when dataset ingestion fails."""


_FILE_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".parquet": pd.read_parquet,
}

_DIRECT_IDENTIFIER_KEYWORDS = (
    "name",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "mobile",
    "ssn",
    "pesel",
    "passport",
    "username",
    "user_name",
    "handle",
    "account_id",
    "customer_id",
    "patient_id",
    "employee_id",
    "member_id",
    "client_id",
)

_QUASI_IDENTIFIER_KEYWORDS = (
    "age",
    "birth_date",
    "date_of_birth",
    "dob",
    "birthdate",
    "year_of_birth",
    "yob",
    "zip",
    "postcode",
    "postal_code",
    "city",
    "state",
    "region",
    "country",
    "sex",
    "gender",
    "marital_status",
)

_SENSITIVE_ATTRIBUTE_KEYWORDS = (
    "salary",
    "income",
    "diagnosis",
    "disease",
    "health",
    "religion",
    "ethnicity",
    "political",
    "biometric",
    "medical",
    "disability",
)

_TECHNICAL_IDENTIFIER_KEYWORDS = (
    "index",
    "row_id",
    "record_id",
    "uuid",
    "guid",
    "internal_id",
    "batch_id",
    "load_id",
    "source_id",
)

_COMPANY_NON_SENSITIVE_KEYWORDS = (
    "company",
    "company_name",
    "business_name",
)

_WEBSITE_KEYWORDS = (
    "website",
    "url",
    "domain",
)

_PERSONAL_WEBSITE_KEYWORDS = (
    "user",
    "person",
    "customer",
    "account",
    "profile",
    "owner",
    "client",
    "employee",
)
_HIGH_UNIQUENESS_THRESHOLD = 80.0

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_URL_OR_DOMAIN_PATTERN = re.compile(
    r"^(https?://)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/.*)?$"
)


def _resolve_reader(file_path: Path) -> tuple[str, Callable[[Path], pd.DataFrame]]:
    suffix = file_path.suffix.lower()
    reader = _FILE_READERS.get(suffix)
    if reader is None:
        raise DatasetIngestionError(
            f"Unsupported file format '{suffix or '<none>'}'. "
            "Supported formats: .csv, .xlsx, .xls, .parquet."
        )
    return suffix, reader


def load_dataset(file_path: str | Path) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load a tabular dataset and return data with basic metadata."""
    path = Path(file_path)
    if not path.exists():
        raise DatasetIngestionError(f"File not found: {path}")
    if not path.is_file():
        raise DatasetIngestionError(f"Expected a file path, got: {path}")

    file_type, reader = _resolve_reader(path)

    try:
        dataframe = reader(path)
    except Exception as exc:  # pragma: no cover - backend errors vary by engine/version.
        raise DatasetIngestionError(f"Failed to read '{path.name}': {exc}") from exc

    metadata = DatasetMetadata(
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns.tolist()],
        file_type=file_type.lstrip("."),
    )
    return dataframe, metadata


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return str(value)


def profile_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build a deterministic, column-level dataset profile summary."""
    total_rows = int(len(dataframe))
    summary_rows: list[dict[str, Any]] = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        min_value: Any = None
        max_value: Any = None
        if is_numeric_dtype(series):
            min_value = _normalize_value(series.min(skipna=True))
            max_value = _normalize_value(series.max(skipna=True))

        sample_values = [_normalize_value(value) for value in series.dropna().head(5).tolist()]

        summary_rows.append(
            {
                "column_name": str(column_name),
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_percent": (null_count / total_rows * 100.0) if total_rows > 0 else 0.0,
                "unique_count": unique_count,
                "unique_percent": (unique_count / total_rows * 100.0) if total_rows > 0 else 0.0,
                "sample_values": sample_values,
                "min_value": min_value,
                "max_value": max_value,
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows,
        columns=[
            "column_name",
            "dtype",
            "null_count",
            "null_percent",
            "unique_count",
            "unique_percent",
            "sample_values",
            "min_value",
            "max_value",
        ],
    )
    return summary_dataframe, summary_rows


def _is_numeric_dtype_name(dtype_name: str) -> bool:
    lowered = dtype_name.lower()
    if "bool" in lowered:
        return False
    numeric_markers = ("int", "float", "double", "decimal", "number")
    return any(marker in lowered for marker in numeric_markers)


def _contains_any_keyword(column_name: str, keywords: tuple[str, ...]) -> bool:
    normalized_name = _normalize_column_name_for_matching(column_name)
    return any(keyword in normalized_name for keyword in keywords)


def _normalize_samples(raw_samples: Any) -> list[str]:
    if isinstance(raw_samples, list):
        return [str(value).strip() for value in raw_samples if value is not None and str(value).strip()]
    return []


def _is_phone_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    digit_count = sum(char.isdigit() for char in stripped)
    if digit_count < 7:
        return False
    return all(char.isdigit() or char in "+-(). " for char in stripped)


def _classify_by_value_patterns(samples: list[str]) -> tuple[str, str] | None:
    if not samples:
        return None
    if any(_EMAIL_PATTERN.match(sample) for sample in samples):
        return "direct_identifier", "value-pattern: email-like sample"
    if any(_is_phone_like(sample) for sample in samples):
        return "direct_identifier", "value-pattern: phone-like sample"
    if any(_UUID_PATTERN.match(sample) for sample in samples):
        return "technical_identifier", "value-pattern: uuid-like sample"
    if any(_URL_OR_DOMAIN_PATTERN.match(sample) for sample in samples):
        return "non_sensitive", "value-pattern: url/domain-like sample"
    return None


def _normalize_column_name_for_matching(column_name: str) -> str:
    lowered = column_name.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def classify_columns(
    profiling_summary: pd.DataFrame,
    presidio_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Classify columns by sensitivity using simple, deterministic heuristics."""
    required_columns = {"column_name", "dtype", "unique_percent"}
    missing_columns = required_columns.difference(profiling_summary.columns)
    if missing_columns:
        missing_joined = ", ".join(sorted(missing_columns))
        raise ValueError(f"Profiling summary missing required columns: {missing_joined}")

    presidio_by_column: dict[str, dict[str, Any]] = {}
    if presidio_summary is not None and not presidio_summary.empty:
        if "column_name" in presidio_summary.columns:
            for record in presidio_summary.to_dict(orient="records"):
                column_name_key = str(record.get("column_name", ""))
                if column_name_key:
                    presidio_by_column[column_name_key] = record

    classification_rows: list[dict[str, str]] = []
    for row in profiling_summary.itertuples(index=False):
        column_name = str(getattr(row, "column_name"))
        dtype_name = str(getattr(row, "dtype"))
        unique_percent = float(getattr(row, "unique_percent"))
        sample_values = _normalize_samples(getattr(row, "sample_values", []))

        if _contains_any_keyword(column_name, _COMPANY_NON_SENSITIVE_KEYWORDS):
            classification = "non_sensitive"
            reason = "name-heuristic: company/business column"
        elif _contains_any_keyword(column_name, _WEBSITE_KEYWORDS) and not _contains_any_keyword(
            column_name, _PERSONAL_WEBSITE_KEYWORDS
        ):
            classification = "non_sensitive"
            reason = "name-heuristic: generic website/url/domain column"
        elif _contains_any_keyword(column_name, _WEBSITE_KEYWORDS) and _contains_any_keyword(
            column_name, _PERSONAL_WEBSITE_KEYWORDS
        ):
            classification = "direct_identifier"
            reason = "name-heuristic: personal website/url/domain column"
        elif _contains_any_keyword(column_name, _TECHNICAL_IDENTIFIER_KEYWORDS):
            classification = "technical_identifier"
            reason = "name-heuristic: technical identifier keyword"
        elif _contains_any_keyword(column_name, _DIRECT_IDENTIFIER_KEYWORDS):
            classification = "direct_identifier"
            reason = "name-heuristic: direct identifier keyword"
        elif _contains_any_keyword(column_name, _SENSITIVE_ATTRIBUTE_KEYWORDS):
            classification = "sensitive_attribute"
            reason = "name-heuristic: sensitive attribute keyword"
        elif _contains_any_keyword(column_name, _QUASI_IDENTIFIER_KEYWORDS):
            classification = "quasi_identifier"
            reason = "name-heuristic: quasi identifier keyword"
        elif pattern_result := _classify_by_value_patterns(sample_values):
            classification, reason = pattern_result
        elif _is_numeric_dtype_name(dtype_name) and unique_percent >= _HIGH_UNIQUENESS_THRESHOLD:
            classification = "quasi_identifier"
            reason = "dtype-heuristic: numeric with high uniqueness"
        else:
            classification = "non_sensitive"
            reason = "fallback: no sensitive heuristic matched"

        presidio_record = presidio_by_column.get(column_name)
        if presidio_record is not None:
            entities = {
                str(entity).strip().upper()
                for entity in (presidio_record.get("presidio_entities_detected") or [])
                if str(entity).strip()
            }
            hit_count = int(presidio_record.get("presidio_hit_count") or 0)

            risk_rank = {
                "non_sensitive": 0,
                "quasi_identifier": 1,
                "technical_identifier": 2,
                "sensitive_attribute": 2,
                "direct_identifier": 3,
            }

            presidio_target: str | None = None
            presidio_reason: str | None = None
            if hit_count > 0 and ("EMAIL_ADDRESS" in entities or "PHONE_NUMBER" in entities):
                presidio_target = "direct_identifier"
                presidio_reason = "presidio: email/phone entity detected"
            elif hit_count > 0 and ("PERSON" in entities or "LOCATION" in entities):
                presidio_target = "quasi_identifier"
                presidio_reason = "presidio: person/location entity detected"
            elif hit_count > 0 and entities and entities.issubset({"URL"}):
                presidio_target = "non_sensitive"
                presidio_reason = "presidio: url-only entities detected"

            if presidio_target is not None:
                if risk_rank[presidio_target] > risk_rank[classification]:
                    classification = presidio_target
                    reason = presidio_reason or reason
                elif presidio_target == classification and presidio_reason is not None:
                    reason = presidio_reason

        classification_rows.append(
            {
                "column_name": column_name,
                "classification": classification,
                "reason": reason,
            }
        )

    classification_dataframe = pd.DataFrame(
        classification_rows,
        columns=["column_name", "classification", "reason"],
    )
    return classification_dataframe, classification_rows



def detect_column_roles(
    profiling_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Infer a simple functional role for each column.

    This is intentionally conservative and additive:
    it does not replace existing classification, it complements it.
    """
    required_classification_columns = {"column_name", "classification"}
    missing_classification = required_classification_columns.difference(classification_summary.columns)
    if missing_classification:
        missing_joined = ", ".join(sorted(missing_classification))
        raise ValueError(
            f"Classification summary missing required columns: {missing_joined}"
        )

    role_rows: list[dict[str, Any]] = []

    for row in classification_summary.itertuples(index=False):
        column_name = str(getattr(row, "column_name"))
        classification = str(getattr(row, "classification"))
        column_name_lower = _normalize_column_name_for_matching(column_name)

        functional_role = "unknown"
        role_reason = "fallback: no role heuristic matched"

        if classification == "direct_identifier":
            if any(
                keyword in column_name_lower
                for keyword in ("email", "name", "phone", "mobile", "username", "handle")
            ):
                functional_role = "identity"
                role_reason = "role-heuristic: human-readable identity field"
            else:
                functional_role = "linkage"
                role_reason = "role-heuristic: identifier used as linkage key"

        elif classification == "technical_identifier":
            functional_role = "administrative"
            role_reason = "role-heuristic: technical/system identifier"

        elif classification == "sensitive_attribute":
            functional_role = "analytical"
            role_reason = "role-heuristic: sensitive attribute used analytically"

        elif classification == "quasi_identifier":
            if any(
                keyword in column_name_lower
                for keyword in (
                    "dob",
                    "birth",
                    "birth_date",
                    "date_of_birth",
                    "birthdate",
                    "year_of_birth",
                    "yob",
                    "age",
                    "sex",
                    "gender",
                    "marital_status",
                    "zip",
                    "postcode",
                    "postal_code",
                    "city",
                    "state",
                    "region",
                    "country",
                )
            ):
                functional_role = "analytical"
                role_reason = "role-heuristic: quasi-identifier with analytical value"
            else:
                functional_role = "analytical"
                role_reason = "role-heuristic: contextual quasi-identifier treated as analytical"

        elif classification == "non_sensitive":
            if any(
                keyword in column_name_lower
                for keyword in (
                    "job_title",
                    "occupation",
                    "education",
                    "education_level",
                    "department",
                    "tenure",
                    "experience",
                    "experience_years",
                    "score",
                    "amount",
                    "spend",
                )
            ):
                functional_role = "analytical"
                role_reason = "role-heuristic: analytical feature"
            else:
                functional_role = "analytical"
                role_reason = "role-heuristic: non-sensitive feature"

        role_rows.append(
            {
                "column_name": column_name,
                "classification": classification,
                "functional_role": functional_role,
                "role_reason": role_reason,
            }
        )

    role_dataframe = pd.DataFrame(
        role_rows,
        columns=[
            "column_name",
            "classification",
            "functional_role",
            "role_reason",
        ],
    )
    return role_dataframe, role_rows


def _is_likely_text_column(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    non_null_values = series.dropna().tolist()
    if not non_null_values:
        return False
    return any(
        str(value).strip() not in ("", "nan", "none", "null")
        for value in non_null_values
    )


def _load_presidio_analyzer() -> tuple[Any | None, bool]:
    try:
        from presidio_analyzer import AnalyzerEngine
        return AnalyzerEngine(), True
    except Exception:
        return None, False
    
    
def detect_text_pii_with_presidio(
    dataframe: pd.DataFrame,
    sample_size: int = 200,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Optionally detect likely PII entities in text-like columns using Presidio."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    analyzer, presidio_available = _load_presidio_analyzer()
    requested_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION", "URL"]
    detection_rows: list[dict[str, Any]] = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        if not _is_likely_text_column(series):
            continue

        column_values = [
            str(value).strip()
            for value in series.dropna().tolist()
            if str(value).strip() not in ("", "nan", "none", "null")
        ][:sample_size]

        entity_counter: Counter[str] = Counter()
        hit_count = 0
        presidio_used = False

        if presidio_available and analyzer is not None and column_values:
            presidio_used = True
            for value in column_values:
                try:
                    results = analyzer.analyze(
                        text=value,
                        entities=requested_entities,
                        language="en",
                    )
                except Exception:
                    results = []

                for result in results:
                    entity_type = str(getattr(result, "entity_type", "")).strip()
                    if entity_type:
                        entity_counter[entity_type] += 1
                        hit_count += 1

        detection_rows.append(
            {
                "column_name": str(column_name),
                "presidio_entities_detected": sorted(entity_counter.keys()),
                "presidio_hit_count": int(hit_count),
                "presidio_used": bool(presidio_used),
            }
        )

    detection_dataframe = pd.DataFrame(
        detection_rows,
        columns=[
            "column_name",
            "presidio_entities_detected",
            "presidio_hit_count",
            "presidio_used",
        ],
    )
    return detection_dataframe, detection_rows
