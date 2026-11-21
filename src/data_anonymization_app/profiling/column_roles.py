from typing import Any
import pandas as pd


def detect_column_roles(
    profiling_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:

    role_rows = []

    for row in classification_summary.itertuples(index=False):
        column_name = str(getattr(row, "column_name"))
        classification = str(getattr(row, "classification"))

        # Default
        functional_role = "unknown"
        role_reason = "default"

        col_lower = column_name.lower()

        # --- Identity (human-readable identifiers)
        if classification == "direct_identifier":
            if any(k in col_lower for k in ["email", "name", "phone", "username"]):
                functional_role = "identity"
                role_reason = "direct identifier: human identity field"
            else:
                functional_role = "linkage"
                role_reason = "direct identifier: likely system ID"

        # --- Technical identifiers
        elif classification == "technical_identifier":
            functional_role = "administrative"
            role_reason = "technical/system column"

        # --- Sensitive attributes
        elif classification == "sensitive_attribute":
            functional_role = "analytical"
            role_reason = "sensitive but used for analysis"

        # --- Quasi identifiers
        elif classification == "quasi_identifier":
            if any(k in col_lower for k in ["dob", "birth", "age"]):
                functional_role = "analytical"
                role_reason = "demographic feature (analytical)"
            else:
                functional_role = "analytical"
                role_reason = "default quasi → analytical"

        # --- Non sensitive
        else:
            functional_role = "analytical"
            role_reason = "non-sensitive feature"

        role_rows.append(
            {
                "column_name": column_name,
                "classification": classification,
                "functional_role": functional_role,
                "role_reason": role_reason,
            }
        )

    role_df = pd.DataFrame(
        role_rows,
        columns=["column_name", "classification", "functional_role", "role_reason"],
    )

    return role_df, role_rows