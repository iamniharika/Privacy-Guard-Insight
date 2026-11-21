from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
_APP_SPEC = importlib.util.spec_from_file_location("app_module_under_test_actions", _APP_PATH)
assert _APP_SPEC is not None
assert _APP_SPEC.loader is not None
_APP_MODULE = importlib.util.module_from_spec(_APP_SPEC)
_APP_SPEC.loader.exec_module(_APP_MODULE)


def test_action_labels_include_mask_and_round_trip() -> None:
    assert "mask" in _APP_MODULE._SUPPORTED_OVERRIDE_ACTIONS
    assert "generalize_light" in _APP_MODULE._SUPPORTED_OVERRIDE_ACTIONS
    assert "generalize_medium" in _APP_MODULE._SUPPORTED_OVERRIDE_ACTIONS
    assert "generalize_strict" in _APP_MODULE._SUPPORTED_OVERRIDE_ACTIONS
    assert _APP_MODULE._friendly_action("mask") == "Mask value (display only)  \U0001f534 Display only"
    assert _APP_MODULE._raw_action_from_label("Mask value (display only)  \U0001f534 Display only") == "mask"
    assert _APP_MODULE._friendly_action("generalize") == "Generalize / reduce precision  \U0001f7e1 Medium protection"
    assert _APP_MODULE._friendly_action("generalize_medium") == "Generalize more  \U0001f7e1 Medium protection"
    assert _APP_MODULE._friendly_action("generalize_strict") == "Generalize strongly  \U0001f7e2 Stronger protection"
    assert _APP_MODULE._raw_action_from_label("Generalize / reduce precision  \U0001f7e1 Medium protection") == "generalize_light"
    assert _APP_MODULE._friendly_action("pseudonymize") == "Replace with pseudonym  \U0001f7e1 Medium protection"


def test_build_external_mask_warning_flags_direct_identifiers() -> None:
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

    warning_text = _APP_MODULE._build_external_mask_warning(reviewed_df, "external sharing")

    assert warning_text is not None
    assert "Masking is not recommended for external sharing" in warning_text
    assert "email" in warning_text
