import html
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

import hashlib

from textwrap import dedent

import base64


# Ensure local src-based package imports work without editable install.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


LOGO_ICON_PATH = ROOT_DIR / "assets" / "logos" / "privacyguard_ai_icon.png"

DEMO_DATASET_PATH = ROOT_DIR / "demo_data" / "privacyguard_ai_demo_synthetic_sensitive_dataset.csv"
DEMO_DATASET_FILENAME = "privacyguard_ai_demo_synthetic_sensitive_dataset.csv"


def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"



from data_anonymization_app.anonymization.service import anonymize_dataset, plan_anonymization
from data_anonymization_app.config.settings import get_settings
from data_anonymization_app.privacy_metrics.service import validate_anonymization_output
from data_anonymization_app.privacy_layers import (
    apply_randomized_response,
    randomized_response_truth_probability,
    train_private_adversarial_model,
)
from data_anonymization_app.profiling.service import (
    classify_columns,
    detect_column_roles,
    detect_text_pii_with_presidio,
    load_dataset,
    profile_dataset,
)
from data_anonymization_app.reports.report_generator import generate_anonymization_report
from data_anonymization_app.reports.run_artifacts import save_run_artifacts

st.markdown(
    dedent(f"""
    <div class="export-note-card">
        ...
    </div>
    """),
    unsafe_allow_html=True,
)

_GENERALIZE_LEVELS = [
    "generalize_light",
    "generalize_medium",
    "generalize_strict",
]
_LEGACY_ACTION_ALIASES = {
    "generalize": "generalize_light",
    "generalize / reduce precision": "generalize_light",
    "generalize_reduce_precision": "generalize_light",
    "reduce precision": "generalize_light",
}
_SUPPORTED_OVERRIDE_ACTIONS = [
    "drop",
    "pseudonymize",
    "generalize_light",
    "generalize_medium",
    "generalize_strict",
    "add_noise",
    "mask",
    "keep",
]
_DATASET_PROTECTION_ACTIONS = ["drop", "pseudonymize", *_GENERALIZE_LEVELS, "add_noise"]
_DISPLAY_PROTECTION_ACTIONS = ["mask"]
_TECHNICAL_IDENTIFIER_CLASS = "technical_identifier"
_CLASSIFICATION_LABELS = {
    "direct_identifier": "Direct identifier",
    "quasi_identifier": "Indirect identifier",
    "sensitive_attribute": "Sensitive information",
    "technical_identifier": "Internal system identifier",
    "non_sensitive": "No sensitivity detected",
}
_ACTION_LABELS = {
    "drop": "Remove column  🟢 Strong protection",
    "pseudonymize": "Replace with pseudonym  🟡 Medium protection",
    "generalize_light": "Generalize / reduce precision  🟡 Medium protection",
    "generalize_medium": "Generalize more  🟡 Medium protection",
    "generalize_strict": "Generalize strongly  🟢 Stronger protection",
    "add_noise": "Add numeric noise  🟡 Medium protection",
    "mask": "Mask value (display only)  🔴 Display only",
    "keep": "Leave unchanged  ⚠ No protection",
}
_REASON_BY_CLASSIFICATION = {
    "direct_identifier": "Direct identifiers should be removed to reduce re-identification risk.",
    "sensitive_attribute": "Sensitive attributes are protected by perturbation where possible.",
    "quasi_identifier": "Indirect identifiers are generalized to reduce linkage risk.",
    "non_sensitive": "Non-sensitive columns can be kept without transformation.",
    "technical_identifier": "Technical/index columns are usually dropped.",
}
_WORKFLOW_STEPS = [
    "Upload & scan",
    "Protect & validate",
    "Export",
]


def _normalize_action_key(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _LEGACY_ACTION_ALIASES.get(normalized, normalized)


def _is_generalize_action(value: Any) -> bool:
    return _normalize_action_key(value) in set(_GENERALIZE_LEVELS)


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f7fb;
            --card-bg: #ffffff;
            --border: #d9e2ec;
            --text: #102a43;
            --muted: #526b82;
            --primary: #163d7a;
            --secondary: #0b6668;
            --success: #2f855a;
            --warning: #b7791f;
            --danger: #c53030;
            accent-color: #163d7a;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        .app-hero {
            background: var(--primary);
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
            margin-bottom: 1rem;
        }
        .app-hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; }
        .app-hero p {
            margin: 0.5rem 0 0;
            font-size: 0.92rem;
            font-weight: 500;
            color: #e8eef8;
            letter-spacing: 0.025em;
            line-height: 1.45;
        }

        .section-title { font-size: 1.58rem; font-weight: 750; color: var(--text); letter-spacing: -0.015em; margin-bottom: 0.28rem; }
        .section-subtitle { color: var(--muted); font-size: 0.94rem; line-height: 1.5; margin-bottom: 1.05rem; }
        .subsection-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0.55rem 0 0.15rem;}

        h3 {
            font-size: 1.05rem !important; font-weight: 700 !important; color: var(--text) !important;
            line-height: 1.35 !important; margin-top: 0.55rem !important; margin-bottom: 0.2rem !important;
        }

        .workflow-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 0.8rem 0 1.5rem; }
        .workflow-pill {
            border: 1px solid var(--border); border-radius: 999px; padding: 0.45rem 0.8rem;
            background: rgba(255, 255, 255, 0.92); color: var(--muted); font-size: 0.9rem; font-weight: 600;
        }
        .workflow-pill.active { background: var(--primary); color: #ffffff; border-color: var(--primary); box-shadow: 0 10px 24px rgba(22, 61, 122, 0.18); }
        .workflow-pill.complete { background: #e6f4f2; color: #075958; border-color: #9accc7; }

        .status-chip { display: inline-block; border-radius: 999px; padding: 0.22rem 0.6rem; font-size: 0.78rem; font-weight: 700; margin-left: 0.45rem; }
        .status-chip.complete { color: var(--success); background: rgba(47, 133, 90, 0.12); }
        .status-chip.current { color: var(--primary); background: rgba(22, 61, 122, 0.12); }
        .status-chip.pending { color: var(--muted); background: rgba(98, 125, 152, 0.14); }

        .info-panel { background: rgba(255, 255, 255, 0.9); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.1rem; margin-bottom: 1rem; }
        .info-panel strong { font-size: 0.94rem; }
        .info-panel span { color: var(--muted); font-size: 0.9rem; line-height: 1.45; }

        .context-strip { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.4rem 0 1rem; }
        .context-chip { background: rgba(22, 61, 122, 0.06); border: 1px solid rgba(22, 61, 122, 0.12); border-radius: 999px; padding: 0.45rem 0.8rem; color: var(--text); font-size: 0.84rem; }
        .context-chip strong { color: var(--primary); }


        /* ── Detection category help (section 3) ── */
        .explanation-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            min-height: 132px;
        }
        .explanation-shell {
            display: flex;
            gap: 1rem;
            align-items: stretch;
        }
        .explanation-shell-left {
            flex: 0 0 32%;
            border-right: 1px solid rgba(217, 226, 236, 0.85);
            padding-right: 1rem;
        }
        .explanation-shell-right {
            flex: 1 1 auto;
        }
        .eyebrow-title {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .explanation-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.45rem;
        }
        .example-block {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin-top: 0.55rem;
        }


        /* ── Segmented control (all instances) ── */
        [data-testid="stBaseButton-segmented_controlActive"] {
            background-color: #163d7a !important;
            color: #ffffff !important;
            border-color: #163d7a !important;
            font-weight: 500 !important;
        }
        [data-testid="stBaseButton-segmented_control"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: rgba(22, 61, 122, 0.65) !important;
            border-color: rgba(217, 226, 236, 0.9) !important;
            font-weight: 400 !important;
        }


        

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #fbfcfe;
            border-right: 1px solid #e6edf5;
        }

        /* Sidebar section heading */
        [data-testid="stSidebar"] h3 {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            color: var(--text) !important;
            line-height: 1.3 !important;
            margin-top: 0.1rem !important;
            margin-bottom: 0.35rem !important;
            letter-spacing: -0.01em !important;
        }

        /* Sidebar helper text like:
        "A quick summary of where the run stands." */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #526b82 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 0.84rem !important;
            line-height: 1.45 !important;
            margin-bottom: 0.95rem !important;
        }

        /* Workflow status lines */
        [data-testid="stSidebar"] p {
            color: var(--muted);
            font-size: 0.87rem !important;
            line-height: 1.45 !important;
            font-weight: 500 !important;
            margin-bottom: 0.72rem !important;
        }

        /* Status pill */
        [data-testid="stSidebar"] .status-chip {
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            min-width: 84px;
            padding: 0.24rem 0.7rem;
            margin-left: 0.55rem;
            font-size: 0.76rem;
            font-weight: 700;
            vertical-align: middle;
        }

        /* Slightly soften sidebar markdown spacing */
        [data-testid="stSidebar"] .element-container:has(.status-chip) {
            margin-bottom: 0.08rem !important;
        }


        

        /* ── File uploader ── */
        [data-testid="stFileUploader"] button {
            background-color: #163d7a !important;
            color: #ffffff !important;
            border-color: #163d7a !important;
            border-radius: 999px !important;
            padding: 0.38rem 1.05rem !important;
            min-height: unset !important;
            line-height: 1.35 !important;
            box-shadow: 0 6px 14px rgba(22, 61, 122, 0.12) !important;
        }

        /* Streamlit puts the visible button label inside child elements,
        so target them directly too. */
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span {
            color: #ffffff !important;
            font-size: 0.84rem !important;
            font-weight: 700 !important;
            line-height: 1.35 !important;
        }
        
        [data-testid="stFileUploader"] button:hover {
            background-color: #1a4a94 !important;
            border-color: #1a4a94 !important;
        }
        [data-testid="stFileUploaderDeleteBtn"] button {
            background-color: transparent !important;
            border-color: transparent !important;
            color: var(--muted) !important;
            border-radius: 50% !important;
        }
        [data-testid="stFileUploaderDeleteBtn"] button:hover {
            background-color: rgba(197, 48, 48, 0.08) !important;
            color: var(--danger) !important;
        }

        /* ── Primary button ── */
        [data-testid="stBaseButton-primary"] {
            background-color: #163d7a !important;
            border-color: #163d7a !important;
            color: #ffffff !important;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background-color: #1a4a94 !important;
            border-color: #1a4a94 !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: #ffffff !important;
            border: 1px solid #c5d2e0 !important;
            color: #163d7a !important;
        }
        [data-testid="stBaseButton-secondary"]:hover {
            background: #edf4fb !important;
            border-color: #8ca8c6 !important;
            color: #102a43 !important;
        }
        [data-testid="stBaseButton-secondary"] * {
            color: #163d7a !important;
        }

        [data-testid="stExpander"] details > summary {
            background: #ffffff !important;
            border: 1px solid #c5d2e0 !important;
            border-radius: 12px !important;
            color: #102a43 !important;
            opacity: 1 !important;
        }
        [data-testid="stExpander"] details > summary:hover {
            background: #edf4fb !important;
            border-color: #8ca8c6 !important;
        }
        [data-testid="stExpander"] details > summary *,
        [data-testid="stExpander"] details > summary svg {
            color: #102a43 !important;
            fill: #102a43 !important;
            opacity: 1 !important;
        }
        [data-testid="stExpander"] details > summary:focus-visible {
            outline: 3px solid rgba(22, 61, 122, 0.35) !important;
            outline-offset: 2px !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background: #ffffff !important;
            border-color: #d9e2ec !important;
            color: #102a43 !important;
        }

        /* ── Step 1 demo dataset block ── */
        .step1-mini-heading {
            font-size: 1.08rem;
            font-weight: 750;
            color: #163d7a;
            margin-bottom: 0.55rem;
            letter-spacing: 0.01em;
        }

        .step1-mini-heading-teal {
            color: #0f8b8d;
        }

        .workflow-card-title {
            font-size: 0.92rem;
            font-weight: 750;
            color: var(--text);
            margin-bottom: 0.28rem;
            letter-spacing: -0.005em;
        }

        .demo-download-shell {
            background: rgba(15, 139, 141, 0.055);
            border: 1px solid rgba(15, 139, 141, 0.08);
            border-radius: 10px;
            padding: 0.72rem 0.9rem;
            min-height: 64px;
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.9rem;
        }


        .demo-download-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.48rem;
            background: #0f8b8d;
            color: #ffffff !important;
            border: 1px solid #0f8b8d;
            border-radius: 999px;
            padding: 0.38rem 1.05rem;
            font-size: 0.84rem;
            font-weight: 700;
            line-height: 1.35;
            text-decoration: none !important;
            white-space: nowrap;
            box-shadow: 0 6px 14px rgba(15, 139, 141, 0.12);
        }

        .demo-download-pill:hover {
            background: #0d7779;
            border-color: #0d7779;
            color: #ffffff !important;
            text-decoration: none !important;
        }

        .demo-download-pill svg {
            width: 15px;
            height: 15px;
            flex-shrink: 0;
        }

        .demo-download-helper {
            color: rgba(15, 110, 112, 0.76);
            font-size: 0.82rem;
            line-height: 1.4;
            font-weight: 400;
        }

        .synthetic-demo-card {
            background: rgba(15, 139, 141, 0.08);
            border: 1px solid rgba(15, 139, 141, 0.22);
            border-left: 6px solid #0f8b8d;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-top: 0.85rem;
            margin-bottom: 0.85rem;
        }

        .synthetic-demo-title {
            font-size: 0.92rem;
            font-weight: 750;
            color: #0f8b8d;
            margin-bottom: 0.28rem;
            letter-spacing: -0.005em;
        }

        .synthetic-demo-text {
            color: rgba(15, 110, 112, 0.82);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        
        
        /* ── Category badges ── */
        .category-badge { display: inline-block; border-radius: 999px; padding: 0.22rem 0.6rem; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.5rem; }
        .category-direct_identifier { background: rgba(197, 48, 48, 0.12); color: var(--danger); }
        .category-sensitive_attribute { background: rgba(183, 121, 31, 0.14); color: var(--warning); }
        .category-quasi_identifier { background: rgba(15, 139, 141, 0.12); color: var(--secondary); }
        .category-technical_identifier { background: rgba(22, 61, 122, 0.1); color: var(--primary); }
        .category-non_sensitive { background: rgba(47, 133, 90, 0.12); color: var(--success); }


        /* ── Overview metric cards (section 2) ── */
        .overview-metric-card {
            background: rgba(22, 61, 122, 0.05);
            border: 1px solid rgba(22, 61, 122, 0.14);
            border-left: 6px solid #163d7a;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 116px;
            margin-bottom: 0.85rem;
            box-shadow: none;
        }

        .overview-metric-card.highlight {
            background: #fbf7ee;
            border-color: rgba(200, 164, 90, 0.24);
            border-left: 6px solid #a16207;
        }

        .overview-metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.35;
            margin-bottom: 0.35rem;
        }

        .overview-metric-value {
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 750;
            line-height: 1.05;
            margin-bottom: 0.25rem;
        }

        .overview-metric-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }



        /* ── Profile cards (section 2) ── */
        .profile-card {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.85rem;
            min-height: 150px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.03);
        }

        .profile-card-title {
            font-size: 1.06rem;
            font-weight: 750;
            color: var(--text);
            margin-bottom: 0.62rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(217, 226, 236, 0.7);
            letter-spacing: -0.01em;
        }

        .profile-card-meta {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.7;
        }

        .profile-card-meta strong {
            color: var(--text);
            font-weight: 700;
        }

        .profile-card-hint {
            margin-top: 0.38rem;
            color: #7b6738;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.35;
        }

        /* ── Summary + detail cards (section 3) ── */
        .summary-card { border: 1px solid var(--border); border-radius: 18px; padding: 0.9rem 1rem; background: rgba(255, 255, 255, 0.97); min-height: 108px; margin-bottom: 0.85rem; }
        .summary-card .metric-value { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.45rem; }
        .summary-card .metric-label { display: inline-block; border-radius: 999px; padding: 0.25rem 0.65rem; font-size: 0.82rem; font-weight: 700; }
        .summary-card .metric-note { color: var(--muted); font-size: 0.88rem; margin-top: 0.35rem; }
        .summary-list { color: var(--muted); font-size: 0.92rem; line-height: 1.5; margin-top: 0.55rem; }

        .detail-card { border: 1px solid var(--border); border-radius: 18px; padding: 0.95rem 1rem; background: rgba(255, 255, 255, 0.97); min-height: 220px; margin-bottom: 0.85rem; }
        .detail-entry { margin-bottom: 0.8rem; }
        .detail-entry:last-child { margin-bottom: 0; }
        .detail-name { font-weight: 700; color: var(--text); }
        .detail-reason { color: var(--muted); font-size: 0.9rem; margin-top: 0.18rem; line-height: 1.45; }
        .detail-method { color: var(--muted); font-size: 0.82rem; margin-top: 0.12rem; line-height: 1.35; }


        /* ── Equal height columns for plan cards ── */
        [data-testid="stHorizontalBlock"]:has(.plan-card) {
            align-items: stretch;
        }
        [data-testid="stHorizontalBlock"]:has(.plan-card) > [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stHorizontalBlock"]:has(.plan-card) > div > [data-testid="stVerticalBlock"] {
            height: 100%;
        }
        [data-testid="stHorizontalBlock"]:has(.plan-card) .plan-card {
            height: 100%;
            min-height: 280px;
        }

       /* ── Plan cards (section 4) ── */
        .plan-metric-card { background: rgba(255, 255, 255, 0.97); border: 1px solid var(--border); border-radius: 18px; padding: 0.82rem 0.95rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); min-height: 130px; }
        .plan-metric-label { display: inline-block; border-radius: 999px; padding: 0.22rem 0.62rem; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.45rem; }
        .plan-metric-label.warning { background: rgba(197, 48, 48, 0.12); color: var(--danger); }
        .plan-metric-value { font-size: 1.42rem; font-weight: 700; color: var(--text); }
        .plan-card { background: rgba(255, 255, 255, 0.97); border: 1px solid var(--border); border-radius: 18px; padding: 0.95rem 1rem; margin-bottom: 0.85rem; min-height: 320px; }
        .plan-card-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }
        .plan-card-meta { color: var(--muted); font-size: 0.8rem; line-height: 1.35; margin-bottom: 0.45rem; }
        .plan-card-count { font-size: 1.25rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem; }
        .plan-card-text { color: var(--muted); font-size: 0.86rem; line-height: 1.45; }
        .plan-column-item { font-size: 0.9rem; font-weight: 600; color: var(--text); line-height: 1.45; margin-bottom: 0.22rem; }
        .plan-column-list { margin-top: 0.65rem; color: var(--muted); font-size: 0.88rem; line-height: 1.5; }

        [data-testid="stHorizontalBlock"]:has(.plan-metric-card) {
            align-items: stretch;
        }
        [data-testid="stHorizontalBlock"]:has(.plan-metric-card) > [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stHorizontalBlock"]:has(.plan-metric-card) > div > [data-testid="stVerticalBlock"] {
            height: 100%;
        }
        [data-testid="stHorizontalBlock"]:has(.plan-metric-card) .plan-metric-card {
            height: 100%;
            min-height: 120px;
        }


        /* ── PMG (Protection Method Guide) ── */
        .pmg-section-label { color: var(--muted); font-size: 0.76rem; font-weight: 700; line-height: 1.3; margin-bottom: 0.28rem; letter-spacing: 0.01em; }
        .pmg-section-value { color: var(--text); font-size: 0.96rem; line-height: 1.5; }
        .pmg-section-block { margin-bottom: 1.2rem; }
        .pmg-section-block:last-child { margin-bottom: 0; }
        .pmg-badges-row { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; }
        .inline-warning-badge { display: inline-flex; align-items: center; justify-content: center; padding: 0.4rem 0.53rem; border-radius: 999px; background: rgba(197, 48, 48, 0.10); color: var(--danger); font-size: 0.75rem; font-weight: 700; line-height: 1; white-space: nowrap; }

        /* ── FPD cards (section 5 Final Protection Decisions) ── */
        .fpd-col-name { font-size: 0.97rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem; }
        .fpd-reason { color: var(--muted); font-size: 0.84rem; line-height: 1.45; margin-top: 0.3rem; }
        .fpd-eyebrow { font-size: 0.72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.18rem; }
        .fpd-suggested-name { font-size: 0.9rem; font-weight: 700; color: var(--text); }
        .fpd-suggested-strength { font-size: 0.78rem; color: var(--muted); margin-left: 0.3rem; }
        .fpd-change-pill { display: inline-block; margin-top: 0.3rem; padding: 0.18rem 0.52rem; border-radius: 999px; background: rgba(22, 61, 122, 0.09); color: var(--primary); font-size: 0.73rem; font-weight: 700; }
        .fpd-warning-pill { display: inline-block; margin-top: 0.3rem; padding: 0.18rem 0.52rem; border-radius: 999px; background: rgba(197, 48, 48, 0.09); color: var(--danger); font-size: 0.73rem; font-weight: 700; }

        /* Closed selectbox selected value */
        [data-baseweb="select"] div[value] {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            color: var(--text) !important;
            letter-spacing: 0.01em !important;
        }

        
        /* ── Decision overview cards (section 5) ── */
        .decision-summary-card { background: rgba(255, 255, 255, 0.97); border: 1px solid var(--border); border-radius: 16px; padding: 0.85rem 0.95rem; min-height: 155px; margin-bottom: 0.8rem; }
        .decision-summary-label { color: var(--muted); font-size: 0.8rem; font-weight: 600; line-height: 1.35; }
        .decision-summary-value { color: var(--text); font-size: 1.45rem; font-weight: 750; line-height: 1.1; margin: 0.35rem 0 0.3rem; }
        .decision-summary-meta { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }
        .decision-summary-safe {
            background: #E0F2EA;
            border-color: #cfe3bf;
        }

        .decision-summary-warn {
            background: #f6efe0;
            border-color: #e2c98b;
        }
        .decision-summary-danger {
            background: #f9e7e7;
            border-color: #e7b8b8;
        }


        /* ── Risk assessment cards ── */
        .risk-metric-card { background: rgba(255, 255, 255, 0.97); border: 1px solid var(--border); border-radius: 16px; padding: 0.85rem 0.95rem; min-height: 108px; margin-bottom: 0.8rem; }
        .risk-metric-label { color: var(--muted); font-size: 0.8rem; font-weight: 600; line-height: 1.35; }
        .risk-metric-value { color: var(--text); font-size: 1.3rem; font-weight: 750; margin-top: 0.32rem; }
        .decision-subtle-note { color: var(--muted); font-size: 0.82rem; line-height: 1.4; margin-top: 0.25rem; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] { background: #fbfcfe; border-right: 1px solid #e6edf5; }

        /* ── Streamlit native components ── */
        [data-testid="stMetric"] { background: rgba(255, 255, 255, 0.96); border: 1px solid var(--border); border-radius: 18px; padding: 0.68rem 0.86rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }
        [data-testid="stMetricLabel"] { font-size: 0.8rem !important; font-weight: 700 !important; color: var(--muted) !important; }
        [data-testid="stMetricValue"] { font-size: 1.42rem !important; font-weight: 700 !important; color: var(--text) !important; }
        [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
        [data-testid="stCaptionContainer"] p { font-size: 0.84rem !important; line-height: 1.45 !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"], div[data-testid="stDataEditor"] { border-radius: 18px; overflow: hidden; border: 1px solid rgba(217, 226, 236, 0.85); }

        /* Keep native form controls readable in light and dark browser themes. */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: var(--text) !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: #9aabba !important;
            color: var(--text) !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
            color: var(--text) !important;
            opacity: 1 !important;
        }
        [role="option"] { color: var(--text) !important; }
        [role="option"]:hover,
        [aria-selected="true"][role="option"] { background: #edf4fb !important; }
        [data-testid="stSlider"] [role="slider"] {
            background: var(--primary) !important;
            border-color: #ffffff !important;
        }
        [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
            color: var(--text) !important;
            opacity: 1 !important;
        }
        [data-testid="stBaseButton-primary"]:disabled,
        [data-testid="stBaseButton-secondary"]:disabled,
        [data-testid="stBaseButton-primary"][disabled],
        [data-testid="stBaseButton-secondary"][disabled] {
            background: #e8edf3 !important;
            border-color: #aebdcb !important;
            color: #526b82 !important;
            opacity: 1 !important;
            cursor: not-allowed !important;
        }
        [data-testid="stBaseButton-primary"]:disabled *,
        [data-testid="stBaseButton-secondary"]:disabled *,
        [data-testid="stBaseButton-primary"][disabled] *,
        [data-testid="stBaseButton-secondary"][disabled] * {
            color: #526b82 !important;
            opacity: 1 !important;
        }
        .stApp button:focus-visible,
        .stApp input:focus-visible,
        .stApp [role="slider"]:focus-visible,
        .stApp [data-baseweb="select"]:focus-within {
            outline: 3px solid #4b83c3 !important;
            outline-offset: 2px !important;
        }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 1rem 2rem; }
            .app-hero { padding: 1.2rem; border-radius: 18px; }
            .app-hero h1 { font-size: 1.75rem; }
            .app-hero p { font-size: 0.88rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <style>
    

    .validation-metric-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .validation-metric-safe {
        background: #edf7e7;
        border-color: #cfe3bf;
    }

    .validation-metric-review {
        background: #fbf7ee;
        border-color: #e6d7b2;
    }

    .validation-metric-danger {
        background: #f9e7e7;
        border-color: #e7b8b8;
    }

    .validation-metric-label {
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .validation-metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        line-height: 1;
        margin-top: 0.55rem;
    }

    .validation-metric-meta {
        font-size: 0.84rem;
        margin-top: 0.5rem;
        line-height: 1.25;
        color: var(--muted);
        font-weight: 500;
    }


    .validation-verdict-row {
        display: flex;
        align-items: center;
        gap: 0.48rem;
        margin-bottom: 0.34rem;
    }

    .validation-verdict-icon {
        width: 1.82rem;
        height: 1.82rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.88rem;
        font-weight: 700;
        flex-shrink: 0;
        border: 1.4px solid rgba(16, 42, 67, 0.65);
        color: rgba(16, 42, 67, 0.82);
        background: rgba(255, 255, 255, 0.45);
    }

    .validation-issue-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(18, 28, 45, 0.05);
        border-left: 2px solid rgba(180, 83, 9, 0.24);
        background: #fcfaf5;
    }

    .validation-issue-danger {
        background: #faf1f1;
        border-color: rgba(191, 90, 90, 0.16);
        border-left-color: rgba(180, 83, 9, 0.38);
        box-shadow: none;
    }

    .validation-issue-warn {
        background: #fcfaf5;
        border-color: rgba(169, 128, 47, 0.12);
        border-left-color: rgba(180, 83, 9, 0.24);
        box-shadow: none;
    }

    .validation-issue-title-danger {
        color: #9c4646;
    }

    .validation-issue-title-warn {
        color: #7b6738;
    }

    /* ── Step 6 inspection intro ── */
    .inspection-intro-card {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(18, 28, 45, 0.06);
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.95rem;
    }

    .inspection-intro-eyebrow {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .inspection-intro-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.45rem;
    }

    .inspection-intro-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 800;
        flex-shrink: 0;
        color: var(--primary);
        border: 1.6px solid rgba(22, 61, 122, 0.5);
        background: rgba(22, 61, 122, 0.02);
    }

    .inspection-intro-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }

    .inspection-intro-text {
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.55;
        margin-top: 0.15rem;
    }


    /* ── Risk / info cards ── */
    .risk-hero-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-left: 6px solid #a16207;
        background: rgba(255, 255, 255, 0.98);
        margin-bottom: 1.05rem;
    }

    .risk-hero-navy {
        border-left-color: #163d7a;
        background: rgba(22, 61, 122, 0.05);
    }

    .risk-hero-navy .risk-hero-text {
        color: rgba(22, 61, 122, 0.75);
    }

    .risk-hero-high {
        border-left-color: #b45309;
        background: #faf1f1;
        border-color: rgba(180, 83, 9, 0.22);
    }

    .risk-hero-moderate {
        border-left-color: #8b5e00;
        background: #fbf7ee;
        border-color: rgba(200, 164, 90, 0.22);
    }

    .risk-hero-low {
        border-left-color: #2a7a52;
        background: #edf7f1;
        border-color: rgba(47, 133, 90, 0.20);
    }

    .risk-hero-eyebrow {
        color: rgba(98, 125, 152, 0.88);
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .risk-hero-value {
        color: var(--text);
        font-size: 1.32rem;
        font-weight: 700;
        margin: 0.14rem 0 0.26rem;
    }

    .risk-hero-text {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }




    /* ── Step 6 dedicated hero card ── */
    
    .step6-hero-card {
        border-radius: 22px;
        padding: 1.05rem 1.15rem;
        border: 1px solid var(--border);
        border-left: 6px solid #a16207;
        background: #fbf7ee;
        margin-bottom: 1.05rem;
    }


    .step6-hero-high {
        background: #faf1f1;
        border-color: rgba(180, 83, 9, 0.22);
        border-left: 6px solid #b45309;
    }

    .step6-hero-moderate {
        background: #fbf7ee;
        border-color: rgba(200, 164, 90, 0.22);
        border-left: 6px solid #a16207;
    }

    .step6-hero-low {
        background: #edf7f1;
        border-color: rgba(47, 133, 90, 0.20);
        border-left: 6px solid #2a7a52;
    }

    .step6-hero-eyebrow {
        color: rgba(98, 125, 152, 0.88);
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.28rem;
    }

    .step6-hero-title {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }

    .step6-hero-text {
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.5;
    }

    

    /* ── Step 7 outputs redesign ── */

    .outputs-hero-card {
        border-radius: 22px;
        padding: 1.05rem 1.15rem;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.98);
        margin-bottom: 1rem;
    }

    .outputs-hero-safe {
        background: #edf7f1;
        border-color: rgba(47, 133, 90, 0.20);
        border-left: 6px solid #2a7a52;
    }

    .outputs-hero-review {
        background: #fbf7ee;
        border-color: rgba(200, 164, 90, 0.22);
        border-left: 6px solid #a16207;
    }

    .outputs-hero-danger {
        background: #faf1f1;
        border-color: rgba(180, 83, 9, 0.22);
        border-left: 6px solid #b45309;
    }

    .outputs-hero-eyebrow {
        color: rgba(98, 125, 152, 0.88);
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.28rem;
    }

    .outputs-hero-title {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }

    .outputs-hero-text {
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.5;
    }

    .outputs-signal-card {
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        min-height: 132px;
        margin-bottom: 0.85rem;
    }

    .outputs-signal-label {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .outputs-signal-value {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.1;
        margin: 0.38rem 0 0.26rem;
    }

    .outputs-signal-meta {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .outputs-breakdown-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid var(--border);
        border-left: 5px solid transparent;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        min-height: 220px;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    .outputs-breakdown-strong {
        border-left-color: #2f855a;
        background: #edf7f1;
        border-color: rgba(47, 133, 90, 0.18);
    }

    .outputs-breakdown-medium {
        border-left-color: #b7791f;
        background: #fbf7ee;
        border-color: rgba(183, 121, 31, 0.20);
    }

    .outputs-breakdown-unchanged {
        border-left-color: #c53030;
        background: #faf1f1;
        border-color: rgba(197, 48, 48, 0.18);
    }

    .outputs-breakdown-title-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.25rem;
    }

    .outputs-breakdown-icon {
        font-size: 1rem;
        line-height: 1;
        flex-shrink: 0;
    }

    .outputs-breakdown-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0;
    }

    .outputs-breakdown-value {
        font-size: 1.45rem;
        font-weight: 750;
        color: var(--text);
        margin-bottom: 0.35rem;
    }

    .outputs-breakdown-meta {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
        margin-bottom: 0.55rem;
    }

    .outputs-breakdown-list {
        color: var(--text);
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    .outputs-breakdown-list div {
        margin-bottom: 0.18rem;
    }

    .outputs-breakdown-list strong {
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .outputs-breakdown-list .breakdown-columns {
        color: #475569;
        font-weight: 500;
    }

    .outputs-note-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.98);
        margin-bottom: 0.85rem;
     }

    .outputs-note-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.28rem;
    }

    .outputs-note-text {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .outputs-improvement-item {
        color: var(--text);
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 0.32rem;
    }


    .outputs-guidance-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        border-left: 5px solid transparent;
        background: rgba(255, 255, 255, 0.98);
        min-height: 122px;
        margin-bottom: 0.85rem;
    }

    .outputs-guidance-allowed {
        border-left-color: #2f855a;
        background: #edf7f1;
        border-color: rgba(47, 133, 90, 0.18);
    }

    .outputs-guidance-restricted {
        border-left-color: #b7791f;
        background: #fbf7ee;
        border-color: rgba(183, 121, 31, 0.20);
    }

    .outputs-guidance-title-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.28rem;
    }

    .outputs-guidance-icon {
        font-size: 1rem;
        line-height: 1;
        flex-shrink: 0;
    }

    .outputs-guidance-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }

    .outputs-guidance-text {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }

   .outputs-nextstep-card {
        border-radius: 18px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(22, 61, 122, 0.18);
        border-left: 6px solid #163d7a; /* darker blue stripe */
        background: rgba(22, 61, 122, 0.05); /* subtle blue tint */
        margin-bottom: 0.85rem;
    }

    .outputs-nextstep-title-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.35rem;
    }

    .outputs-nextstep-icon {
        font-size: 1rem;
        line-height: 1;
        flex-shrink: 0;
    }

    .outputs-nextstep-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        line-height: 1.3;
    }

        .outputs-nextstep-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .outputs-nextstep-list {
        margin-top: 0.8rem;
    }

    .outputs-nextstep-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin-bottom: 0.62rem;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .outputs-nextstep-bullet {
        color: #163d7a;
        font-weight: 700;
        min-width: 1rem;
        flex-shrink: 0;
    }

    .outputs-field-list {
        color: #6b7280;
    }

    .outputs-nextstep-item-text {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #334155;
    }

    

    /* ── Step 8 export redesign ── */

    .export-hero-card {
        border-radius: 22px;
        padding: 1.02rem 1.12rem;
        border: 1px solid rgba(22, 61, 122, 0.16);
        border-left: 6px solid #163d7a;
        background: rgba(22, 61, 122, 0.05);
        margin-bottom: 0.9rem;
    }

    .export-hero-eyebrow {
        color: rgba(98, 125, 152, 0.88);
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.28rem;
    }

    .export-hero-title {
        color: var(--text);
        font-size: 1.28rem;
        font-weight: 750;
        line-height: 1.18;
        margin-bottom: 0.28rem;
    }

    .export-hero-text {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .export-mini-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.82rem 0.95rem;
        min-height: 112px;
        margin-bottom: 0.75rem;
    }

    .export-mini-label {
        color: var(--muted);
        font-size: 0.77rem;
        font-weight: 700;
        line-height: 1.3;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .export-mini-value {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 750;
        line-height: 1.18;
        margin: 0.34rem 0 0.18rem;
    }

    .export-mini-value-run {
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1.35;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        word-break: break-word;
    }

    .export-mini-meta {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.42;
    }

    .export-bundle-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.92rem 1rem;
        margin-bottom: 0.8rem;
    }

    .export-bundle-title-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.22rem;
    }

    .export-bundle-icon {
        font-size: 1rem;
        line-height: 1;
        flex-shrink: 0;
    }

    .export-bundle-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        line-height: 1.3;
    }

    .export-bundle-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.68rem;
    }

    .export-file-list {
        margin-top: 0.4rem;
        padding-top: 0.2rem;
        border-top: none;
    }

    .export-file-item {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 0.22rem;
    }

    .export-file-item span {
        color: var(--muted);
    }

    .export-note-card {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.98);
        margin-bottom: 0.8rem;
    }

    .export-note-card-last {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(22, 61, 122, 0.16);
        border-left: 6px solid #163d7a;
        background: rgba(22, 61, 122, 0.05);
        margin-bottom: 0.8rem;
    }

    .export-note-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.28rem;
    }

    .export-note-text {
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.5;
    }


    .export-hero-helper {
        margin-top: 0.7rem;
        padding-top: 0.6rem;
        border-top: 1px solid rgba(217, 226, 236, 0.7);
    }

    .export-helper-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.2rem;
    }

    .export-helper-text {
        font-size: 0.85rem;
        color: var(--muted);
        line-height: 1.45;
    }

    .helper-subtle {
        display: block;
        margin-top: 0.25rem;
        font-size: 0.8rem;
    }

    
    .export-combined-header {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }

    .export-combined-section {
        flex: 1;
    }

    .export-combined-divider {
        margin: 0.5rem 0 0.6rem;
        border-top: 1px solid rgba(217, 226, 236, 0.7);
    }

    .export-combined-body {
        display: flex;
        gap: 1.5rem;
    }

    .export-combined-left,
    .export-combined-right {
        flex: 1;
    }
    


    /* ── Section 8 download buttons ── */
    [data-testid="stDownloadButton"] button {
        width: 100% !important;
        min-height: 2.65rem !important;
        border-radius: 12px !important;
        background: #163d7a !important;
        color: #ffffff !important;
        border: 1px solid #163d7a !important;
        font-weight: 650 !important;
        font-size: 0.88rem !important;
        box-shadow: none !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #1a4a94 !important;
        border-color: #1a4a94 !important;
        color: #ffffff !important;
    }

    [data-testid="stDownloadButton"] button:focus {
        box-shadow: 0 0 0 0.18rem rgba(22, 61, 122, 0.14) !important;
    }


    
    

    /* ── Premium dataframe styling ── */

    .table-card {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.6rem 0.6rem 0.2rem;
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }

    

    /* Target Streamlit dataframe */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        margin-top: 0.35rem;
    }

    
    /* Header */
    div[data-testid="stDataFrame"] thead th {
        background: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }

    
    /* Rows */
    div[data-testid="stDataFrame"] tbody td {
        font-size: 0.82rem !important;
        color: #334155 !important;
        border-bottom: 1px solid #f1f5f9 !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }


    /* Zebra striping */
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #fcfdff !important;
    }

    /* Hover */
    div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #f1f5f9 !important;
    }

    /* Index column */
    div[data-testid="stDataFrame"] tbody td:first-child {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
    }

    /* Scrollbar (subtle) */
    div[data-testid="stDataFrame"] ::-webkit-scrollbar {
        height: 6px;
    }

    div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 6px;
    }

    /* Section 6 & 7 redesign */
    .decision-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.55rem;
    }

    .decision-chip {
        display: inline-block;
        border-radius: 999px;
        padding: 0.34rem 0.72rem;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.96);
        color: var(--text);
    }

    


    /* SECTION 5 – Policy alignment stronger green */
    .policy-alignment-card.risk-hero-low {
        border-left-color: #2a7a52;
    }

    .policy-alignment-card.risk-hero-moderate {
        border-left-color: #8b5e00 !important;
    }

    .policy-alignment-card.risk-hero-high {
        border-left-color: #b45309;
    }




    </style>

    """,
    unsafe_allow_html=True,
)

def _render_step_header(step_number: int, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{step_number}. {title}</div>
        <div class="section-subtitle">{description}</div>
        """,
        unsafe_allow_html=True,
    )


# Spacing between sections
def _render_section_spacer(size: str = "md") -> None:
    sizes = {"sm": ".5rem", "md": "2rem", "lg": "4rem"}
    st.markdown(
        f'<div style="height:{sizes.get(size, "1.75rem")};"></div>',
        unsafe_allow_html=True,
    )  



def _build_plan_signature(
    plan_df: pd.DataFrame,
    planning_context: dict[str, object],
) -> str:
    plan_parts = []
    for row in plan_df.itertuples(index=False):
        plan_parts.append(
            "|".join(
                [
                    str(getattr(row, "column_name", "")),
                    str(getattr(row, "recommended_action", "")),
                    str(getattr(row, "classification", "")),
                    str(getattr(row, "functional_role", "")),
                ]
            )
        )

    context_part = "|".join(
        [
            str(planning_context.get("intended_use", "")),
            str(planning_context.get("need_record_linkage", "")),
            str(planning_context.get("dataset_domain", "")),
            str(planning_context.get("dataset_size_rows", "")),
        ]
    )

    return "||".join([context_part] + plan_parts)


def _short_signature(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]



def _build_run_input_signature(
    uploaded_file: Any,
    planning_context: dict[str, Any],
) -> str:
    file_hasher = hashlib.md5()
    try:
        file_hasher.update(uploaded_file.getbuffer())
    except Exception:
        file_hasher.update(uploaded_file.getvalue())

    payload = {
        "file_name": str(getattr(uploaded_file, "name", "")),
        "file_digest": file_hasher.hexdigest(),
        "planning_context": planning_context,
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _build_reviewed_plan_signature(reviewed_df: pd.DataFrame) -> str:
    signature_rows: list[str] = []
    for row in reviewed_df.itertuples(index=False):
        signature_rows.append(
            "|".join(
                [
                    str(getattr(row, "column_name", "")),
                    str(getattr(row, "classification", "")),
                    str(getattr(row, "recommended_action", "")),
                    str(getattr(row, "final_action", "")),
                ]
            )
        )
    return hashlib.md5("||".join(signature_rows).encode("utf-8")).hexdigest()


def _clear_saved_run_outputs() -> None:
    st.session_state["anonymization_completed"] = False
    st.session_state["anonymized_df"] = None
    st.session_state["transformation_log"] = None
    st.session_state["validation_df"] = None
    st.session_state["validation_summary"] = None
    st.session_state["report_markdown"] = None
    st.session_state["report_json"] = None
    st.session_state["run_id"] = None
    st.session_state["run_dir"] = None
    st.session_state["artifact_paths"] = None
    st.session_state["_completed_run_input_signature"] = None
    st.session_state["_completed_review_plan_signature"] = None


def _recommended_guide_methods(planning_context: dict[str, object]) -> list[str]:
    intended_use = str(planning_context.get("intended_use", "")).strip().lower()
    need_record_linkage = bool(planning_context.get("need_record_linkage", False))

    if intended_use == "external sharing":
        return ["drop", "generalize_light"]

    if intended_use == "ml training":
        if need_record_linkage:
            return ["pseudonymize", "add_noise"]
        return ["add_noise", "generalize_light"]

    if intended_use == "internal analysis":
        if need_record_linkage:
            return ["pseudonymize", "add_noise"]
        return ["keep", "add_noise", "generalize_light"]

    if intended_use == "dashboard analytics":
        return ["keep", "add_noise", "generalize_light"]

    if intended_use == "research":
        return ["generalize_light", "add_noise"]

    return []



def _format_action_card_explanation(
    *,
    action: str,
    intended_use: str | None,
    record_linkage_required: bool,
) -> str:
    action = _normalize_action_key(action)
    intended_use = (intended_use or "").strip().lower()

    if action == "drop":
        if intended_use == "external sharing":
            return "Removes identifying fields entirely to ensure safe external data sharing."
        return "Removes fields that are not needed and present high identification risk."

    if action == "pseudonymize":
        if record_linkage_required:
            return "Replaces identifiers with consistent tokens so records can still be linked without exposing identities."
        return "Replaces identifiers to reduce direct identity exposure."

    if _is_generalize_action(action):
        if intended_use == "external sharing":
            return "Reduces detail to lower re-identification risk when data is shared externally."
        if intended_use in {"ml training", "internal analysis", "dashboard analytics"}:
            return "Reduces identifiability while preserving useful analytical patterns."
        return "Reduces precision to limit identification risk while keeping data useful."

    if action == "add_noise":
        return "Adds controlled variation to numeric values to protect sensitive data while preserving trends."

    if action == "keep":
        if intended_use in {"ml training", "internal analysis"}:
            return "Keeps low-risk fields unchanged to preserve analytical value."
        return "Leaves fields unchanged where no significant risk was detected."

    if action == "mask":
        return "Hides values for display purposes but does not fully anonymize the underlying data."

    return "Recommended based on detected privacy risk and selected use case."



def _format_planning_explanation(
    *,
    action: str,
    reason_code: str | None,
    classification: str | None,
    functional_role: str | None,
    intended_use: str | None,
    record_linkage_required: bool | None = None,
) -> str:
    action = _normalize_action_key(action)
    reason_code = (reason_code or "").strip().upper()
    classification = (classification or "").strip().lower()
    functional_role = (functional_role or "").strip().lower()
    intended_use = (intended_use or "").strip().lower()
    record_linkage_required = bool(record_linkage_required)

    if reason_code == "LOW_RISK_ANALYTICAL_KEEP":
        return (
            "Kept unchanged because this is a low-risk analytical feature that remains useful in the selected use case."
        )

    if reason_code == "LOW_RISK_DEMOGRAPHIC_KEEP":
        if intended_use in {"ml training", "internal analysis", "dashboard analytics"}:
            return (
                "Kept unchanged because this demographic field has relatively low standalone identification risk and remains useful for analysis. "
                "Note: keeping this field in exact form may still contribute to residual linkage risk, which can limit external sharing."
            )
        return (
            "Kept unchanged because this field has relatively low identification risk in the selected use case. "
            "Note: keeping this field in exact form may still contribute to residual linkage risk, which can limit external sharing."
        )

    if reason_code == "ANALYTICAL_QUASI_GENERALIZE":
        if intended_use == "external sharing":
            return (
                "Generalized because this field can contribute to re-identification and should be shared in reduced-detail form externally."
            )
        return (
            "Generalized because this field can help identify individuals when combined with others, while still being useful for analysis."
        )

    if reason_code in {"SENSITIVE_ANALYTICAL_NOISE", "ML_SENSITIVE_NOISE", "DASHBOARD_SENSITIVE_NOISE"}:
        return (
            "Protected with numeric noise because this field is analytically useful, but exact values should not remain exposed."
        )

    if reason_code == "EXTERNAL_IDENTITY_REMOVE":
        return (
            "Removed because this field directly identifies individuals or enables linkage, which is not appropriate for external sharing."
        )

    if reason_code == "LINKAGE_REQUIRED_PSEUDONYMIZE":
        if record_linkage_required:
            return (
                "Pseudonymized to preserve record linkage while reducing direct identity exposure."
            )
        return (
            "Pseudonymized to reduce direct identity exposure."
        )

    if reason_code == "EXTERNAL_TECHNICAL_REMOVE":
        return (
            "Removed because internal system identifiers are usually unnecessary and risky in externally shared data."
        )

    if reason_code == "EXTERNAL_QUASI_GENERALIZE":
        return (
            "Generalized because indirect identifiers should be shared in reduced-detail form for external use."
        )

    if reason_code == "EXTERNAL_SENSITIVE_GENERALIZE":
        return (
            "Generalized because sensitive fields should not be shared in exact form externally."
        )

    if reason_code == "RESEARCH_SENSITIVE_GENERALIZE":
        return (
            "Generalized because this sensitive field should be reduced in detail for research use."
        )

    if reason_code == "SMALL_DATASET_SENSITIVE_GENERALIZE":
        return (
            "Generalized because small datasets are more vulnerable to re-identification and need stronger protection for sensitive fields."
        )

    if reason_code == "ANALYTICAL_SAFETY_NO_PSEUDONYMIZATION":
        return (
            "Generalized instead of pseudonymized because analytical fields should preserve signal rather than become artificial identifiers."
        )

    if reason_code == "USER_OVERRIDE":
        return "This recommendation was manually adjusted during review."

    if reason_code == "DEFAULT_CLASSIFICATION_MAPPING":
        if action == "keep":
            return "Kept unchanged because this field has low identification risk in the selected use case."
        if action == "drop":
            return "Removed because this field presents identification risk and is not needed in the selected output."
        if _is_generalize_action(action):
            return "Generalized to reduce re-identification risk while preserving useful patterns."
        if action == "add_noise":
            return "Protected with numeric noise to reduce exposure of exact values while preserving broader trends."
        if action == "pseudonymize":
            return "Pseudonymized to reduce direct identity exposure while preserving linkage."
        if action == "mask":
            return "Masked to reduce direct visibility of identifying content."

    # Final fallback
    if action == "drop":
        return "Removed because this field presents identification risk and is not needed in the selected output."
    if action == "pseudonymize":
        return "Pseudonymized to reduce direct identity exposure while preserving linkage."
    if _is_generalize_action(action):
        return "Generalized to reduce re-identification risk while preserving useful patterns."
    if action == "add_noise":
        return "Protected with numeric noise to reduce exposure of exact values while preserving broader trends."
    if action == "keep":
        return (
            "Kept unchanged because this field has relatively low identification risk and remains useful for analysis. "
            "Note: keeping this field in exact form may still contribute to residual linkage risk, which can limit external sharing."
        )
    if action == "mask":
        return "Masked to reduce direct visibility of identifying content."

    return "Recommended based on detected privacy risk and the selected use case."



def _friendly_classification(value: Any) -> str:
    return _CLASSIFICATION_LABELS.get(str(value), str(value))


def _friendly_action(value: Any) -> str:
    normalized = _normalize_action_key(value)
    return _ACTION_LABELS.get(normalized, str(value))


def _action_summary_explanation(action: str) -> str:
    explanations = {
        "drop": "Safest option when the field is not needed for the final dataset.",
        "pseudonymize": "Keeps records linkable without exposing real identifiers.",
        "generalize_light": "Light reduction in detail while preserving useful analytical patterns.",
        "generalize_medium": "Stronger reduction in detail with less exactness in the final dataset.",
        "generalize_strict": "Strongest reduction in detail for tighter privacy protection.",
        "add_noise": "Slightly changes numeric values while keeping broader trends intact.",
        "mask": "Hides part of a value for display, but is not full anonymization.",
        "keep": "Leaves the field unchanged because the current scan did not require protection.",
    }
    return explanations.get(_normalize_action_key(action), "Applies the recommended protection for this field.")


def _action_strength_label(action_key: str) -> str:
    strength_labels = {
        "drop": "🟢 Strong protection",
        "pseudonymize": "🟢 Strong protection",
        "generalize_light": "🟡 Medium protection",
        "generalize_medium": "🟡 Medium protection",
        "generalize_strict": "🟢 Stronger protection",
        "add_noise": "🟡 Medium protection",
        "mask": "🟡 Display protection",
        "keep": "⚠ No protection",
    }
    return strength_labels.get(_normalize_action_key(action_key), "Recommended protection")


def _action_display_parts(action_key: str) -> tuple[str, str]:
    action_names = {
        "drop": "Remove column",
        "pseudonymize": "Replace with pseudonym",
        "generalize_light": "Generalize / reduce precision",
        "generalize_medium": "Generalize more",
        "generalize_strict": "Generalize strongly",
        "add_noise": "Add numeric noise",
        "mask": "Mask value (display only)",
        "keep": "Leave unchanged",
    }
    normalized = _normalize_action_key(action_key)
    return (
        action_names.get(normalized, str(normalized).replace("_", " ").title()),
        _action_strength_label(normalized),
    )


def _build_plan_action_cards(plan_df: pd.DataFrame) -> list[dict[str, Any]]:
    ordered_action_keys = (
        plan_df["recommended_action"]
        .dropna()
        .astype(str)
        .map(_normalize_action_key)
        .drop_duplicates()
        .tolist()
    )

    action_cards: list[dict[str, Any]] = []

    for action_key in ordered_action_keys:
        assigned_columns = plan_df.loc[
            plan_df["recommended_action"].astype(str).map(_normalize_action_key) == action_key, "column_name"
        ].astype(str).tolist()
        friendly_label, _ = _action_display_parts(action_key)
        strength_label = _action_strength_label(action_key)

        action_cards.append(
            {   "action": action_key,
                "action_key": action_key,
                "friendly_label": friendly_label,
                "count": len(assigned_columns),
                "explanation": _action_summary_explanation(action_key),
                "strength_label": strength_label,
                "columns": assigned_columns,
                  
            }
        )

    return action_cards


def _raw_action_from_label(value: Any) -> Any:
    reverse_labels = {label: action for action, label in _ACTION_LABELS.items()}
    raw_value = reverse_labels.get(str(value), value)
    return _normalize_action_key(raw_value)


def _stringify_for_display(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, list):
        return ", ".join("" if item is None else str(item) for item in value)

    if isinstance(value, dict):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    try:
        if pd.isna(value):
            return value
    except TypeError:
        pass

    return str(value)


def _prepare_dataframe_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    safe_df = dataframe.copy(deep=True)
    object_columns = safe_df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        safe_df[column] = safe_df[column].map(_stringify_for_display)
    return safe_df


def _build_presidio_influence_column(classification_df: pd.DataFrame) -> pd.Series:
    if "reason" not in classification_df.columns:
        return pd.Series(["no"] * len(classification_df), index=classification_df.index)
    return classification_df["reason"].map(
        lambda value: "yes" if "presidio" in str(value).lower() else "no"
    )


def _build_review_table(
    plan_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    presidio_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    review_df = plan_df.copy(deep=True)
    classification_reason_df = classification_df[["column_name"]].copy()
    if "reason" in classification_df.columns:
        classification_reason_df["reason"] = classification_df["reason"]
    else:
        classification_reason_df["reason"] = classification_reason_df["column_name"].map(
            lambda _: "Action derived from classification."
        )

    review_df = review_df.merge(classification_reason_df, on="column_name", how="left")
    review_df["reason"] = review_df.apply(
        lambda row: row["reason"]
        if pd.notna(row["reason"])
        else _REASON_BY_CLASSIFICATION.get(
            str(row["classification"]), "Action derived from classification."
        ),
        axis=1,
    )
    review_df["presidio_influenced"] = review_df["reason"].map(
        lambda value: "yes" if "presidio" in str(value).lower() else "no"
    )

    if presidio_summary_df is not None and not presidio_summary_df.empty:
        presidio_columns = ["column_name", "presidio_entities_detected", "presidio_hit_count"]
        available_columns = [column for column in presidio_columns if column in presidio_summary_df.columns]
        if "column_name" in available_columns:
            review_df = review_df.merge(
                presidio_summary_df[available_columns].copy(),
                on="column_name",
                how="left",
            )
        else:
            review_df["presidio_entities_detected"] = None
            review_df["presidio_hit_count"] = None
    else:
        review_df["presidio_entities_detected"] = None
        review_df["presidio_hit_count"] = None

    review_df["presidio_entities_detected"] = review_df["presidio_entities_detected"].map(
        _stringify_for_display
    )
    if "final_action" not in review_df.columns:
        review_df["final_action"] = review_df["recommended_action"]
    selected_columns = [
        "column_name",
        "classification",
        "reason",
        "presidio_influenced",
        "presidio_entities_detected",
        "presidio_hit_count",
        "recommended_action",
        "final_action",
    ]
    selected_columns.extend(
        column
        for column in [
            "functional_role",
            "planning_reason_code",
            "policy_context",
            "record_linkage_required",
        ]
        if column in review_df.columns
    )
    return review_df[selected_columns]


def _build_external_mask_warning(reviewed_df: pd.DataFrame, intended_use: str) -> str | None:
    if str(intended_use).strip().lower() != "external sharing":
        return None
    required_columns = {"column_name", "classification", "final_action"}
    if not required_columns.issubset(reviewed_df.columns):
        return None

    direct_identifier_mask = (
        (reviewed_df["classification"] == "direct_identifier")
        & (reviewed_df["final_action"] == "mask")
    )
    if not direct_identifier_mask.any():
        return None

    column_list = ", ".join(
        reviewed_df.loc[direct_identifier_mask, "column_name"].astype(str).tolist()
    )
    return (
        "Masking is not recommended for external sharing because it may still expose personal data "
        f"under GDPR. Review these direct identifiers: {column_list}"
    )


def _build_step5_context_hint(planning_context: dict[str, Any]) -> str:
    intended_use = str(planning_context.get("intended_use", "")).strip().lower()
    need_record_linkage = bool(planning_context.get("need_record_linkage", False))

    if intended_use == "external sharing":
        return (
            "Based on your configuration: External sharing usually requires stronger protection. "
            "Direct identifiers are typically removed rather than masked or kept."
        )
    if intended_use == "ml training" and need_record_linkage:
        return (
            "Based on your configuration: Intended use is ML training and record linkage is required. "
            "Direct identifiers are usually replaced with pseudonyms so records can still be linked "
            "without exposing real identities."
        )
    if intended_use == "dashboard analytics" and not need_record_linkage:
        return (
            "Based on your configuration: Intended use is dashboard analytics and record linkage is not "
            "required. Direct identifiers are usually removed, and masking may be suitable only for "
            "display-oriented internal views."
        )
    if need_record_linkage:
        return (
            "Based on your configuration: Record linkage is required. Pseudonyms are often preferred for "
            "direct identifiers when you need to reconnect the same records later."
        )
    if intended_use == "research":
        return (
            "Based on your configuration: Research datasets often benefit from generalized values so the "
            "dataset stays useful while reducing the chance of identifying individuals."
        )
    return (
        "Based on your configuration: If records do not need to be linked later, direct identifiers are "
        "usually removed and indirect identifiers are often generalized."
    )


def _build_privacy_impact_preview(reviewed_df: pd.DataFrame) -> tuple[dict[str, int], str, str]:
    direct_identifiers_exposed = int(
        (
            (reviewed_df["classification"] == "direct_identifier")
            & (~reviewed_df["final_action"].isin(["drop", "pseudonymize"]))
        ).sum()
    )
    indirect_identifiers_present = int(
        (
            (reviewed_df["classification"] == "quasi_identifier")
            & (reviewed_df["final_action"].isin(["keep", "mask"]))
        ).sum()
    )
    sensitive_fields_weakly_protected = int(
        (
            (reviewed_df["classification"] == "sensitive_attribute")
            & (reviewed_df["final_action"].isin(["keep", "mask"]))
        ).sum()
    )
    display_only_protections = int((reviewed_df["final_action"] == "mask").sum())

    if direct_identifiers_exposed > 0:
        estimated_risk = "High"
        preview_message = (
            "Some direct identifiers are still exposed. Consider pseudonymizing or removing them."
        )
    elif indirect_identifiers_present > 0 or sensitive_fields_weakly_protected > 0:
        estimated_risk = "Moderate"
        preview_message = "Masking is display-oriented and may still leave data within GDPR scope."
    else:
        estimated_risk = "Low"
        preview_message = "Your current selections appear aligned with stronger privacy protection."

    preview_counts = {
        "Direct identifiers still exposed": direct_identifiers_exposed,
        "Indirect identifiers still present": indirect_identifiers_present,
        "Sensitive fields still weakly protected": sensitive_fields_weakly_protected,
        "Display-only protections selected": display_only_protections,
    }
    return preview_counts, estimated_risk, preview_message


def _render_context_strip(items: list[tuple[str, str]]) -> None:
    chip_markup = "".join(
        f'<div class="context-chip"><strong>{label}:</strong> {value}</div>'
        for label, value in items
    )
    st.markdown(f'<div class="context-strip">{chip_markup}</div>', unsafe_allow_html=True)


def _render_overview_metric_cards(
    metadata: Any,
    classification_df: pd.DataFrame,
) -> None:
    detected_sensitive_columns = int(
        classification_df["classification"].isin(
            ["direct_identifier", "quasi_identifier", "sensitive_attribute"]
        ).sum()
    )

    metric_items = [
        {
            "label": "Rows",
            "value": f"{metadata.rows:,}",
            "meta": "Total records in the uploaded dataset.",
            "card_class": "",
        },
        {
            "label": "Columns",
            "value": f"{metadata.columns:,}",
            "meta": "Fields available before protection is applied.",
            "card_class": "",
        },
        {
            "label": "File type",
            "value": str(metadata.file_type).upper(),
            "meta": "Uploaded source format.",
            "card_class": "",
        },
        {
            "label": "Detected sensitive columns",
            "value": f"{detected_sensitive_columns} of {metadata.columns}",
            "meta": "Columns flagged as direct, indirect, or sensitive.",
            "card_class": "highlight",
        },
    ]

    metric_columns = st.columns(4, gap="large")
    for column, item in zip(metric_columns, metric_items):
        with column:
            st.markdown(
                f"""
                <div class="overview-metric-card {html.escape(item['card_class'])}">
                    <div class="overview-metric-label">{html.escape(item["label"])}</div>
                    <div class="overview-metric-value">{html.escape(item["value"])}</div>
                    <div class="overview-metric-meta">{html.escape(item["meta"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_classification_pie_chart(classification_df: pd.DataFrame) -> None:
    counts = (
        classification_df["classification"]
        .map(_CLASSIFICATION_LABELS)
        .fillna("Other")
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Columns")
    )
    if counts.empty:
        st.caption("No column categories are available to chart.")
        return

    figure = px.pie(
        counts,
        names="Category",
        values="Columns",
        hole=0.55,
        color_discrete_sequence=["#163d7a", "#0f8b8d", "#63b3ed", "#ed8936", "#a0aec0"],
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    figure.update_layout(
        height=330,
        margin=dict(t=12, b=12, l=12, r=12),
        legend_title_text="",
        template="plotly_white",
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )



def _render_profiling_summary(profiling_df: pd.DataFrame) -> None:
    if profiling_df.empty:
        st.info("Profiling details are not available for this dataset.")
        return

    _DTYPE_LABELS = {
        "object": "Text",
        "int64": "Integer",
        "int32": "Integer",
        "float64": "Decimal",
        "float32": "Decimal",
        "bool": "True / False",
        "datetime64[ns]": "Date / Time",
        "datetime64": "Date / Time",
        "category": "Category",
    }

    summary_df = profiling_df.copy(deep=True)
    summary_df["null_percent"] = pd.to_numeric(summary_df["null_percent"], errors="coerce").fillna(0.0)
    summary_df["unique_percent"] = pd.to_numeric(summary_df["unique_percent"], errors="coerce").fillna(0.0)

    summary_df = summary_df[summary_df["column_name"] != "Index"].reset_index(drop=True)

    profile_columns = st.columns(3, gap="large")
    for index, row in enumerate(summary_df.itertuples(index=False)):
        dtype_label = _DTYPE_LABELS.get(str(row.dtype), str(row.dtype))
        unique_pct = round(float(row.unique_percent))
        null_pct = round(float(row.null_percent))
        missing_text = "None" if null_pct == 0 else f"{null_pct}%"

        uniqueness_hint = ""
        if unique_pct >= 80:
            uniqueness_hint = '<div class="profile-card-hint">High uniqueness</div>'

        target_column = profile_columns[index % 3]
        with target_column:
            st.markdown(
                f"""
                <div class="profile-card">
                    <div class="profile-card-title">{html.escape(str(row.column_name))}</div>
                    <div class="profile-card-meta"><strong>Type:</strong> {html.escape(dtype_label)}</div>
                    <div class="profile-card-meta"><strong>Missing values:</strong> {html.escape(missing_text)}</div>
                    <div class="profile-card-meta"><strong>Unique values:</strong> {unique_pct}%</div>
                    {uniqueness_hint}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _category_badge_markup(classification: str) -> str:
    return (
        f'<div class="category-badge category-{classification}">'
        f"{_friendly_classification(classification)}"
        "</div>"
    )


def _simplified_category_label(classification: str) -> str:
    labels = {
        "direct_identifier": "Direct identifier",
        "quasi_identifier": "Indirect identifier",
        "sensitive_attribute": "Sensitive information",
        "technical_identifier": "Internal system identifier",
        "non_sensitive": "No sensitivity detected",
    }
    return labels.get(classification, classification.replace("_", " ").title())


def _render_subsection_heading(title: str) -> None:
    st.markdown(f'<div class="subsection-title">{title}</div>', unsafe_allow_html=True)


def _format_column_list_for_ui(value: Any, empty_label: str = "None") -> str:
    if value is None:
        return empty_label

    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else empty_label

    text = str(value).strip()
    if not text or text == "[]":
        return empty_label

    return text


def _friendly_validation_warning_text(warning: str) -> str:
    text = str(warning).strip()

    replacements = {
        "Quasi-identifier columns still present after anonymization:": "Indirect identifiers still present after anonymization:",
        "Exact quasi-identifiers remain (higher re-identification risk)": "Indirect identifiers still present after anonymization:",
        "Generalized quasi-identifiers still present:": "Indirect identifiers were generalized, but privacy risk still remains:",
        "Residual technical identifiers not dropped:": "Internal system identifiers still present after anonymization:",
        "Residual technical identifiers not safely protected for current context:": "Internal system identifiers still present after anonymization:",
        "Contextual technical-identifier validation used strict fallback because required metadata was missing for:": "Context needed for technical-identifier validation was missing:",
        "Residual direct identifiers not dropped or pseudonymized:": "Direct identifiers still present after anonymization:",
        "High-uniqueness columns remain after anonymization:": "Highly unique fields still remain after anonymization:",
    }

    for old, new in replacements.items():
        if text.startswith(old):
            return text.replace(old, new, 1)

    return text




def _get_validation_metric_state(label: str, value: int) -> tuple[str, str]:
    if label in {"Direct identifiers", "Technical fields"}:
        if value > 0:
            return "validation-metric-danger", "Critical issue"
        return "validation-metric-safe", "None remaining"

    if label in {"High-uniqueness fields", "Indirect identifiers still exact", "Failed protection attempts"}:
        if value > 0:
            return "validation-metric-review", "Needs review"
        return "validation-metric-safe", "None remaining"

    if label == "Indirect identifiers generalized":
        if value > 0:
            return "validation-metric-safe", "Reduced but still present"
        return "validation-metric-safe", "None remaining"

    if label == "Indirect identifiers":
        if value > 0:
            return "validation-metric-review", "Needs review"
        return "validation-metric-safe", "None remaining"

    return "validation-metric-safe", ""


def _render_validation_metric_row(metric_items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(metric_items), gap="large")

    for column, (label, value_text) in zip(columns, metric_items):
        value_int = int(value_text) if str(value_text).isdigit() else 0
        card_class, meta_text = _get_validation_metric_state(label, value_int)

        with column:
            st.markdown(
                f"""
                <div class="validation-metric-card {card_class}">
                    <div class="validation-metric-label">{html.escape(label)}</div>
                    <div class="validation-metric-value">{html.escape(value_text)}</div>
                    <div class="validation-metric-meta">{html.escape(meta_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _render_inspection_intro_card() -> None:
    st.markdown(
        """
        <div class="inspection-intro-card">
            <div class="inspection-intro-eyebrow">Post-anonymization review</div>
            <div class="inspection-intro-row">
                <div class="inspection-intro-icon">⚠</div>
                <div class="inspection-intro-title">Inspect residual re-identification risk</div>
            </div>
            <div class="inspection-intro-text">
                Your selected protection methods have been applied. This step helps you assess whether any
                remaining patterns in the data could still allow individuals to be singled out or re-identified.
            </div>
            <div class="inspection-intro-text" style="margin-top:0.65rem;">
                The suggestions below are optional and focus on further reducing risk only when stricter
                sharing requirements apply.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _get_validation_verdict_icon(validation_summary: dict[str, Any]) -> tuple[str, str]:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    high_uniqueness_count = int(
        validation_summary.get("high_uniqueness_columns_after_anonymization_count", 0) or 0
    )
    indirect_identifier_count = int(
        validation_summary.get("quasi_identifier_columns_remaining_count", 0) or 0
    )

    if residual_direct_count > 0 or residual_technical_count > 0:
        return "✕", "risk-hero-high"

    if high_uniqueness_count > 0 or indirect_identifier_count > 0:
        return "⚠", "risk-hero-moderate"

    return "✓", "risk-hero-low"



def _render_validation_verdict_card(validation_summary: dict[str, Any]) -> None:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    high_uniqueness_count = int(
        validation_summary.get("high_uniqueness_columns_after_anonymization_count", 0) or 0
    )
    indirect_identifier_count = int(
        validation_summary.get("quasi_identifier_columns_remaining_count", 0) or 0
    )

    if residual_direct_count > 0 or residual_technical_count > 0:
        verdict = "High residual risk"
        hero_class = "risk-hero-high"
        helper_text = (
            "Some direct or internal system identifiers still remain in the protected dataset. "
            "This should be resolved before sharing."
        )
        summary_html = (
            "<strong>This is not just an optional improvement.</strong> "
            "Identifiers that directly expose or link records are still present."
        )
    elif high_uniqueness_count > 0 or indirect_identifier_count > 0:
        verdict = "Residual risk remains"
        hero_class = "risk-hero-moderate"
        helper_text = (
            "The selected protection methods correctly removed direct and internal system identifiers, "
            "but some re-identification risk still remains in the protected dataset."
        )
        summary_html = (
            "<strong>This does not mean your protection choices were wrong.</strong> "
            "It means some inherent re-identification risk still remains after the recommended protection was applied."
        )
    else:
        verdict = "Low residual risk"
        hero_class = "risk-hero-low"
        helper_text = (
            "No residual direct identifiers, internal system identifiers, or additional warning conditions were detected."
        )
        summary_html = (
            "<strong>Current result:</strong> the protected dataset passed the residual-risk check at the current rule level."
        )

    verdict_icon, _ = _get_validation_verdict_icon(validation_summary)

    st.markdown(
        f"""
        <div class="risk-hero-card {hero_class}">
            <div class="risk-hero-eyebrow">Residual risk status</div>
            <div class="validation-verdict-row">
                <div class="validation-verdict-icon">{html.escape(verdict_icon)}</div>
                <div class="risk-hero-value" style="margin:0;">{html.escape(verdict)}</div>
            </div>
            <div class="risk-hero-text">{html.escape(helper_text)}</div>
            <div class="risk-hero-text" style="margin-top:0.55rem;">
                {summary_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_validation_issue_cards(validation_summary: dict[str, Any]) -> None:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    high_uniqueness_columns = validation_summary.get("high_uniqueness_columns_after_anonymization", []) or []

    exact_indirect_columns = validation_summary.get("quasi_identifier_exact_remaining_columns", []) or []
    generalized_indirect_columns = validation_summary.get("quasi_identifier_generalized_remaining_columns", []) or []
    failed_protection_columns = validation_summary.get("failed_protection_attempt_columns", []) or []

    items: list[dict[str, str]] = []

    if residual_direct_count > 0 or residual_technical_count > 0:
        items.append(
            {
                "title": "Identifiers still remain",
                "body_prefix": "Direct or internal system identifiers are still present:",
                "field_list": (
                    f"direct identifiers = {residual_direct_count}, "
                    f"internal system identifiers = {residual_technical_count}"
                ),
                "action": "Go back to Final Protection Decisions and remove or pseudonymize the remaining identifying fields before sharing.",
                "tone_class": "validation-issue-danger",
                "priority_label": "Required before sharing",
            }
        )

    if failed_protection_columns:
        items.append(
            {
                "title": "Requested protection could not be applied",
                "body_prefix": "Some requested protections could not be fully applied for these fields:",
                "field_list": _format_column_list_for_ui(failed_protection_columns),
                "action": "Go back to Final Protection Decisions, review these fields, choose a different protection method if needed, and rerun anonymization.",
                "tone_class": "validation-issue-warn",
                "priority_label": "Needs review",
            }
        )

    if high_uniqueness_columns:
        items.append(
            {
                "title": "Highly unique fields remain",
                "body_prefix": "These fields remain highly unique. This can still allow re-identification in some cases:",
                "field_list": _format_column_list_for_ui(high_uniqueness_columns),
                "action": "Review these fields before broader sharing. Strengthen protection only if your sharing context needs it.",
                "tone_class": "validation-issue-warn",
                "priority_label": "Optional improvement",
            }
        )

    if exact_indirect_columns:
        items.append(
            {
                "title": "Exact indirect identifiers remain",
                "body_prefix": "These fields still remain in exact form and can contribute more strongly to linkage risk:",
                "field_list": _format_column_list_for_ui(exact_indirect_columns),
                "action": "Go back to Final Protection Decisions, increase the generalization level for these fields, and rerun anonymization before sharing externally.",
                "tone_class": "validation-issue-warn",
                "priority_label": "Needs attention",
            }
        )

    if generalized_indirect_columns:
        items.append(
            {
                "title": "Generalized indirect identifiers remain",
                "body_prefix": "These fields are already generalized, but they still remain in the dataset:",
                "field_list": _format_column_list_for_ui(generalized_indirect_columns),
                "action": "Optional improvement: increase the generalization level further only if stricter disclosure requirements apply.",
                "tone_class": "validation-issue-warn",
                "priority_label": "Optional improvement",
            }
        )

    if not items:
        return

    _render_subsection_heading("Residual risk indicators")
    st.caption(f"{len(items)} residual risk indicators are shown below.")

    for item in items:
        st.markdown(
            f"""
            <div class="validation-issue-card {html.escape(item['tone_class'])}">
                <div class="validation-issue-priority">{html.escape(item['priority_label'])}</div>
                <div class="validation-issue-title">{html.escape(item['title'])}</div>
                <div class="validation-issue-text">
                    <strong>{html.escape(item['body_prefix'])}</strong>
                    {html.escape(item['field_list'])}
                </div>
                <div class="validation-issue-text" style="margin-top:0.45rem;">
                    <strong>What you can do:</strong> {html.escape(item['action'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _format_validation_detail_columns(value: Any) -> str:
    if value is None:
        return "None"

    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned_items.append(_friendly_validation_warning_text(text))
        return " | ".join(cleaned_items) if cleaned_items else "None"

    text = str(value).strip()
    if not text or text == "[]":
        return "None"

    return _friendly_validation_warning_text(text)


def _render_validation_warning_note(validation_summary: dict[str, Any]) -> None:
    warnings = validation_summary.get("validation_warnings", []) or []
    if not warnings:
        return

    normalized_warnings = [_friendly_validation_warning_text(warning) for warning in warnings]
    joined = " | ".join(normalized_warnings)

    st.markdown(
        f"""
        <div class="decision-subtle-note">
            Validator checks triggered: {html.escape(joined)}
        </div>
        """,
        unsafe_allow_html=True,
    )




def _friendly_detection_reason(classification: str, raw_reason: str) -> tuple[str, str | None]:
    reason = str(raw_reason or "").strip().lower()
    method_label: str | None = None

    if "presidio" in reason:
        method_label = "Detected using: text scan"
    elif "value-pattern" in reason:
        method_label = "Detected using: value pattern"
    elif "name-heuristic" in reason or "keyword" in reason:
        method_label = "Detected using: column pattern"

    if classification == "direct_identifier":
        if "presidio" in reason or "email" in reason or "phone" in reason:
            return "Contains personal data in text values", method_label
        return "Clearly identifies a person", method_label
    if classification == "quasi_identifier":
        return "May identify a person when combined with other fields", method_label
    if classification == "sensitive_attribute":
        return "Contains sensitive personal information", method_label
    if classification == "technical_identifier":
        return "Used to track records inside systems", method_label
    if classification == "non_sensitive":
        return "No sensitive signal detected in the current scan", method_label
    return "Flagged by the current detection scan", method_label


def _friendly_review_reason(classification: str, raw_reason: str) -> str:
    reason = str(raw_reason or "").strip().lower()
    if "presidio" in reason or "text scan" in reason:
        return "Sensitive text patterns were detected in this column."
    if classification == "direct_identifier":
        return "This column can identify a person directly."
    if classification == "quasi_identifier":
        return "This column may identify a person when combined with other fields."
    if classification == "sensitive_attribute":
        return "This column contains sensitive personal information."
    if classification == "technical_identifier":
        return "This column behaves like an internal system field."
    if classification == "non_sensitive":
        return "No sensitive signal was detected in this column."
    return "This column was included in the current review."


def _is_changed_from_recommendation(recommended_action: str, final_action: str) -> bool:
    return _normalize_action_key(recommended_action) != _normalize_action_key(final_action)


def _protection_rank(action: str) -> int:
    action = _normalize_action_key(action)
    ranking = {
        "drop": 8,
        "pseudonymize": 7,
        "generalize_strict": 6,
        "generalize_medium": 5,
        "generalize_light": 4,
        "add_noise": 3,
        "mask": 2,
        "keep": 1,
    }
    return ranking.get(action, 0)


def _is_higher_risk_choice(
    classification: str,
    final_action: str,
    recommended_action: str,
) -> bool:
    final_rank = _protection_rank(final_action)
    recommended_rank = _protection_rank(recommended_action)
    return final_rank < recommended_rank


def _protection_strength_rank(action: str) -> int:
    normalized = _normalize_action_key(action)

    rank_map = {
        "drop": 8,
        "remove column": 8,
        "drop column": 8,
        "remove_column": 8,
        "pseudonymize": 7,
        "replace with pseudonym": 7,
        "replace_with_pseudonym": 7,
        "generalize_strict": 6,
        "generalize strongly": 6,
        "generalize_medium": 5,
        "generalize more": 5,
        "generalize_light": 4,
        "add_noise": 3,
        "add numeric noise": 3,
        "mask": 2,
        "mask for display": 2,
        "display masking": 2,
        "mask_display_only": 2,
        "keep": 1,
        "leave unchanged": 1,
        "leave_unchanged": 1,
        "none": 1,
    }

    return rank_map.get(normalized, 0)


def _build_decision_overview(reviewed_df: pd.DataFrame) -> list[dict[str, str]]:
    total_columns = len(reviewed_df)

    recommended_actions = (
        reviewed_df.get("recommended_action", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .map(_normalize_action_key)
    )
    final_actions = (
        reviewed_df.get("final_action", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .map(_normalize_action_key)
    )

    recommendations_followed = int((recommended_actions == final_actions).sum())
    actions_changed = int(total_columns - recommendations_followed)

    higher_risk_overrides = 0
    for recommended_action, final_action in zip(recommended_actions, final_actions):
        recommended_rank = _protection_strength_rank(recommended_action)
        final_rank = _protection_strength_rank(final_action)

        if final_rank < recommended_rank:
            higher_risk_overrides += 1

    higher_risk_tone = "decision-summary-safe" if higher_risk_overrides == 0 else "decision-summary-warn"

    return [
        {
            "label": "Columns under review",
            "value": str(total_columns),
            "meta": "Fields currently included in the final review step.",
            "tone_class": "",
        },
        {
            "label": "Recommendations followed",
            "value": str(recommendations_followed),
            "meta": "Selections that still match the suggested protection for this use case.",
            "tone_class": "",
        },
        {
            "label": "Actions changed by you",
            "value": str(actions_changed),
            "meta": "Columns where you overrode the suggested protection.",
            "tone_class": "",
        },
        {
            "label": "Higher-risk overrides",
            "value": str(higher_risk_overrides),
            "meta": (
                "Selections that are less protective than the suggested policy. "
                "Residual dataset risk is checked after protection."
            ),
            "tone_class": higher_risk_tone,
        },
    ]


def _build_policy_alignment_preview(reviewed_df: pd.DataFrame) -> tuple[dict[str, int], str]:
    policy_risk_items = _build_policy_risk_items(reviewed_df)

    changed_columns: set[str] = set()
    higher_risk_columns: set[str] = set()
    display_only_columns: set[str] = set()
    has_high_risk = False
    has_moderate_risk = False

    for item in policy_risk_items:
        columns = {str(column) for column in (item.get("columns", []) or [])}
        kind = str(item.get("kind", "")).strip()
        risk_level = str(item.get("risk_level", "")).strip().lower()

        if kind in {"weaker_than_recommended", "changed_but_not_weaker"}:
            changed_columns.update(columns)

        if kind == "weaker_than_recommended":
            higher_risk_columns.update(columns)

        if kind == "display_only_selected":
            display_only_columns.update(columns)

        if risk_level == "high":
            has_high_risk = True
        elif risk_level == "moderate":
            has_moderate_risk = True

    preview_counts = {
        "Actions changed by user": len(changed_columns),
        "Higher-risk overrides": len(higher_risk_columns),
        "Display-only protections selected": len(display_only_columns),
    }

    if has_high_risk:
        alignment_status = "High"
    elif has_moderate_risk:
        alignment_status = "Moderate"
    else:
        alignment_status = "Low"

    return preview_counts, alignment_status


def _build_policy_risk_items(reviewed_df: pd.DataFrame) -> list[dict[str, Any]]:
    recommended_actions = (
        reviewed_df.get("recommended_action", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .map(_normalize_action_key)
    )
    final_actions = (
        reviewed_df.get("final_action", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .map(_normalize_action_key)
    )
    column_names = reviewed_df.get("column_name", pd.Series(dtype="object")).fillna("").astype(str)

    weaker_columns: list[str] = []
    changed_but_not_weaker_columns: list[str] = []
    display_only_columns: list[str] = []

    for column_name, recommended_action, final_action in zip(
        column_names,
        recommended_actions,
        final_actions,
    ):
        recommended_rank = _protection_strength_rank(recommended_action)
        final_rank = _protection_strength_rank(final_action)
        has_changed = final_action != recommended_action

        if final_action == "mask":
            display_only_columns.append(str(column_name))

        if not has_changed:
            continue

        if final_rank < recommended_rank:
            weaker_columns.append(str(column_name))
        else:
            changed_but_not_weaker_columns.append(str(column_name))

    items: list[dict[str, Any]] = []
    if weaker_columns:
        items.append(
            {
                "kind": "weaker_than_recommended",
                "columns": sorted(weaker_columns),
                "risk_category": "policy_alignment",
                "risk_level": "high",
            }
        )
    if display_only_columns:
        items.append(
            {
                "kind": "display_only_selected",
                "columns": sorted(display_only_columns),
                "risk_category": "policy_alignment",
                "risk_level": "moderate",
            }
        )
    if changed_but_not_weaker_columns:
        items.append(
            {
                "kind": "changed_but_not_weaker",
                "columns": sorted(changed_but_not_weaker_columns),
                "risk_category": "policy_alignment",
                "risk_level": "low",
            }
        )
    return items

def _build_detection_entries(
    classification_df: pd.DataFrame, classification: str, limit: int = 4
) -> list[tuple[str, str, str | None]]:
    filtered_df = classification_df.loc[
        classification_df["classification"] == classification,
        ["column_name", "reason"],
    ].copy(deep=True)
    if filtered_df.empty:
        return []

    filtered_df["reason"] = filtered_df["reason"].fillna("Flagged by the detection scan.")
    entries = []
    for row in filtered_df.itertuples(index=False):
        primary_reason, method_label = _friendly_detection_reason(
            classification,
            str(row.reason),
        )
        entries.append((str(row.column_name), primary_reason, method_label))
    return entries[:limit]


def _render_detection_summary_cards(classification_df: pd.DataFrame) -> None:
    """Renders combined detection cards — summary count + column details in one card per category.
    Dynamically handles any number of categories, max 3 columns per row with wrapping.
    Replaces the previous two-function approach (_render_detection_summary_cards +
    _render_detection_category_column_cards) to keep summary and detail aligned.
    """
    summary_items = [
        ("direct_identifier",    "Direct identifiers",   "Highest re-identification risk"),
        ("sensitive_attribute",  "Sensitive information","Usually needs stronger protection"),
        ("quasi_identifier",     "Indirect identifiers", "Can identify people when combined"),
        ("technical_identifier", "System identifiers",   "Internal fields, usually dropped"),
    ]

    # Only include categories that actually have columns detected
    active_items = [
        (cls, label, note)
        for cls, label, note in summary_items
        if int((classification_df["classification"] == cls).sum()) > 0
    ]

    if not active_items:
        st.caption("No sensitive columns detected.")
        return

    num_cols = min(3, len(active_items))
    columns = st.columns(num_cols, gap="large")

    for i, (classification, label, note) in enumerate(active_items):
        col = columns[i % num_cols]
        count = int((classification_df["classification"] == classification).sum())
        entries = _build_detection_entries(classification_df, classification)
        remaining_count = max(count - len(entries), 0)

        if not entries:
            body_markup = '<div class="detail-reason">No columns detected in this category.</div>'
        else:
            entry_markup = []
            for column_name, reason, method_label in entries:
                method_markup = (
                    f'<div class="detail-method">{html.escape(method_label)}</div>'
                    if method_label else ""
                )
                entry_markup.append(
                    f'<div class="detail-entry">'
                    f'<div class="detail-name">{html.escape(column_name)}</div>'
                    f'<div class="detail-reason">{html.escape(reason)}</div>'
                    f'{method_markup}'
                    f'</div>'
                )
            if remaining_count > 0:
                entry_markup.append(f'<div class="detail-method">+{remaining_count} more</div>')
            body_markup = "".join(entry_markup)

        with col:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="metric-label category-{classification}">{html.escape(label)}</div>
                    <div class="metric-value">{count}</div>
                    <div class="metric-note">{note}</div>
                </div>
                <div class="detail-card" style="margin-top:0;">{body_markup}</div>
                """,
                unsafe_allow_html=True,
            )


def _render_detection_category_column_cards(classification_df: pd.DataFrame) -> None:
    """Intentionally empty — merged into _render_detection_summary_cards."""
    pass


def _render_detection_category_help() -> None:
    _render_subsection_heading("This is how Data is Categorized")
    _render_section_spacer("sm")
    explanation_items = [
        (
            "direct_identifier",
            "email address, phone number, employee ID",
            "These fields can directly point to a specific person.",
            "They usually need the strongest action because they create the clearest identification risk.",
        ),
        (
            "quasi_identifier",
            "age band, city, region, education level",
            "These fields may not identify someone alone, but combinations can narrow a person down.",
            "They often need generalization so the dataset stays useful without being too specific.",
        ),
        (
            "sensitive_attribute",
            "salary, health status, protected personal details",
            "These fields contain information people would usually expect to be handled more carefully.",
            "They matter because even without a direct identifier, the content itself can be sensitive.",
        ),
        (
            "technical_identifier",
            "UUID, record key, internal row ID",
            "These are system-generated values used to manage records behind the scenes.",
            "They are often low business value for analysis and can make records easier to trace.",
        ),
        (
            "non_sensitive",
            "product category, generic operational labels",
            "These fields were not flagged by the current scan.",
            "They usually need less intervention, but they should still be reviewed in context.",
        ),
    ]
    for classification, examples, meaning, why_it_matters in explanation_items:
        st.markdown(
            f"""
            <div class="explanation-card">
                <div class="explanation-shell">
                    <div class="explanation-shell-left">
                        <div class="category-badge category-{classification}">{_simplified_category_label(classification)}</div>
                        <div class="eyebrow-title">Example data</div>
                        <div class="example-block">{examples}</div>
                    </div>
                    <div class="explanation-shell-right">
                        <div class="eyebrow-title">Category characteristics</div>
                        <div class="explanation-text">{meaning}</div>
                        <div class="explanation-text">{why_it_matters}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_overview_metrics(metadata: Any, dataframe: pd.DataFrame) -> list[tuple[str, str]]:
    missing_cells = int(dataframe.isna().sum().sum())
    return [
        ("Rows", f"{metadata.rows:,}"),
        ("Columns", f"{metadata.columns:,}"),
        ("File type", str(metadata.file_type).upper()),
        ("Missing values", f"{missing_cells:,}"),
    ]


def _build_detection_summary(
    classification_df: pd.DataFrame,
    presidio_summary_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    direct_identifier_count = int(
        (classification_df["classification"] == "direct_identifier").sum()
    )
    sensitive_attribute_count = int(
        (classification_df["classification"] == "sensitive_attribute").sum()
    )
    quasi_identifier_count = int(
        (classification_df["classification"] == "quasi_identifier").sum()
    )
    presidio_hits_total = 0
    if not presidio_summary_df.empty and "presidio_hit_count" in presidio_summary_df.columns:
        presidio_hits_total = int(
            pd.to_numeric(presidio_summary_df["presidio_hit_count"], errors="coerce").fillna(0).sum()
        )

    return [
        ("Direct identifiers", f"{direct_identifier_count:,}"),
        ("Sensitive fields", f"{sensitive_attribute_count:,}"),
        ("Indirect identifiers", f"{quasi_identifier_count:,}"),
        ("Text detection hits", f"{presidio_hits_total:,}"),
    ]


def _render_metric_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def _friendly_display_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_df = dataframe.copy(deep=True)
    if "classification" in display_df.columns:
        display_df["classification"] = display_df["classification"].map(_friendly_classification)
    if "recommended_action" in display_df.columns:
        display_df["recommended_action"] = display_df["recommended_action"].map(_friendly_action)
    if "final_action" in display_df.columns:
        display_df["final_action"] = display_df["final_action"].map(_friendly_action)
    if "action_requested" in display_df.columns:
        display_df["action_requested"] = display_df["action_requested"].map(_friendly_action)
    if "action_applied" in display_df.columns:
        display_df["action_applied"] = display_df["action_applied"].map(_friendly_action)
    return _prepare_dataframe_for_display(display_df)


def _render_workflow_progress(current_stage: int) -> None:
    step_markup: list[str] = []
    for index, step in enumerate(_WORKFLOW_STEPS, start=1):
        state = "pending"
        if index < current_stage:
            state = "complete"
        elif index == current_stage:
            state = "active"
        step_markup.append(f'<div class="workflow-pill {state}">{index}. {step}</div>')
    st.markdown(
        f'<div class="workflow-row">{"".join(step_markup)}</div>',
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> str:
    return f'<span class="status-chip {status.lower()}">{status}</span>'


def _render_sidebar_status(
    statuses: list[tuple[str, str]],
    placeholder: Any | None = None,
) -> None:
    sidebar_target = placeholder.container() if placeholder is not None else st.sidebar
    with sidebar_target:
        st.markdown("### Workflow status")
        st.caption("A quick summary of where the run stands.")

        for label, status in statuses:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;gap:0.55rem;margin-bottom:0.72rem;">
                    <div style="color:var(--text);font-size:0.9rem;font-weight:500;line-height:1.35;">
                        {html.escape(label)}
                    </div>
                    {_status_badge(status)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_info_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-panel">
            <strong>{title}</strong><br>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_demo_dataset_download() -> None:
    st.markdown(
        """
        <div style="height:1.65rem;"></div>
        <div class="step1-mini-heading step1-mini-heading-teal">
            Need test data?
        </div>
        """,
        unsafe_allow_html=True,
    )

    if DEMO_DATASET_PATH.exists():
        demo_dataset_b64 = base64.b64encode(DEMO_DATASET_PATH.read_bytes()).decode("utf-8")
        download_href = (
            f"data:text/csv;base64,{demo_dataset_b64}"
        )

        st.markdown(
            f"""
            <div class="demo-download-shell">
                <a
                    class="demo-download-pill"
                    href="{download_href}"
                    download="{html.escape(DEMO_DATASET_FILENAME)}"
                >
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M12 4v10" stroke="currentColor" stroke-width="2.4"
                            stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M7.5 9.5L12 14l4.5-4.5" stroke="currentColor" stroke-width="2.4"
                            stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M5 18h14" stroke="currentColor" stroke-width="2.4"
                            stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Download Demo Dataset
                </a>
                <div class="demo-download-helper">
                    Download Test Dataset and upload it in Dataset file section.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="demo-download-shell">
                <div class="demo-download-helper">
                    Demo dataset file is not available in this deployment.
                    Expected path: demo_data/privacyguard_ai_demo_synthetic_sensitive_dataset.csv
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="synthetic-demo-card">
            <div class="synthetic-demo-title">Synthetic demo dataset</div>
            <div class="synthetic-demo-text">
                Download a fully synthetic CSV created for this demo, then upload it in the
                Dataset file field above. The sample contains fake identifiers, quasi-identifiers,
                sensitive fields, and free-text notes, so you can test detection, planning,
                anonymization, validation, and export end to end.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_privacy_x_factor_lab(
    dataframe: pd.DataFrame,
    classification_df: pd.DataFrame,
    source_signature: str,
) -> None:
    with st.expander("Privacy X-Factor Lab | Local DP + private model training", expanded=False):
        local_tab, model_tab = st.tabs(
            ["Local differential privacy", "Model privacy: GRL + DP-SGD"]
        )

        low_cardinality_columns = [
            str(column)
            for column in dataframe.columns
            if 2 <= dataframe[column].nunique(dropna=False) <= 8
        ]
        with local_tab:
            st.markdown("#### Randomized response")
            st.caption(
                "Randomizes selected low-cardinality fields on-device. Each selected field receives "
                "its own local privacy budget; unselected fields remain unchanged. The category domain "
                "is assumed to be fixed and public."
            )
            if not low_cardinality_columns:
                st.info("No columns with 2–8 categories are available for randomized response.")
            else:
                with st.form("x_factor_rr_form"):
                    selected_columns = st.multiselect(
                        "Sensitive categorical fields",
                        options=low_cardinality_columns,
                        key="x_factor_rr_columns",
                    )
                    epsilon = st.slider(
                        "Local ε per selected field",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        help="Smaller ε means more randomization and less disclosure per field.",
                        key="x_factor_rr_epsilon",
                    )
                    if selected_columns:
                        truth_probabilities = [
                            randomized_response_truth_probability(
                                epsilon,
                                int(dataframe[column].nunique(dropna=False)),
                            )
                            for column in selected_columns
                        ]
                        st.caption(
                            f"Composed ε upper bound across these fields: "
                            f"{epsilon * len(selected_columns):.2f}. "
                            f"Truth probability across the selected fields: "
                            f"{min(truth_probabilities):.1%}–{max(truth_probabilities):.1%}."
                        )
                    st.warning(
                        "This export only protects the selected fields. Repeated releases compose privacy loss; "
                        "review the total budget before sharing."
                    )
                    apply_clicked = st.form_submit_button(
                        "Apply randomized response",
                        type="primary",
                        disabled=not selected_columns,
                    )
                    if apply_clicked:
                        try:
                            protected_df, report = apply_randomized_response(
                                dataframe,
                                selected_columns,
                                epsilon,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["x_factor_rr_output"] = {
                                "source_signature": source_signature,
                                "columns": selected_columns,
                                "epsilon": epsilon,
                                "dataframe": protected_df,
                                "report": report,
                            }

                rr_output = st.session_state.get("x_factor_rr_output")
                if (
                    rr_output
                    and rr_output.get("source_signature") == source_signature
                    and rr_output.get("columns") == selected_columns
                    and rr_output.get("epsilon") == epsilon
                ):
                    st.success(
                        "Randomized response applied. "
                        f"Composed ε upper bound: "
                        f"{rr_output['report']['composed_epsilon_upper_bound']:.2f}."
                    )
                    st.download_button(
                        "Download locally randomized CSV",
                        data=rr_output["dataframe"].to_csv(index=False).encode("utf-8"),
                        file_name="privacyguard_ai_randomized_response.csv",
                        mime="text/csv",
                        key="x_factor_rr_download",
                    )

        binary_columns = [
            str(column)
            for column in dataframe.columns
            if dataframe[column].notna().all()
            and dataframe[column].nunique(dropna=True) == 2
        ]
        with model_tab:
            st.markdown("#### Adversarial representation with DP-SGD")
            st.caption(
                "Trains a small utility classifier and a sensitive-attribute adversary. "
                "Gradient reversal discourages sensitive information in the representation; "
                "per-example clipping and secure Gaussian noise protect model parameters."
            )
            if len(binary_columns) < 2:
                st.info(
                    "Model training requires two complete binary fields: one utility target "
                    "and one sensitive attribute."
                )
            else:
                with st.form("x_factor_model_form"):
                    target_column = st.selectbox(
                        "Utility target (binary)",
                        options=binary_columns,
                        index=0,
                        key="x_factor_model_target",
                    )
                    sensitive_options = [
                        column for column in binary_columns if column != target_column
                    ]
                    sensitive_column = st.selectbox(
                        "Sensitive attribute (binary)",
                        options=sensitive_options,
                        index=0,
                        key="x_factor_model_sensitive",
                    )
                    direct_identifier_columns = set(
                        classification_df.loc[
                            classification_df["classification"].isin(
                                ["direct_identifier", "technical_identifier"]
                            ),
                            "column_name",
                        ].astype(str)
                    )
                    feature_options = [
                        str(column)
                        for column in dataframe.columns
                        if column not in {target_column, sensitive_column}
                        and column not in direct_identifier_columns
                        and pd.api.types.is_numeric_dtype(dataframe[column])
                    ]
                    feature_columns = st.multiselect(
                        "Numeric model features",
                        options=feature_options,
                        key=f"x_factor_model_features_{target_column}_{sensitive_column}",
                        help="Features are clipped per row before training; select only fields approved for this experiment.",
                    )
                    model_settings = st.columns(3)
                    with model_settings[0]:
                        target_epsilon = st.slider(
                            "Target model ε",
                            min_value=0.5,
                            max_value=10.0,
                            value=3.0,
                            step=0.5,
                            key="x_factor_model_epsilon",
                        )
                    with model_settings[1]:
                        delta = st.number_input(
                            "δ",
                            min_value=1e-7,
                            max_value=1e-3,
                            value=1e-5,
                            format="%.1e",
                            key="x_factor_model_delta",
                        )
                    with model_settings[2]:
                        epochs = st.slider(
                            "Training epochs",
                            min_value=1,
                            max_value=10,
                            value=5,
                            key="x_factor_model_epochs",
                        )
                    st.warning(
                        "The ε,δ guarantee applies to model parameters on the 80% training split, "
                        "conditional on the selected schema. Validation scores are not differentially private."
                    )
                    train_clicked = st.form_submit_button(
                        "Train GRL + DP-SGD model",
                        type="primary",
                        disabled=not feature_columns,
                    )
                    if train_clicked:
                        try:
                            model_report = train_private_adversarial_model(
                                dataframe,
                                target_column=target_column,
                                sensitive_column=sensitive_column,
                                feature_columns=feature_columns,
                                target_epsilon=target_epsilon,
                                delta=float(delta),
                                epochs=epochs,
                                random_state=None,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["x_factor_model_report"] = {
                                "source_signature": source_signature,
                                "target_column": target_column,
                                "sensitive_column": sensitive_column,
                                "feature_columns": feature_columns,
                                "target_epsilon": target_epsilon,
                                "delta": float(delta),
                                "epochs": epochs,
                                "report": model_report,
                            }

                model_output = st.session_state.get("x_factor_model_report")
                if (
                    model_output
                    and model_output.get("source_signature") == source_signature
                    and model_output.get("target_column") == target_column
                    and model_output.get("sensitive_column") == sensitive_column
                    and model_output.get("feature_columns") == feature_columns
                    and model_output.get("target_epsilon") == target_epsilon
                    and model_output.get("delta") == float(delta)
                    and model_output.get("epochs") == epochs
                ):
                    report = model_output["report"]
                    result_columns = st.columns(3)
                    result_columns[0].metric(
                        "Accounted model ε",
                        f"{report['achieved_epsilon']:.2f}",
                        help=f"δ = {report['delta']:.1e}; Poisson-sampled Gaussian RDP accounting.",
                    )
                    result_columns[1].metric(
                        "Utility accuracy",
                        f"{report['utility_accuracy']:.1%}",
                        help="Measured on a held-out split; this score is not differentially private.",
                    )
                    result_columns[2].metric(
                        "Sensitive probe accuracy",
                        f"{report['sensitive_adversary_accuracy']:.1%}",
                        help="Held-out accuracy of the adversary head; compare with the baseline below.",
                    )
                    st.caption(
                        f"Sensitive majority baseline: {report['sensitive_majority_baseline']:.1%}. "
                        f"Training rows: {report['train_rows']}; validation rows: "
                        f"{report['validation_rows']}; steps: {report['training_steps']}."
                    )
                    st.info(
                        "This run reports an accounted DP-SGD budget and adversary probe result. "
                        "It does not run a membership-inference attack or claim an MIA AUC of 0.50."
                    )
                    st.download_button(
                        "Download model privacy report (JSON)",
                        data=json.dumps(report, indent=2).encode("utf-8"),
                        file_name="privacyguard_ai_dp_sgd_report.json",
                        mime="application/json",
                        key="x_factor_model_report_download",
                    )


def _build_outputs_readiness_summary(
    planning_context: dict[str, Any],
    validation_summary: dict[str, Any],
    reviewed_df: pd.DataFrame,
) -> dict[str, str]:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    high_uniqueness_count = int(
        validation_summary.get("high_uniqueness_columns_after_anonymization_count", 0) or 0
    )
    exact_indirect_count = int(
        validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0
    )
    generalized_indirect_count = int(
        validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0
    )
    failed_attempt_count = int(
        validation_summary.get("failed_protection_attempt_count", 0) or 0
    )
    k_anonymity = validation_summary.get("k_anonymity") or {}
    records_below_k = int(k_anonymity.get("records_below_k") or 0)

    intended_use = str(planning_context.get("intended_use", "")).strip().lower()

    if residual_direct_count > 0 or residual_technical_count > 0:
        return {
            "hero_class": "outputs-hero-danger",
            "title": "Not ready for external sharing" if intended_use == "external sharing" else "Dataset not ready for release",
            "eyebrow": "Protection outcome",
            "text": (
                "Direct or internal system identifiers still remain in the protected dataset. "
                "This output should not be shared until those fields are removed or more strongly protected."
            ),
            "recommended_use": "Internal review only",
        }

    if failed_attempt_count > 0:
        return {
            "hero_class": "outputs-hero-review",
            "title": "Protection applied with unresolved execution issues",
            "eyebrow": "Protection outcome",
            "text": (
                "Some requested protection methods could not be applied as planned. "
                "Review those fields before broader sharing because the final dataset may not match the intended protection level."
            ),
            "recommended_use": "Internal review or controlled access only until unresolved fields are reviewed",
        }

    if exact_indirect_count > 0 or generalized_indirect_count > 0 or records_below_k > 0:
        if intended_use == "external sharing":
            return {
                "hero_class": "outputs-hero-review",
                "title": "Not ready for external sharing",
                "eyebrow": "Protection outcome",
                "text": (
                    "Direct identifiers were removed, but residual re-identification risk remains. "
                    + (
                        f"{records_below_k} records fall in groups smaller than k={k_anonymity.get('k', 5)}. "
                        if records_below_k > 0
                        else "Indirect identifiers or highly unique fields are still present and may allow records "
                        "to be singled out or linked."
                    )
                ),
                "recommended_use": "Internal review or controlled access only until additional protection is applied",
            }
        return {
            "hero_class": "outputs-hero-review",
            "title": "Ready for the selected use case with residual risk",
            "eyebrow": "Protection outcome",
            "text": (
                (
                    f"{records_below_k} records fall in groups smaller than k={k_anonymity.get('k', 5)}. "
                    if records_below_k > 0
                    else "The main protection steps were applied, but some indirect identifiers or highly unique fields "
                    "still remain. "
                )
                + "This may be acceptable for internal use, but stricter sharing scenarios require further protection "
                "because some fields can still support re-identification."
            ),
                "recommended_use": "Selected use case with caution",
            }

    if high_uniqueness_count > 0:
        return {
            "hero_class": "outputs-hero-safe",
            "title": "Protected dataset ready with review warnings",
            "eyebrow": "Protection outcome",
            "text": (
                "No residual identifier or quasi-identifier warnings were flagged, but some highly unique fields "
                "still remain. Review them before broader disclosure."
            ),
            "recommended_use": "Selected use case; review high-uniqueness fields before broader disclosure",
        }

    return {
        "hero_class": "outputs-hero-safe",
        "title": "Protected dataset ready",
        "eyebrow": "Protection outcome",
        "text": (
            "No residual direct identifiers or internal system identifiers were flagged at the current rule level. "
            "No additional residual-risk indicators were detected."
        ),
        "recommended_use": "Selected use case",
    }



def _build_outputs_signal_items(
    planning_context: dict[str, Any],
    validation_summary: dict[str, Any],
    reviewed_df: pd.DataFrame,
    transformation_log_df: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    high_uniqueness_count = int(
        validation_summary.get("high_uniqueness_columns_after_anonymization_count", 0) or 0
    )
    exact_indirect_count = int(
        validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0
    )
    generalized_indirect_count = int(
        validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0
    )
    failed_attempt_count = int(
        validation_summary.get("failed_protection_attempt_count", 0) or 0
    )
    k_anonymity = validation_summary.get("k_anonymity") or {}
    records_below_k = int(k_anonymity.get("records_below_k") or 0)

    if residual_direct_count > 0 or residual_technical_count > 0:
        residual_risk = "High"
        residual_risk_meta = "Identifiers still remain and should be resolved before sharing."
    elif failed_attempt_count > 0:
        residual_risk = "Moderate"
        residual_risk_meta = (
            "Some requested protection methods could not be applied, so review is needed before broader sharing."
        )
    elif high_uniqueness_count > 0 or exact_indirect_count > 0 or records_below_k > 0:
        residual_risk = "Moderate"
        residual_risk_meta = (
            f"{records_below_k} records fall in groups smaller than k={k_anonymity.get('k', 5)}."
            if records_below_k > 0
            else "Direct identifiers were handled, but some exact indirect identifiers or uniqueness risk still remain."
        )
    elif generalized_indirect_count > 0:
        residual_risk = "Moderate"
        residual_risk_meta = (
            "Direct identifiers were handled, and remaining indirect identifiers are generalized, but some linkage risk still remains."
        )
    else:
        residual_risk = "Low"
        residual_risk_meta = "No major residual issues were flagged at the current rule level."

    total_count = int(len(reviewed_df))

    if (
        transformation_log_df is not None
        and not transformation_log_df.empty
        and "action_applied" in transformation_log_df.columns
    ):
        protected_count = int((transformation_log_df["action_applied"].astype(str) != "keep").sum())
    else:
        protected_count = int((reviewed_df["final_action"].astype(str) != "keep").sum())

    utility_score = "High" if protected_count <= max(1, total_count // 2) else "Moderate"
    utility_meta = (
        "Most analytical detail remains in the final dataset."
        if utility_score == "High"
        else "Several fields were transformed or removed to reduce privacy risk."
    )

    intended_use = str(planning_context.get("intended_use", "")).strip().lower()
    use_case_label = intended_use.replace("_", " ").title() if intended_use else "Selected use case"

    return [
        {
            "label": "Residual risk",
            "value": residual_risk,
            "meta": residual_risk_meta,
        },
        {
            "label": "Data utility",
            "value": utility_score,
            "meta": utility_meta,
        },
        {
            "label": "Applied protection coverage",
            "value": f"{protected_count} / {total_count}",
            "meta": "Columns with actual applied transformations other than leave unchanged.",
        },
        {
            "label": "Selected policy context",
            "value": use_case_label,
            "meta": "The target use case selected earlier in the workflow.",
        },
    ]



def _build_outputs_breakdown_cards(
    reviewed_df: pd.DataFrame,
    transformation_log_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    source_df = transformation_log_df.copy() if not transformation_log_df.empty else reviewed_df.copy()

    action_column = "action_applied" if "action_applied" in source_df.columns else "final_action"
    if action_column not in source_df.columns:
        source_df[action_column] = "keep"

    grouped_columns: dict[str, list[str]] = {}
    for _, row in source_df.iterrows():
        action = _normalize_action_key(row.get(action_column, "keep"))
        column_name = str(row.get("column_name", ""))
        grouped_columns.setdefault(action, []).append(column_name)

    drop_columns = grouped_columns.get("drop", [])
    pseudonymize_columns = grouped_columns.get("pseudonymize", [])
    generalize_columns = (
        grouped_columns.get("generalize_light", [])
        + grouped_columns.get("generalize_medium", [])
        + grouped_columns.get("generalize_strict", [])
    )
    noise_columns = grouped_columns.get("add_noise", [])
    mask_columns = grouped_columns.get("mask", [])
    keep_columns = grouped_columns.get("keep", [])

    medium_columns = generalize_columns + noise_columns

    cards = [
        {
            "title": "Strong protection",
            "icon": "🔒",
            "tone_class": "outputs-breakdown-strong",
            "value": str(len(drop_columns) + len(pseudonymize_columns)),
            "meta": "Columns removed or pseudonymized to eliminate direct identification risk.",
            "items": (
                [f"Removed: {', '.join(drop_columns[:4])}"] if drop_columns else []
            ) + (
                [f"Pseudonymized: {', '.join(pseudonymize_columns[:4])}"] if pseudonymize_columns else []
            ),
        },
        {
            "title": "Medium protection",
            "icon": "⚠️",
            "tone_class": "outputs-breakdown-medium",
            "value": str(len(medium_columns)),
            "meta": "Columns generalized or perturbed to reduce precision while preserving utility.",
            "items": (
                [f"Generalized: {', '.join(generalize_columns[:4])}"] if generalize_columns else []
            ) + (
                [f"Noise added: {', '.join(noise_columns[:4])}"] if noise_columns else []
            ),
        },
        {
            "title": "Display-only protection",
            "icon": "🟠",
            "tone_class": "outputs-breakdown-medium",
            "value": str(len(mask_columns)),
            "meta": "Columns masked for display. This reduces visibility, but does not anonymize the dataset for sharing.",
            "items": [f"Masked: {', '.join(mask_columns[:4])}"] if mask_columns else [],
        },
        {
            "title": "Left unchanged",
            "icon": "❗",
            "tone_class": "outputs-breakdown-unchanged",
            "value": str(len(keep_columns)),
            "meta": "Columns retained in original form. These may still contribute to re-identification risk when combined.",
            "items": [f"Unchanged: {', '.join(keep_columns[:5])}"] if keep_columns else [],
        },
    ]

    cards = [
        card for card in cards
        if card["title"] != "Display-only protection" or mask_columns
    ]

    for card in cards:
        if not card["items"]:
            card["items"] = ["No columns in this group."]

    return cards




def _build_outputs_improvement_items(
    validation_summary: dict[str, Any],
    reviewed_df: pd.DataFrame,
) -> list[str]:
    suggestions: list[str] = []

    exact_indirect_cols = validation_summary.get("quasi_identifier_exact_remaining_columns", []) or []
    generalized_indirect_cols = validation_summary.get("quasi_identifier_generalized_remaining_columns", []) or []
    high_uniqueness_cols = validation_summary.get("high_uniqueness_columns_after_anonymization", []) or []
    failed_protection_cols = validation_summary.get("failed_protection_attempt_columns", []) or []

    if failed_protection_cols:
        suggestions.append(
            "Requested protection could not be applied for: "
            + ", ".join(map(str, failed_protection_cols[:4]))
            + ". Review these fields before export."
        )

    if exact_indirect_cols:
        suggestions.append(
            "Exact indirect identifiers still remain: "
            + ", ".join(map(str, exact_indirect_cols[:4]))
            + ". Go back to Final Protection Decisions, increase the generalization level for these fields, and rerun anonymization before broader sharing."
        )

    if high_uniqueness_cols:
        suggestions.append(
            "Review highly unique fields that still remain: "
            + ", ".join(map(str, high_uniqueness_cols[:4]))
            + ". These fields can still make records easier to re-identify in some cases."
        )

    if generalized_indirect_cols:
        suggestions.append(
            "Generalized indirect identifiers are still present: "
            + ", ".join(map(str, generalized_indirect_cols[:4]))
            + ". Increase the generalization level further only if stricter disclosure requirements apply."
        )

    remaining_direct_identifier_df = reviewed_df[
        (reviewed_df["classification"] == "direct_identifier")
        & (~reviewed_df["final_action"].isin(["drop", "pseudonymize"]))
    ][["column_name"]].copy()

    if not remaining_direct_identifier_df.empty:
        suggestions.append(
            "Direct identifiers still use a weaker-than-recommended action. Remove or pseudonymize them before wider sharing."
        )

    if not suggestions:
        suggestions.append(
            "No additional improvements are currently required at the present rule level."
        )

    return suggestions[:3]




def _build_outputs_audit_preview_df(
    transformation_log_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
    planning_context: dict[str, Any],
) -> pd.DataFrame:
    if not transformation_log_df.empty:
        log_df = transformation_log_df.copy()

        if "action_requested" not in log_df.columns:
            log_df["action_requested"] = log_df.get("recommended_action", None)

        preview_df = log_df[
            ["column_name", "classification", "action_requested", "action_applied", "notes"]
        ].copy()

        preview_df = _friendly_display_dataframe(preview_df).rename(
            columns={
                "column_name": "Column",
                "classification": "Detected category",
                "action_requested": "Requested action",
                "action_applied": "Applied action",
                "notes": "Notes",
            }
        )
        return _prepare_dataframe_for_display(preview_df)

    audit_df = reviewed_df.copy()
    audit_df["Why this was recommended"] = audit_df.apply(
        lambda row: _format_planning_explanation(
            action=str(row.get("recommended_action", "")),
            reason_code=str(row.get("planning_reason_code", "")),
            classification=str(row.get("classification", "")),
            functional_role=str(row.get("functional_role", "")),
            intended_use=planning_context.get("intended_use", ""),
            record_linkage_required=planning_context.get("need_record_linkage", False),
        ),
        axis=1,
    )

    audit_df = audit_df[
        ["column_name", "classification", "recommended_action", "final_action", "Why this was recommended"]
    ].copy()

    audit_df = _friendly_display_dataframe(audit_df).rename(
        columns={
            "column_name": "Column",
            "classification": "Detected category",
            "recommended_action": "Suggested action",
            "final_action": "Applied action",
        }
    )
    return _prepare_dataframe_for_display(audit_df)


def _format_preview_columns(
    df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
    transformation_log_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if (
        transformation_log_df is not None
        and not transformation_log_df.empty
        and "action_applied" in transformation_log_df.columns
    ):
        action_map = dict(
            zip(
                transformation_log_df["column_name"].astype(str),
                transformation_log_df["action_applied"].astype(str),
            )
        )
    else:
        action_map = dict(
            zip(
                reviewed_df["column_name"].astype(str),
                reviewed_df["final_action"].astype(str),
            )
        )

    def decorate(col: str) -> str:
        action = _normalize_action_key(action_map.get(str(col), "keep"))

        if action == "drop":
            return f"{col} 🔴"
        elif action == "pseudonymize":
            return f"{col} 🟢"
        elif _is_generalize_action(action):
            return f"{col} 🟡"
        elif action == "add_noise":
            return f"{col} 🟡"
        elif action == "mask":
            return f"{col} 🟠"
        else:
            return f"{col} 🟣"

    df.columns = [decorate(str(c)) for c in df.columns]
    return df



def _build_step6_risk_summary(validation_summary: dict[str, Any]) -> dict[str, Any]:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    exact_indirect_count = int(validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0)
    generalized_indirect_count = int(validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0)
    failed_attempt_count = int(validation_summary.get("failed_protection_attempt_count", 0) or 0)
    k_anonymity = validation_summary.get("k_anonymity") or {}
    records_below_k = int(k_anonymity.get("records_below_k") or 0)

    if residual_direct_count > 0 or residual_technical_count > 0:
        return {
            "level": "High",
            "hero_class": "step6-hero-high",
            "title": "Residual privacy risk is high",
            "text": "Direct or internal system identifiers still remain in the protected dataset.",
            "next_step": "Resolve those identifiers before any sharing or downstream release.",
            "indirect_status": "Critical identifiers still present",
            "execution_status": "Review needed" if failed_attempt_count > 0 else "No execution issues",
            "usage_internal": "Not recommended",
            "usage_external": "Not allowed",
        }

    if failed_attempt_count > 0:
        return {
            "level": "Moderate",
            "hero_class": "step6-hero-moderate",
            "title": "Moderate residual risk",
            "text": "Some requested protection methods could not be applied as requested during anonymization.",
            "next_step": "Go back to Final Protection Decisions, review the affected fields, and rerun anonymization.",
            "indirect_status": "Indirect identifier review depends on the applied actions below",
            "execution_status": "Review needed",
            "usage_internal": "Review first",
            "usage_external": "Not allowed",
        }

    if exact_indirect_count > 0:
        return {
            "level": "Moderate",
            "hero_class": "step6-hero-moderate",
            "title": "Residual privacy risk remains",
            "text": "Some indirect identifiers remain in exact form or retain high uniqueness, which can still support record linkage.",
            "next_step": "Go back to Final Protection Decisions, increase protection for remaining exact or highly unique fields (for example by further generalization or a stronger dataset-level protection), and rerun anonymization.",
            "indirect_status": f"{exact_indirect_count} exact indirect identifiers remain",
            "execution_status": "No execution issues",
            "usage_internal": "Acceptable with caution",
            "usage_external": "Not allowed",
        }

    if records_below_k > 0:
        return {
            "level": "Moderate",
            "hero_class": "step6-hero-moderate",
            "title": "Small quasi-identifier groups remain",
            "text": (
                f"{records_below_k} records are in groups smaller than "
                f"k={k_anonymity.get('k', 5)} across the remaining indirect identifiers."
            ),
            "next_step": "Review the measured k-anonymity result and consider stronger generalization before sharing.",
            "indirect_status": f"{records_below_k} records in groups below k",
            "execution_status": "No execution issues",
            "usage_internal": "Acceptable with caution",
            "usage_external": "Not recommended",
        }

    if generalized_indirect_count > 0:
        return {
            "level": "Moderate",
            "hero_class": "step6-hero-moderate",
            "title": "Residual privacy risk is reduced but still present",
            "text": "No critical identifiers remain, but generalized or reduced indirect identifiers still remain.",
            "next_step": "Proceed for internal use if appropriate. Increase the generalization level further only if stricter sharing requirements apply.",
            "indirect_status": f"{generalized_indirect_count} generalized indirect identifiers remain",
            "execution_status": "No execution issues",
            "usage_internal": "Acceptable",
            "usage_external": "Not recommended",
        }

    return {
        "level": "Low",
        "hero_class": "step6-hero-low",
        "title": "Residual privacy risk is low",
        "text": "No residual direct identifiers, technical identifiers, failed protection attempts, or quasi-identifier warnings were flagged at the current rule level.",
        "next_step": "No additional privacy tightening is currently required.",
        "indirect_status": "No indirect identifiers remain",
        "execution_status": "No execution issues",
        "usage_internal": "Acceptable",
        "usage_external": "More acceptable at current rule level",
    }


def _render_step6_risk_banner(risk_summary: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="step6-hero-card {html.escape(risk_summary['hero_class'])}">
            <div class="step6-hero-eyebrow">Privacy risk assessment</div>
            <div class="step6-hero-title">{html.escape(risk_summary['title'])}</div>
            <div class="step6-hero-text">{html.escape(risk_summary['text'])}</div>
            <div class="step6-hero-text" style="margin-top:0.55rem;">
                <strong>Recommended next step:</strong> {html.escape(risk_summary['next_step'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_step6_risk_items(validation_summary: dict[str, Any]) -> list[dict[str, Any]]:
    items = validation_summary.get("validation_warning_items") or []
    if items:
        return items

    warnings = validation_summary.get("validation_warnings", []) or []
    result: list[dict[str, Any]] = []
    for warning in warnings:
        result.append(
            {
                "kind": "legacy",
                "columns": [],
                "risk_category": "unknown",
                "risk_level": "moderate",
                "text": str(warning),
            }
        )
    return result


def _render_step6_signal_cards(validation_summary: dict[str, Any], risk_summary: dict[str, Any]) -> None:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    failed_attempt_count = int(validation_summary.get("failed_protection_attempt_count", 0) or 0)

    direct_status = "Present" if (residual_direct_count > 0 or residual_technical_count > 0) else "Removed"
    execution_status = "Review needed" if failed_attempt_count > 0 else "None"
    k_anonymity = validation_summary.get("k_anonymity") or {}
    if k_anonymity.get("available"):
        k_status = f"{k_anonymity['records_below_k_percent']:.1f}% below k={k_anonymity['k']}"
        k_meta = (
            f"{k_anonymity['records_below_k']} of {k_anonymity['row_count']} records "
            "are in smaller quasi-identifier groups."
        )
        if validation_summary.get("k_anonymity_comparison_available"):
            before = validation_summary["k_anonymity_before"]
            k_meta = (
                f"Before protection: {before['records_below_k_percent']:.1f}%; "
                f"after: {k_anonymity['records_below_k_percent']:.1f}%. "
                f"{k_anonymity['records_below_k']} of {k_anonymity['row_count']} "
                "records are in smaller groups."
            )
    elif k_anonymity.get("applicable"):
        k_status = "Not available"
        k_meta = "No rows were available for the equivalence-class check."
    else:
        k_status = "Not applicable"
        k_meta = "No remaining indirect-identifier columns were available to measure."

    direct_pill = _category_badge_markup("direct_identifier")
    indirect_pill = _category_badge_markup("quasi_identifier")

    items = [
        {
            "label_html": direct_pill,
            "value": direct_status,
            "meta": "Whether direct or internal system identifiers still remain.",
        },
        {
            "label_html": indirect_pill,
            "value": risk_summary["indirect_status"],
            "meta": "Whether Indirect identifiers remain in exact or generalized form.",
        },
        {
            "label_html": "⚙️ Execution issues",
            "value": execution_status,
            "meta": "Whether any requested protection could not be applied.",
        },
        {
            "label_html": "k=5 equivalence-class check",
            "value": k_status,
            "meta": k_meta,
        },
    ]

    cols = st.columns(4, gap="large")
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="outputs-signal-card">
                    <div class="outputs-signal-label">{item["label_html"]}</div>
                    <div class="outputs-signal-value" style="font-size:1.15rem;">{html.escape(item["value"])}</div>
                    <div class="outputs-signal-meta">{html.escape(item["meta"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _filter_linkage_risk_items(risk_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in risk_items
        if item.get("risk_category") in {"quasi_identifier", "uniqueness"}
    ]


def _build_step6_remaining_risk_groups(validation_summary: dict[str, Any]) -> list[dict[str, Any]]:
    risk_items = _get_step6_risk_items(validation_summary)
    grouped_columns: dict[str, set[str]] = {
        "exact_indirect_identifiers": set(),
        "generalized_indirect_identifiers": set(),
        "high_uniqueness": set(),
    }

    structured_items = validation_summary.get("validation_warning_items") or []
    if structured_items:
        for item in risk_items:
            kind = str(item.get("kind", "")).strip()
            if kind not in grouped_columns:
                continue
            grouped_columns[kind].update(str(col) for col in (item.get("columns", []) or []))
    else:
        for warning in validation_summary.get("validation_warnings", []) or []:
            text = _friendly_validation_warning_text(str(warning))
            prefix, _, suffix = text.partition(":")
            columns = [
                str(column).strip()
                for column in suffix.split(",")
                if str(column).strip()
            ]

            if prefix.strip() == "Indirect identifiers still present after anonymization":
                grouped_columns["exact_indirect_identifiers"].update(columns)
            elif prefix.strip() == "Indirect identifiers were generalized, but privacy risk still remains":
                grouped_columns["generalized_indirect_identifiers"].update(columns)
            elif prefix.strip() == "Highly unique fields still remain after anonymization":
                grouped_columns["high_uniqueness"].update(columns)

    group_specs = [
        {
            "kind": "exact_indirect_identifiers",
            "title": "Exact indirect identifiers still unchanged",
            "helper_text": "These fields remain in exact form and are the strongest contributors to linkage risk.",
            "label_html": _category_badge_markup("quasi_identifier"),
        },
        {
            "kind": "generalized_indirect_identifiers",
            "title": "Generalized fields still contributing some residual risk",
            "helper_text": "These fields were generalized, but can still support linkage when combined with other information.",
            "label_html": _category_badge_markup("quasi_identifier"),
        },
        {
            "kind": "high_uniqueness",
            "title": "High uniqueness signals",
            "helper_text": "These fields remain highly unique after anonymization and can still increase re-identification risk.",
            "label_html": '<span class="category-badge" style="background:#fff4e5;color:#b26a00;">High uniqueness</span>',
        },
    ]

    groups: list[dict[str, Any]] = []
    for spec in group_specs:
        columns = sorted(grouped_columns.get(spec["kind"], set()))
        if not columns:
            continue
        groups.append(
            {
                "kind": spec["kind"],
                "title": spec["title"],
                "helper_text": spec["helper_text"],
                "label_html": spec["label_html"],
                "columns": columns,
            }
        )

    return groups


def _render_step6_remaining_risk_fields(validation_summary: dict[str, Any]) -> None:
    risk_groups = _build_step6_remaining_risk_groups(validation_summary)
    indirect_pill = _category_badge_markup("quasi_identifier")

    if not risk_groups:
        st.markdown(
            (
                '<div class="outputs-note-card">'
                '<div class="outputs-note-title">⚠️ What still carries risk</div>'
                '<div class="outputs-note-text">'
                f'{indirect_pill}<br><br>'
                'No remaining indirect-identifier fields are currently flagged.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        return

    group_blocks: list[str] = []

    for idx, group in enumerate(risk_groups):
        items_html = "".join(
            f"<div class='outputs-improvement-item outputs-field-list'>• {html.escape(str(field))}</div>"
            for field in group["columns"]
        )

        divider_html = (
            '<div style="height:1px;background:#edf2f7;margin:0.95rem 0;"></div>'
            if idx > 0
            else ""
        )

        group_blocks.append(
            divider_html
            + '<div>'
            + f'<div class="outputs-note-text" style="font-weight:700;color:#102a43;">{html.escape(group["title"])}</div>'
            + '<div class="outputs-note-text" style="margin-top:0.2rem;">'
            + f'{group["label_html"]}<br><br>{html.escape(group["helper_text"])}'
            + '</div>'
            + f'<div style="margin-top:0.65rem;">{items_html}</div>'
            + '</div>'
        )

    st.markdown(
        (
            '<div class="outputs-note-card">'
            '<div class="outputs-note-title">⚠️ What still carries risk</div>'
            '<div class="outputs-note-text">'
            'This breakdown shows which fields still contribute to re-identification risk and how they do so.'
            '</div>'
            f'<div style="margin-top:0.85rem;">{"".join(group_blocks)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )



def _render_step6_validator_warnings(validation_summary: dict[str, Any]) -> None:
    structured_items = validation_summary.get("validation_warning_items")
    warnings_list = structured_items if structured_items else validation_summary.get("validation_warnings", []) or []
    if not warnings_list:
        return

    direct_pill = _category_badge_markup("direct_identifier")
    indirect_pill = _category_badge_markup("quasi_identifier")

    warning_rows: list[str] = []

    if structured_items:
        for idx, item in enumerate(structured_items):
            kind = str(item.get("kind", "")).strip()
            columns = item.get("columns", []) or []

            if kind == "direct_identifiers":
                label_html = direct_pill
                message_html = "still present after anonymization"
            elif kind == "exact_indirect_identifiers":
                label_html = indirect_pill
                message_html = "remain unchanged (higher re-identification risk)"
            elif kind == "generalized_indirect_identifiers":
                label_html = indirect_pill
                message_html = "were generalized, but privacy risk still remains"
            elif kind == "high_uniqueness":
                label_html = '<span class="category-badge" style="background:#fff4e5;color:#b26a00;">High uniqueness</span>'
                message_html = "fields still remain after anonymization"
            elif kind == "technical_identifiers":
                label_html = ""
                message_html = "Internal system identifiers still present after anonymization"
            elif kind == "technical_context_missing":
                label_html = ""
                message_html = "Technical identifier context was missing during validation"
            else:
                label_html = ""
                message_html = kind

            if label_html:
                row_header = f'{label_html}<span>{html.escape(message_html)}</span>'
            else:
                row_header = f'<span>{html.escape(message_html)}</span>'

            
            columns_html = ""
            if columns:
                columns_str = ", ".join(sorted(columns))
                columns_html = (
                    f'<div class="outputs-note-text" style="margin-top:0.25rem;">'
                    f'{html.escape(columns_str)}'
                    f'</div>'
                )

            divider_html = (
                '<div style="height:1px;background:#edf2f7;margin:0.85rem 0;"></div>'
                if idx > 0
                else ""
            )

            warning_rows.append(
                f'{divider_html}'
                f'<div class="outputs-improvement-item" style="margin-bottom:1rem;">'
                f'<div style="display:flex;align-items:center;gap:0.60rem;flex-wrap:wrap;">'
                f'{row_header}'
                f'</div>'
                f'{columns_html}'
                f'</div>'
            )
    else:
        normalized_warnings = [_friendly_validation_warning_text(warning) for warning in warnings_list]

        for idx, item in enumerate(normalized_warnings):
            text = str(item).strip()

            if ":" in text:
                prefix, suffix = text.split(":", 1)
                prefix = prefix.strip()
                suffix = suffix.strip()
            else:
                prefix, suffix = text, ""


            if prefix == "Direct identifiers still present after anonymization":
                label_html = direct_pill
                message_html = "still present after anonymization"

            elif prefix == "Indirect identifiers still present after anonymization":
                label_html = indirect_pill
                message_html = "remain unchanged (higher re-identification risk)"

            elif prefix in {
                "Generalized quasi-identifiers still present",
                "Generalized indirect identifiers still present",
                "Indirect identifiers were generalized, but privacy risk still remains",
            }:
                label_html = indirect_pill
                message_html = "were generalized, but privacy risk still remains"

            elif prefix == "Highly unique fields still remain after anonymization":
                label_html = '<span class="category-badge" style="background:#fff4e5;color:#b26a00;">High uniqueness</span>'
                message_html = "fields still remain after anonymization"

            elif prefix == "Internal system identifiers still present after anonymization":
                label_html = ""
                message_html = "Internal system identifiers still present after anonymization"

            else:
                label_html = ""
                message_html = prefix

            if label_html:
                row_header = f'{label_html}<span>{html.escape(message_html)}</span>'
            else:
                row_header = f'<span>{html.escape(message_html)}</span>'

            columns_html = ""
            if suffix:
                columns_html = (
                    f'<div class="outputs-note-text" style="margin-top:0.25rem;">'
                    f'{html.escape(suffix)}'
                    f'</div>'
                )

            divider_html = (
                '<div style="height:1px;background:#edf2f7;margin:0.85rem 0;"></div>'
                if idx > 0
                else ""
            )

            warning_rows.append(
                f'{divider_html}'
                f'<div class="outputs-improvement-item" style="margin-bottom:1rem;">'
                f'<div style="display:flex;align-items:center;gap:0.60rem;flex-wrap:wrap;">'
                f'{row_header}'
                f'</div>'
                f'{columns_html}'
                f'</div>'
            )

    warnings_markup = "".join(warning_rows)

    st.markdown(
        (
            '<div class="outputs-note-card">'
            '<div class="outputs-note-title">🔍 Validator warnings raised</div>'
            '<div class="outputs-note-text">'
            'These checks explain why the dataset still carries residual privacy risk at the current rule level.'
            '</div>'
            f'<div style="margin-top:1.3rem;">{warnings_markup}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )



def _render_step6_usage_impact(risk_summary: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="outputs-note-card">
            <div class="outputs-note-title">Usage impact</div>
            <div class="outputs-note-text">
                <strong>Internal analysis:</strong> {html.escape(risk_summary['usage_internal'])}<br><br>
                <strong>External sharing:</strong> {html.escape(risk_summary['usage_external'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _build_step7_use_decision_matrix(
    planning_context: dict[str, Any],
    validation_summary: dict[str, Any],
) -> pd.DataFrame:
    residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
    residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
    failed_attempt_count = int(validation_summary.get("failed_protection_attempt_count", 0) or 0)
    exact_indirect_count = int(validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0)
    generalized_indirect_count = int(validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0)
    k_anonymity = validation_summary.get("k_anonymity") or {}
    records_below_k = int(k_anonymity.get("records_below_k") or 0)

    high_block = residual_direct_count > 0 or residual_technical_count > 0

    rows = []

    if high_block:
        rows = [
            {"Use case": "Internal analytics", "Status": "⚠️ Restricted", "Reason": "Identifiers still remain."},
            {"Use case": "ML training", "Status": "⚠️ Restricted", "Reason": "Identifiers still remain."},
            {"Use case": "External sharing", "Status": "❌ Not allowed", "Reason": "Identifiers still remain."},
            {"Use case": "Public release", "Status": "❌ Not allowed", "Reason": "Identifiers still remain."},
        ]
    elif failed_attempt_count > 0:
        rows = [
            {"Use case": "Internal analytics", "Status": "🔍 Review first", "Reason": "Requested protection could not be fully applied."},
            {"Use case": "ML training", "Status": "🔍 Review first", "Reason": "Requested protection could not be fully applied."},
            {"Use case": "External sharing", "Status": "❌ Not allowed", "Reason": "Unresolved protection execution issues remain."},
            {"Use case": "Public release", "Status": "❌ Not allowed", "Reason": "Unresolved protection execution issues remain."},
        ]
    elif exact_indirect_count > 0 or records_below_k > 0:
        reason = (
            f"Records remain in quasi-identifier groups smaller than k={k_anonymity.get('k', 5)}."
            if records_below_k > 0
            else "Exact or highly linkable fields remain, which can still support linkage in some scenarios."
        )
        k_only_warning = records_below_k > 0 and exact_indirect_count == 0
        rows = [
            {"Use case": "Internal analytics", "Status": "⚠️ Allowed with caution", "Reason": reason},
            {"Use case": "ML training", "Status": "⚠️ Allowed with caution", "Reason": reason},
            {
                "Use case": "External sharing",
                "Status": "⚠️ Not recommended" if k_only_warning else "❌ Not allowed",
                "Reason": reason,
            },
            {
                "Use case": "Public release",
                "Status": "🔍 Review policy" if k_only_warning else "❌ Not allowed",
                "Reason": reason if k_only_warning else "Residual linkage risk remains.",
            },
        ]
    elif generalized_indirect_count > 0:
        rows = [
            {"Use case": "Internal analytics", "Status": "✅ Allowed", "Reason": "Remaining indirect identifiers are generalized or reduced."},
            {"Use case": "ML training", "Status": "✅ Allowed", "Reason": "Remaining indirect identifiers are generalized or reduced."},
            {"Use case": "External sharing", "Status": "⚠️ Not recommended", "Reason": "Generalized indirect identifiers still remain."},
            {"Use case": "Public release", "Status": "❌ Not allowed", "Reason": "Residual re-identification risk still remains."},
        ]
    else:
        rows = [
            {"Use case": "Internal analytics", "Status": "✅ Allowed", "Reason": "No major residual issues flagged."},
            {"Use case": "ML training", "Status": "✅ Allowed", "Reason": "Protected dataset appears acceptable at current rule level."},
            {"Use case": "External sharing", "Status": "✅ More acceptable", "Reason": "No major residual issues flagged at current rule level."},
            {"Use case": "Public release", "Status": "🔍 Review policy", "Reason": "Final release policy should still be checked."},
        ]

    return pd.DataFrame(rows)



def _build_step7_next_steps(validation_summary: dict[str, Any]) -> list[str]:
    exact_cols = validation_summary.get("quasi_identifier_exact_remaining_columns", []) or []
    generalized_cols = validation_summary.get("quasi_identifier_generalized_remaining_columns", []) or []
    failed_cols = validation_summary.get("failed_protection_attempt_columns", []) or []
    k_anonymity = validation_summary.get("k_anonymity") or {}
    records_below_k = int(k_anonymity.get("records_below_k") or 0)

    if failed_cols:
        return [
            "Review fields where requested protection could not be applied. 🔍",
            "Choose an alternative protection method or extend anonymization support. ⚙️",
            "Re-run anonymization before broader use. 🔁",
        ]

    if exact_cols:
        return [
            "Go back to Final Protection Decisions, increase the generalization level for the remaining exact indirect identifiers, and rerun anonymization. ⚙️",
            "Re-run anonymization. 🔁",
            "Re-check the validation summary after protection. 🔍",
        ]

    if records_below_k > 0:
        return [
            f"Review the {records_below_k} records in equivalence classes smaller than k={k_anonymity.get('k', 5)}. 🔍",
            "Consider stronger generalization for the remaining indirect identifiers. ⚙️",
            "Re-run validation before external sharing. 🔁",
        ]

    if generalized_cols:
        return [
            "Proceed with internal-only usage if that fits the project context. ✅",
            "For external sharing, increase the generalization level for the remaining indirect identifiers only if stricter sharing requirements apply. ⚙️",
            "Re-run anonymization and re-check risk. 🔁",
        ]

    return [
        "Proceed with the protected dataset for the selected use case. ✅",
        "Export the audit package for record keeping. 📁",
        "Retain the run artifacts as supporting documentation. 📄",
    ]




# MAIN BODY 

def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="PrivacyGuard AI", page_icon=LOGO_ICON_PATH, layout="wide")
    _inject_app_styles()
    
    logo_uri = image_to_data_uri(LOGO_ICON_PATH)

    st.markdown(
        f"""
        <div class="app-hero">
            <div style="display:flex;align-items:center;gap:1.05rem;">
                <img src="{logo_uri}" alt="PrivacyGuard AI logo" style="width:68px;height:68px;border-radius:17px;flex-shrink:0;">
                <div style="display:flex;flex-direction:column;justify-content:center;">
                    <h1 style="margin:0 0 0.07rem 0; line-height:1;">PrivacyGuard AI</h1>
                    <p style="margin:0; line-height:1.35;">Privacy-first dataset protection | Evidence-based risk review</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # st.caption(f"Environment: `{settings.environment}`")
    
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        _render_workflow_progress(1)

    if "anonymization_completed" not in st.session_state:
        st.session_state["anonymization_completed"] = False

    if "anonymized_df" not in st.session_state:
        st.session_state["anonymized_df"] = None

    if "transformation_log" not in st.session_state:
        st.session_state["transformation_log"] = None

    if "validation_df" not in st.session_state:
        st.session_state["validation_df"] = None

    if "validation_summary" not in st.session_state:
        st.session_state["validation_summary"] = None

    if "report_markdown" not in st.session_state:
        st.session_state["report_markdown"] = None

    if "report_json" not in st.session_state:
        st.session_state["report_json"] = None

    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None

    if "run_dir" not in st.session_state:
        st.session_state["run_dir"] = None

    if "artifact_paths" not in st.session_state:
        st.session_state["artifact_paths"] = None

    if "_active_run_input_signature" not in st.session_state:
        st.session_state["_active_run_input_signature"] = None

    if "_completed_run_input_signature" not in st.session_state:
        st.session_state["_completed_run_input_signature"] = None

    if "_completed_review_plan_signature" not in st.session_state:
        st.session_state["_completed_review_plan_signature"] = None


    _render_section_spacer("lg")


    with st.container():
        _render_step_header(
            1,
            "Upload & scan",
            "Choose a dataset and policy, then click Scan dataset to see its privacy results.",
        )

        _render_section_spacer("md")

        # Labels row — both aligned to same level
        label_col1, label_col2 = st.columns([1.2, 1], gap="large")
        with label_col1:
            st.markdown(
                '<div class="step1-mini-heading">Dataset file</div>',
                unsafe_allow_html=True,
            )
        with label_col2:
            st.markdown(
                '<div class="step1-mini-heading">Policy context</div>',
                unsafe_allow_html=True,
            )

        # Content row
        upload_col, policy_col = st.columns([1.2, 1], gap="large")

        with upload_col:
            uploaded_file = st.file_uploader(
                "Dataset file",
                type=["csv", "xlsx", "parquet"],
                accept_multiple_files=False,
                help="Supported formats: CSV, XLSX, and Parquet.",
                label_visibility="collapsed",
            )

            if uploaded_file is None:
                _render_demo_dataset_download()

        with policy_col:
            st.markdown(
                '<div style="font-size:0.82rem;color:var(--muted);margin-bottom:0.2rem;">Primary use case</div>',
                unsafe_allow_html=True,
            )
            intended_use = st.selectbox(
                "Primary use case",
                options=["internal analysis", "external sharing", "ML training", "research", "dashboard analytics"],
                index=0,
                label_visibility="collapsed",
            )
            st.markdown(
                '<div style="font-size:0.82rem;color:var(--muted);margin-bottom:0.2rem;">Will you need to link records again later?</div>',
                unsafe_allow_html=True,
            )
            linkage_key = "need_record_linkage_toggle"
            if linkage_key not in st.session_state:
                st.session_state[linkage_key] = "no"
            segmented_control = getattr(st, "segmented_control", None)
            if callable(segmented_control):
                need_record_linkage = st.segmented_control(
                    "Will you need to link records again later?",
                    options=["yes", "no"],
                    key=linkage_key,
                    label_visibility="collapsed",
                )
            else:
                need_record_linkage = st.radio(
                    "Will you need to link records again later?",
                    options=["yes", "no"],
                    horizontal=True,
                    key=linkage_key,
                    label_visibility="collapsed",
                )
            st.markdown(
                '<div style="font-size:0.82rem;color:var(--muted);margin-bottom:0.2rem;">Business area</div>',
                unsafe_allow_html=True,
            )
            dataset_domain = st.selectbox(
                "Business area",
                options=["general", "HR", "healthcare", "finance", "customer analytics"],
                index=0,
                label_visibility="collapsed",
            )
    if uploaded_file is None:
        st.session_state.pop("_scan_input_signature", None)
        st.session_state.pop("_scan_requested_signature", None)
        st.session_state.pop("_pipeline_analysis_cache", None)
        return

    scan_context = {
        "intended_use": intended_use,
        "need_record_linkage": need_record_linkage == "yes",
        "dataset_domain": dataset_domain,
    }
    scan_input_signature = _build_run_input_signature(uploaded_file, scan_context)
    if st.session_state.get("_scan_input_signature") != scan_input_signature:
        st.session_state["_scan_input_signature"] = scan_input_signature
        st.session_state.pop("_pipeline_analysis_cache", None)

    scan_clicked = st.button(
        "Scan dataset",
        type="primary",
        use_container_width=True,
        key="scan_dataset_button",
    )
    if scan_clicked:
        st.session_state["_scan_requested_signature"] = scan_input_signature

    if st.session_state.get("_scan_requested_signature") != scan_input_signature:
        st.info("Your dataset is ready. Click **Scan dataset** to view its privacy results.")
        return


    suffix = Path(uploaded_file.name).suffix.lower()

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = Path(tmp.name)

        dataframe, metadata = load_dataset(temp_path)
        planning_context = {
            "intended_use": intended_use,
            "need_record_linkage": need_record_linkage == "yes",
            "dataset_domain": dataset_domain,
            "dataset_size_rows": metadata.rows,
        }
        current_run_input_signature = _build_run_input_signature(
            uploaded_file,
            planning_context,
        )
        if st.session_state.get("_active_run_input_signature") != current_run_input_signature:
            _clear_saved_run_outputs()
            st.session_state.pop("_pipeline_analysis_cache", None)
            st.session_state["_active_run_input_signature"] = current_run_input_signature

        cached_pipeline = st.session_state.get("_pipeline_analysis_cache")
        if (
            isinstance(cached_pipeline, dict)
            and cached_pipeline.get("signature") == current_run_input_signature
        ):
            profiling_df = cached_pipeline["profiling_df"]
            presidio_summary_df = cached_pipeline["presidio_summary_df"]
            classification_df = cached_pipeline["classification_df"]
            plan_df = cached_pipeline["plan_df"]
        else:
            with st.spinner("Scanning dataset for sensitive fields…"):
                profiling_df, _ = profile_dataset(dataframe)
                presidio_summary_df, _ = detect_text_pii_with_presidio(dataframe)

                classifier_params = inspect.signature(classify_columns).parameters
                if "presidio_summary" in classifier_params:
                    classification_df, _ = classify_columns(
                        profiling_df,
                        presidio_summary=presidio_summary_df,
                    )
                else:
                    classification_df, _ = classify_columns(profiling_df)

                role_df, _ = detect_column_roles(
                    profiling_df,
                    classification_df,
                    context=planning_context,
                )
                planner_input_df = classification_df.merge(
                    role_df[["column_name", "functional_role", "role_reason"]],
                    on="column_name",
                    how="left",
                )

                planner_params = inspect.signature(plan_anonymization).parameters
                if "context" in planner_params:
                    plan_df, _ = plan_anonymization(
                        planner_input_df,
                        context=planning_context,
                    )
                elif "policy_context" in planner_params:
                    plan_df, _ = plan_anonymization(
                        planner_input_df,
                        policy_context=planning_context,
                    )
                else:
                    plan_df, _ = plan_anonymization(planner_input_df)

            st.session_state["_pipeline_analysis_cache"] = {
                "signature": current_run_input_signature,
                "profiling_df": profiling_df,
                "presidio_summary_df": presidio_summary_df,
                "classification_df": classification_df,
                "plan_df": plan_df,
            }
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    with progress_placeholder.container():
        _render_workflow_progress(2)

    _render_section_spacer("md")
    _render_privacy_x_factor_lab(
        dataframe,
        classification_df,
        current_run_input_signature,
    )

    _render_section_spacer("lg")


    with st.container():
        _render_subsection_heading("Privacy overview")

        _render_overview_metric_cards(
            metadata=metadata,
            classification_df=classification_df,
        )
        _render_classification_pie_chart(classification_df)

        _render_context_strip(
            [
                ("Use case", str(planning_context["intended_use"])),
                (
                    "Linkage",
                    "Required" if planning_context["need_record_linkage"] else "Not required",
                ),
                ("Domain", str(planning_context["dataset_domain"])),
            ]
        )

        _render_section_spacer("sm")

        with st.expander("Dataset preview and profiling details", expanded=False):
            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(_prepare_dataframe_for_display(dataframe.head(10)), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Data types, missing values, and uniqueness by column.")
            _render_profiling_summary(profiling_df)
            profiling_display_df = _prepare_dataframe_for_display(profiling_df)
            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(profiling_display_df, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)


    _render_section_spacer("lg")




    with st.container():
        _render_subsection_heading("Privacy risks found")
        _render_detection_summary_cards(classification_df)


        _render_section_spacer("md")
        

        with st.expander("How field categories are identified", expanded=False):
            _render_detection_category_help()


        _render_section_spacer("md")


        with st.expander("Text scan details", expanded=False):
            presidio_hits_total = 0
            if not presidio_summary_df.empty:
                presidio_hits_total = int(
                    pd.to_numeric(
                        presidio_summary_df["presidio_hit_count"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
            if presidio_hits_total == 0:
                st.caption("No sensitive patterns were detected in free-text fields.")
            else:
                st.caption(f"Detected {presidio_hits_total:,} text-based sensitive patterns.")
            # presidio table
            # Only show rows where text scan actually found something
            if presidio_hits_total > 0:
                presidio_display_df = presidio_summary_df[
                    presidio_summary_df["presidio_hit_count"] > 0
                ][["column_name", "presidio_entities_detected", "presidio_hit_count"]].copy()
                presidio_display_df["presidio_entities_detected"] = presidio_display_df[
                    "presidio_entities_detected"
                ].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )
                presidio_display_df = presidio_display_df.rename(columns={
                    "column_name": "Column",
                    "presidio_entities_detected": "What was found",
                    "presidio_hit_count": "Occurrences",
                })
                st.dataframe(
                    _prepare_dataframe_for_display(presidio_display_df),
                    width="stretch",
                )

        

        with st.expander("View detailed detection results", expanded=False):
            detection_display_df = classification_df[
                ["column_name", "classification", "reason"]
            ].copy()
            detection_display_df["classification"] = detection_display_df["classification"].map(
                _friendly_classification
            )
            detection_display_df["reason"] = detection_display_df["reason"].map(
                lambda r: (
                    "Identified by column name pattern"
                    if "name-heuristic" in str(r)
                    else "No sensitive signal detected"
                    if "fallback" in str(r)
                    else str(r)
                )
            )
            detection_display_df = detection_display_df.rename(columns={
                "column_name": "Column",
                "classification": "Detected category",
                "reason": "How it was detected",
            })
            st.dataframe(
                _prepare_dataframe_for_display(detection_display_df),
                width="stretch",
            )



    _render_section_spacer("lg")

    with st.container():
        _render_step_header(
            2,
            "Protect & validate",
            "Review the recommended actions, make changes if needed, then run protection and check the result.",
        )

        
        _render_section_spacer("sm")


        plan_action_cards = _build_plan_action_cards(plan_df)

        _render_subsection_heading("Columns by data category")
        metric_columns = st.columns(4, gap="large")
        metric_items = [
            ("Direct identifiers", "direct_identifier", "category-direct_identifier"),
            ("Sensitive fields", "sensitive_attribute", "category-sensitive_attribute"),
            ("Indirect identifiers", "quasi_identifier", "category-quasi_identifier"),
        ]
        for column, (label, cls_key, label_class) in zip(metric_columns, metric_items):
            matched_cols = classification_df[
                classification_df["classification"] == cls_key
            ]["column_name"].tolist()
            count = len(matched_cols)
            col_names = ", ".join(matched_cols[:4])
            if len(matched_cols) > 4:
                col_names += f" +{len(matched_cols)-4} more"
            with column:
                st.markdown(
                    f"""
                    <div class="plan-metric-card">
                        <div class="plan-metric-label {label_class}">{html.escape(label)}</div>
                        <div class="plan-metric-value">{count}</div>
                        <div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem;line-height:1.45;">{html.escape(col_names) if col_names else "-"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with metric_columns[3]:
            st.markdown(
                """
                <div class="plan-metric-card">
                    <div class="plan-metric-label warning">Warnings</div>
                    <div class="plan-metric-value">0</div>
                    <div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem;">-</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _render_section_spacer("md")


        _render_subsection_heading("Recommended actions based on Policy Context")
        action_columns = st.columns(min(3, max(len(plan_action_cards), 1)), gap="large")
        for index, action_card in enumerate(plan_action_cards):
            assigned_cols = list(action_card["columns"])
            visible_cols = assigned_cols[:5]
            remaining = max(len(assigned_cols) - 5, 0)
            col_items_html = "".join(
                f'<div class="plan-column-item">{html.escape(c)}</div>'
                for c in visible_cols
            )
            if remaining > 0:
                col_items_html += f'<div class="plan-card-text">+{remaining} more</div>'
            if not assigned_cols:
                col_items_html = '<div class="plan-card-text">No columns assigned.</div>'
            with action_columns[index % len(action_columns)]:
                st.markdown(
                    f"""
                    <div class="plan-card">
                        <div class="plan-card-title">{html.escape(str(action_card["friendly_label"]))}</div>
                        <div class="plan-card-meta">{html.escape(str(action_card["strength_label"]))}</div>
                        <div class="plan-card-count">{int(action_card["count"])} {"column" if int(action_card["count"]) == 1 else "columns"}</div>
                        <div class="plan-card-text">
                        {html.escape(
                            _format_action_card_explanation(
                                action=str(action_card.get("action", "")),
                                intended_use=planning_context.get("intended_use", ""),
                                record_linkage_required=planning_context.get("need_record_linkage", False),
                            )
                        )}
                        </div>
                        <div style="border-top:1px solid rgba(217,226,236,0.7);margin-top:0.65rem;padding-top:0.55rem;">
                            <div style="font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;">Applies to</div>
                            {col_items_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        _render_section_spacer("sm")

       
        # How to use this plan" card
        st.markdown(
            """
            <div class="risk-hero-card risk-hero-navy">
                <div style="font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:0.2rem;">How to use this plan</div>
                <div class="risk-hero-text">This plan is PrivacyGuard AI's recommended starting point based on detected risk and your selected use case. You can review and change these choices in the next step before anonymization runs.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        _render_section_spacer("sm")

        with st.expander("View full planning table", expanded=False):
            planning_display_df = plan_df.copy()

            planning_display_df["display_explanation"] = planning_display_df.apply(
                lambda row: _format_planning_explanation(
                    action=str(row.get("recommended_action", "")),
                    reason_code=str(row.get("planning_reason_code", "")),
                    classification=str(row.get("classification", "")),
                    functional_role=str(row.get("functional_role", "")),
                    intended_use=planning_context.get("intended_use", ""),
                    record_linkage_required=planning_context.get("need_record_linkage", False),
                ),
                axis=1,
            )

            planning_display_df = planning_display_df[
                ["column_name", "classification", "recommended_action", "display_explanation"]
            ].copy()

            planning_display_df["classification"] = planning_display_df["classification"].map(_friendly_classification)
            planning_display_df["recommended_action"] = planning_display_df["recommended_action"].map(_friendly_action)

            planning_display_df = planning_display_df.rename(columns={
                "column_name": "Column",
                "classification": "Detected category",
                "recommended_action": "Suggested action",
                "display_explanation": "Why this was recommended",
            })

            st.dataframe(_prepare_dataframe_for_display(planning_display_df), width="stretch")



    _render_section_spacer("lg")


    with st.container():
        _render_subsection_heading("Review protection choices")
        guidance_items = [
            {
                "title": "Remove column",
                "best_used_for": "External sharing, public datasets",
                "categories": ["Direct identifier", "Internal system identifier"],
                "why": "Removes the field completely from the dataset.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Replace with pseudonym",
                "best_used_for": "ML training, internal analytics, datasets that require record linkage",
                "categories": ["Direct identifier"],
                "why": "Keeps records linkable while removing direct identity.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Generalize / reduce precision",
                "best_used_for": "Demographic analysis, statistics, research",
                "categories": ["Indirect identifier", "Sensitive information"],
                "why": "Light reduction in detail, such as exact values into narrower ranges or groups.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Generalize more",
                "best_used_for": "Stricter internal review or lower-detail sharing",
                "categories": ["Indirect identifier", "Sensitive information"],
                "why": "Applies stronger reduction in detail and leaves less exact information in the dataset.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Generalize strongly",
                "best_used_for": "Tighter sharing controls and the strongest reduction in detail",
                "categories": ["Indirect identifier", "Sensitive information"],
                "why": "Applies the strongest available reduction in detail for these fields.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Add numeric noise",
                "best_used_for": "ML training or aggregated numeric analysis",
                "categories": ["Sensitive information", "Indirect identifier"],
                "why": "Slightly alters numeric values while preserving broad patterns.",
                "caution": None,
                "display_only_badge": False,
            },
            {
                "title": "Mask value (display only)",
                "best_used_for": "Dashboards or internal reports",
                "categories": ["Direct identifier", "Sensitive information"],
                "why": "Hides part of the value visually, but does not anonymize the dataset.",
                "caution": "Masking is display-only and does not protect the dataset for sharing.",
                "display_only_badge": True,
            },
        ]
        
        _render_section_spacer("sm")

        _render_subsection_heading("Protection Method Guide")
        import streamlit.components.v1 as components

        recommended_methods = _recommended_guide_methods(planning_context)
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
            body {{ background: transparent; }}
            .pmg-wrap {{ border: 1px solid rgba(217,226,236,0.85); border-radius: 14px; overflow: hidden; background: rgba(255,255,255,0.97); }}
            .pmg-tab-bar {{ display: flex; background: rgba(244,247,251,0.96); border-bottom: 1px solid rgba(217,226,236,0.85); padding: 6px 8px 0; gap: 2px; flex-wrap: wrap; }}
            .pmg-tab {{ display: flex; align-items: center; gap: 7px; padding: 7px 13px 9px; border-radius: 8px 8px 0 0; border: 1px solid transparent; border-bottom: none; background: transparent; color: rgba(98,125,152,0.9); font-size: 0.84rem; font-weight: 500; cursor: pointer; transition: all 0.12s; position: relative; bottom: -1px; font-family: inherit; }}
            .pmg-tab:hover {{ background: rgba(255,255,255,0.8); color: rgba(16,42,67,0.9); }}
            .pmg-tab.active {{ background: #163d7a; border-color: #163d7a; border-bottom-color: #163d7a; color: #ffffff; font-weight: 700; }}
            .pmg-tab.active:hover {{ background: #1a4a94; border-color: #1a4a94; border-bottom-color: #1a4a94; }}
            .pmg-tab.active .pmg-tab-icon svg {{ stroke: #ffffff; }}
            .pmg-body-inner {{ padding: 20px 20px 20px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; }}
            .pmg-body-header {{ grid-column: 1/-1; display: flex; align-items: flex_start; gap: 10px; margin-bottom: 26px; padding-bottom: 16px; border-bottom: 1px solid rgba(217,226,236,0.7); }}
            .pmg-icon-wrap {{ width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; background: rgba(220,232,248,0.5); border: 1px solid rgba(150,185,225,0.5); color: #163d7a; }}
            .pmg-icon-wrap svg {{ stroke: #163d7a; }}
            .pmg-method-name {{ font-size: 0.97rem; font-weight: 700; color: #163d7a; line-height: 1.2; }}
            .pmg-method-sub {{ font-size: 0.78rem; color: rgba(98,125,152,1); margin-top: 2px; }}
            .pmg-method-note {{ font-size: 0.74rem; color: #0f6e56; margin-top: 6px; font-weight: 600; line-height: 1.35; }}
            .pmg-col {{ padding-right: 18px; }}
            .pmg-col:not(:last-child) {{ border-right: 1px solid rgba(217,226,236,0.7); margin-right: 18px; }}
            .pmg-col-label {{ font-size: 0.72rem; font-weight: 700; color: rgba(98,125,152,0.85); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
            .pmg-col-value {{ font-size: 0.9rem; color: rgba(98,125,152,1); line-height: 1.55; }}
            .pmg-caution-row {{ grid-column: 1/-1; margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(217,226,236,0.7); display: flex; align-items: flex-start; gap: 8px; }}
            .pmg-caution-dot {{ width: 18px; height: 18px; border-radius: 50%; flex-shrink: 0; margin-top: 1px; background: rgba(183,121,31,0.15); color: #b7791f; font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
            .pmg-caution-text {{ font-size: 0.8rem; color: rgba(98,125,152,1); line-height: 1.5; }}
            .pmg-display-pill {{ display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; background: rgba(197,48,48,0.1); color: #c53030; font-size: 0.72rem; font-weight: 700; margin-left: 8px; }}
            .category-badge {{ display: inline-block; border-radius: 999px; padding: 0.22rem 0.6rem; font-size: 0.78rem; font-weight: 700; margin: 2px 3px 2px 0; }}
            .category-direct_identifier {{ background: rgba(197,48,48,0.12); color: #c53030; }}
            .category-sensitive_attribute {{ background: rgba(183,121,31,0.14); color: #b7791f; }}
            .category-quasi_identifier {{ background: rgba(15,139,141,0.12); color: #0f8b8d; }}
            .category-technical_identifier {{ background: rgba(22,61,122,0.1); color: #163d7a; }}
            </style>
            </head>
            <body>
            <div class="pmg-wrap">
                <div class="pmg-tab-bar" id="pmg-tabs"></div>
                <div id="pmg-body"></div>
            </div>
            <script>
            const recommendedMethods = {recommended_methods};

            const BIN = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path><path d="M10 11v6M14 11v6"></path><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"></path></svg>`;
            const PSEUDO = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3l4 4-4 4"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><path d="M7 21l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>`;
            const GENERALIZE = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/></svg>`;
            const NOISE = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`;
            const MASK = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`;

            const methods = [
            {{ 
                key:"drop",
                icon:BIN,
                label:"Remove column",
                sub:"Drop entirely",
                bestFor:"External sharing, public datasets, fields not needed downstream",
                cats:[
                {{t:"Direct identifier",c:"category-direct_identifier"}},
                {{t:"Internal system identifier",c:"category-technical_identifier"}}
                ],
                why:"Removes the field completely from the dataset.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"pseudonymize",
                icon:PSEUDO,
                label:"Replace with pseudonym",
                sub:"Consistent token",
                bestFor:"ML training, internal analytics, datasets that require record linkage",
                cats:[
                {{t:"Direct identifier",c:"category-direct_identifier"}}
                ],
                why:"Keeps records linkable while removing direct identity.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"generalize_light",
                icon:GENERALIZE,
                label:"Generalize / reduce precision",
                sub:"Light reduction",
                bestFor:"Indirect identifiers, demographic analysis, research, external sharing",
                cats:[
                {{t:"Indirect identifier",c:"category-quasi_identifier"}},
                {{t:"Sensitive information",c:"category-sensitive_attribute"}}
                ],
                why:"Applies a light reduction in detail, such as exact values into narrower ranges or groups.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"generalize_medium",
                icon:GENERALIZE,
                label:"Generalize more",
                sub:"Stronger reduction",
                bestFor:"Broader internal sharing, tighter quasi-identifier protection",
                cats:[
                {{t:"Indirect identifier",c:"category-quasi_identifier"}},
                {{t:"Sensitive information",c:"category-sensitive_attribute"}}
                ],
                why:"Applies stronger reduction in detail and leaves less exact information in the dataset.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"generalize_strict",
                icon:GENERALIZE,
                label:"Generalize strongly",
                sub:"Strongest reduction",
                bestFor:"Stricter sharing requirements and the lowest detail level",
                cats:[
                {{t:"Indirect identifier",c:"category-quasi_identifier"}},
                {{t:"Sensitive information",c:"category-sensitive_attribute"}}
                ],
                why:"Applies the strongest available reduction in detail for these fields.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"add_noise",
                icon:NOISE,
                label:"Add numeric noise",
                sub:"Statistical perturbation",
                bestFor:"ML training, aggregated numeric analysis, sensitive numeric features",
                cats:[
                {{t:"Sensitive information",c:"category-sensitive_attribute"}},
                {{t:"Indirect identifier",c:"category-quasi_identifier"}}
                ],
                why:"Slightly alters numeric values while preserving broad patterns.",
                caution:null,
                displayOnly:false
            }},
            {{ 
                key:"mask",
                icon:MASK,
                label:"Mask value (display only)",
                sub:"Display only",
                bestFor:"Dashboards, internal reports, limited display use",
                cats:[
                {{t:"Direct identifier",c:"category-direct_identifier"}},
                {{t:"Sensitive information",c:"category-sensitive_attribute"}}
                ],
                why:"Hides part of the value visually, but does not anonymize the dataset.",
                caution:"Masking is display-only and does not protect the dataset for sharing.",
                displayOnly:true
            }}
            ];

            let active = 0;

            function renderTabs() {{
                document.getElementById("pmg-tabs").innerHTML = methods.map((m,i) =>
                    `<button class="pmg-tab${{i===active ? " active" : ""}}" onclick="sel(${{i}})">
                        <span class="pmg-tab-icon">${{m.icon}}</span>${{m.label}}
                    </button>`
                ).join("");
            }}

            function renderBody() {{
                const m = methods[active];
                const badges = m.cats.map(c => `<span class="category-badge ${{c.c}}">${{c.t}}</span>`).join("");
                const pill = m.displayOnly ? `<span class="pmg-display-pill">&#9888; Display-only</span>` : "";
                const caution = m.caution ? `<div class="pmg-caution-row"><div class="pmg-caution-dot">!</div><div class="pmg-caution-text">${{m.caution}}</div></div>` : "";
                const recommendedNote = recommendedMethods.includes(m.key)
                    ? `<div class="pmg-method-note">Recommended for your current use case</div>`
                    : "";

                document.getElementById("pmg-body").innerHTML = `
                    <div class="pmg-body-inner">
                        <div class="pmg-body-header">
                            <div class="pmg-icon-wrap">${{m.icon}}</div>
                            <div>
                                <div class="pmg-method-name">${{m.label}}${{pill}}</div>
                                <div class="pmg-method-sub">${{m.sub}}</div>
                                ${{recommendedNote}}
                            </div>
                        </div>
                        <div class="pmg-col"><div class="pmg-col-label">Best used for</div><div class="pmg-col-value">${{m.bestFor}}</div></div>
                        <div class="pmg-col"><div class="pmg-col-label">Suitable data categories</div><div class="pmg-col-value">${{badges}}</div></div>
                        <div class="pmg-col" style="padding-right:0;border-right:none"><div class="pmg-col-label">Why this method</div><div class="pmg-col-value">${{m.why}}</div></div>
                        ${{caution}}
                    </div>`;
            }}

            function sel(i) {{
                active = i;
                renderTabs();
                renderBody();
            }}

            renderTabs();
            renderBody();
            </script>
            </body></html>
            """,
            height=350,
            scrolling=False,
        )
        


        context_hint_text = _build_step5_context_hint(planning_context)
        st.markdown(
            f"""
            <div class="risk-hero-card risk-hero-navy">
                <div class="risk-hero-eyebrow"></div>
                <div style="margin-top:0.28rem;display:flex;flex-direction:column;gap:0.75rem;">
                    <div>
                        <div style="font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:0.2rem;">Context hint</div>
                        <div class="risk-hero-text">{html.escape(context_hint_text)}</div>
                    </div>
                    <div style="border-top:1px solid rgba(15,110,86,0.15);padding-top:0.65rem;">
                        <div style="font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:0.2rem;">Important note</div>
                        <div class="risk-hero-text">Pseudonymization and masking can still leave data within GDPR scope. For external sharing, stronger protection is usually needed.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        

        _render_section_spacer("md")


        _render_subsection_heading("Final Protection Decisions")
        st.caption(
            "Suggested action = the system recommendation. Final protection action = what will actually be applied. "
            "Use this step to review and override where needed."
        )
        
        review_df = _build_review_table(plan_df, classification_df, presidio_summary_df)
        review_df = review_df.merge(
            plan_df[
                [
                    "column_name",
                    "functional_role",
                    "planning_reason_code",
                    "planning_reason",
                ]
            ],
            on="column_name",
            how="left",
        )

        REVIEW_SIGNATURE_KEY = "_review_plan_signature_v2"

        current_plan_signature = _build_plan_signature(plan_df, planning_context)
        current_plan_signature_short = _short_signature(current_plan_signature)
        previous_plan_signature = st.session_state.get(REVIEW_SIGNATURE_KEY)

        if previous_plan_signature != current_plan_signature:
            keys_to_delete = [
                key for key in st.session_state.keys()
                if key.startswith("final_action_")
            ]
            for key in keys_to_delete:
                del st.session_state[key]

            st.session_state[REVIEW_SIGNATURE_KEY] = current_plan_signature

            
                # Final actions will be reconstructed from widget state after rendering all selectors.
        friendly_action_options = [_friendly_action(action) for action in _SUPPORTED_OVERRIDE_ACTIONS]

        for row in review_df.itertuples(index=False):
            suggested_action_name, suggested_action_strength = _action_display_parts(
                str(row.recommended_action)
            )
            current_action_label = _friendly_action(row.final_action)
            state_key = f"final_action_{current_plan_signature_short}_{row.column_name}"

            review_explanation = _format_planning_explanation(
                action=str(row.recommended_action),
                reason_code=str(getattr(row, "planning_reason_code", "")),
                classification=str(row.classification),
                functional_role=str(getattr(row, "functional_role", "")),
                intended_use=planning_context.get("intended_use", ""),
                record_linkage_required=planning_context.get("need_record_linkage", False),
            )

            left_col, right_col = st.columns([1.4, 1], gap="small")

            with left_col:
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.97);border:1px solid rgba(217,226,236,0.85);
                        border-radius:16px;padding:0.9rem 1.1rem;">
                        <div class="fpd-col-name">{html.escape(row.column_name)}</div>
                        {_category_badge_markup(str(row.classification))}
                        <div class="fpd-reason">{html.escape(review_explanation)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_col:
                current_state = st.session_state.get(state_key, current_action_label)
                current_raw = (
                    _raw_action_from_label(current_state)
                    if current_state in friendly_action_options
                    else str(row.final_action)
                )
                has_changed = _is_changed_from_recommendation(
                    str(row.recommended_action),
                    current_raw,
                )
                has_warning = _is_higher_risk_choice(
                    str(row.classification),
                    current_raw,
                    str(row.recommended_action),
                )

                changed_pill_inline = (
                    '<span class="fpd-change-pill" style="margin-left:0.4rem;vertical-align:middle;">Changed</span>'
                    if has_changed else ""
                )
                warning_pill_inline = (
                    '<span class="fpd-warning-pill" style="margin-left:0.4rem;vertical-align:middle;">Higher risk</span>'
                    if has_warning else ""
                )

                final_action_line = (
                    '<div style="display:flex;align-items:center;margin-top:0.75rem;gap:0.35rem;flex-wrap:wrap;">'
                    '<span class="fpd-eyebrow" style="margin:0;">Final protection action</span>'
                    f"{changed_pill_inline}{warning_pill_inline}"
                    "</div>"
                )

                st.markdown(
                    f"""
                    <div style="background:rgba(248,250,253,0.9);
                        border:1px solid rgba(217,226,236,0.85);
                        border-bottom:0;
                        border-radius:16px 16px 0 0;
                        padding:0.9rem 1.1rem 0.75rem;">
                        <div class="fpd-eyebrow">Suggested action</div>
                        <div class="fpd-suggested-name">{html.escape(suggested_action_name)}</div>
                        <div class="fpd-suggested-strength">{html.escape(suggested_action_strength)}</div>
                        {final_action_line}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.selectbox(
                    f"Final protection action for {row.column_name}",
                    options=friendly_action_options,
                    index=(
                        friendly_action_options.index(current_state)
                        if current_state in friendly_action_options
                        else friendly_action_options.index(current_action_label)
                    ),
                    key=state_key,
                    label_visibility="collapsed",
                    help="This is the action that will actually be applied during anonymization.",
                )

                st.markdown(
                    """
                    <div style="background:rgba(248,250,253,0.9);
                        border:1px solid rgba(217,226,236,0.85);
                        border-top:0;
                        border-radius:0 0 16px 16px;
                        height:10px;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        reviewed_df = review_df.copy(deep=True)

        reviewed_df["final_action"] = reviewed_df["column_name"].apply(
            lambda column_name: _raw_action_from_label(
                st.session_state.get(
                    f"final_action_{current_plan_signature_short}_{column_name}",
                    _friendly_action(
                        reviewed_df.loc[
                            reviewed_df["column_name"] == column_name,
                            "final_action",
                        ].iloc[0]
                    ),
                )
            )
        )
        current_review_plan_signature = _build_reviewed_plan_signature(reviewed_df)

        if (
            st.session_state.get("anonymization_completed", False)
            and st.session_state.get("_completed_review_plan_signature") is not None
            and st.session_state.get("_completed_review_plan_signature") != current_review_plan_signature
        ):
            _clear_saved_run_outputs()

        external_mask_warning = _build_external_mask_warning(
            reviewed_df,
            planning_context.get("intended_use", ""),
        )


    if external_mask_warning is not None:
        st.warning(external_mask_warning)

    _render_section_spacer("md")

    _render_subsection_heading("Decision overview")
    decision_overview_items = _build_decision_overview(reviewed_df)
    overview_columns = st.columns(len(decision_overview_items), gap="large")
    for column, item in zip(overview_columns, decision_overview_items):
        with column:
            tone_class = item.get("tone_class", "")
            st.markdown(
                f"""
                <div class="decision-summary-card {html.escape(tone_class)}">
                    <div class="decision-summary-label">{html.escape(item["label"])}</div>
                    <div class="decision-summary-value">{html.escape(item["value"])}</div>
                    <div class="decision-summary-meta">{html.escape(item["meta"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    preview_counts, estimated_risk = _build_policy_alignment_preview(reviewed_df)

    alignment_label = {
        "High": "Policy deviation detected",
        "Moderate": "Partially aligned",
        "Low": "Aligned",
    }.get(estimated_risk, "Aligned")

    risk_class = {
        "High": "risk-hero-high",
        "Moderate": "risk-hero-moderate",
        "Low": "risk-hero-low",
    }.get(estimated_risk, "risk-hero-low")


    display_only_count = preview_counts.get("Display-only protections selected", 0)
    changed_non_weaker_count = preview_counts.get("Changed (non-weaker) actions", 0)
    actions_changed_count = preview_counts.get("Actions changed by user", 0)
    higher_risk_count = preview_counts.get("Higher-risk overrides", 0)

    if estimated_risk == "High":
        alignment_text = "Some final protection choices are weaker than the recommended policy for this use case."
    elif estimated_risk == "Moderate" and changed_non_weaker_count == 0 and actions_changed_count == 0 and display_only_count > 0:
        alignment_text = (
            "Display-only protection is selected for one or more fields. "
            "This does not anonymize the dataset for sharing, even if it may still match the current recommendation."
        )
    elif estimated_risk == "Moderate":
        alignment_text = (
            "Some final protection choices differ from the recommendation, "
            "but they are not less protective than the suggested policy."
        )
    else:
        alignment_text = "Your final protection choices follow the recommended policy for this use case."



    _render_subsection_heading("Pre-run policy alignment")
    st.markdown(
        f"""
        <div class="risk-hero-card policy-alignment-card {risk_class}">
            <div class="risk-hero-eyebrow">Policy alignment</div>
            <div class="risk-hero-value">{html.escape(alignment_label)}</div>
            <div class="risk-hero-text">
                {html.escape(alignment_text)}
            </div>
            <div class="risk-hero-text" style="margin-top:0.55rem;">
                This checks policy alignment only. The validation summary shows what risk remains after protection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if preview_counts["Display-only protections selected"] > 0:
        st.markdown(
            f"""
            <div class="decision-subtle-note">
                Display-only protections selected: {preview_counts["Display-only protections selected"]}.
                Masking is display-only and does not protect the dataset for sharing.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "This preview shows whether your final choices match the selected policy before anonymization runs."
    )

    if estimated_risk == "High":
        st.warning(
            "Some final protection choices are less protective than the selected policy. "
            "Review them before running anonymization."
        )
    elif estimated_risk == "Moderate" and changed_non_weaker_count == 0 and actions_changed_count == 0 and display_only_count > 0:
        st.info(
            "Display-only masking is selected for one or more fields. "
            "Review whether those fields need full dataset protection before running anonymization."
        )
    elif estimated_risk == "Moderate":
        st.info(
            "You changed one or more suggested actions. Your selections may still be acceptable, "
            "but they do not exactly match the recommended plan."
        )
    else:
        st.success("Your current protection decisions are aligned with the selected policy.")

    st.caption(
        "Your selections are used directly when you choose Run anonymization with reviewed plan. "
        "Residual dataset risk is reviewed after protection is applied."
    )

    with st.expander("View full review details", expanded=False):
        audit_df = reviewed_df.copy()

        audit_df["display_explanation"] = audit_df.apply(
            lambda row: _format_planning_explanation(
                action=str(row.get("recommended_action", "")),
                reason_code=str(row.get("planning_reason_code", "")),
                classification=str(row.get("classification", "")),
                functional_role=str(row.get("functional_role", "")),
                intended_use=planning_context.get("intended_use", ""),
                record_linkage_required=planning_context.get("need_record_linkage", False),
            ),
            axis=1,
        )

        audit_df = audit_df[
            [
                "column_name",
                "classification",
                "recommended_action",
                "final_action",
                "display_explanation",
            ]
        ].copy()

        audit_df["classification"] = audit_df["classification"].map(_friendly_classification)
        audit_df["recommended_action"] = audit_df["recommended_action"].map(_friendly_action)
        audit_df["final_action"] = audit_df["final_action"].map(_friendly_action)

        audit_df = audit_df.rename(columns={
            "column_name": "Column",
            "classification": "Detected category",
            "recommended_action": "Suggested action",
            "final_action": "Your final choice",
            "display_explanation": "Why this was recommended",
        })

        st.dataframe(_prepare_dataframe_for_display(audit_df), width="stretch")

        action_guide_df = pd.DataFrame(
            [
                {
                    "Action label": _friendly_action(action),
                    "System value": action,
                    "Protection type": (
                        "Dataset protection"
                        if action in _DATASET_PROTECTION_ACTIONS
                        else "Display protection"
                        if action in _DISPLAY_PROTECTION_ACTIONS
                        else "No protection change"
                    ),
                }
                for action in _SUPPORTED_OVERRIDE_ACTIONS
            ]
        )
        st.table(_prepare_dataframe_for_display(action_guide_df))


    _render_section_spacer("md")

    
    st.markdown(
    """
    <div class="risk-hero-card risk-hero-navy">
        <div style="font-size:0.84rem;color:var(--muted);line-height:1.5;">
            When you run anonymization, your protected dataset, a transformation log,
            and an audit report will be generated automatically.
            This may take a few seconds depending on dataset size.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


    #Anonymization Button
    run_anonymization_clicked = st.button(
        "Run anonymization with reviewed plan",
        type="primary",
        help="Applies your final protection choices and generates the anonymized dataset.",
    )


    if run_anonymization_clicked:

        invalid_actions = ~reviewed_df["final_action"].isin(_SUPPORTED_OVERRIDE_ACTIONS)
        if invalid_actions.any():
            st.error("Something went wrong with your protection selections. Please reload the page and try again.")
            return

        reviewed_plan_columns = [
            "column_name",
            "classification",
            "recommended_action",
            "final_action",
        ]
        reviewed_plan_columns.extend(
            column
            for column in [
                "functional_role",
                "policy_context",
                "record_linkage_required",
            ]
            if column in reviewed_df.columns
        )
        reviewed_plan_df = reviewed_df[reviewed_plan_columns].copy()
        
        try:
            anonymized_df, transformation_log = anonymize_dataset(dataframe, reviewed_plan_df)
        except Exception as exc:
            st.error(f"Anonymization failed: {exc}")
            return

        try:
            validation_df, _, validation_summary = validate_anonymization_output(
                classification_df,
                reviewed_plan_df,
                anonymized_df,
                transformation_log=transformation_log,
                original_dataframe=dataframe,
            )
        except Exception as exc:
            st.error(f"Validation failed: {exc}")
            return

        try:
            report_markdown, report_json = generate_anonymization_report(
                metadata=metadata,
                planning_context=planning_context,
                profiling_df=profiling_df,
                presidio_summary_df=presidio_summary_df,
                classification_df=classification_df,
                plan_df=reviewed_plan_df,
                transformation_log=transformation_log,
                validation_df=validation_df,
                validation_summary=validation_summary,
                anonymized_dataframe=anonymized_df,
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
            return

        try:
            run_id, run_dir, artifact_paths = save_run_artifacts(
                metadata=metadata,
                planning_context=planning_context,
                anonymized_dataframe=anonymized_df,
                transformation_log=transformation_log,
                validation_summary=validation_summary,
                report_markdown=report_markdown,
                report_json=report_json,
            )
        except Exception as exc:
            st.error(f"Run artifact save failed: {exc}")
            return

        transformation_log_df = pd.DataFrame(transformation_log)



        with progress_placeholder.container():
            _render_workflow_progress(3)

        # ✅ SAVE RESULTS TO SESSION STATE 
        st.session_state["anonymization_completed"] = True
        st.session_state["anonymized_df"] = anonymized_df
        st.session_state["transformation_log"] = transformation_log
        st.session_state["validation_df"] = validation_df
        st.session_state["validation_summary"] = validation_summary
        st.session_state["report_markdown"] = report_markdown
        st.session_state["report_json"] = report_json
        st.session_state["run_id"] = run_id
        st.session_state["run_dir"] = run_dir
        st.session_state["artifact_paths"] = artifact_paths
        st.session_state["_completed_run_input_signature"] = current_run_input_signature
        st.session_state["_completed_review_plan_signature"] = current_review_plan_signature


    # ✅ NEW GATE — prevents reset on rerun
    if not st.session_state.get("anonymization_completed", False):
        return

    if st.session_state.get("_completed_run_input_signature") != current_run_input_signature:
        return

    if st.session_state.get("_completed_review_plan_signature") != current_review_plan_signature:
        return

    # ✅ RESTORE DATA AFTER RERUN
    anonymized_df = st.session_state["anonymized_df"]
    transformation_log = st.session_state["transformation_log"]
    validation_df = st.session_state["validation_df"]
    validation_summary = st.session_state["validation_summary"]
    report_markdown = st.session_state["report_markdown"]
    report_json = st.session_state["report_json"]
    run_id = st.session_state["run_id"]
    run_dir = st.session_state["run_dir"]
    artifact_paths = st.session_state["artifact_paths"]

    transformation_log_df = pd.DataFrame(transformation_log)




    _render_section_spacer("lg")

    with st.container():
        _render_subsection_heading("Validation summary")

        _render_section_spacer("sm")

        risk_summary = _build_step6_risk_summary(validation_summary)

        _render_step6_risk_banner(risk_summary)
        _render_section_spacer("sm")

        _render_step6_signal_cards(validation_summary, risk_summary)
        _render_section_spacer("sm")

        _render_subsection_heading("What still carries risk")
        _render_step6_remaining_risk_fields(validation_summary)
        _render_section_spacer("sm")

        if risk_summary["usage_internal"] in {"Not recommended", "Acceptable with caution"}:
            internal_guidance = "⚠️ " + risk_summary["usage_internal"]
        else:
            internal_guidance = "✅ " + risk_summary["usage_internal"]

        if risk_summary["usage_external"] in {"Not allowed", "Not recommended"}:
            external_guidance = "❌ " + risk_summary["usage_external"] if risk_summary["usage_external"] == "Not allowed" else "⚠️ " + risk_summary["usage_external"]
        else:
            external_guidance = "✅ " + risk_summary["usage_external"]

        impact_col_1, impact_col_2 = st.columns(2, gap="large")

        with impact_col_1:
            st.markdown(
                f"""
                <div class="outputs-note-card">
                    <div class="outputs-note-title">What this means for your dataset</div>
                    <div class="outputs-note-text">
                        <strong>Internal analysis:</strong> {html.escape(internal_guidance)}<br><br>
                        <strong>External sharing:</strong> {html.escape(external_guidance)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
        residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
        exact_indirect_count = int(validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0)
        generalized_indirect_count = int(validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0)
        failed_attempt_count = int(validation_summary.get("failed_protection_attempt_count", 0) or 0)

        if residual_direct_count > 0 or residual_technical_count > 0:
            action_now = (
                "Some direct or internal system identifiers still remain. "
                "Go back to Final Protection Decisions and strengthen protection before using or sharing this dataset."
            )
        elif failed_attempt_count > 0:
            action_now = (
                "Some requested protections could not be fully applied. "
                "Review those fields before relying on this version for broader use."
            )
        elif exact_indirect_count > 0:
            action_now = (
                "Exact indirect identifiers still remain. "
                "Increase protection for remaining exact or highly unique fields "
                "(for example by further generalization or masking), and rerun anonymization before external sharing."
            )
        elif generalized_indirect_count > 0:
            action_now = (
                "Generalized indirect identifiers still remain. "
                "Internal use may still be appropriate, but broader sharing may require a higher generalization level."
            )
        else:
            action_now = (
                "No major residual issues were flagged at the current rule level. "
                "Review the dataset readiness guidance before using or sharing this output."
            )

        with impact_col_2:
            st.markdown(
                f"""
                <div class="outputs-note-card">
                    <div class="outputs-note-title">👉 Recommended action now</div>
                    <div class="outputs-note-text">{html.escape(action_now)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        _render_section_spacer("sm")


        _render_step6_validator_warnings(validation_summary)
        

        _render_section_spacer("sm")



        with st.expander("Advanced diagnostics", expanded=False):
            validation_detail_df = validation_df.copy()

            validation_detail_df["check_name"] = validation_detail_df["check_name"].replace(
                {
                    "residual_direct_identifier_count": "Residual direct identifiers",
                    "residual_technical_identifier_count": "Residual internal system identifiers",
                    "contextual_technical_validation_incomplete": "Technical identifier context missing",
                    "high_uniqueness_columns_after_anonymization": "Highly unique fields after anonymization",
                    "quasi_identifier_columns_remaining": "Indirect identifiers still present",
                    "quasi_identifier_exact_remaining": "Indirect identifiers still exact",
                    "quasi_identifier_generalized_remaining": "Indirect identifiers generalized but still present",
                    "records_in_equivalence_classes_below_k": "Records in groups below k",
                    "failed_protection_attempts": "Failed protection attempts",
                    "validation_warnings": "Residual-risk warnings raised",
                }
            )

            validation_detail_df["columns"] = validation_detail_df["columns"].apply(
                _format_validation_detail_columns
            )

            validation_detail_df = validation_detail_df.rename(
                columns={
                    "check_name": "Residual-risk check",
                    "value": "Count",
                    "columns": "Affected fields or details",
                }
            )

            validation_detail_df = validation_detail_df[
                ~(
                    (validation_detail_df["Count"] == 0)
                    & (validation_detail_df["Affected fields or details"] == "None")
                )
            ]

            st.dataframe(
                _prepare_dataframe_for_display(validation_detail_df),
                width="stretch",
            )

        
        st.info("Review the risk result before using or sharing the protected dataset.")

    _render_section_spacer("lg")

    with st.container():
        _render_subsection_heading("Dataset readiness")

        outputs_summary = _build_outputs_readiness_summary(
            planning_context=planning_context,
            validation_summary=validation_summary,
            reviewed_df=reviewed_df,
        )
        breakdown_cards = _build_outputs_breakdown_cards(
            reviewed_df=reviewed_df,
            transformation_log_df=transformation_log_df,
        )
        use_decision_df = _build_step7_use_decision_matrix(
            planning_context=planning_context,
            validation_summary=validation_summary,
        )
        next_steps = _build_step7_next_steps(validation_summary)

        residual_direct_count = int(validation_summary.get("residual_direct_identifier_count", 0) or 0)
        residual_technical_count = int(validation_summary.get("residual_technical_identifier_count", 0) or 0)
        exact_indirect_count = int(validation_summary.get("quasi_identifier_exact_remaining_count", 0) or 0)
        generalized_indirect_count = int(validation_summary.get("quasi_identifier_generalized_remaining_count", 0) or 0)
        failed_attempt_count = int(validation_summary.get("failed_protection_attempt_count", 0) or 0)



        st.markdown(
            f"""
            <div class="risk-hero-card {html.escape(outputs_summary['hero_class'])}">
                <div class="risk-hero-eyebrow">Dataset readiness for use</div>
                <div class="risk-hero-value">{html.escape(outputs_summary['title'])}</div>
                <div class="risk-hero-text">{html.escape(outputs_summary['text'])}</div>
                <div class="risk-hero-text" style="margin-top:0.5rem;">
                    <strong>Recommended fit:</strong> {html.escape(outputs_summary['recommended_use'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if residual_direct_count > 0 or residual_technical_count > 0:
            allowed_use = "Restricted internal review only"
            restricted_use = "External sharing and public release are not allowed"
        elif failed_attempt_count > 0:
            allowed_use = "Internal review or controlled access after checking the flagged fields"
            restricted_use = "Broader sharing should wait until the flagged fields are reviewed and rerun"
        elif exact_indirect_count > 0:
            allowed_use = "Internal analysis with caution for limited, controlled use"
            restricted_use = "External sharing is not recommended because exact or highly linkable fields remain, which could enable re-identification when combined with external data."
        elif generalized_indirect_count > 0:
            allowed_use = "Internal analytics and controlled internal use"
            restricted_use = "External sharing is still not recommended without further review"
        else:
            allowed_use = "Selected use case appears acceptable at the current rule level"
            restricted_use = "Still confirm final organizational policy before broad release"

        
        
        guidance_col_1, guidance_col_2 = st.columns(2, gap="large")
        with guidance_col_1:
            st.markdown(
                f"""
                <div class="outputs-guidance-card outputs-guidance-allowed">
                    <div class="outputs-guidance-title-row">
                        <div class="outputs-guidance-icon">✅</div>
                        <div class="outputs-guidance-title">Allowed use</div>
                    </div>
                    <div class="outputs-guidance-text">{html.escape(allowed_use)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with guidance_col_2:
            st.markdown(
                f"""
                <div class="outputs-guidance-card outputs-guidance-restricted">
                    <div class="outputs-guidance-title-row">
                        <div class="outputs-guidance-icon">⚠️</div>
                        <div class="outputs-guidance-title">Restricted or not recommended</div>
                    </div>
                    <div class="outputs-guidance-text">{html.escape(restricted_use)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        _render_section_spacer("sm")




        decision_matrix_df = use_decision_df

        _render_subsection_heading("Allowed use and release guidance")
        st.caption("This table translates the privacy result into practical usage guidance for this protected dataset.")
        st.markdown('<div class="table-card">', unsafe_allow_html=True)
        st.dataframe(
            _prepare_dataframe_for_display(decision_matrix_df),
            width="stretch",
            height=220,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        _render_section_spacer("md")

        _render_subsection_heading("Protected dataset preview")
        st.caption(
            "Preview of the dataset after applied transformations. Columns left unchanged may still carry re-identification risk."
        )

        preview_mode = st.radio(
            "Preview mode",
            options=["Protected dataset", "Original dataset", "Side-by-side preview"],
            horizontal=True,
            key="outputs_preview_mode",
            label_visibility="collapsed",
        )

        if preview_mode == "Protected dataset":
            formatted_protected_preview = _format_preview_columns(
                _prepare_dataframe_for_display(anonymized_df.head(100)),
                reviewed_df,
                transformation_log_df=transformation_log_df,
            )

            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(
                formatted_protected_preview,
                width="stretch",
                height=420,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        elif preview_mode == "Original dataset":
            formatted_original_preview = _format_preview_columns(
                _prepare_dataframe_for_display(dataframe.head(100)),
                reviewed_df,
                transformation_log_df=transformation_log_df,
            )

            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(
                formatted_original_preview,
                width="stretch",
                height=420,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            preview_left, preview_right = st.columns(2, gap="large")

            with preview_left:
                st.markdown(
                    '<div class="outputs-note-title" style="margin-bottom:0.45rem;">Original data</div>',
                    unsafe_allow_html=True,
                )
                formatted_original_preview = _format_preview_columns(
                    _prepare_dataframe_for_display(dataframe.head(40)),
                    reviewed_df,
                    transformation_log_df=transformation_log_df,
                )
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                st.dataframe(
                    formatted_original_preview,
                    width="stretch",
                    height=360,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with preview_right:
                st.markdown(
                    '<div class="outputs-note-title" style="margin-bottom:0.45rem;">Protected data</div>',
                    unsafe_allow_html=True,
                )
                formatted_protected_preview = _format_preview_columns(
                    _prepare_dataframe_for_display(anonymized_df.head(40)),
                    reviewed_df,
                    transformation_log_df=transformation_log_df,
                )
                st.markdown('<div class="table-card">', unsafe_allow_html=True)
                st.dataframe(
                    formatted_protected_preview,
                    width="stretch",
                    height=360,
                )
                st.markdown('</div>', unsafe_allow_html=True)

        _render_section_spacer("md")

        _render_subsection_heading("How your data was transformed")
        breakdown_columns = st.columns(len(breakdown_cards), gap="large")
        for column, card in zip(breakdown_columns, breakdown_cards):
            


            items_html = "".join(
                f"<div><strong>{html.escape(item.split(':')[0])}:</strong> <span class='breakdown-columns'>{html.escape(':'.join(item.split(':')[1:]).strip())}</span></div>"
                if ":" in item else f"<div>{html.escape(item)}</div>"
                for item in card["items"]
            )


            with column:
                st.markdown(
                    f"""
                    <div class="outputs-breakdown-card {html.escape(card['tone_class'])}">
                        <div class="outputs-breakdown-title-row">
                            <div class="outputs-breakdown-icon">{html.escape(card["icon"])}</div>
                            <div class="outputs-breakdown-title">{html.escape(card["title"])}</div>
                        </div>
                        <div class="outputs-breakdown-value">{html.escape(card["value"])}</div>
                        <div class="outputs-breakdown-meta">{html.escape(card["meta"])}</div>
                        <div class="outputs-breakdown-list">{items_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        _render_section_spacer("sm")

        total_columns = int(len(classification_df))
        presidio_hits_total = 0
        if not presidio_summary_df.empty and "presidio_hit_count" in presidio_summary_df.columns:
            presidio_hits_total = int(
                pd.to_numeric(
                    presidio_summary_df["presidio_hit_count"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        summary_col_1, summary_col_2 = st.columns([1, 1], gap="large")
        with summary_col_1:
            st.markdown(
                f"""
                <div class="outputs-note-card">
                    <div class="outputs-note-title">Run summary</div>
                    <div class="outputs-note-text">
                        Total columns reviewed: <strong>{total_columns}</strong><br>
                        Final protected columns: <strong>{anonymized_df.shape[1]}</strong><br>
                        Rows in protected dataset: <strong>{anonymized_df.shape[0]}</strong><br>
                        Saved files in this run: <strong>{len(artifact_paths)}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with summary_col_2:
            st.markdown(
                f"""
                <div class="outputs-note-card">
                    <div class="outputs-note-title">Detection and audit signals</div>
                    <div class="outputs-note-text">
                        Text entities detected earlier in workflow: <strong>{"Yes" if presidio_hits_total > 0 else "No"}</strong><br>
                        Text detection hit count: <strong>{presidio_hits_total}</strong><br>
                        Residual warnings after anonymization: <strong>{len(validation_summary.get("validation_warnings", []))}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )



        _render_section_spacer("md")




        next_steps_markup = "".join(
            f'<div class="outputs-nextstep-item">'
            f'<div class="outputs-nextstep-bullet">→</div>'
            f'<div class="outputs-nextstep-item-text">{html.escape(item)}</div>'
            f'</div>'
            for item in next_steps
        )

        _render_subsection_heading("Recommended next step")
        st.markdown(
            (
                '<div class="outputs-nextstep-card">'
                '<div class="outputs-nextstep-title-row">'
                '<div class="outputs-nextstep-icon">👉</div>'
                '<div class="outputs-nextstep-title">What to do next</div>'
                '</div>'
                '<div class="outputs-nextstep-text">'
                'Use this guidance to decide whether to proceed with this version or go back to Final Protection Decisions and rerun anonymization.'
                '</div>'
                f'<div class="outputs-nextstep-list">{next_steps_markup}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )



        remaining_direct_identifier_df = reviewed_df[
            (reviewed_df["classification"] == "direct_identifier")
            & (~reviewed_df["final_action"].isin(["drop", "pseudonymize"]))
        ][["column_name", "final_action"]].copy()

        if not remaining_direct_identifier_df.empty:
            remaining_direct_identifier_df["final_action"] = remaining_direct_identifier_df["final_action"].map(
                _friendly_action
            )
            column_list = ", ".join(remaining_direct_identifier_df["column_name"].astype(str).tolist())
            st.warning(
                "Some directly identifying columns still have a weaker final action than removal or pseudonymization. "
                f"Review these columns before wider sharing: {column_list}"
            )



        _render_section_spacer("sm")


        with st.expander("View transformation details and audit preview", expanded=False):
            audit_preview_df = _build_outputs_audit_preview_df(
                transformation_log_df=transformation_log_df,
                reviewed_df=reviewed_df,
                planning_context=planning_context,
            )
            st.markdown('<div class="table-card">', unsafe_allow_html=True)
            st.dataframe(
                audit_preview_df,
                width="stretch",
                height=320,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            report_preview = report_markdown[:2000]
            if len(report_markdown) > 2000:
                report_preview += "\n... (truncated)"
            st.code(report_preview, language="markdown")




    _render_section_spacer("lg")



    with st.container():
        _render_step_header(
            3,
            "Export",
            "Download the protected dataset and audit report.",
        )

        warning_count = len(validation_summary.get("validation_warnings", []))
        saved_file_names = [Path(path).name for path in artifact_paths.values()]
        csv_data = anonymized_df.to_csv(index=False).encode("utf-8")

        st.markdown(
            """
    <div class="export-hero-card">
        <div class="export-hero-eyebrow">Final outputs</div>
        <div class="export-hero-title">Your anonymization package is ready</div>
        <div class="export-hero-text">
            Export the protected dataset and the supporting audit files from this completed run.
            The saved run folder can also be kept as a traceable internal record.
        </div>
        <div class="export-hero-helper">
            <div class="export-helper-title">What most users need</div>
            <div class="export-helper-text">
                <strong>Usually download:</strong> protected dataset + audit report.<br>
                <span class="helper-subtle">
                    Use the transformation log when you need technical traceability or implementation review.
                </span>
            </div>
        </div>
    </div>
            """,
            unsafe_allow_html=True,
        )


        _render_section_spacer("md")


        st.markdown(
            f"""
    <div class="export-note-card">
        <div class="export-combined-header">
            <div class="export-combined-section">
                <div class="export-note-title">🗂️ Saved artifacts in this run</div>
            </div>
            <div class="export-combined-section">
                <div class="export-note-title">📑 Run metadata</div>
            </div>
        </div>
        <div class="export-combined-divider"></div>
        <div class="export-combined-body">
            <div class="export-combined-left">
                <div class="export-note-text">
                    <strong>{len(saved_file_names)}</strong> files were saved as part of the run record.
                </div>
                <div class="export-file-list">
                    {''.join(f'<div class="export-file-item">• {html.escape(name)}</div>' for name in saved_file_names)}
                </div>
            </div>
            <div class="export-combined-right">
                <div class="export-note-text">
                    <strong>Run ID:</strong> {html.escape(str(run_id))}<br>
                    <strong>Saved location:</strong> {html.escape(str(run_dir))}<br>
                    <strong>Rows:</strong> {anonymized_df.shape[0]}<br>
                    <strong>Columns:</strong> {anonymized_df.shape[1]}
                </div>
            </div>
        </div>
    </div>
            """,
            unsafe_allow_html=True,
        )


        _render_section_spacer("sm")



        top_cols = st.columns(4, gap="large")
        top_items = [
            {
                "label": "Run ID 🆔",
                "value": str(run_id),
                "meta": "Unique identifier for this anonymization run.",
                "value_class": "export-mini-value export-mini-value-run",
            },
            {
                "label": "Saved files 🗂️",
                "value": str(len(artifact_paths)),
                "meta": "Artifacts stored for this completed run.",
                "value_class": "export-mini-value",
            },
            {
                "label": "Warnings raised ⚠️",
                "value": str(warning_count),
                "meta": "Residual warnings recorded after anonymization.",
                "value_class": "export-mini-value",
            },
            {
                "label": "Saved location 📍",
                "value": "Ready",
                "meta": str(run_dir),
                "value_class": "export-mini-value",
            },
        ]

        for col, item in zip(top_cols, top_items):
            with col:
                st.markdown(
                    f"""
                    <div class="export-mini-card">
                        <div class="export-mini-label">{html.escape(item["label"])}</div>
                        <div class="{html.escape(item['value_class'])}">{html.escape(item["value"])}</div>
                        <div class="export-mini-meta">{html.escape(item["meta"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        _render_section_spacer("sm")

        if not transformation_log_df.empty:
            log_json_bytes = json.dumps(
                transformation_log,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")

            if "action_requested" not in transformation_log_df.columns:
                transformation_log_df["action_requested"] = transformation_log_df.get(
                    "recommended_action",
                    None,
                )

            transformation_log_export_df = transformation_log_df[
                [
                    "column_name",
                    "classification",
                    "action_requested",
                    "action_applied",
                    "notes",
                ]
            ].copy()

            transformation_log_export_df = _friendly_display_dataframe(
                transformation_log_export_df
            ).rename(
                columns={
                    "column_name": "Column",
                    "classification": "Detected category",
                    "action_requested": "Requested action",
                    "action_applied": "Applied action",
                    "notes": "Notes",
                }
            )

            log_csv_bytes = transformation_log_export_df.to_csv(index=False).encode("utf-8")
        else:
            log_json_bytes = None
            log_csv_bytes = None



        _render_section_spacer("md")




        _render_subsection_heading("Download package")

        bundle_col_1, bundle_col_2, bundle_col_3 = st.columns(3, gap="large")

        with bundle_col_1:
            st.markdown(
                """
                <div class="export-bundle-card">
                    <div class="export-bundle-title-row">
                        <div class="export-bundle-icon">📦</div>
                        <div class="export-bundle-title">Protected dataset</div>
                    </div>
                    <div class="export-bundle-text">
                        Export the anonymized dataset used as the final protected output of this run.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                label="Download anonymized CSV",
                data=csv_data,
                file_name="anonymized_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown(
                """
                    <div class="export-file-list">
                        <div class="export-file-item"><span>Includes:</span> protected tabular dataset</div>
                        <div class="export-file-item"><span>Best for:</span> downstream analysis, ML, controlled sharing</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bundle_col_2:
            st.markdown(
                """
                <div class="export-bundle-card">
                    <div class="export-bundle-title-row">
                        <div class="export-bundle-icon">🧾</div>
                        <div class="export-bundle-title">Transformation log</div>
                    </div>
                    <div class="export-bundle-text">
                        Structured record of requested versus applied protection decisions.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if log_json_bytes is not None:
                st.download_button(
                    label="Download log (JSON)",
                    data=log_json_bytes,
                    file_name="transformation_log.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if log_csv_bytes is not None:
                st.download_button(
                    label="Download log (CSV)",
                    data=log_csv_bytes,
                    file_name="transformation_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(
                """
                    <div class="export-file-list">
                        <div class="export-file-item"><span>Includes:</span> requested action, applied action, notes</div>
                        <div class="export-file-item"><span>Best for:</span> traceability and implementation review</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bundle_col_3:
            st.markdown(
                """
                <div class="export-bundle-card">
                    <div class="export-bundle-title-row">
                        <div class="export-bundle-icon">📘</div>
                        <div class="export-bundle-title">Audit report</div>
                    </div>
                    <div class="export-bundle-text">
                        Summary package for documentation, review, and governance support.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                label="Download audit report (.md)",
                data=report_markdown.encode("utf-8"),
                file_name="anonymization_audit_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

            st.download_button(
                label="Download audit report (.json)",
                data=json.dumps(report_json, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="anonymization_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown(
                """
                    <div class="export-file-list">
                        <div class="export-file-item"><span>Includes:</span> dataset summary, privacy checks, run outputs</div>
                        <div class="export-file-item"><span>Best for:</span> audit trail and internal documentation</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _render_section_spacer("lg")

    st.markdown(
        """
        <div class="export-note-card-last">
            <div class="export-note-title">📌 Recommended record-keeping</div>
            <div class="export-note-text">
                Keep the saved run folder together with the exported files when you need a complete audit trail.
                This is especially useful when the anonymization run may need to be reviewed later,
                repeated under the same conditions, or explained to internal stakeholders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_section_spacer("lg")


    
if __name__ == "__main__":
    main()
