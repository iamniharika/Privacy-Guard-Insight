<p align="center">
  <img src="assets/logos/privacyguard_ai_logo.svg" alt="PrivacyGuard AI logo" width="1100">
</p>

# PrivacyGuard AI — Detect sensitive data | Assess privacy risk | Protect datasets

PrivacyGuard AI is a local-first Streamlit app for privacy-aware dataset transformation, residual-risk review, and audit reporting. It combines automated column classification, policy-aware transformation planning, output validation, and reproducible run artifacts.

In local development, uploaded files are processed on the user's machine. In the deployed demo, files are processed during the Streamlit session and are not intentionally stored in a permanent database or external cloud storage by the app.


## Why this project exists

Data anonymization is often treated as a simple transformation step (e.g., masking or hashing), but in practice it involves trade-offs between data utility, privacy risk, and policy constraints.

Common problems in real workflows:
- No clear distinction between recommended vs applied anonymization actions
- Lack of validation after transformations (false sense of privacy)
- No audit trail for decisions and outputs
- Difficulty aligning technical transformations with policy requirements

This project addresses these gaps by modeling anonymization as a decision pipeline with validation and auditability.




## Live Demo

Use the hosted demo to try PrivacyGuard AI directly in your browser:

**[Open the PrivacyGuard AI Live Demo](https://privacy-guard-insight-h2r4ydel3bzgrynakava3w.streamlit.app/)**

## Demo Dataset

This repository includes fully synthetic demo datasets under `demo_data/`.

The recommended demo file is:

```text
demo_data/privacyguard_ai_demo_synthetic_sensitive_dataset.csv
```

The dataset contains synthetic names, emails, phone numbers, addresses, quasi-identifiers, sensitive attributes, analytical fields, and free-text notes with embedded synthetic PII. It exercises detection, classification, planning, transformation, validation, and audit export.

No real personal data is included.

**To regenerate the main demo dataset locally, run:**

```bash
python scripts/generate_demo_dataset.py

```

## What makes PrivacyGuard AI stand out

- **Policy-aware protection:** recommendations account for intended use, record-linkage needs, and detected field roles.
- **Measured re-identification signal:** validation compares before/after equivalence classes for remaining indirect identifiers and reports how many records fall in groups smaller than `k=5`.
- **Local differential privacy:** the X-Factor Lab can apply generalized randomized response to selected low-cardinality fields, with per-field ε and a composed budget.
- **Private model experiment:** a small adversarial network uses gradient reversal with per-example gradient clipping and secure Gaussian noise; an RDP accountant reports the model-training ε,δ budget.
- **Transparent execution:** requested and applied actions are compared, and residual issues are called out rather than hidden behind a “safe” label.
- **Audit-ready output:** each completed run can include the protected dataset, transformation log, validation summary, and report.

### Privacy-layer scope

The model experiment is a small NumPy implementation, not a PyTorch training pipeline. Its ε,δ accounting applies to model parameters on the sampled training split, conditional on the selected feature schema. Held-out utility and adversary scores are not differentially private. The app does **not** run membership-inference attacks, so no MIA AUC (including `0.50`) is claimed. Randomized response assumes the selected field's category domain is fixed and public. The k-anonymity diagnostic is a risk signal, not a guarantee against linkage using external data or attribute inference.



## Project Overview
This project implements a policy-aware anonymization pipeline, focusing on transparency, reproducibility, and practical data protection workflows. It is designed for managers, analysts and data engineers who need to safely prepare datasets for external sharing, analytics, or machine learning.

The system analyzes datasets, detects potential personal identifiers (PII), recommends anonymization actions based on policy context, applies transformations, validates privacy risks, and generates an audit report for traceability.


## What makes this different

Unlike typical anonymization tools, this system:

- Separates recommendation, execution, and validation stages
- Tracks deviations between suggested and applied protection actions
- Re-evaluates privacy risk after transformations (not assumed safe)
- Produces audit-ready artifacts for traceability and reproducibility


## Key Features

- Policy-aware anonymization planning  
- Automated column classification  
- PII detection using Microsoft Presidio  
- Deterministic pseudonymization with persistent vault  
- Privacy validation checks (residual identifiers, uniqueness risk)  
- Automatic anonymization audit report generation  
- Per-run artifact storage for reproducibility  
- Local-first architecture (no cloud APIs / external services required)


## Example Scenario

A data analyst needs to share a customer dataset externally.

The pipeline:

1. Detects columns such as `email` or `customer_id` as **direct identifiers** / **indirect identifiers**
2. Detects fields like `salary` as **sensitive attributes**
3. Generates an anonymization plan based on policy context
4. Applies transformations such as:
   - pseudonymization for identifiers
   - generalization or noise for sensitive attributes
5. Runs privacy validation checks
6. Generates an anonymization audit report
7. Saves reproducible run artifacts

**direct identifiers**   -> Uniquely identifies a person on its own / Risk for Immediate identification     (name, email)
**indirect identifiers** -> Can identify when combined with others / Risk for Linkage / re-identification   (age, city)
**sensitive attributes** -> Private information about a person / Risk for Disclosure harm (not identity)    (salary, disease)

The output supports more informed sharing decisions, but privacy validation is a risk assessment—not a guarantee that a dataset is safe to share.



## Architecture
The system is structured as a data processing pipeline:

Streamlit UI
|
v
Pipeline Orchestrator
|
v
Profiling -> PII Detection -> Classification
|
v
Policy-Aware Anonymization Planner
|
v
Transformation Engine
|
v
Privacy Validation
|
v
Audit Report Generator
|
v
Run Artifact Storage



## Implementation Approach / Design Decisions

This project was developed using an AI-assisted development workflow, with a strong emphasis on system design, data logic, and validation.

The system architecture, pipeline structure, and anonymization logic were designed end-to-end based on an understanding of real-world data workflows, privacy requirements (GDPR/RODO), and the trade-offs between data utility and privacy risk.

AI-assisted tools (e.g., Codex) were used to accelerate implementation, but the following responsibilities remained manual and deliberate:

- defining the full pipeline structure (profiling → classification → planning → execution → validation → reporting)
- designing anonymization strategies based on data types and policy context
- ensuring alignment between recommended actions, applied transformations, and validation results
- structuring auditability and reproducibility (run artifacts, logs, reports)
- iterating on UI/UX to clearly communicate risk, decisions, and outcomes

All generated code was reviewed, adapted, and validated to ensure correctness, consistency, and alignment with the intended system behavior.

The focus of this project is not on writing code from scratch, but on designing and implementing a reliable, transparent, and audit-ready data processing system.
This approach reflects a system-oriented mindset, focusing on how data workflows behave in practice rather than on isolated code implementation.



## Technology Stack

- Python
- Pandas
- Streamlit
- Microsoft Presidio
- spaCy
- Pytest



## Run Artifacts

Each anonymization execution produces a reproducible run directory:

runs/<run_id>/
|-- anonymized_dataset.csv
|-- transformation_log.json
|-- validation_summary.json
|-- anonymization_report.md
|-- anonymization_report.json
`-- metadata.json


## Example Workflow

1. Upload dataset
2. Profile dataset structure
3. Detect PII signals
4. Classify columns (direct identifier, quasi identifier, etc.)
5. Generate anonymization plan
6. Apply transformations
7. Validate privacy risk
8. Generate anonymization audit report
9. Store run artifacts



## Project Modules

- **profiling/**  
  Dataset profiling and column statistics.

- **anonymization/**  
  Anonymization planning and transformation engine.

- **privacy_metrics/**  
Validation checks for residual identifiers, uniqueness, and k-anonymity signals.

- **reports/**  
  Audit report generation and run artifact persistence.

- **storage/**  
  Persistent pseudonymization vault.

- **config/**  
  Configuration management.




## Persistent Pseudonymization Vault

Deterministic tokenization is supported through a local vault:

data/mapping_vault.json

This ensures:
- consistent token mapping across runs  
- reversible pseudonymization if access to the vault is available  
- isolation by column namespace


## Generated Audit Report (Example)

Each run produces a structured anonymization report including:

- dataset overview  
- anonymization policy context  
- column classification summary  
- anonymization plan  
- transformation log  
- privacy validation results  
- warnings and residual risks



## Quick Start - Installation

Clone repository:

```bash
git clone <repository-url> or ssh

<repository-url>

cd privacyguard-ai
```
Python 3.10+ (recommended 3.11)



## Setup - Create environment / Install dependencies / Install NLP model for Presidio

Python 3.11 is recommended.

The spaCy English model is required by Microsoft Presidio to detect sensitive entities such as names, locations, and organizations.

### Option 1 - Conda environment from environment.yml recommended

```bash
conda env create -f environment.yml
conda activate privacyguard-ai-env
python -m spacy download en_core_web_lg

```

**`Run Tests & Start the App`**
```bash
pytest
streamlit run app.py
```


### Option 2 - Python venv on Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

```
**`Run Tests & Start the App`**
```bash
PYTHONPATH=. pytest
PYTHONPATH=. streamlit run app.py
```


### Option 3 - Python venv on WSL / Linux / macOS 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

```

### Option 3 - Conda environment from environment.yml

```bash
conda env create -f environment.yml
conda activate privacyguard-ai-env

Download the spaCy English model used by Presidio:
python -m spacy download en_core_web_lg

```
**`Run Tests & Start the App`**
```bash
PYTHONPATH=. pytest
PYTHONPATH=. streamlit run app.py
```


### Option 4 - Manual Conda environment
**Use this option only if you do not want to create the environment from environment.yml.**

```bash
conda create -n privacyguard-ai-env python=3.11
conda activate privacyguard-ai-env
pip install -r requirements.txt
python -m spacy download en_core_web_lg

```
**`Run Tests & Start the App`**
```bash
PYTHONPATH=. pytest
PYTHONPATH=. streamlit run app.py
```



## Note

The spaCy model is required for Presidio-based entity detection
(e.g. names, locations, organizations). Install it during setup
using:

```bash
python -m spacy download en_core_web_lg
```


## Run the App

```bash
streamlit run app.py
```


## Test Coverage

The project includes automated tests covering anonymization logic, validation checks, and reporting components.

Run tests with:

pytest



## Use Cases

- preparing datasets for external sharing  
- anonymizing internal datasets for research  
- preparing data for machine learning  
- generating privacy audit documentation



## Core Pipeline Stages

The anonymization workflow consists of the following stages:

1. Dataset ingestion
2. Profiling and column statistics
3. PII signal detection (Presidio)
4. Column classification
5. Policy-aware anonymization planning
6. Transformation execution
7. Privacy validation checks
8. Audit report generation
9. Run artifact persistence



## Supported Actions

- `drop`: remove column from output
- `keep`: leave column unchanged
- `generalize`: reduce precision (numeric bucketing)
- `add_noise`: apply deterministic numeric perturbation
- `pseudonymize`: replace values with deterministic tokens using the persistent vault


## UI Workflow

The app keeps the interface to three stages. Uploading a file alone does not run analysis or show scan results; click **Scan dataset** to start the scan.

1. **Upload & scan:** Upload a dataset, select the policy context, and click **Scan dataset** to view the privacy overview and category chart.
2. **Protect & validate:** Review or adjust recommended protections, apply them, and check residual privacy risk.
3. **Export:** Download the protected dataset and audit report.

## Policy Questionnaire Note

Policy inputs are incorporated into the anonymization planning logic to guide recommended protection strategies and risk thresholds. The current implementation focuses on practical, transparent decision support rather than strict automated enforcement.



## Project Status

This project is currently a **feature-complete v1 anonymization pipeline**.

Future improvements may include:
- stronger privacy metrics (k-anonymity / l-diversity)
- batch processing mode
- REST API interface
- domain-specific anonymization policies



## Security Note

Local artifact directories are intentionally excluded from version control:

data/ - persistent pseudonymization vault  
runs/ - per-run artifact outputs

These files may contain reversible mappings or sensitive metadata and should not be committed to Git repositories.
