# Explaining and Predicting State-Level Variation in UPI Adoption Intensity Across India

Code and data for the Master's thesis of the same name (Amsterdam Business School, University of Amsterdam).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME/HEAD?labpath=run_pipeline.ipynb)

Click the badge above to run the full pipeline in your browser, no installation needed. Takes a minute or two to launch, then a few minutes to run.

## What this is

A two-stage analysis of UPI (Unified Payments Interface) adoption intensity across 36 Indian states and union territories, April 2023 to December 2025.

- **Stage 1 (explanatory):** between-state, pooled, and within-state regressions testing six candidate predictors, using a state-month panel.
- **Stage 2 (predictive):** Ridge regression vs. Random Forest, compared via leave-one-state-out cross-validation on the four predictors Stage 1 confirms.

Full methodology and results are in the thesis itself.

## Running it yourself

**Option 1 — Binder (easiest):** click the badge above.

**Option 2 — locally:**
```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
pip install -r requirements.txt
python full_pipeline_v2.py
```

Both produce identical output: every table and figure in the thesis, plus exported CSVs (`table_3_2_summary_stats.csv`, `stage2_predictions.csv`, `ridge_alpha_sensitivity.csv`, and others).

## Data

All source files are included in this repo (~11MB total), so no separate download is needed. Original sources and dataset IDs:

| File | Source |
|---|---|
| `Ecosystem-Statistics-UPI-*.xlsx` (33 files) | NPCI, [State-wise UPI Product Statistics](https://www.npci.org.in/product/ecosystem-statistics/upi) |
| `Digital-NFS-data.csv` | NPCI, via [Dataful](https://dataful.in/datasets/336/) |
| `gst-collection-data.csv` | GST Council, via [Dataful](https://dataful.in/datasets/15167/) |
| `ATM-data.csv` | Reserve Bank of India, via [Dataful](https://dataful.in/datasets/131/) |
| `Internet-data.csv` | Telecom Regulatory Authority of India, via [Dataful](https://dataful.in/datasets/19277/) |
| `Population-data.csv` | Ministry of Health and Family Welfare, via [Dataful](https://dataful.in/datasets/18521/) |
| `Urbanisation-data.csv` | Ministry of Health and Family Welfare, via [Dataful](https://dataful.in/datasets/18520/) |
| `unemployment-data.csv` | Periodic Labour Force Survey, via [Dataful](https://dataful.in/datasets/20533/) |
| `gross-state-value-data.csv` | Ministry of Statistics and Programme Implementation, via [Dataful](https://dataful.in/datasets/21439/) |

## Reproducibility

- All Random Forest models use a fixed random seed (`random_state=42`), so results are exactly repeatable.
- Package versions are pinned in `requirements.txt`.
- The full run is deterministic: identical input files always produce identical output.

## Citation

If you use this code or data, please cite the thesis:

> [Your name]. (2026). *Explaining and Predicting State-Level Variation in UPI Adoption Intensity Across India.* Master's thesis, Amsterdam Business School, University of Amsterdam.

## License

[Choose a license, e.g. MIT for code, CC-BY for the thesis text, before publishing]
