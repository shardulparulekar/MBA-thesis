# Explaining and Predicting State-Level Variation in UPI Adoption Intensity Across India

Code and data for the Master's thesis of the same name (Amsterdam Business School, University of Amsterdam).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shardulparulekar/MBA-thesis/HEAD?labpath=run_pipeline.ipynb)

Click the badge above to run the full pipeline in your browser, no installation needed. Takes a minute or two to launch, then a few minutes to run.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21905716.svg)](https://doi.org/10.5281/zenodo.21905716)

The DOI badge above is a permanent, citable snapshot of this exact code and data, archived by Zenodo. Use it when citing this repository; the live GitHub link may change, but this reference won't. 

## What this is

A two-stage analysis of UPI (Unified Payments Interface) adoption intensity across 36 Indian states and union territories, April 2023 to December 2025.

- **Stage 1 (explanatory):** between-state, pooled, and within-state regressions testing six candidate predictors, using a state-month panel.
- **Stage 2 (predictive):** Ridge regression vs. Random Forest, compared via leave-one-state-out cross-validation on the four predictors Stage 1 confirms.

Full methodology and results are in the thesis itself.

## Running it yourself

**Option 1 — Binder (easiest):** click the badge above.

**Option 2 — locally:**
```bash
git clone https://github.com/shardulparulekar/MBA-thesis.git
cd MBA-thesis
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

> Shardul Parulekar. (2026). *Explaining and Predicting State-Level Variation in UPI Adoption Intensity Across India.* Master's thesis, Amsterdam Business School, University of Amsterdam.

## License

## License

Code (`full_pipeline_v2.py`, `run_pipeline.ipynb`) is released under the MIT License (see `LICENSE`).

Data files are compiled from publicly available government sources (NPCI, RBI, TRAI, Ministry of Health and Family Welfare, GST Council, PLFS, MoSPI), accessed via Dataful — see the Data section above for original sources. They are shared here for reproducibility of this thesis; usage of the underlying data remains subject to the original sources' own terms.
