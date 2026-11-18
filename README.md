# Predicting Hospital Readmission

An early-stage course project using the UCI Diabetes 130-US Hospitals dataset to study 30-day readmission.

## Reproduce the initial baseline

1. Download the dataset from https://doi.org/10.24432/C5230J and extract `diabetic_data.csv` into `data/`.
2. Create a Python environment and install `pip install -r requirements.txt`.
3. Run `python src/baseline.py`. The script writes summary statistics and preliminary holdout metrics to `results/baseline.json`.

The split groups encounters by patient. Preprocessing is fitted on the training data only. The preliminary logistic regression uses a limited set of encounter variables; it is not the planned full model comparison. This historical dataset and a single random patient split do not establish clinical validity. Results at a 0.5 threshold should not be interpreted as an optimized clinical decision rule.
