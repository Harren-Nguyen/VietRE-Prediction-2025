# Vietnam Real Estate Price Prediction

A small machine learning project that estimates residential property listing prices in Vietnam from basic property and location information. The current version uses **HistGradientBoosting (HistGB)** as its main model and compares it with simpler baselines.

> **This is an experimental project, not a professional valuation tool.** Predictions are estimates based on the available data and should not replace independent valuation or legal checks.

## Contents

- [1. Overview](#1-overview)
- [2. Dataset and Features](#2-dataset-and-features)
- [3. How the Model Works](#3-how-the-model-works)
- [4. Evaluation and Results](#4-evaluation-and-results)
- [5. Installation and Usage](#5-installation-and-usage)
- [6. Project Structure and Outputs](#6-project-structure-and-outputs)
- [7. Limitations and Future Improvements](#7-limitations-and-future-improvements)

## 1. Overview

The goal is to build a practical regression pipeline that estimates a property's price from a small set of structural and geographical features.

The project focuses on three things:

- Building a complete workflow from raw listing data to price prediction.
- Comparing HistGB with simple statistical and linear baselines.
- Giving users a basic indication of prediction uncertainty and whether an input is outside the data the model has seen.

The target variable is `price`, measured in **billion VND**.

## 2. Dataset and Features

Place the dataset (after being renamed as below) at:

```text
dataset/vietnam_real_estate_refined_v4.csv
```

The CSV is expected to contain six columns:

```csv
address, area, floors, bedrooms, bathrooms, price
```

The address is split into three location features, giving the model seven inputs:

| Feature | Description |
|---|---|
| `area` | Property area (m²) |
| `floors` | Number of floors |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `street` | Street extracted from the address |
| `ward` | Ward/commune extracted from the address |
| `province` | Province/city extracted from the address |

The preprocessing code converts numerical fields, removes records with missing or invalid values, applies configured range checks, normalizes address components, and removes duplicates. These are project-specific cleaning rules; see `src/features.py` for the exact implementation.

The development dataset snapshot contained 161,632 raw rows before cleaning. The final usable row count is printed during training and may change when the dataset is updated.

## 3. How the Model Works

### Price per square metre

Instead of predicting total price directly, the model learns log price per square metre. If `y` is the total price in billion VND and `A` is area in m²:

\[
r = \frac{1000y}{A}, \qquad z = \log(r)
\]

The model predicts `z`, then converts it back to total price:

\[
\hat{y} = \frac{A\exp(f(X))}{1000}
\]

This lets the model learn price intensity per square metre while returning an estimated total price.

### Location encoding and HistGB

The `street`, `ward`, and `province` features are categorical. The pipeline uses scikit-learn's **TargetEncoder** to represent these categories using smoothed target statistics. It is included inside the training pipeline so encoding is fitted within each training fold.

The main estimator is **HistGradientBoostingRegressor** with absolute-error loss. It builds decision trees sequentially, with each new tree improving the current predictions. Histogram-based splitting makes it suitable for larger tabular datasets.

The **baseline** configuration in the current source is:

| Parameter | Value |
|---|---:|
| `loss` | `absolute_error` |
| `learning_rate` | 0.05 |
| `max_iter` | 1,200 |
| `max_leaf_nodes` | 63 |
| `min_samples_leaf` | 40 |
| `l2_regularization` | 1.0 |

When tuning is enabled, the search can select a different configuration. Ridge with one-hot encoding is kept as a comparison baseline, not combined with HistGB.

### Prediction intervals and data-support checks

The project uses out-of-fold predictions from the training data to estimate multiplicative prediction intervals. In simplified form, an interval around prediction \(\hat y\) is:

\[
[L,U] = [\hat y\exp(-q),\;\hat y\exp(q)]
\]

The model also checks whether numerical inputs or location categories are poorly represented in the training data and can show warnings. These checks are intended as guidance, not a guarantee that a prediction is reliable.

The current source also contains a limited heuristic adjustment for unusually high floor and room counts. This should be treated cautiously because tree-based models are not naturally designed for reliable extrapolation.

## 4. Evaluation and Results

The main metrics are:

**Median Absolute Percentage Error (MedAPE)** — lower is better.

\[
MedAPE = 100\times median\left(\left|\frac{\hat y_i-y_i}{y_i}\right|\right)
\]

**Within20** — percentage of predictions whose absolute percentage error is no greater than 20%; higher is better.

\[
Within20 = \frac{100}{N}\sum_{i=1}^{N}\mathbb{1}\left(\left|\frac{\hat y_i-y_i}{y_i}\right|\leq0.20\right)
\]

The project also reports prediction-interval coverage: the percentage of holdout prices that fall inside the estimated interval. Coverage should be considered together with interval width.

### Experiment setup

The current training configuration uses a 20% holdout set, 3-fold cross-validation, and random seed 42. Hyperparameter tuning is controlled by `RUN_TUNING` in `src/train.py`. The holdout set is reserved for final evaluation rather than tuning.

### Current Model Training Results


| Model | Holdout MedAPE (%) ↓ | Holdout Within20 (%) ↑ |
|---|---:|---:|
| Median-price baseline | 45.496 | 27.127 |
| Median-price-per-m² baseline | 36.099 | 36.099 |
| Ridge + one-hot | 20.532 | 48.971 |
| **HistGB** | **15.495** | **59.612** |

Cross-validation results are saved in `reports/cv_results.csv`; holdout metrics and interval coverage are saved in `models/meta.json`.

## 5. Installation and Usage

### Requirements

- Python 3.10.11 (the original development environment)
- NumPy, pandas, SciPy, scikit-learn, Matplotlib, and Joblib

Install the dependencies listed in `requirements.txt`. For reproducible experiments, record the exact package versions used in your environment.

### Setup (Git Bash)

```bash
git clone https://github.com/Harren-Nguyen/VietRE-Prediction-2025.git
cd VietRE-Predict-2025
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Put the CSV file in the `dataset/` directory, then run commands from the repository root.

### Train

```bash
python -m src.train
```

To disable tuning, set `RUN_TUNING = False` in `src/train.py`. The number of search iterations can be changed through `TUNE_ITERATIONS`.

### Predict

Run the built-in examples:

```bash
python -m src.predict
```

Or provide a property directly:

```bash
python -m src.predict --address "Street, Ward, Province" --area 60 --floors 3 --bedrooms 3 --bathrooms 3
```

The output includes an estimated price, prediction intervals, and warnings when the input has limited support in the training data.

## 6. Project Structure and Outputs

```text
VietRE-Predict-2025/
├── dataset/
│   └── vietnam_real_estate_refined_v4.csv
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── models/       # generated model files
├── reports/      # generated evaluation reports
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

Training generates the following files:

| Location | File | Purpose |
|---|---|---|
| `models/` | `house_price_model.joblib` | Trained model |
| `models/` | `meta.json` | Metrics, calibration information, and model metadata |
| `models/` | `reference.json` | Training-data reference used by support/OOD checks |
| `reports/` | `eda.png` | Exploratory data plots |
| `reports/` | `cv_results.csv` | Cross-validation comparison |
| `reports/` | `error_analysis.png` | Prediction error plots |
| `reports/` | `worst_predictions.csv` | Largest holdout prediction errors |
| `reports/` | `feature_importance.csv`, `feature_importance.png` | Permutation feature importance |

## 7. Limitations and Future Improvements

This project is an early-stage experiment and has several limitations:

- **Limited inputs:** Important factors such as exact coordinates, frontage, alley width, legal status, building condition, and nearby amenities are not included.
- **Listing prices:** If the dataset contains asking prices, predictions reflect asking-price patterns rather than verified transaction values.
- **Generalization:** A random holdout split does not show how well the model performs in entirely unseen areas or future market conditions.
- **Uncertainty:** Prediction intervals and data-support warnings are useful indicators, but they do not guarantee that an individual estimate is accurate.
- **Extrapolation:** The heuristic adjustment for unusually high floor and room counts requires further validation.

Possible next steps include adding better location features, testing spatial or time-based splits, and evaluating the model on independently verified data.

## References

- [scikit-learn: HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [scikit-learn: TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- [scikit-learn: RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

---

