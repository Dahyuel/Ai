# Smartphone Price Classification

A machine learning project that classifies smartphones as **Expensive** or **Non-Expensive** based on their technical specifications. Includes multiple ML models with hyperparameter tuning, exploratory data analysis, and interactive GUI applications.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter%20%7C%20CustomTkinter-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Input Fields](#input-fields)
- [How It Works](#how-it-works)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This project builds a binary classifier that predicts whether a smartphone falls into the **Expensive** or **Non-Expensive** category from its hardware and software specifications. It ships with a data preprocessing pipeline, four tuned classification models, an EDA module that generates eight visualizations, model persistence utilities, and two GUI front-ends (a chat-style assistant and a form-based predictor).

---

## Features

- **4 ML Models** — Random Forest, K-Nearest Neighbors, Logistic Regression, and SVM, each with GridSearchCV tuning.
- **Reusable Preprocessing Pipeline** — `preprocess()` handles deduplication, binary encoding, label encoding, numeric extraction, and scaling (`preprocessing.py:50`).
- **EDA Visualizations** — Eight plots covering target distribution, correlations, feature distributions, boxplots, and outliers (`EDA.py`).
- **Model Persistence** — Trains and serializes all models plus their scalers/encoders with joblib (`save_models.py`).
- **Two GUI Applications** — A Tkinter chatbot (`chatbot_gui.py`) and a CustomTkinter assistant (`modern_gui.py`).
- **Verification Scripts** — Quick model sanity checks and a full end-to-end verification (`test_model.py`, `verify_models.py`).

---

## Models

| Model | File | Reported Accuracy | Hyperparameters Tuned |
|-------|------|-------------------|-----------------------|
| **Random Forest** | `Random Forest` | ~89% | `n_estimators`, `max_depth`, `min_samples_split` |
| **KNN** | `KNN_Tuned.py` | ~85% | `n_neighbors`, `weights`, `metric` |
| **Logistic Regression** | `Logistic_Regression_Tuned.py` | ~85% | `C`, `penalty`, `solver` |
| **SVM** | `SVM.py` | ~89% | `C`, `gamma`, `kernel` |

A shared `ModelHandler` class in `models.py:11` exposes `train_knn`, `train_logistic_regression`, and `train_random_forest` methods with a consistent train/evaluate interface.

---

## Project Structure

```
Spec-to-Price-main/
├── chatbot_gui.py                 # Tkinter chat-style GUI with model selector
├── modern_gui.py                  # CustomTkinter assistant (dark theme)
├── gui.py                         # Basic Tkinter interface
├── models.py                      # ModelHandler class (train/evaluate/predict)
├── preprocessing.py               # Data cleaning + encoding pipeline
├── EDA.py                         # Exploratory data analysis (8 plots)
│
├── # Machine Learning Models
├── Random Forest                  # Random Forest classifier
├── KNN_Tuned.py                   # KNN with GridSearchCV
├── Logistic_Regression_Tuned.py   # LR with GridSearchCV
├── SVM.py                         # SVM with GridSearchCV
│
├── # Utilities
├── save_models.py                 # Persist trained models to disk
├── test_model.py                  # Quick single-model test
├── verify_models.py               # End-to-end verification
├── run_chatbot.bat                # Windows launcher (modern_gui.py)
├── main                           # XOR access-code snippet
│
├── Datasets/
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
│
├── eda_plots/                     # Generated EDA visualizations
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_correlations.png
│   ├── 04_feature_distributions.png
│   ├── 05_boxplots_by_price.png
│   ├── 06_brand_distribution.png
│   ├── 07_binary_features_by_price.png
│   └── 08_outlier_boxplots.png
│
├── PROJECT_REPORT.md              # Detailed analysis and methodology
├── requirements.txt
└── README.md
```

`saved_models/` is created at runtime by `save_models.py` and is not tracked in the repository.

---

## Prerequisites

- Python 3.8 or newer
- `pip`
- Tkinter (bundled with most Python installs; see [Troubleshooting](#troubleshooting))

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Spec-to-Price-main
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

Optional packages used by specific scripts:

```bash
pip install matplotlib seaborn      # required by EDA.py
pip install joblib                  # required by save_models.py
pip install customtkinter           # required by modern_gui.py
```

### 4. Verify the install

```bash
python -c "import pandas, sklearn, numpy; print('Ready!')"
```

---

## Usage

### Launch a GUI

```bash
python chatbot_gui.py     # Tkinter chat-style interface
python modern_gui.py      # CustomTkinter assistant (dark theme)
python gui.py             # Basic form interface
```

On Windows you can also double-click `run_chatbot.bat` (edit the hard-coded interpreter path first).

### Run the EDA

```bash
python EDA.py
```

Generates the eight PNG files into `eda_plots/`.

### Train and evaluate individual models

```bash
python "Random Forest"              # Random Forest
python KNN_Tuned.py                 # KNN + GridSearchCV
python Logistic_Regression_Tuned.py # Logistic Regression + GridSearchCV
python SVM.py                       # SVM + GridSearchCV
```

### Save trained models

```bash
python save_models.py
```

Writes to `saved_models/`:

```
label_encoders.joblib
feature_cols.joblib
random_forest.joblib
knn.joblib + knn_scaler.joblib
logistic_regression.joblib + lr_scaler.joblib
svm.joblib + svm_scaler.joblib
```

Load them later with:

```python
import joblib

model = joblib.load("saved_models/random_forest.joblib")
encoders = joblib.load("saved_models/label_encoders.joblib")
```

### Verify everything works

```bash
python test_model.py        # quick Random Forest sanity check
python verify_models.py     # exercises ModelHandler end-to-end
```

---

## Input Fields

### Binary options

| Field | Description |
|-------|-------------|
| Dual SIM | Dual SIM support |
| 4G / 5G | Network connectivity |
| Vo5G | Voice over 5G |
| NFC | Near Field Communication |
| IR Blaster | Infrared blaster |
| Memory Card | SD card support |

### Numeric specifications

| Field | Example |
|-------|---------|
| RAM | 8 GB |
| Storage | 256 GB |
| Battery | 5000 mAh |
| Screen Size | 6.5 inches |
| Resolution | 2400 x 1080 |
| Refresh Rate | 120 Hz |
| Camera MP | 48 MP |
| Core Count | 8 |
| Clock Speed | 2.8 GHz |
| Fast Charging | 65 W |

### Categorical

| Field | Example Values |
|-------|----------------|
| Brand | Samsung, Apple, Xiaomi, OnePlus |
| Processor Brand | Snapdragon, MediaTek, Exynos |
| Processor Series | 8 Gen 2, Dimensity 9200 |
| Notch Type | Punch Hole, Water Drop, None |
| OS | Android, iOS |

---

## How It Works

1. **Preprocessing** (`preprocessing.py`)
   - Removes duplicates and unwanted tier columns.
   - Converts Yes/No binaries to `1`/`0`.
   - Label-encodes categorical columns, storing encoders for reuse on test data.
   - Extracts numeric values from strings (e.g. `"8 GB"` → `8`, `"1 TB"` → `1024`).
   - Fills missing memory-card sizes with `0`.

2. **Model training**
   - 80/20 train/test split with `random_state=42`.
   - `StandardScaler` fitted on the training set only for KNN, LR, and SVM.
   - `GridSearchCV` with 5-fold cross-validation for hyperparameter search.

3. **Evaluation**
   - Accuracy, precision, recall, F1-score, and confusion matrix.
   - Feature-importance ranking from Random Forest.
   - Cross-validated scores for every model.

---

## Results

### Model comparison

| Model | Validation Accuracy | Test Accuracy | CV Mean |
|-------|---------------------|---------------|---------|
| **Random Forest** | ~90% | ~88% | ~89% |
| SVM (RBF) | ~88% | ~86% | ~87% |
| KNN (Tuned) | ~86% | ~84% | ~85% |
| Logistic Regression | ~85% | ~83% | ~84% |

### Key findings

- **RAM size** and **storage size** are the strongest predictors of price category.
- **5G capability** and **NFC** correlate strongly with premium pricing.
- **Random Forest** gives the best accuracy, needs no feature scaling, and provides interpretable feature importances.
- All four models exceed 80% accuracy, confirming the features are highly predictive.

Top Random Forest features: `RAM Size GB`, `Storage Size GB`, `primary_rear_camera_mp`, `battery_capacity`, `Clock_Speed_GHz`.

See `PROJECT_REPORT.md` for the full analysis.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No module named 'customtkinter'` | `pip install customtkinter` (only needed for `modern_gui.py`) |
| `No module named 'joblib'` | `pip install joblib` |
| `No module named 'matplotlib'` / `seaborn` | `pip install matplotlib seaborn` |
| GUI does not open | Install Tkinter (see below) |
| `FileNotFoundError: Datasets/train.csv` | Run scripts from the project root |
| Plots not showing | Ensure `eda_plots/` is writable |

### Platform-specific Tkinter install

**Windows / macOS:** Tkinter ships with the standard Python installer.

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install python3-tk
```

**Linux (Fedora):**

```bash
sudo dnf install python3-tkinter
```

---

## Roadmap

- [ ] Add ensemble methods (XGBoost, LightGBM).
- [ ] Handle class imbalance with SMOTE.
- [ ] Engineer feature interactions.
- [ ] Evaluate deep learning for larger datasets.
- [ ] Deploy the GUI as a web application.

---

## Documentation

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — full methodology, EDA findings, and model analysis.
- `eda_plots/` — all generated visualizations.
- [`requirements.txt`](requirements.txt) — runtime dependencies.

---

## License

Released under the MIT License. Add a `LICENSE` file to the repository root if one is not already present.

---

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/).
- GUI built with [Tkinter](https://docs.python.org/3/library/tkinter.html) and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).

---

Made with Python, scikit-learn, and Tkinter.