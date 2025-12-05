# 📱 Smartphone Price Classification Project

A machine learning project that classifies smartphones as **Expensive** or **Non-Expensive** based on their specifications. Includes a modern GUI chatbot, 4 ML models with hyperparameter tuning, and comprehensive EDA.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20|%20KNN%20|%20LR%20|%20SVM-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Input Fields](#-input-fields)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- 🎨 **Modern Dark Theme GUI** - Sleek chatbot interface
- 🧠 **4 ML Models** - Random Forest, KNN, Logistic Regression, SVM
- 📊 **Hyperparameter Tuning** - GridSearchCV for all models
- � **EDA Visualizations** - 8 plots for data analysis
- � **Model Persistence** - Save/load trained models
- � **Comprehensive Report** - Full project documentation

---

## 🤖 Models

| Model | File | Accuracy | Hyperparameters Tuned |
|-------|------|----------|----------------------|
| **Random Forest** | `Random Forest` | ~89% | n_estimators, max_depth, min_samples_split |
| **KNN** | `KNN_Tuned.py` | ~85% | n_neighbors, weights, metric |
| **Logistic Regression** | `Logistic_Regression_Tuned.py` | ~85% | C, penalty, solver |
| **SVM** | `SVM.py` | ~89% | C, gamma, kernel |

---

## 📁 Project Structure

```
Spec-to-Price-main/
├── chatbot_gui.py              # 🎮 Modern GUI with model selector
├── preprocessing.py            # 🔧 Data preprocessing pipeline
│
├── # Machine Learning Models
├── Random Forest               # RF classifier
├── KNN_Tuned.py                # KNN with GridSearchCV
├── Logistic_Regression_Tuned.py # LR with GridSearchCV
├── SVM.py                      # SVM with GridSearchCV
│
├── # Analysis & Utilities
├── EDA.py                      # 📊 Exploratory Data Analysis
├── save_models.py              # 💾 Save trained models
├── PROJECT_REPORT.md           # 📝 Detailed project report
│
├── Datasets/
│   ├── train.csv               # Training data (867 samples)
│   └── test.csv                # Test data (153 samples)
│
├── eda_plots/                  # 📈 Generated EDA visualizations
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_correlations.png
│   ├── 04_feature_distributions.png
│   ├── 05_boxplots_by_price.png
│   ├── 06_brand_distribution.png
│   ├── 07_binary_features_by_price.png
│   └── 08_outlier_boxplots.png
│
├── saved_models/               # 💾 Saved model files (after running save_models.py)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn
```

### Step 2: Verify Installation

```bash
python -c "import pandas; import sklearn; print('Ready!')"
```

---

## 🎮 Usage

### Launch GUI (Recommended)
```bash
python chatbot_gui.py
```
- Select model from dropdown (RF, KNN, LR, SVM)
- Fill in phone specifications
- Click "Predict Price Category"

### Run EDA Analysis
```bash
python EDA.py
```
Generates 8 visualizations in `eda_plots/`

### Run Individual Models
```bash
python SVM.py                      # SVM with hyperparameter tuning
python KNN_Tuned.py                # KNN with tuning
python Logistic_Regression_Tuned.py # LR with tuning
```

### Save Trained Models
```bash
python save_models.py
```
Saves all models to `saved_models/` directory

---

## 📝 Input Fields

### Binary Options
| Field | Description |
|-------|-------------|
| Dual SIM | Dual SIM support |
| 4G / 5G | Network connectivity |
| NFC | Near Field Communication |
| IR Blaster | Infrared blaster |
| Memory Card | SD card support |

### Numeric Specifications
| Field | Example |
|-------|---------|
| RAM | 8 GB |
| Storage | 256 GB |
| Battery | 5000 mAh |
| Screen Size | 6.5 inches |
| Refresh Rate | 120 Hz |
| Camera MP | 48 MP |

### Categorical
| Field | Example Values |
|-------|----------------|
| Brand | Samsung, Apple, Xiaomi, OnePlus |
| Processor | Snapdragon, MediaTek, Exynos |
| Notch Type | Punch Hole, Water Drop, None |
| OS | Android, iOS |

---

## ⚙️ How It Works

1. **Preprocessing** (`preprocessing.py`)
   - Remove duplicates
   - Encode categorical variables (LabelEncoder)
   - Convert binary Yes/No to 1/0
   - Extract numeric values from strings

2. **Model Training**
   - Train/test split (80/20)
   - StandardScaler for KNN, LR, SVM
   - GridSearchCV for hyperparameter tuning
   - 5-fold cross-validation

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Feature Importance

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| GUI doesn't open | Install tkinter: `pip install tk` |
| Model loading fails | Check `Datasets/train.csv` exists |
| Plots not showing | Install matplotlib: `pip install matplotlib` |

### Platform-Specific

**Windows:**
```bash
pip install tk
```

**Linux:**
```bash
sudo apt-get install python3-tk
```

**macOS:** tkinter comes with Python

---

## 📊 Key Findings

- **RAM** and **Storage** are the strongest price predictors
- **5G capability** indicates expensive phones
- **Random Forest** and **SVM** achieve best accuracy (~89%)
- NFC presence strongly correlates with premium pricing

---

## 📄 Documentation

For detailed analysis, see:
- `PROJECT_REPORT.md` - Full project documentation
- `eda_plots/` - All EDA visualizations

---

Made with ❤️ using Python, Scikit-learn & Tkinter
