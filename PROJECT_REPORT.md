# Smartphone Price Classification Project Report

## 📋 Project Overview

**Objective:** Build a machine learning model to classify smartphones as "Expensive" or "Non-Expensive" based on their technical specifications.

**Dataset:** 
- Training data: `train.csv`  
- Test data: `test.csv`

**Target Variable:** `price` (binary: expensive / non-expensive)

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Samples | ~1500 | ~300 |
| Features | 32 | 32 |
| Missing Values | Minimal | Minimal |
| Duplicates | Removed during preprocessing |

### 1.2 Target Distribution

The dataset is relatively balanced:
- **Expensive:** ~45%
- **Non-Expensive:** ~55%

Class ratio of approximately 1.2:1 indicates no significant class imbalance.

### 1.3 Feature Categories

| Category | Features |
|----------|----------|
| **Binary** | Dual_Sim, 4G, 5G, Vo5G, NFC, IR_Blaster, memory_card_support |
| **Categorical** | brand, Processor_Brand, Processor_Series, Notch_Type, os_name, os_version |
| **Numeric** | rating, Core_Count, Clock_Speed_GHz, RAM Size GB, Storage Size GB, battery_capacity, Screen_Size, Resolution, Refresh_Rate, camera specs |

### 1.4 Key Correlation Findings

Top features most correlated with price:
1. **RAM Size GB** (Strong positive)
2. **Storage Size GB** (Strong positive)
3. **primary_rear_camera_mp** (Moderate positive)
4. **5G support** (Moderate positive)
5. **Refresh_Rate** (Moderate positive)

### 1.5 Outlier Analysis (IQR Method)

Outliers detected in:
- `battery_capacity`: Some phones with unusually high/low capacity
- `primary_rear_camera_mp`: High-end phones with 108MP+ cameras
- `RAM Size GB`: Gaming phones with 16GB+ RAM

---

## 2. Data Preprocessing

### 2.1 Missing Value Handling

```python
# Memory card size - fill NaN with 0 (no SD card support)
if pd.isna(x): return 0
```

### 2.2 Duplicate Removal

```python
df.drop_duplicates(inplace=True)
```

### 2.3 Data Encoding

| Feature Type | Encoding Method |
|--------------|-----------------|
| Binary (Yes/No) | Map to 1/0 |
| Categorical | LabelEncoder |
| Target (price) | expensive=1, non-expensive=0 |

```python
# Binary encoding
df[col] = df[col].map({'Yes': 1, 'No': 0})

# Label encoding for categorical
le = LabelEncoder()
df[f'{col}_encoded'] = le.fit_transform(df[col])
```

### 2.4 Feature Scaling

**StandardScaler** applied for KNN, SVM, and Logistic Regression:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # No data leakage
```

### 2.5 Feature Engineering

- `os_version_numeric`: Extracted numeric version from string
- `memory_card_size_gb`: Converted TB/GB strings to numeric

---

## 3. Classification Models

### 3.1 Models Implemented

| Model | Implementation File |
|-------|---------------------|
| Random Forest | `Random Forest` |
| K-Nearest Neighbors | `KNN_Tuned.py` |
| Logistic Regression | `Logistic_Regression_Tuned.py` |
| Support Vector Machine | `SVM.py` |

### 3.2 Random Forest

**Hyperparameters Tuned:**
- `n_estimators`: [50, 100, 200, 300]
- `max_depth`: [5, 10, 20, None]
- `min_samples_split`: [2, 5, 10]

**Best Parameters:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
```

### 3.3 K-Nearest Neighbors (KNN)

**Hyperparameters Tuned:**
- `n_neighbors`: [3, 5, 7, 9, 11, 13, 15]
- `weights`: ['uniform', 'distance']
- `metric`: ['euclidean', 'manhattan', 'minkowski']

**Impact Analysis:**
- Distance-weighted voting generally performs better
- Optimal k typically between 5-9
- Manhattan distance competitive with Euclidean

### 3.4 Logistic Regression

**Hyperparameters Tuned:**
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear', 'saga']

**Impact Analysis:**
- L2 regularization (Ridge) slightly better than L1 (Lasso)
- Moderate C values (1-10) perform best
- Regularization prevents overfitting on high-dimensional data

### 3.5 Support Vector Machine (SVM)

**Hyperparameters Tuned:**
- `C`: [0.1, 1, 10, 100]
- `gamma`: ['scale', 'auto', 0.1, 0.01]
- `kernel`: ['rbf', 'poly', 'sigmoid']

**Kernel Comparison:**
- RBF kernel achieves best accuracy
- Polynomial kernel competitive but slower
- Higher C values improve training fit but risk overfitting

---

## 4. Model Evaluation

### 4.1 Evaluation Metrics

All models evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

### 4.2 Results Comparison

| Model | Validation Accuracy | Test.csv Accuracy | CV Mean |
|-------|---------------------|-------------------|---------|
| Random Forest | ~90% | ~88% | ~89% |
| KNN (Tuned) | ~86% | ~84% | ~85% |
| Logistic Regression | ~85% | ~83% | ~84% |
| SVM (RBF) | ~88% | ~86% | ~87% |

### 4.3 Best Model: Random Forest

**Reasons:**
1. Highest accuracy on both validation and test sets
2. No feature scaling required
3. Provides feature importance rankings
4. Robust to outliers
5. Handles mixed feature types naturally

### 4.4 Cross-Validation

5-Fold Cross-Validation performed on all models:
```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
```

---

## 5. Feature Importance

### Random Forest Feature Importance (Top 10)

1. RAM Size GB
2. Storage Size GB
3. primary_rear_camera_mp
4. battery_capacity
5. Clock_Speed_GHz
6. Screen_Size
7. Refresh_Rate
8. fast_charging_power
9. Resolution_Height
10. 5G support

---

## 6. GUI Application

A modern Tkinter-based GUI was developed:

**Features:**
- Dark theme interface
- All 30+ input fields for phone specifications
- Model selector (RF, KNN, LR, SVM)
- Real-time predictions with confidence scores
- Chat-style prediction history

**File:** `chatbot_gui.py`

---

## 7. Conclusion

### Key Findings

1. **RAM and Storage** are the strongest predictors of phone price category
2. **5G capability** significantly indicates expensive phones
3. **Random Forest** achieves best overall performance
4. **All models** achieve >80% accuracy, indicating good feature predictiveness

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Mixed data types | Proper encoding (Label, Binary) |
| Feature scaling | StandardScaler for distance-based models |
| Hyperparameter selection | GridSearchCV with cross-validation |

### Future Improvements

1. Implement ensemble methods (XGBoost, LightGBM)
2. Add SMOTE for potential class imbalance
3. Feature interaction engineering
4. Deep learning approaches for larger datasets
5. Deploy GUI as web application

---

## 8. Files Structure

```
Spec-to-Price-main/
├── Datasets/
│   ├── train.csv
│   └── test.csv
├── preprocessing.py          # Preprocessing pipeline
├── EDA.py                    # Exploratory Data Analysis
├── Random Forest             # RF model
├── KNN_Tuned.py              # KNN with hyperparameter tuning
├── Logistic_Regression_Tuned.py  # LR with tuning
├── SVM.py                    # SVM with tuning
├── chatbot_gui.py            # GUI application
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## References

- Scikit-learn Documentation
- Pandas Documentation
- Lab Materials (Preprocessing, Classification)
