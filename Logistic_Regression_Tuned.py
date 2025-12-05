"""
Logistic Regression Classification Model with Full Hyperparameter Tuning
Smartphone Price Category Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("LOGISTIC REGRESSION CLASSIFICATION MODEL")
print("="*60)

# ============================================
# 1. LOAD AND PREPROCESS DATA
# ============================================

print("\n[1] Loading and preprocessing data...")

df = pd.read_csv('Datasets/train.csv')
df_processed, encoders = preprocess(df, fit=True)

feature_cols = [
    col for col in df_processed.columns 
    if col not in ['price', 'price_encoded'] and df_processed[col].dtype != 'object'
]

X = df_processed[feature_cols]
y = df_processed['price_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")
print(f"    Features: {len(feature_cols)}")

# ============================================
# 2. BASELINE MODEL
# ============================================

print("\n" + "="*60)
print("[2] BASELINE LOGISTIC REGRESSION")
print("="*60)

lr_baseline = LogisticRegression(max_iter=300, random_state=42)
lr_baseline.fit(X_train_scaled, y_train)

pred_baseline = lr_baseline.predict(X_test_scaled)
acc_baseline = accuracy_score(y_test, pred_baseline)

print(f"\nBaseline Accuracy: {acc_baseline:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_baseline))

# ============================================
# 3. HYPERPARAMETER TUNING
# ============================================

print("\n" + "="*60)
print("[3] HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*60)

# Define comprehensive parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],           # Regularization strength
    'penalty': ['l1', 'l2'],                        # Regularization type
    'solver': ['liblinear', 'saga'],                # Optimization algorithm
    'max_iter': [300, 500]                          # Maximum iterations
}

print("\nParameter grid:")
for param, values in param_grid.items():
    print(f"    {param}: {values}")

print("\nStarting Grid Search...")

# Grid Search
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Results
print("\n" + "-"*40)
print("GRID SEARCH RESULTS:")
print("-"*40)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Best model
best_lr = grid_search.best_estimator_
pred_tuned = best_lr.predict(X_test_scaled)
acc_tuned = accuracy_score(y_test, pred_tuned)

print(f"\nTuned Validation Accuracy: {acc_tuned:.4f}")
print(f"Improvement: {(acc_tuned - acc_baseline)*100:.2f}%")

# ============================================
# 4. HYPERPARAMETER IMPACT ANALYSIS
# ============================================

print("\n" + "="*60)
print("[4] HYPERPARAMETER IMPACT ANALYSIS")
print("="*60)

results_df = pd.DataFrame(grid_search.cv_results_)

# Analyze C (regularization) impact
print("\n[Effect of C (Regularization Strength)]:")
c_results = results_df.groupby('param_C')['mean_test_score'].mean()
for c, score in c_results.items():
    print(f"    C={c}: Mean Accuracy = {score:.4f}")

# Analyze penalty impact
print("\n[Effect of Penalty (Regularization Type)]:")
p_results = results_df.groupby('param_penalty')['mean_test_score'].mean()
for p, score in p_results.items():
    print(f"    {p}: Mean Accuracy = {score:.4f}")

# Analyze solver impact
print("\n[Effect of Solver]:")
s_results = results_df.groupby('param_solver')['mean_test_score'].mean()
for s, score in s_results.items():
    print(f"    {s}: Mean Accuracy = {score:.4f}")

# ============================================
# 5. FEATURE IMPORTANCE (Coefficients)
# ============================================

print("\n" + "="*60)
print("[5] FEATURE IMPORTANCE (Coefficients)")
print("="*60)

# Get absolute coefficients
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': best_lr.coef_[0],
    'Abs_Coefficient': np.abs(best_lr.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 15 Most Important Features:")
print(coef_df.head(15).to_string(index=False))

# ============================================
# 6. CROSS-VALIDATION
# ============================================

print("\n" + "="*60)
print("[6] CROSS-VALIDATION")
print("="*60)

cv_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================
# 7. TEST ON TEST.CSV
# ============================================

print("\n" + "="*60)
print("[7] EVALUATION ON TEST.CSV")
print("="*60)

testdf = pd.read_csv("Datasets/test.csv")
yt = testdf['price']
yt_encoded = yt.map({'expensive': 1, 'non-expensive': 0})

testdf_features = testdf.drop(columns=['price'])
test_processed, _ = preprocess(testdf_features, label_encoders=encoders, fit=False)

for col in feature_cols:
    if col not in test_processed.columns:
        test_processed[col] = 0

X_test_final = test_processed[feature_cols]
X_test_final_scaled = scaler.transform(X_test_final)

y_pred = best_lr.predict(X_test_final_scaled)

print(f"\nTest.csv Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(yt_encoded, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(yt_encoded, y_pred))

# ============================================
# 8. VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("[8] GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Logistic Regression Hyperparameter Analysis', fontsize=16, fontweight='bold')

# Plot 1: C value vs Accuracy
ax1 = axes[0, 0]
ax1.semilogx(c_results.index, c_results.values, 'o-', color='#3498db', linewidth=2, markersize=8)
ax1.set_xlabel('C (Regularization Strength)')
ax1.set_ylabel('Mean CV Accuracy')
ax1.set_title('Effect of C on Accuracy')
ax1.grid(True, alpha=0.3)

# Plot 2: Penalty comparison
ax2 = axes[0, 1]
p_results.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
ax2.set_xlabel('Penalty Type')
ax2.set_ylabel('Mean CV Accuracy')
ax2.set_title('L1 vs L2 Regularization')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# Plot 3: Top 10 Feature Coefficients
ax3 = axes[1, 0]
top_features = coef_df.head(10)
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_features['Coefficient']]
ax3.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Top 10 Feature Coefficients')
ax3.axvline(x=0, color='black', linewidth=0.5)
ax3.invert_yaxis()

# Plot 4: Confusion Matrix
ax4 = axes[1, 1]
cm = confusion_matrix(yt_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Non-Expensive', 'Expensive'],
            yticklabels=['Non-Expensive', 'Expensive'])
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title('Confusion Matrix (Test Set)')

plt.tight_layout()
plt.savefig('logistic_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'logistic_regression_analysis.png'")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Model: Logistic Regression
Best C: {grid_search.best_params_['C']}
Best Penalty: {grid_search.best_params_['penalty']}
Best Solver: {grid_search.best_params_['solver']}

Performance:
  - Baseline Accuracy: {acc_baseline:.4f}
  - Tuned Accuracy: {acc_tuned:.4f}
  - Test.csv Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}
  - CV Mean: {cv_scores.mean():.4f}
""")

print("="*60)
print("LOGISTIC REGRESSION MODEL COMPLETE")
print("="*60)
