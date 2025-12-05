"""
K-Nearest Neighbors (KNN) Classification Model with Full Hyperparameter Tuning
Smartphone Price Category Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("K-NEAREST NEIGHBORS (KNN) CLASSIFICATION MODEL")
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

# Scale features (CRITICAL for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")
print(f"    Features: {len(feature_cols)}")

# ============================================
# 2. BASELINE KNN MODEL
# ============================================

print("\n" + "="*60)
print("[2] BASELINE KNN MODEL (k=5)")
print("="*60)

knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train_scaled, y_train)

pred_baseline = knn_baseline.predict(X_test_scaled)
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
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],       # Number of neighbors
    'weights': ['uniform', 'distance'],             # Weight function
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
    'p': [1, 2]  # Power parameter for Minkowski
}

print("\nParameter grid:")
for param, values in param_grid.items():
    print(f"    {param}: {values}")

total_combos = (len(param_grid['n_neighbors']) * len(param_grid['weights']) * 
                len(param_grid['metric']) * len(param_grid['p']))
print(f"\nTotal combinations: {total_combos}")
print("Starting Grid Search...")

# Grid Search
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
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
best_knn = grid_search.best_estimator_
pred_tuned = best_knn.predict(X_test_scaled)
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

# Analyze n_neighbors impact
print("\n[Effect of n_neighbors (k)]:")
k_results = results_df.groupby('param_n_neighbors')['mean_test_score'].mean()
for k, score in k_results.items():
    print(f"    k={k}: Mean Accuracy = {score:.4f}")

# Analyze weights impact
print("\n[Effect of weights]:")
w_results = results_df.groupby('param_weights')['mean_test_score'].mean()
for w, score in w_results.items():
    print(f"    {w}: Mean Accuracy = {score:.4f}")

# Analyze metric impact
print("\n[Effect of distance metric]:")
m_results = results_df.groupby('param_metric')['mean_test_score'].mean()
for m, score in m_results.items():
    print(f"    {m}: Mean Accuracy = {score:.4f}")

# ============================================
# 5. CROSS-VALIDATION
# ============================================

print("\n" + "="*60)
print("[5] CROSS-VALIDATION")
print("="*60)

cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================
# 6. TEST ON TEST.CSV
# ============================================

print("\n" + "="*60)
print("[6] EVALUATION ON TEST.CSV")
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

y_pred = best_knn.predict(X_test_final_scaled)

print(f"\nTest.csv Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(yt_encoded, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(yt_encoded, y_pred))

# ============================================
# 7. VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("[7] GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KNN Hyperparameter Analysis', fontsize=16, fontweight='bold')

# Plot 1: K value vs Accuracy
ax1 = axes[0, 0]
ax1.plot(k_results.index, k_results.values, 'o-', color='#3498db', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Neighbors (k)')
ax1.set_ylabel('Mean CV Accuracy')
ax1.set_title('Effect of k on Accuracy')
ax1.grid(True, alpha=0.3)

# Plot 2: Weights comparison
ax2 = axes[0, 1]
w_results.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
ax2.set_xlabel('Weight Function')
ax2.set_ylabel('Mean CV Accuracy')
ax2.set_title('Effect of Weights on Accuracy')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# Plot 3: Metric comparison
ax3 = axes[1, 0]
m_results.plot(kind='bar', ax=ax3, color=['#9b59b6', '#f39c12', '#1abc9c'])
ax3.set_xlabel('Distance Metric')
ax3.set_ylabel('Mean CV Accuracy')
ax3.set_title('Effect of Metric on Accuracy')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

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
plt.savefig('knn_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'knn_analysis.png'")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Model: K-Nearest Neighbors (KNN)
Best k: {grid_search.best_params_['n_neighbors']}
Best Weights: {grid_search.best_params_['weights']}
Best Metric: {grid_search.best_params_['metric']}

Performance:
  - Baseline Accuracy (k=5): {acc_baseline:.4f}
  - Tuned Accuracy: {acc_tuned:.4f}
  - Test.csv Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}
  - CV Mean: {cv_scores.mean():.4f}
""")

print("="*60)
print("KNN MODEL COMPLETE")
print("="*60)
