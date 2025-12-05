"""
Support Vector Machine (SVM) Classification Model
Smartphone Price Category Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*60)
print("SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION MODEL")
print("="*60)

# ============================================
# 1. LOAD AND PREPROCESS DATA
# ============================================

print("\n[1] Loading and preprocessing data...")

# Load training data
df = pd.read_csv('Datasets/train.csv')
print(f"    Training data shape: {df.shape}")

# Preprocess train set (fit encoders)
df_processed, encoders = preprocess(df, fit=True)

# Select features (exclude non-numeric and target)
feature_cols = [
    col for col in df_processed.columns 
    if col not in ['price', 'price_encoded'] and df_processed[col].dtype != 'object'
]

X = df_processed[feature_cols]
y = df_processed['price_encoded']

print(f"    Features selected: {len(feature_cols)}")
print(f"    Target distribution: {y.value_counts().to_dict()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================
# 2. FEATURE SCALING (Critical for SVM!)
# ============================================

print("\n[2] Scaling features with StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("    Features scaled successfully!")

# ============================================
# 3. SVM MODEL - BASELINE
# ============================================

print("\n[3] Training baseline SVM model...")
print("="*60)

# Train basic SVM with RBF kernel
svm_baseline = SVC(kernel='rbf', random_state=42)
svm_baseline.fit(X_train_scaled, y_train)

# Evaluation
pred_baseline = svm_baseline.predict(X_test_scaled)
acc_baseline = accuracy_score(y_test, pred_baseline)

print(f"\nBaseline SVM (RBF kernel):")
print(f"    Accuracy: {acc_baseline:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test, pred_baseline))
print("    Confusion Matrix:")
print(confusion_matrix(y_test, pred_baseline))

# ============================================
# 4. HYPERPARAMETER TUNING
# ============================================

print("\n" + "="*60)
print("[4] HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*60)

# Define parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 0.01],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']     # Kernel type
}

print("\nParameter grid:")
for param, values in param_grid.items():
    print(f"    {param}: {values}")

print(f"\nTotal combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
print("Starting Grid Search (this may take a few minutes)...")

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("\n" + "-"*40)
print("GRID SEARCH RESULTS:")
print("-"*40)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Get best model
best_svm = grid_search.best_estimator_

# Evaluate tuned model on validation set
pred_tuned = best_svm.predict(X_test_scaled)
acc_tuned = accuracy_score(y_test, pred_tuned)

print(f"\nTuned SVM Validation Accuracy: {acc_tuned:.4f}")
print(f"Improvement over baseline: {(acc_tuned - acc_baseline)*100:.2f}%")

# ============================================
# 5. CROSS-VALIDATION ANALYSIS
# ============================================

print("\n" + "="*60)
print("[5] CROSS-VALIDATION ANALYSIS")
print("="*60)

cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================
# 6. TESTING DIFFERENT KERNELS
# ============================================

print("\n" + "="*60)
print("[6] KERNEL COMPARISON")
print("="*60)

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_results = []

for kernel in kernels:
    svm_temp = SVC(kernel=kernel, C=grid_search.best_params_['C'], random_state=42)
    svm_temp.fit(X_train_scaled, y_train)
    pred_temp = svm_temp.predict(X_test_scaled)
    acc_temp = accuracy_score(y_test, pred_temp)
    kernel_results.append({'Kernel': kernel, 'Accuracy': acc_temp})
    print(f"    {kernel:10s} kernel: Accuracy = {acc_temp:.4f}")

# ============================================
# 7. EVALUATE ON TEST.CSV FILE
# ============================================

print("\n" + "="*60)
print("[7] EVALUATION ON TEST.CSV")
print("="*60)

# Load test data
testdf = pd.read_csv("Datasets/test.csv")
print(f"\nTest data shape: {testdf.shape}")

# Extract and encode labels
yt = testdf['price']
yt_encoded = yt.map({'expensive': 1, 'non-expensive': 0})

# Remove label column
testdf_features = testdf.drop(columns=['price'])

# Preprocess using existing encoders
test_processed, _ = preprocess(
    testdf_features,
    label_encoders=encoders,
    fit=False
)

# Ensure all training features exist
for col in feature_cols:
    if col not in test_processed.columns:
        print(f"    [WARN] Adding missing column: {col}")
        test_processed[col] = 0

# Get features in correct order
X_test_final = test_processed[feature_cols]

# Scale with training scaler
X_test_final_scaled = scaler.transform(X_test_final)

# Predict with best SVM model
y_pred = best_svm.predict(X_test_final_scaled)

# Evaluate
print("\nFINAL TEST RESULTS (test.csv):")
print("-"*40)
print(f"Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}")
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

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SVM Model Analysis', fontsize=16, fontweight='bold')

# Plot 1: Kernel Comparison
ax1 = axes[0, 0]
kernel_df = pd.DataFrame(kernel_results)
bars = ax1.bar(kernel_df['Kernel'], kernel_df['Accuracy'], color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax1.set_xlabel('Kernel Type')
ax1.set_ylabel('Accuracy')
ax1.set_title('Kernel Comparison')
ax1.set_ylim(0.7, 1.0)
for bar, acc in zip(bars, kernel_df['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

# Plot 2: Confusion Matrix Heatmap
ax2 = axes[0, 1]
cm = confusion_matrix(yt_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Non-Expensive', 'Expensive'],
            yticklabels=['Non-Expensive', 'Expensive'])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix (Test Set)')

# Plot 3: Cross-Validation Scores
ax3 = axes[1, 0]
folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax3.bar(folds, cv_scores, color='#3498db', alpha=0.7)
ax3.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('5-Fold Cross-Validation Scores')
ax3.set_ylim(0.8, 1.0)
ax3.legend()
for bar, score in zip(bars, cv_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: C Parameter Impact (from grid search results)
ax4 = axes[1, 1]
results_df = pd.DataFrame(grid_search.cv_results_)
c_results = results_df.groupby('param_C')['mean_test_score'].mean()
ax4.plot(c_results.index.astype(str), c_results.values, 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax4.set_xlabel('C (Regularization)')
ax4.set_ylabel('Mean CV Accuracy')
ax4.set_title('Effect of C Parameter on Accuracy')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'svm_analysis.png'")

# ============================================
# 9. SUMMARY
# ============================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Model: Support Vector Machine (SVM)
Best Kernel: {grid_search.best_params_['kernel']}
Best C: {grid_search.best_params_['C']}
Best Gamma: {grid_search.best_params_['gamma']}

Performance:
  - Baseline Accuracy: {acc_baseline:.4f}
  - Tuned Accuracy: {acc_tuned:.4f}
  - Test.csv Accuracy: {accuracy_score(yt_encoded, y_pred):.4f}
  - CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})
""")

print("="*60)
print("SVM MODEL COMPLETE")
print("="*60)
