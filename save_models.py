"""
Save Trained Models Script
Saves all trained models using joblib for later use
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess

print("="*60)
print("SAVING TRAINED MODELS")
print("="*60)

# Create models directory
os.makedirs('saved_models', exist_ok=True)

# Load and preprocess data
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

# Save encoders and feature columns
print("\n[2] Saving encoders and feature list...")
joblib.dump(encoders, 'saved_models/label_encoders.joblib')
joblib.dump(feature_cols, 'saved_models/feature_cols.joblib')
print("    ✓ Saved: label_encoders.joblib")
print("    ✓ Saved: feature_cols.joblib")

# Train and save Random Forest
print("\n[3] Training and saving Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'saved_models/random_forest.joblib')
print(f"    ✓ Saved: random_forest.joblib (Accuracy: {rf_model.score(X_test, y_test):.4f})")

# Train and save KNN
print("\n[4] Training and saving KNN...")
knn_scaler = StandardScaler()
X_train_scaled = knn_scaler.fit_transform(X_train)
X_test_scaled = knn_scaler.transform(X_test)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
joblib.dump(knn_model, 'saved_models/knn.joblib')
joblib.dump(knn_scaler, 'saved_models/knn_scaler.joblib')
print(f"    ✓ Saved: knn.joblib (Accuracy: {knn_model.score(X_test_scaled, y_test):.4f})")
print("    ✓ Saved: knn_scaler.joblib")

# Train and save Logistic Regression
print("\n[5] Training and saving Logistic Regression...")
lr_scaler = StandardScaler()
X_train_scaled_lr = lr_scaler.fit_transform(X_train)
X_test_scaled_lr = lr_scaler.transform(X_test)
lr_model = LogisticRegression(max_iter=300, random_state=42)
lr_model.fit(X_train_scaled_lr, y_train)
joblib.dump(lr_model, 'saved_models/logistic_regression.joblib')
joblib.dump(lr_scaler, 'saved_models/lr_scaler.joblib')
print(f"    ✓ Saved: logistic_regression.joblib (Accuracy: {lr_model.score(X_test_scaled_lr, y_test):.4f})")
print("    ✓ Saved: lr_scaler.joblib")

# Train and save SVM
print("\n[6] Training and saving SVM...")
svm_scaler = StandardScaler()
X_train_scaled_svm = svm_scaler.fit_transform(X_train)
X_test_scaled_svm = svm_scaler.transform(X_test)
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled_svm, y_train)
joblib.dump(svm_model, 'saved_models/svm.joblib')
joblib.dump(svm_scaler, 'saved_models/svm_scaler.joblib')
print(f"    ✓ Saved: svm.joblib (Accuracy: {svm_model.score(X_test_scaled_svm, y_test):.4f})")
print("    ✓ Saved: svm_scaler.joblib")

# Summary
print("\n" + "="*60)
print("MODELS SAVED SUCCESSFULLY")
print("="*60)
print("""
Saved files in 'saved_models/' directory:
  - label_encoders.joblib
  - feature_cols.joblib
  - random_forest.joblib
  - knn.joblib + knn_scaler.joblib
  - logistic_regression.joblib + lr_scaler.joblib
  - svm.joblib + svm_scaler.joblib

To load models later:
  model = joblib.load('saved_models/random_forest.joblib')
  encoders = joblib.load('saved_models/label_encoders.joblib')
""")
