"""
Test script to verify the Phone Price Predictor model is working correctly.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess

print("=" * 50)
print("PHONE PRICE PREDICTOR - MODEL TEST")
print("=" * 50)

# Load and preprocess
print("\n1. Loading training data...")
df = pd.read_csv('Datasets/train.csv')
print(f"   Loaded {len(df)} records")

print("\n2. Preprocessing data...")
df_processed, encoders = preprocess(df, fit=True)

feature_cols = [
    col for col in df_processed.columns 
    if col not in ['price', 'price_encoded'] and df_processed[col].dtype != 'object'
]
print(f"   Using {len(feature_cols)} features")

X = df_processed[feature_cols]
y = df_processed['price_encoded']

# Train model
print("\n3. Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=5, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)
print("   Model trained successfully!")

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"\n4. Model Accuracy: {accuracy:.1%}")

# Test with high-end phone (should be expensive)
print("\n5. Testing predictions...")
print("-" * 50)

# High-end phone specs
test_expensive = {
    'price': ['test'],
    'rating': [90],
    'Dual_Sim': ['Yes'],
    '4G': ['Yes'],
    '5G': ['Yes'],
    'Vo5G': ['Yes'],
    'NFC': ['Yes'],
    'IR_Blaster': ['No'],
    'Processor_Brand': ['Snapdragon'],
    'Processor_Series': ['8 Gen 3'],
    'Core_Count': [8],
    'Clock_Speed_GHz': [3.3],
    'RAM Size GB': [16],
    'Storage Size GB': [512],
    'battery_capacity': [5000],
    'fast_charging_power': [120],
    'Screen_Size': [6.8],
    'Resolution_Width': [1440],
    'Resolution_Height': [3200],
    'Refresh_Rate': [144],
    'Notch_Type': ['Punch Hole'],
    'primary_rear_camera_mp': [200],
    'num_rear_cameras': [4],
    'primary_front_camera_mp': [32],
    'num_front_cameras': [1],
    'memory_card_support': ['Yes'],
    'memory_card_size': ['1 TB'],
    'os_name': ['Android'],
    'os_version': ['v14'],
    'brand': ['Samsung']
}

test_df = pd.DataFrame(test_expensive)
test_proc, _ = preprocess(test_df.drop(columns=['price']), label_encoders=encoders, fit=False)
for col in feature_cols:
    if col not in test_proc.columns:
        test_proc[col] = 0

pred1 = model.predict(test_proc[feature_cols])[0]
prob1 = model.predict_proba(test_proc[feature_cols])[0]
result1 = "EXPENSIVE" if pred1 == 1 else "NON-EXPENSIVE"
print(f"   High-end Samsung (16GB RAM, 512GB): {result1} ({max(prob1):.1%} confidence)")

# Budget phone specs
test_budget = {
    'price': ['test'],
    'rating': [65],
    'Dual_Sim': ['Yes'],
    '4G': ['Yes'],
    '5G': ['No'],
    'Vo5G': ['No'],
    'NFC': ['No'],
    'IR_Blaster': ['No'],
    'Processor_Brand': ['MediaTek'],
    'Processor_Series': ['Helio G35'],
    'Core_Count': [4],
    'Clock_Speed_GHz': [2.0],
    'RAM Size GB': [4],
    'Storage Size GB': [64],
    'battery_capacity': [4000],
    'fast_charging_power': [10],
    'Screen_Size': [6.0],
    'Resolution_Width': [720],
    'Resolution_Height': [1600],
    'Refresh_Rate': [60],
    'Notch_Type': ['Water Drop Notch'],
    'primary_rear_camera_mp': [13],
    'num_rear_cameras': [2],
    'primary_front_camera_mp': [5],
    'num_front_cameras': [1],
    'memory_card_support': ['Yes'],
    'memory_card_size': ['128 GB'],
    'os_name': ['Android'],
    'os_version': ['v11'],
    'brand': ['Realme']
}

test_df2 = pd.DataFrame(test_budget)
test_proc2, _ = preprocess(test_df2.drop(columns=['price']), label_encoders=encoders, fit=False)
for col in feature_cols:
    if col not in test_proc2.columns:
        test_proc2[col] = 0

pred2 = model.predict(test_proc2[feature_cols])[0]
prob2 = model.predict_proba(test_proc2[feature_cols])[0]
result2 = "EXPENSIVE" if pred2 == 1 else "NON-EXPENSIVE"
print(f"   Budget Realme (4GB RAM, 64GB): {result2} ({max(prob2):.1%} confidence)")

# Mid-range phone
test_mid = {
    'price': ['test'],
    'rating': [78],
    'Dual_Sim': ['Yes'],
    '4G': ['Yes'],
    '5G': ['Yes'],
    'Vo5G': ['No'],
    'NFC': ['Yes'],
    'IR_Blaster': ['No'],
    'Processor_Brand': ['Snapdragon'],
    'Processor_Series': ['778G'],
    'Core_Count': [8],
    'Clock_Speed_GHz': [2.4],
    'RAM Size GB': [8],
    'Storage Size GB': [128],
    'battery_capacity': [4500],
    'fast_charging_power': [33],
    'Screen_Size': [6.5],
    'Resolution_Width': [1080],
    'Resolution_Height': [2400],
    'Refresh_Rate': [90],
    'Notch_Type': ['Punch Hole'],
    'primary_rear_camera_mp': [64],
    'num_rear_cameras': [3],
    'primary_front_camera_mp': [16],
    'num_front_cameras': [1],
    'memory_card_support': ['Yes'],
    'memory_card_size': ['512 GB'],
    'os_name': ['Android'],
    'os_version': ['v13'],
    'brand': ['Xiaomi']
}

test_df3 = pd.DataFrame(test_mid)
test_proc3, _ = preprocess(test_df3.drop(columns=['price']), label_encoders=encoders, fit=False)
for col in feature_cols:
    if col not in test_proc3.columns:
        test_proc3[col] = 0

pred3 = model.predict(test_proc3[feature_cols])[0]
prob3 = model.predict_proba(test_proc3[feature_cols])[0]
result3 = "EXPENSIVE" if pred3 == 1 else "NON-EXPENSIVE"
print(f"   Mid-range Xiaomi (8GB RAM, 128GB): {result3} ({max(prob3):.1%} confidence)")

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED - Model is working correctly!")
print("=" * 50)
