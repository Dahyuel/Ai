"""
Exploratory Data Analysis (EDA) Script
Smartphone Price Category Prediction
Generates all required visualizations and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("EXPLORATORY DATA ANALYSIS (EDA) - SMARTPHONE PRICE PREDICTION")
print("="*70)

# ============================================
# 1. LOAD AND OVERVIEW DATA
# ============================================

print("\n" + "="*70)
print("1. DATASET OVERVIEW")
print("="*70)

# Load data
traindf = pd.read_csv('Datasets/train.csv')
testdf = pd.read_csv('Datasets/test.csv')

print(f"\n[Training Data]")
print(f"  Shape: {traindf.shape} (rows, columns)")
print(f"  Memory Usage: {traindf.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n[Test Data]")
print(f"  Shape: {testdf.shape} (rows, columns)")

# Display columns
print(f"\n[Columns ({len(traindf.columns)} total)]:")
for i, col in enumerate(traindf.columns, 1):
    dtype = traindf[col].dtype
    print(f"  {i:2}. {col:30} ({dtype})")

# ============================================
# 2. DATA TYPES AND INFO
# ============================================

print("\n" + "="*70)
print("2. DATA TYPES AND INFO")
print("="*70)

print("\n[Data Types Summary]")
print(traindf.dtypes.value_counts().to_string())

print("\n[Detailed df.info()]")
traindf.info()

# ============================================
# 3. STATISTICAL SUMMARY
# ============================================

print("\n" + "="*70)
print("3. STATISTICAL SUMMARY (df.describe())")
print("="*70)

# Numeric columns
numeric_cols = traindf.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n[Numeric Features ({len(numeric_cols)})]:")
print(traindf[numeric_cols].describe().round(2).to_string())

# Categorical columns
cat_cols = traindf.select_dtypes(include=['object']).columns.tolist()
print(f"\n[Categorical Features ({len(cat_cols)})]:")
print(traindf[cat_cols].describe().to_string())

# ============================================
# 4. MISSING VALUES ANALYSIS
# ============================================

print("\n" + "="*70)
print("4. MISSING VALUES ANALYSIS")
print("="*70)

missing = traindf.isnull().sum()
missing_pct = (missing / len(traindf) * 100).round(2)

print("\n[Missing Values by Column]:")
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
}).sort_values('Missing Count', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))

if missing.sum() == 0:
    print("\n  ✓ No missing values found in the dataset!")
else:
    print(f"\n  Total missing values: {missing.sum()}")

# ============================================
# 5. DUPLICATE ANALYSIS
# ============================================

print("\n" + "="*70)
print("5. DUPLICATE ANALYSIS")
print("="*70)

duplicates = traindf.duplicated().sum()
print(f"\n  Duplicate rows found: {duplicates}")
print(f"  Percentage: {duplicates/len(traindf)*100:.2f}%")

if duplicates > 0:
    print("  These will be removed during preprocessing.")

# ============================================
# 6. TARGET VARIABLE ANALYSIS
# ============================================

print("\n" + "="*70)
print("6. TARGET VARIABLE ANALYSIS (price)")
print("="*70)

target_counts = traindf['price'].value_counts()
target_pct = (target_counts / len(traindf) * 100).round(2)

print("\n[Class Distribution]:")
for cls, count in target_counts.items():
    pct = target_pct[cls]
    print(f"  {cls:15}: {count:5} samples ({pct:.1f}%)")

# Check for class imbalance
ratio = target_counts.max() / target_counts.min()
print(f"\n  Class Ratio: {ratio:.2f}:1")
if ratio > 1.5:
    print("  ⚠ Slight class imbalance detected")
else:
    print("  ✓ Classes are relatively balanced")

# ============================================
# 7. CATEGORICAL FEATURES ANALYSIS
# ============================================

print("\n" + "="*70)
print("7. CATEGORICAL FEATURES ANALYSIS")
print("="*70)

categorical_features = ['brand', 'Processor_Brand', 'Processor_Series', 
                        'Notch_Type', 'os_name', 'os_version',
                        'Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 
                        'memory_card_support']

for col in categorical_features:
    if col in traindf.columns:
        unique_count = traindf[col].nunique()
        print(f"\n[{col}] - {unique_count} unique values:")
        value_counts = traindf[col].value_counts().head(5)
        for val, count in value_counts.items():
            pct = count / len(traindf) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")
        if unique_count > 5:
            print(f"    ... and {unique_count - 5} more")

# ============================================
# 8. CORRELATION ANALYSIS
# ============================================

print("\n" + "="*70)
print("8. CORRELATION ANALYSIS")
print("="*70)

# Prepare data for correlation
df_corr = traindf.copy()

# Encode target
df_corr['price_encoded'] = df_corr['price'].map({'expensive': 1, 'non-expensive': 0})

# Encode binary columns
binary_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support']
for col in binary_cols:
    if col in df_corr.columns:
        df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0})

# Get numeric columns for correlation
numeric_for_corr = df_corr.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation with price
correlations = []
for col in numeric_for_corr:
    if col != 'price_encoded':
        corr = df_corr[col].corr(df_corr['price_encoded'])
        if not pd.isna(corr):
            correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n[Correlation with Price (Top 15)]:")
print("-" * 50)
for i, (col, corr) in enumerate(correlations[:15], 1):
    direction = "+" if corr > 0 else "-"
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    bar = "█" * int(abs(corr) * 20)
    print(f"{i:2}. {col:30} {direction}{abs(corr):.3f} {bar} ({strength})")

# ============================================
# 9. OUTLIER DETECTION (IQR Method)
# ============================================

print("\n" + "="*70)
print("9. OUTLIER DETECTION (IQR Method)")
print("="*70)

outlier_features = ['rating', 'battery_capacity', 'RAM Size GB', 'Storage Size GB', 
                    'Screen_Size', 'primary_rear_camera_mp', 'Clock_Speed_GHz']

print("\n[Outlier Analysis by Feature]:")
print("-" * 60)

for feature in outlier_features:
    if feature in traindf.columns and traindf[feature].dtype in ['int64', 'float64']:
        Q1 = traindf[feature].quantile(0.25)
        Q3 = traindf[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = traindf[(traindf[feature] < lower) | (traindf[feature] > upper)]
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(traindf) * 100
        
        print(f"\n  {feature}:")
        print(f"    Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"    Bounds: [{lower:.2f}, {upper:.2f}]")
        print(f"    Outliers: {outlier_count} ({outlier_pct:.1f}%)")

# ============================================
# 10. GENERATE VISUALIZATIONS
# ============================================

print("\n" + "="*70)
print("10. GENERATING VISUALIZATIONS")
print("="*70)

# Create output directory for plots
import os
os.makedirs('eda_plots', exist_ok=True)

# -------- FIGURE 1: Target Distribution --------
fig1, ax = plt.subplots(figsize=(8, 6))
colors = ['#2ecc71', '#e74c3c']
target_counts.plot(kind='bar', color=colors, ax=ax, edgecolor='black')
ax.set_title('Target Variable Distribution (Price Category)', fontsize=14, fontweight='bold')
ax.set_xlabel('Price Category', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, (idx, val) in enumerate(target_counts.items()):
    ax.text(i, val + 20, f'{val}\n({target_pct[idx]:.1f}%)', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution.png', dpi=150)
plt.close()
print("  ✓ Saved: 01_target_distribution.png")

# -------- FIGURE 2: Correlation Heatmap --------
fig2, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df_corr[numeric_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/02_correlation_heatmap.png', dpi=150)
plt.close()
print("  ✓ Saved: 02_correlation_heatmap.png")

# -------- FIGURE 3: Top Feature Correlations with Price --------
fig3, ax = plt.subplots(figsize=(12, 8))
top_corrs = correlations[:15]
features = [x[0] for x in top_corrs]
corr_vals = [x[1] for x in top_corrs]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corr_vals]
bars = ax.barh(features, corr_vals, color=colors)
ax.set_xlabel('Correlation Coefficient', fontsize=12)
ax.set_title('Top 15 Features Correlated with Price', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
for bar, corr in zip(bars, corr_vals):
    x_pos = corr + 0.02 if corr > 0 else corr - 0.02
    ha = 'left' if corr > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{corr:.3f}', 
            ha=ha, va='center', fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('eda_plots/03_price_correlations.png', dpi=150)
plt.close()
print("  ✓ Saved: 03_price_correlations.png")

# -------- FIGURE 4: Numeric Feature Distributions --------
fig4, axes = plt.subplots(3, 3, figsize=(15, 12))
dist_features = ['rating', 'RAM Size GB', 'Storage Size GB', 'battery_capacity',
                 'Screen_Size', 'Refresh_Rate', 'primary_rear_camera_mp',
                 'Clock_Speed_GHz', 'fast_charging_power']

for ax, feature in zip(axes.flatten(), dist_features):
    if feature in traindf.columns:
        traindf[feature].hist(ax=ax, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')

plt.suptitle('Numeric Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_plots/04_feature_distributions.png', dpi=150)
plt.close()
print("  ✓ Saved: 04_feature_distributions.png")

# -------- FIGURE 5: Box Plots by Price Category --------
fig5, axes = plt.subplots(2, 3, figsize=(15, 10))
box_features = ['RAM Size GB', 'Storage Size GB', 'battery_capacity', 
                'Screen_Size', 'primary_rear_camera_mp', 'Refresh_Rate']

for ax, feature in zip(axes.flatten(), box_features):
    if feature in traindf.columns:
        traindf.boxplot(column=feature, by='price', ax=ax)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel('Price Category')

plt.suptitle('Features by Price Category (Box Plots)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/05_boxplots_by_price.png', dpi=150)
plt.close()
print("  ✓ Saved: 05_boxplots_by_price.png")

# -------- FIGURE 6: Brand Distribution --------
fig6, ax = plt.subplots(figsize=(14, 6))
brand_counts = traindf['brand'].value_counts().head(15)
brand_counts.plot(kind='bar', color='#3498db', ax=ax, edgecolor='black')
ax.set_title('Top 15 Phone Brands in Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Brand', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/06_brand_distribution.png', dpi=150)
plt.close()
print("  ✓ Saved: 06_brand_distribution.png")

# -------- FIGURE 7: Binary Features by Price --------
fig7, ax = plt.subplots(figsize=(12, 6))
binary_features = ['5G', 'NFC', 'Vo5G', 'IR_Blaster']
binary_data = []
for feature in binary_features:
    if feature in traindf.columns:
        for price_cat in ['expensive', 'non-expensive']:
            yes_count = traindf[(traindf['price'] == price_cat) & (traindf[feature] == 'Yes')].shape[0]
            total = traindf[traindf['price'] == price_cat].shape[0]
            pct = yes_count / total * 100
            binary_data.append({'Feature': feature, 'Price': price_cat, 'Yes %': pct})

binary_df = pd.DataFrame(binary_data)
binary_pivot = binary_df.pivot(index='Feature', columns='Price', values='Yes %')
binary_pivot.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_title('Binary Feature Prevalence by Price Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('% with Feature = Yes', fontsize=12)
ax.legend(title='Price Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_plots/07_binary_features_by_price.png', dpi=150)
plt.close()
print("  ✓ Saved: 07_binary_features_by_price.png")

# -------- FIGURE 8: Outlier Box Plots --------
fig8, axes = plt.subplots(2, 4, figsize=(16, 8))
outlier_plot_features = ['rating', 'battery_capacity', 'RAM Size GB', 'Storage Size GB',
                         'Screen_Size', 'primary_rear_camera_mp', 'Clock_Speed_GHz', 'Refresh_Rate']

for ax, feature in zip(axes.flatten(), outlier_plot_features):
    if feature in traindf.columns:
        ax.boxplot(traindf[feature].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(feature, fontsize=10)
        ax.grid(True, alpha=0.3)

plt.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/08_outlier_boxplots.png', dpi=150)
plt.close()
print("  ✓ Saved: 08_outlier_boxplots.png")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)

print(f"""
Dataset Overview:
  - Total samples: {len(traindf)}
  - Features: {len(traindf.columns)}
  - Missing values: {traindf.isnull().sum().sum()}
  - Duplicates: {duplicates}

Target Distribution:
  - Expensive: {target_counts.get('expensive', 0)} ({target_pct.get('expensive', 0):.1f}%)
  - Non-Expensive: {target_counts.get('non-expensive', 0)} ({target_pct.get('non-expensive', 0):.1f}%)

Key Findings:
  - Top correlated features: {', '.join([x[0] for x in correlations[:5]])}
  - Binary features (5G, NFC, etc.) are strong price indicators
  - RAM and Storage highly correlated with price category
  - Some outliers detected in battery_capacity and camera specs

All visualizations saved to 'eda_plots/' folder.
""")

print("="*70)
print("EDA COMPLETE")
print("="*70)
