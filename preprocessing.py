import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


''' how to do multi lined comment '''


'''
# Load data
traindf = pd.read_csv('Datasets/train.csv')
testdf = pd.read_csv('Datasets/test.csv')



DOING ANALYSIS ON DATA

# Check basic info for train.csv
print(traindf.info())
print(traindf.isnull().sum())
print(traindf['price'].value_counts()) 




# Check basic info for test.csv
print(testdf.info())
print(testdf.isnull().sum())
print(testdf['price'].value_counts()) 





# Display data info
print("Original Data Types:")
print(traindf.dtypes)
print("\n" + "="*50 + "\n")
'''







def preprocess(df, label_encoders=None, fit=True):
    """
    Preprocess the input dataframe.
    If fit=True → fit LabelEncoders and scalers.
    If fit=False → use existing label_encoders (for test data).
    """
    df = df.copy()

    # === Remove unwanted columns ===
    tier_columns = ['RAM Tier', 'Performance_Tier']
    df.drop(columns=[c for c in tier_columns if c in df.columns], inplace=True)

    # === Remove duplicates ===
    df.drop_duplicates(inplace=True)

    # === Binary Columns ===
    binary_columns = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # === Price (target) ===
    if 'price' in df.columns:
        df['price_encoded'] = df['price'].map({'expensive': 1, 'non-expensive': 0})

    # === Label Encoding ===
    categorical_columns = [
        'Processor_Brand', 'Processor_Series', 'Notch_Type', 'os_name', 'brand'
    ]

    if label_encoders is None:
        label_encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[f'{col}_encoded'] = df[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else -1)


    # === OS Version ===
    if 'os_version' in df.columns:
        df['os_version_numeric'] = df['os_version'].str.extract('(\d+)').astype(float)

    # === Memory Card Size to numeric ===
    if 'memory_card_size' in df.columns:
        def convert_size(x):
            if pd.isna(x):
                return 0
            s = str(x).lower()
            if 'tb' in s:
                return float(s.replace('tb', '').strip()) * 1024
            elif 'gb' in s:
                return float(s.replace('gb', '').strip())
            return float(s)
        df['memory_card_size_gb'] = df['memory_card_size'].apply(convert_size)

    # === Return processed DF and encoders ===
    return df, label_encoders









'''
# Set style
plt.style.use('default')
sns.set_style("whitegrid")

# ============================================
# 1. SIMPLE CORRELATION WITH PRICE - ALL FEATURES
# ============================================

print("="*50)
print("CORRELATION WITH PRICE - ALL FEATURES")
print("="*50)

# Use the ENCODED price column, not the original string column
price_col = 'price_encoded'  # Changed from 'price' to 'price_encoded'
if 'price_encoded' not in df_encoded.columns:
    # If price_encoded doesn't exist, create it from price column
    if 'price' in df_encoded.columns:
        df_encoded['price_encoded'] = df_encoded['price'].map({'expensive': 1, 'non-expensive': 0})
        print("Created price_encoded column from price")
    else:
        # Try to find price column
        price_col = [col for col in df_encoded.columns if 'price' in col.lower()]
        if price_col:
            price_col = price_col[0]
        else:
            print("ERROR: No price column found!")
            exit()

# Get numerical columns (excluding price itself)
num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col != price_col]

print(f"Analyzing {len(num_cols)} numerical features...")

# Calculate correlations with price
correlations = []
for col in num_cols:
    corr = df_encoded[col].corr(df_encoded[price_col])
    if not pd.isna(corr):
        correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Print ALL correlations
print(f"\nALL FEATURES CORRELATION WITH PRICE (sorted by strength):")
print("-" * 80)
for i, (col, corr) in enumerate(correlations, 1):
    direction = "POSITIVE" if corr > 0 else "NEGATIVE"
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    print(f"{i:3}. {col:35} : {corr:7.3f} ({direction}, {strength})")

# Create correlation plot for ALL features
plt.figure(figsize=(14, len(correlations) * 0.3))  # Adjust height based on number of features

# Get ALL features (not just top 20)
all_features = [col for col, _ in correlations]
all_corrs = [corr for _, corr in correlations]

# Create horizontal bar plot for ALL features
y_pos = np.arange(len(all_features))
colors = ['red' if c < 0 else 'green' for c in all_corrs]

plt.barh(y_pos, all_corrs, color=colors, alpha=0.7, height=0.8)
plt.yticks(y_pos, all_features, fontsize=9)
plt.xlabel('Correlation Coefficient with Price', fontsize=11)
plt.title(f'Correlation with Price for ALL {len(all_features)} Features', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add correlation values on bars
for i, (corr, color) in enumerate(zip(all_corrs, colors)):
    if corr < 0:
        plt.text(corr - 0.005, i, f'{corr:.3f}', 
                ha='right', va='center', color=color, fontsize=8, fontweight='bold')
    else:
        plt.text(corr + 0.005, i, f'{corr:.3f}', 
                ha='left', va='center', color=color, fontsize=8, fontweight='bold')

# Add grid for better readability
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# Create a separate summary table for quick reference
print("\n" + "="*80)
print("CORRELATION SUMMARY STATISTICS")
print("="*80)

# Calculate correlation statistics
corr_values = [corr for _, corr in correlations]

print(f"\nTotal Features Analyzed: {len(correlations)}")
print(f"Average Absolute Correlation: {np.mean(np.abs(corr_values)):.3f}")
print(f"Maximum Correlation: {max(corr_values, key=abs):.3f}")
print(f"Minimum Correlation: {min(corr_values, key=abs):.3f}")

# Count by strength
strong_pos = sum(1 for c in corr_values if c > 0.5)
strong_neg = sum(1 for c in corr_values if c < -0.5)
moderate_pos = sum(1 for c in corr_values if 0.3 < c <= 0.5)
moderate_neg = sum(1 for c in corr_values if -0.5 <= c < -0.3)
weak = sum(1 for c in corr_values if abs(c) <= 0.3)

print(f"\nCorrelation Strength Distribution:")
print(f"  Strong Positive (>0.5): {strong_pos} features")
print(f"  Moderate Positive (0.3-0.5): {moderate_pos} features")
print(f"  Weak (|r| ≤ 0.3): {weak} features")
print(f"  Moderate Negative (-0.5 - -0.3): {moderate_neg} features")
print(f"  Strong Negative (< -0.5): {strong_neg} features")

# Show top 5 positive and negative
print(f"\nTop 5 Positive Correlations:")
for col, corr in correlations[:5]:
    if corr > 0:
        print(f"  {col:35}: {corr:.3f}")

print(f"\nTop 5 Negative Correlations:")
# Find negative correlations
negative_corrs = [(col, corr) for col, corr in correlations if corr < 0]
for col, corr in negative_corrs[:5]:
    print(f"  {col:35}: {corr:.3f}")

# ============================================
# 2. SIMPLE OUTLIER DETECTION
# ============================================

print("\n" + "="*50)
print("OUTLIER DETECTION")
print("="*50)

# Select top 5 features for outlier detection (by variance OR correlation)
# You can choose: use top correlated or highest variance features

# Option A: Use top 5 correlated features
features_for_outliers = [col for col, _ in correlations[:5]]

# Option B: Or use features with highest variance (uncomment below):
# variances = {col: df_encoded[col].var() for col, _ in correlations[:20] if df_encoded[col].nunique() > 5}
# features_for_outliers = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:5]
# features_for_outliers = [f[0] for f in features_for_outliers]

print(f"\nChecking outliers in: {features_for_outliers}")

# Create outlier plots
fig, axes = plt.subplots(1, len(features_for_outliers), figsize=(15, 4))
fig.suptitle('Outlier Detection in Top 5 Features', fontsize=14, fontweight='bold')

if len(features_for_outliers) == 1:
    axes = [axes]  # Make it iterable

for idx, feature in enumerate(features_for_outliers):
    ax = axes[idx] if len(features_for_outliers) > 1 else axes
    
    # Create boxplot
    bp = ax.boxplot(df_encoded[feature].dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Calculate outliers
    Q1 = df_encoded[feature].quantile(0.25)
    Q3 = df_encoded[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:  # Avoid division by zero
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_encoded[(df_encoded[feature] < lower_bound) | (df_encoded[feature] > upper_bound)]
        outlier_count = len(outliers)
        
        ax.set_title(f'{feature}\n({outlier_count} outliers)', fontsize=10)
        ax.set_ylabel('Value')
    else:
        ax.set_title(f'{feature}\n(No variance)', fontsize=10)
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print outlier summary
print("\nOutlier Summary:")
print("-" * 50)
for feature in features_for_outliers:
    Q1 = df_encoded[feature].quantile(0.25)
    Q3 = df_encoded[feature].quantize(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_encoded[(df_encoded[feature] < lower_bound) | (df_encoded[feature] > upper_bound)]
        outlier_count = len(outliers)
        
        print(f"{feature:25}: {outlier_count:3} outliers ({outlier_count/len(df_encoded)*100:.1f}%)")
        if outlier_count > 0:
            print(f"    Range: [{df_encoded[feature].min():.2f}, {df_encoded[feature].max():.2f}]")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    Mean: {df_encoded[feature].mean():.2f}, Std: {df_encoded[feature].std():.2f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
'''