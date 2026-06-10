import pandas as pd
import numpy as np
from pathlib import Path

def analyze_parquet(filepath, expected_cols=None):
    """Analyze a parquet file for data quality."""
    print(f"\n{'='*80}")
    print(f"FILE: {Path(filepath).name}")
    print(f"{'='*80}")
    
    # Load the file
    df = pd.read_parquet(filepath)
    
    # 1. Shape
    print(f"\n1. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 2. Column list
    print(f"\n2. COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col} ({df[col].dtype})")
    
    # 3. Descriptive stats for numeric columns
    print(f"\n3. DESCRIPTIVE STATS (Numeric Columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
        null_pct = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
        
        for col in numeric_cols:
            count = stats_df.loc['count', col]
            mean = stats_df.loc['mean', col]
            std = stats_df.loc['std', col]
            min_val = stats_df.loc['min', col]
            max_val = stats_df.loc['max', col]
            null_pct_val = null_pct[col]
            
            print(f"\n   {col}:")
            print(f"      count: {count:.0f}, mean: {mean:.4f}, std: {std:.4f}")
            print(f"      min: {min_val:.4f}, max: {max_val:.4f}, % nulls: {null_pct_val:.2f}%")
    else:
        print("   No numeric columns found")
    
    # 4. Flag suspicious columns
    print(f"\n4. FLAGGED ISSUES:")
    flags = []
    
    for col in numeric_cols:
        col_data = df[col]
        null_pct_val = (col_data.isnull().sum() / len(df) * 100)
        
        # Check for all-zero
        if (col_data.dropna() == 0).all():
            flags.append(f"   [ALL-ZERO] {col}")
        
        # Check for all-constant
        elif col_data.nunique() == 1:
            flags.append(f"   [ALL-CONSTANT] {col} = {col_data.dropna().iloc[0]}")
        
        # Check for >10% null
        if null_pct_val > 10:
            flags.append(f"   [>10% NULL] {col}: {null_pct_val:.2f}% null")
        
        # Check for values outside [0,1] (assuming normalized data)
        min_val = col_data.min()
        max_val = col_data.max()
        if (min_val < 0 or max_val > 1) and not pd.isna(min_val) and not pd.isna(max_val):
            # Only flag if this looks like it should be normalized
            if 'pct' in col.lower() or 'rate' in col.lower() or 'entropy' in col.lower() or 'ratio' in col.lower():
                flags.append(f"   [OUT-OF-BOUNDS] {col}: min={min_val:.4f}, max={max_val:.4f} (expected [0,1])")
    
    if flags:
        for flag in flags:
            print(flag)
    else:
        print("   None detected")
    
    # 5. Check for expected columns
    if expected_cols:
        print(f"\n5. EXPECTED COLUMNS CHECK:")
        missing = set(expected_cols) - set(df.columns)
        present = set(expected_cols) & set(df.columns)
        
        if missing:
            print(f"   MISSING ({len(missing)}):")
            for col in sorted(missing):
                print(f"      - {col}")
        
        if present:
            print(f"   PRESENT ({len(present)}):")
            for col in sorted(present):
                print(f"      - {col}")

# Expected columns for each file
user_cols = [
    'cuisine_entropy', 'unique_cbg_count', 'unique_state_count', 'unique_county_count',
    'unique_tract_count', 'cbg_entropy', 'mean_cbg_distance', 'food_emphasis',
    'service_emphasis', 'ambiance_emphasis', 'photo_posting_rate', 'pct_5star',
    'pct_1star', 'rating_entropy', 'price_deviation'
]

item_cols = [
    'pct_5star_restaurant', 'rating_bimodality', 'avg_review_length',
    'avg_photos_per_review', 'recency_boost', 'rating_trend', 'velocity_ratio',
    'text_engagement'
]

# Analyze both files
analyze_parquet(
    '/home/swami/Work/Projects/foodie_revamp/data/user_extended_features.parquet',
    expected_cols=user_cols
)

analyze_parquet(
    '/home/swami/Work/Projects/foodie_revamp/data/item_extended_features.parquet',
    expected_cols=item_cols
)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
