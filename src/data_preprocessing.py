"""
===============================================================================
Module: Data Preprocessing
Description: Handles data cleaning, missing values, duplicates, 
             feature engineering, encoding, and scaling
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load raw CSV data into a Pandas DataFrame."""
    print("=" * 60)
    print("STEP 1: Data Loading")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print()
    return df


def handle_missing_values(df):
    """Handle missing values using appropriate strategies."""
    print("=" * 60)
    print("STEP 2: Handling Missing Values")
    print("=" * 60)

    missing_before = df.isnull().sum()
    print(f"Missing values BEFORE cleaning:")
    print(missing_before[missing_before > 0])

    # Numeric columns: fill with median
    numeric_cols = ['Price_USD', 'Rating', 'Battery_mAh', 'Camera_MP', 'Weight_g']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  -> {col}: Filled {missing_before[col]} missing with median ({median_val})")

    missing_after = df.isnull().sum()
    print(f"\nMissing values AFTER cleaning: {missing_after.sum()}")
    print()
    return df


def remove_duplicates(df):
    """Remove duplicate records from the dataset."""
    print("=" * 60)
    print("STEP 3: Removing Duplicates")
    print("=" * 60)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    removed = before - after

    print(f"Records before: {before}")
    print(f"Records after:  {after}")
    print(f"Duplicates removed: {removed}")
    print()
    return df


def data_type_conversion(df):
    """Convert columns to appropriate data types."""
    print("=" * 60)
    print("STEP 4: Data Type Conversion")
    print("=" * 60)

    # Integer columns
    int_cols = ['RAM_GB', 'Storage_GB', 'Battery_mAh', 'Camera_MP', 
                'Weight_g', 'Processor_Score', 'Release_Year',
                '5G_Support', 'Wireless_Charging', 'Fast_Charging', 
                'Water_Resistant', 'Num_Reviews']

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Float columns
    float_cols = ['Price_USD', 'Rating', 'Screen_Size_inch', 'Engagement_Score']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print("Data types converted successfully!")
    print(f"Data types:\n{df.dtypes}")
    print()
    return df


def feature_engineering(df):
    """Create new features from existing ones."""
    print("=" * 60)
    print("STEP 5: Feature Engineering")
    print("=" * 60)

    # Price per GB of RAM
    df['Price_per_RAM'] = round(df['Price_USD'] / df['RAM_GB'], 2)

    # Price per GB of Storage
    df['Price_per_Storage'] = round(df['Price_USD'] / df['Storage_GB'], 2)

    # Camera quality score
    df['Camera_Score'] = round((df['Camera_MP'] / df['Camera_MP'].max()) * 10, 2)

    # Battery efficiency (Battery / Weight)
    df['Battery_Efficiency'] = round(df['Battery_mAh'] / df['Weight_g'], 2)

    # Feature score (composite of key specs)
    df['Feature_Score'] = round(
        (df['RAM_GB'] / df['RAM_GB'].max() * 2) +
        (df['Storage_GB'] / df['Storage_GB'].max() * 2) +
        (df['Camera_MP'] / df['Camera_MP'].max() * 2) +
        (df['Battery_mAh'] / df['Battery_mAh'].max() * 1.5) +
        (df['Processor_Score'] / 10 * 2) +
        (df['5G_Support'] * 0.5)
    , 2)

    # Value Score (Feature Score / Price)
    df['Value_Score'] = round(df['Feature_Score'] / (df['Price_USD'] / 100), 2)

    # Age of product
    current_year = 2025
    df['Product_Age'] = current_year - df['Release_Year']

    print("New features created:")
    print("  -> Price_per_RAM: Price per GB of RAM")
    print("  -> Price_per_Storage: Price per GB of Storage")
    print("  -> Camera_Score: Normalized camera quality (0-10)")
    print("  -> Battery_Efficiency: Battery capacity per gram")
    print("  -> Feature_Score: Composite specification score")
    print("  -> Value_Score: Feature score per $100")
    print("  -> Product_Age: Years since release")
    print()
    return df


def encode_categorical(df):
    """Encode categorical variables."""
    print("=" * 60)
    print("STEP 6: Encoding Categorical Variables")
    print("=" * 60)

    # Save original brand names for later use
    df['Brand_Original'] = df['Brand'].copy()
    df['Country_Original'] = df['Country'].copy()

    # Label encoding for clustering
    le_brand = LabelEncoder()
    le_country = LabelEncoder()

    df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'])
    df['Country_Encoded'] = le_country.fit_transform(df['Country'])

    print("Categorical variables encoded:")
    print(f"  -> Brand: {len(le_brand.classes_)} unique brands")
    print(f"  -> Country: {len(le_country.classes_)} unique countries")
    print()
    return df, le_brand, le_country


def scale_features(df, feature_cols):
    """Standardize numerical features for clustering."""
    print("=" * 60)
    print("STEP 7: Feature Scaling")
    print("=" * 60)

    scaler = StandardScaler()
    df_scaled = df.copy()

    scaled_values = scaler.fit_transform(df[feature_cols])

    # Create scaled column names
    scaled_col_names = [f"{col}_scaled" for col in feature_cols]
    for i, col_name in enumerate(scaled_col_names):
        df_scaled[col_name] = scaled_values[:, i]

    print(f"Features scaled using StandardScaler:")
    for col in feature_cols:
        print(f"  -> {col}")
    print()
    return df_scaled, scaler, scaled_col_names


def preprocess_pipeline(filepath, save_path=None):
    """Complete preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load
    df = load_data(filepath)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Data type conversion
    df = data_type_conversion(df)

    # Step 5: Feature engineering
    df = feature_engineering(df)

    # Step 6: Encode categorical
    df, le_brand, le_country = encode_categorical(df)

    # Select features for clustering
    clustering_features = [
        'Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
        'Camera_MP', 'Screen_Size_inch', 'Weight_g', 'Processor_Score',
        '5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant',
        'Engagement_Score', 'Brand_Encoded', 'Country_Encoded',
        'Feature_Score', 'Value_Score'
    ]

    # Step 7: Scale features
    df_scaled, scaler, scaled_cols = scale_features(df, clustering_features)

    # Save processed data
    if save_path:
        df_scaled.to_csv(save_path, index=False)
        print(f"Processed data saved to: {save_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Final dataset shape: {df_scaled.shape}")

    return df_scaled, scaler, scaled_cols, le_brand, le_country, clustering_features


if __name__ == "__main__":
    df_processed, scaler, scaled_cols, le_brand, le_country, features = preprocess_pipeline(
        "data/mobile_products_raw.csv",
        "data/mobile_products_processed.csv"
    )
