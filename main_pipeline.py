"""
===============================================================================
Module: Main Pipeline Runner
Description: End-to-end execution of the complete ML pipeline
===============================================================================
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_preprocessing import preprocess_pipeline
from clustering_model import run_clustering_pipeline
from recommendation_system import MobileRecommendationSystem, test_recommendation_system
import joblib


def run_full_pipeline():
    """Execute the complete end-to-end pipeline."""
    print("\n" + "=" * 70)
    print("MOBILE PRODUCT SEGMENTATION & RECOMMENDATION SYSTEM")
    print("End-to-End Machine Learning Pipeline")
    print("=" * 70 + "\n")

    # Step 1: Data Preprocessing
    print("\n>>> STEP 1: DATA PREPROCESSING <<<")
    df_processed, scaler, scaled_cols, le_brand, le_country, features = preprocess_pipeline(
        "data/mobile_products_raw.csv",
        save_path="data/mobile_products_processed.csv"
    )

    # Step 2: Clustering
    print("\n>>> STEP 2: CLUSTERING ANALYSIS <<<")
    df_clustered, kmeans_model, insights = run_clustering_pipeline(df_processed, features, n_clusters=4)
    df_clustered.to_csv("data/mobile_products_clustered.csv", index=False)

    # Step 3: Recommendation System
    print("\n>>> STEP 3: RECOMMENDATION SYSTEM <<<")
    rec_sys = test_recommendation_system(df_clustered, features)
    joblib.dump(rec_sys, "models/recommendation_system.pkl")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"✅ Raw Dataset: {df_processed.shape[0]} products, {df_processed.shape[1]} features")
    print(f"✅ Clusters Identified: 4 market segments")
    print(f"✅ Recommendation System: Cosine Similarity based")
    print(f"✅ Models Saved: models/kmeans_model.pkl, models/recommendation_system.pkl")
    print(f"✅ Data Files: data/mobile_products_raw.csv, data/mobile_products_processed.csv, data/mobile_products_clustered.csv")
    print(f"✅ Reports: reports/ (EDA visualizations, cluster analysis)")
    print(f"✅ Streamlit App: app/streamlit_app.py")
    print("=" * 70)
    print("\n🚀 To launch the interactive app, run:")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 70 + "\n")

    return df_clustered, kmeans_model, rec_sys, insights


if __name__ == "__main__":
    run_full_pipeline()
