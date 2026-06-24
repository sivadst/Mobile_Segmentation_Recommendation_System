"""
===============================================================================
Module: Recommendation System
Description: Similarity-based product recommendation using Cosine Similarity
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class MobileRecommendationSystem:
    """Similarity-based product recommendation system for mobile phones."""

    def __init__(self, df, feature_cols):
        """Initialize with processed data and feature columns."""
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.similarity_matrix = None
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        """Build cosine similarity matrix based on product features."""
        print("Building similarity matrix...")

        # Use scaled features if available, otherwise scale
        scaled_cols = [f"{col}_scaled" for col in self.feature_cols]
        if all(col in self.df.columns for col in scaled_cols):
            features = self.df[scaled_cols].values
        else:
            scaler = StandardScaler()
            features = scaler.fit_transform(self.df[self.feature_cols])

        self.similarity_matrix = cosine_similarity(features)
        print(f"Similarity matrix built: {self.similarity_matrix.shape}")

    def get_recommendations_by_product(self, product_id, n_recommendations=5):
        """Get similar products based on a product ID."""
        if product_id not in self.df['Product_ID'].values:
            return None, f"Product ID '{product_id}' not found."

        idx = self.df[self.df['Product_ID'] == product_id].index[0]
        product = self.df.iloc[idx]

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the product itself and get top N
        sim_scores = sim_scores[1:n_recommendations + 1]
        product_indices = [i[0] for i in sim_scores]
        similarity_values = [i[1] for i in sim_scores]

        recommendations = self.df.iloc[product_indices][[
            'Product_ID', 'Brand', 'Model', 'Price_USD', 'Rating', 
            'RAM_GB', 'Storage_GB', 'Camera_MP', 'Battery_mAh', 'Cluster'
        ]].copy()
        recommendations['Similarity_Score'] = [round(s, 4) for s in similarity_values]
        recommendations = recommendations.reset_index(drop=True)

        return recommendations, product

    def get_recommendations_by_features(self, price_range=None, ram=None, storage=None,
                                         brand=None, camera=None, n_recommendations=5):
        """Get recommendations based on user-specified features."""
        filtered_df = self.df.copy()

        # Apply filters
        if price_range:
            filtered_df = filtered_df[
                (filtered_df['Price_USD'] >= price_range[0]) & 
                (filtered_df['Price_USD'] <= price_range[1])
            ]
        if ram:
            filtered_df = filtered_df[filtered_df['RAM_GB'] == ram]
        if storage:
            filtered_df = filtered_df[filtered_df['Storage_GB'] == storage]
        if brand:
            filtered_df = filtered_df[filtered_df['Brand'] == brand]
        if camera:
            filtered_df = filtered_df[filtered_df['Camera_MP'] >= camera]

        if len(filtered_df) == 0:
            return None, "No products match the specified criteria."

        # Sort by rating and engagement
        filtered_df = filtered_df.sort_values(
            by=['Rating', 'Engagement_Score'], 
            ascending=[False, False]
        )

        recommendations = filtered_df.head(n_recommendations)[[
            'Product_ID', 'Brand', 'Model', 'Price_USD', 'Rating',
            'RAM_GB', 'Storage_GB', 'Camera_MP', 'Battery_mAh', 'Cluster',
            'Feature_Score', 'Value_Score'
        ]].copy()

        return recommendations.reset_index(drop=True), None

    def get_recommendations_by_cluster(self, cluster_id, n_recommendations=5):
        """Get top products from a specific cluster."""
        cluster_df = self.df[self.df['Cluster'] == cluster_id]

        if len(cluster_df) == 0:
            return None, f"Cluster {cluster_id} not found."

        # Sort by engagement score and rating
        top_products = cluster_df.sort_values(
            by=['Engagement_Score', 'Rating'], 
            ascending=[False, False]
        ).head(n_recommendations)

        recommendations = top_products[[
            'Product_ID', 'Brand', 'Model', 'Price_USD', 'Rating',
            'RAM_GB', 'Storage_GB', 'Camera_MP', 'Battery_mAh',
            'Feature_Score', 'Value_Score', 'Engagement_Score'
        ]].copy()

        return recommendations.reset_index(drop=True), None

    def get_cross_cluster_recommendations(self, product_id, n_recommendations=5):
        """Get recommendations from different clusters (diverse recommendations)."""
        if product_id not in self.df['Product_ID'].values:
            return None, f"Product ID '{product_id}' not found."

        idx = self.df[self.df['Product_ID'] == product_id].index[0]
        source_cluster = self.df.iloc[idx]['Cluster']

        # Get products from other clusters
        other_clusters = self.df[self.df['Cluster'] != source_cluster]

        if len(other_clusters) == 0:
            return None, "No products in other clusters."

        # Build similarity for cross-cluster
        scaled_cols = [f"{col}_scaled" for col in self.feature_cols]
        if all(col in self.df.columns for col in scaled_cols):
            all_features = self.df[scaled_cols].values
            target_features = all_features[idx].reshape(1, -1)
            other_features = other_clusters[[f"{col}_scaled" for col in self.feature_cols]].values
        else:
            scaler = StandardScaler()
            all_features = scaler.fit_transform(self.df[self.feature_cols])
            target_features = all_features[idx].reshape(1, -1)
            other_indices = other_clusters.index
            other_features = all_features[other_indices]

        similarities = cosine_similarity(target_features, other_features)[0]

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        similarity_values = similarities[top_indices]

        recommendations = other_clusters.iloc[top_indices][[
            'Product_ID', 'Brand', 'Model', 'Price_USD', 'Rating',
            'RAM_GB', 'Storage_GB', 'Camera_MP', 'Battery_mAh', 'Cluster'
        ]].copy()
        recommendations['Similarity_Score'] = [round(s, 4) for s in similarity_values]

        return recommendations.reset_index(drop=True), None

    def validate_recommendations(self, product_id, recommendations):
        """Validate recommendation relevance."""
        if recommendations is None:
            return None

        product = self.df[self.df['Product_ID'] == product_id].iloc[0]

        metrics = {
            'source_product': product_id,
            'source_price': product['Price_USD'],
            'source_rating': product['Rating'],
            'source_cluster': product['Cluster'],
            'avg_recommended_price': recommendations['Price_USD'].mean(),
            'avg_recommended_rating': recommendations['Rating'].mean(),
            'avg_similarity': recommendations['Similarity_Score'].mean() if 'Similarity_Score' in recommendations.columns else None,
            'price_diff_pct': abs(recommendations['Price_USD'].mean() - product['Price_USD']) / product['Price_USD'] * 100,
            'rating_diff': abs(recommendations['Rating'].mean() - product['Rating']),
            'cluster_diversity': recommendations['Cluster'].nunique()
        }

        return metrics


def test_recommendation_system(df, feature_cols):
    """Test the recommendation system with sample queries."""
    print("=" * 70)
    print("RECOMMENDATION SYSTEM TESTING")
    print("=" * 70)

    rec_sys = MobileRecommendationSystem(df, feature_cols)

    # Test 1: Recommend by product ID
    print("\n" + "─" * 60)
    print("TEST 1: Recommendations by Product ID")
    print("─" * 60)
    sample_product = df.iloc[0]['Product_ID']
    recs, product = rec_sys.get_recommendations_by_product(sample_product, 5)

    print(f"\nSource Product: {product['Brand']} {product['Model']}")
    print(f"Price: ${product['Price_USD']}, Rating: {product['Rating']}, Cluster: {product['Cluster']}")
    print(f"\nRecommended Products:")
    print(recs.to_string(index=False))

    # Validate
    metrics = rec_sys.validate_recommendations(sample_product, recs)
    print(f"\nValidation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  -> {k}: {v:.2f}")
        else:
            print(f"  -> {k}: {v}")

    # Test 2: Recommend by features
    print("\n" + "─" * 60)
    print("TEST 2: Recommendations by Features")
    print("─" * 60)
    recs, _ = rec_sys.get_recommendations_by_features(
        price_range=(400, 700), ram=8, storage=128, n_recommendations=5
    )
    print(f"\nFilters: Price $400-$700, RAM=8GB, Storage=128GB")
    print(recs.to_string(index=False))

    # Test 3: Recommend by cluster
    print("\n" + "─" * 60)
    print("TEST 3: Top Products by Cluster")
    print("─" * 60)
    for c in sorted(df['Cluster'].unique()):
        recs, _ = rec_sys.get_recommendations_by_cluster(c, 3)
        print(f"\nCluster {c} - Top 3 Products:")
        print(recs[['Brand', 'Model', 'Price_USD', 'Rating']].to_string(index=False))

    # Test 4: Cross-cluster recommendations
    print("\n" + "─" * 60)
    print("TEST 4: Cross-Cluster Recommendations")
    print("─" * 60)
    recs, _ = rec_sys.get_cross_cluster_recommendations(sample_product, 5)
    print(f"\nCross-cluster recommendations for {sample_product}:")
    print(recs.to_string(index=False))

    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM TESTING COMPLETE!")
    print("=" * 70)

    return rec_sys


if __name__ == "__main__":
    df = pd.read_csv("data/mobile_products_clustered.csv")

    clustering_features = [
        'Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
        'Camera_MP', 'Screen_Size_inch', 'Weight_g', 'Processor_Score',
        '5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant',
        'Engagement_Score', 'Brand_Encoded', 'Country_Encoded',
        'Feature_Score', 'Value_Score'
    ]

    rec_sys = test_recommendation_system(df, clustering_features)
    joblib.dump(rec_sys, "models/recommendation_system.pkl")
    print("\nRecommendation system saved: models/recommendation_system.pkl")
