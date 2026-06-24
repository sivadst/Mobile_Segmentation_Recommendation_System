"""
===============================================================================
Module: Clustering Model (K-Means)
Description: Product/User segmentation using K-Means clustering
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def find_optimal_clusters(df, feature_cols, max_k=10):
    """Find optimal number of clusters using Elbow Method and Silhouette Score."""
    print("=" * 70)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 70)

    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    X = df[scaled_cols].values

    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []

    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))

        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}, "
              f"Calinski-Harabasz={calinski_scores[-1]:.2f}, Davies-Bouldin={davies_bouldin_scores[-1]:.4f}")

    # Plot metrics
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', name='Inertia (Elbow)'))
    fig.update_layout(title='Elbow Method for Optimal K', xaxis_title='Number of Clusters (K)',
                      yaxis_title='Inertia (Within-cluster Sum of Squares)')
    fig.write_html("reports/elbow_method.html")
    try:
        fig.write_image("reports/elbow_method.png", width=900, height=500)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: elbow_method.html & .png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers', 
                                name='Silhouette Score', line=dict(color='green')))
    fig2.update_layout(title='Silhouette Score vs Number of Clusters', xaxis_title='Number of Clusters (K)',
                       yaxis_title='Silhouette Score')
    fig2.write_html("reports/silhouette_analysis.html")
    try:
        fig2.write_image("reports/silhouette_analysis.png", width=900, height=500)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: silhouette_analysis.html & .png")

    # Find best K by silhouette score
    best_k = list(K_range)[np.argmax(silhouette_scores)]
    print(f"\nBest K by Silhouette Score: {best_k} (Score: {max(silhouette_scores):.4f})")

    return best_k, inertias, silhouette_scores


def apply_kmeans(df, feature_cols, n_clusters=4):
    """Apply K-Means clustering and analyze clusters."""
    print("\n" + "=" * 70)
    print(f"APPLYING K-MEANS CLUSTERING (K={n_clusters})")
    print("=" * 70)

    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    X = df[scaled_cols].values

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    # Calculate metrics
    sil_score = silhouette_score(X, df['Cluster'])
    ch_score = calinski_harabasz_score(X, df['Cluster'])
    db_score = davies_bouldin_score(X, df['Cluster'])

    print(f"Clustering Metrics:")
    print(f"  -> Silhouette Score: {sil_score:.4f}")
    print(f"  -> Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"  -> Davies-Bouldin Index: {db_score:.4f}")
    print(f"  -> Inertia: {kmeans.inertia_:.2f}")

    # Cluster distribution
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print(f"\nCluster Distribution:")
    for c, count in cluster_counts.items():
        print(f"  Cluster {c}: {count} products ({count/len(df)*100:.1f}%)")

    # Save model
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    print(f"\n-> Model saved: models/kmeans_model.pkl")

    return df, kmeans, sil_score


def analyze_clusters(df, n_clusters=4):
    """Analyze each cluster and generate business insights."""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS & BUSINESS INSIGHTS")
    print("=" * 70)

    cluster_insights = {}

    # Key metrics for analysis
    analysis_cols = ['Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
                     'Camera_MP', 'Processor_Score', '5G_Support', 'Wireless_Charging',
                     'Fast_Charging', 'Water_Resistant', 'Engagement_Score', 
                     'Feature_Score', 'Value_Score']

    for c in range(n_clusters):
        cluster_df = df[df['Cluster'] == c]

        print(f"\n{'─' * 60}")
        print(f"CLUSTER {c} - ANALYSIS")
        print(f"{'─' * 60}")
        print(f"Products in cluster: {len(cluster_df)}")

        # Average metrics
        avg_metrics = cluster_df[analysis_cols].mean()
        print(f"\nAverage Metrics:")
        for col in analysis_cols:
            if col in ['5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant']:
                print(f"  -> {col}: {avg_metrics[col]*100:.1f}%")
            else:
                print(f"  -> {col}: {avg_metrics[col]:.2f}")

        # Top brands in cluster
        top_brands = cluster_df['Brand'].value_counts().head(3)
        print(f"\nTop Brands:")
        for brand, count in top_brands.items():
            print(f"  -> {brand}: {count} products")

        # Price range
        price_min = cluster_df['Price_USD'].min()
        price_max = cluster_df['Price_USD'].max()
        print(f"\nPrice Range: ${price_min:.2f} - ${price_max:.2f}")

        # Segment classification
        avg_price = avg_metrics['Price_USD']
        avg_rating = avg_metrics['Rating']
        avg_feature = avg_metrics['Feature_Score']

        if avg_price >= 800 and avg_feature >= 6:
            segment = "PREMIUM FLAGSHIP"
            description = "High-end devices with top specifications, premium pricing, and excellent ratings."
        elif avg_price >= 500 and avg_feature >= 5:
            segment = "UPPER MID-RANGE"
            description = "Well-balanced devices with strong specs and good value proposition."
        elif avg_price >= 250 and avg_feature >= 3.5:
            segment = "MID-RANGE"
            description = "Affordable devices with decent specifications for everyday use."
        else:
            segment = "BUDGET / ENTRY-LEVEL"
            description = "Cost-effective devices with basic features, ideal for price-sensitive consumers."

        print(f"\n🎯 SEGMENT: {segment}")
        print(f"📋 Description: {description}")

        cluster_insights[c] = {
            'segment': segment,
            'description': description,
            'count': len(cluster_df),
            'avg_price': avg_price,
            'avg_rating': avg_rating,
            'avg_feature': avg_feature,
            'top_brands': list(top_brands.index),
            'price_range': (price_min, price_max)
        }

    return cluster_insights


def visualize_clusters(df, n_clusters=4):
    """Create cluster visualizations."""
    print("\n" + "=" * 70)
    print("CLUSTER VISUALIZATIONS")
    print("=" * 70)

    # 1. 3D Scatter: Price vs Rating vs Feature Score
    fig = px.scatter_3d(df, x='Price_USD', y='Rating', z='Feature_Score',
                        color='Cluster', hover_data=['Brand', 'Model'],
                        title='3D Cluster Visualization: Price vs Rating vs Feature Score',
                        color_continuous_scale='Viridis')
    fig.write_html("reports/cluster_3d.html")
    print(f"  -> Saved: cluster_3d.html")

    # 2. 2D Scatter: Price vs Rating colored by cluster
    fig = px.scatter(df, x='Price_USD', y='Rating', color='Cluster',
                     size='Engagement_Score', hover_data=['Brand', 'Model', 'Feature_Score'],
                     title='Cluster Visualization: Price vs Rating',
                     color_continuous_scale='Viridis')
    fig.write_html("reports/cluster_2d.html")
    try:
        fig.write_image("reports/cluster_2d.png", width=1100, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: cluster_2d.html & .png")

    # 3. Cluster-wise box plots
    fig = px.box(df, x='Cluster', y='Price_USD', color='Cluster',
                 title='Price Distribution by Cluster',
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.write_html("reports/cluster_price_box.html")
    try:
        fig.write_image("reports/cluster_price_box.png", width=900, height=600)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: cluster_price_box.html & .png")

    # 4. Radar chart for cluster profiles
    categories = ['Price', 'Rating', 'RAM', 'Storage', 'Camera', 'Battery', 'Processor']

    fig = go.Figure()
    colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA']

    for c in range(n_clusters):
        cluster_df = df[df['Cluster'] == c]
        values = [
            cluster_df['Price_USD'].mean() / df['Price_USD'].max() * 10,
            cluster_df['Rating'].mean() / 5 * 10,
            cluster_df['RAM_GB'].mean() / df['RAM_GB'].max() * 10,
            cluster_df['Storage_GB'].mean() / df['Storage_GB'].max() * 10,
            cluster_df['Camera_MP'].mean() / df['Camera_MP'].max() * 10,
            cluster_df['Battery_mAh'].mean() / df['Battery_mAh'].max() * 10,
            cluster_df['Processor_Score'].mean() / 10 * 10
        ]
        values += values[:1]  # Complete the circle

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Cluster {c}',
            line_color=colors[c % len(colors)]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title='Cluster Profile Radar Chart',
        showlegend=True
    )
    fig.write_html("reports/cluster_radar.html")
    try:
        fig.write_image("reports/cluster_radar.png", width=900, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: cluster_radar.html & .png")

    # 5. Cluster segment summary
    segment_data = []
    for c in range(n_clusters):
        cluster_df = df[df['Cluster'] == c]
        segment_data.append({
            'Cluster': f'Cluster {c}',
            'Count': len(cluster_df),
            'Avg_Price': round(cluster_df['Price_USD'].mean(), 2),
            'Avg_Rating': round(cluster_df['Rating'].mean(), 2),
            'Avg_Feature': round(cluster_df['Feature_Score'].mean(), 2),
            '5G_%': round(cluster_df['5G_Support'].mean() * 100, 1),
            'Wireless_%': round(cluster_df['Wireless_Charging'].mean() * 100, 1)
        })

    segment_df = pd.DataFrame(segment_data)
    fig = px.bar(segment_df, x='Cluster', y=['Avg_Price', 'Avg_Rating', 'Avg_Feature'],
                 barmode='group', title='Cluster Comparison: Price, Rating & Feature Score')
    fig.write_html("reports/cluster_comparison.html")
    try:
        fig.write_image("reports/cluster_comparison.png", width=1000, height=600)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: cluster_comparison.html & .png")


def run_clustering_pipeline(df, feature_cols, n_clusters=4):
    """Complete clustering pipeline."""
    print("\n" + "=" * 70)
    print("CLUSTERING PIPELINE")
    print("=" * 70)

    # Find optimal clusters
    best_k, inertias, sil_scores = find_optimal_clusters(df, feature_cols)

    # Apply K-Means
    df_clustered, kmeans_model, sil_score = apply_kmeans(df, feature_cols, n_clusters)

    # Analyze clusters
    insights = analyze_clusters(df_clustered, n_clusters)

    # Visualize
    visualize_clusters(df_clustered, n_clusters)

    print("\n" + "=" * 70)
    print("CLUSTERING PIPELINE COMPLETE!")
    print("=" * 70)

    return df_clustered, kmeans_model, insights


if __name__ == "__main__":
    df = pd.read_csv("data/mobile_products_processed.csv")

    clustering_features = [
        'Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
        'Camera_MP', 'Screen_Size_inch', 'Weight_g', 'Processor_Score',
        '5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant',
        'Engagement_Score', 'Brand_Encoded', 'Country_Encoded',
        'Feature_Score', 'Value_Score'
    ]

    df_clustered, model, insights = run_clustering_pipeline(df, clustering_features, n_clusters=4)
    df_clustered.to_csv("data/mobile_products_clustered.csv", index=False)
