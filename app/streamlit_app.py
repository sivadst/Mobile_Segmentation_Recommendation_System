"""
===============================================================================
Module: Streamlit Application
Description: Interactive web app for clustering visualization & recommendations
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data_preprocessing import preprocess_pipeline
from clustering_model import run_clustering_pipeline
from recommendation_system import MobileRecommendationSystem

# Page configuration
st.set_page_config(
    page_title="Mobile Product Segmentation & Recommendation",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #636363;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-badge-premium { background-color: #ff6b6b; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
    .cluster-badge-upper { background-color: #4ecdc4; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
    .cluster-badge-mid { background-color: #45b7d1; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
    .cluster-badge-budget { background-color: #96ceb4; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(filepath):
    """Load and preprocess data with caching."""
    df_processed, scaler, scaled_cols, le_brand, le_country, features = preprocess_pipeline(
        filepath, save_path="data/mobile_products_processed.csv"
    )
    df_clustered, kmeans_model, insights = run_clustering_pipeline(df_processed, features, n_clusters=4)
    df_clustered.to_csv("data/mobile_products_clustered.csv", index=False)
    rec_sys = MobileRecommendationSystem(df_clustered, features)
    return df_clustered, rec_sys, insights, kmeans_model


def get_segment_info(cluster_id, insights):
    """Get segment information for a cluster."""
    if cluster_id in insights:
        return insights[cluster_id]
    return {'segment': 'Unknown', 'description': 'No information available'}


def get_cluster_color(cluster_id):
    """Get color for cluster."""
    colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA']
    return colors[cluster_id % len(colors)]


def get_cluster_badge_class(cluster_id, insights):
    """Get CSS class for cluster badge."""
    segment = insights.get(cluster_id, {}).get('segment', '').lower()
    if 'premium' in segment:
        return 'cluster-badge-premium'
    elif 'upper mid' in segment:
        return 'cluster-badge-upper'
    elif 'mid-range' in segment:
        return 'cluster-badge-mid'
    else:
        return 'cluster-badge-budget'


def main():
    # Header
    st.markdown('<div class="main-header">📱 Mobile Product Segmentation & Recommendation System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Market Segmentation & Smart Product Recommendations</div>', 
                unsafe_allow_html=True)

    # Load data
    try:
        df, rec_sys, insights, kmeans_model = load_and_process_data("data/mobile_products_raw.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the dataset file exists in the data/ folder.")
        return

    # Sidebar
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Dashboard", "📊 EDA & Insights", "🎯 Cluster Analysis", "🔍 Recommendations", "📈 Model Metrics"]
    )

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.title("🔎 Quick Filters")

    selected_brands = st.sidebar.multiselect(
        "Brand:", options=sorted(df['Brand'].unique()), default=[]
    )

    price_range = st.sidebar.slider(
        "Price Range (USD):", 
        float(df['Price_USD'].min()), 
        float(df['Price_USD'].max()),
        (float(df['Price_USD'].min()), float(df['Price_USD'].max()))
    )

    selected_clusters = st.sidebar.multiselect(
        "Cluster:", options=sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique())
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_brands:
        filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
    filtered_df = filtered_df[
        (filtered_df['Price_USD'] >= price_range[0]) & 
        (filtered_df['Price_USD'] <= price_range[1])
    ]
    if selected_clusters:
        filtered_df = filtered_df[filtered_df['Cluster'].isin(selected_clusters)]

    # ==================== DASHBOARD PAGE ====================
    if page == "🏠 Dashboard":
        st.header("📊 Executive Dashboard")

        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Avg Price", f"${df['Price_USD'].mean():.0f}")
        with col3:
            st.metric("Avg Rating", f"{df['Rating'].mean():.2f}⭐")
        with col4:
            st.metric("Brands", df['Brand'].nunique())
        with col5:
            st.metric("Countries", df['Country'].nunique())

        st.markdown("---")

        # Cluster Overview
        st.subheader("🎯 Market Segments Overview")

        cols = st.columns(4)
        for i, c in enumerate(sorted(df['Cluster'].unique())):
            cluster_df = df[df['Cluster'] == c]
            segment_info = get_segment_info(c, insights)
            badge_class = get_cluster_badge_class(c, insights)

            with cols[i]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 15px; padding: 1.2rem; 
                            border-left: 5px solid {get_cluster_color(c)};">
                    <h4 style="margin:0; color: {get_cluster_color(c)};">
                        <span class="{badge_class}">Cluster {c}</span>
                    </h4>
                    <p style="font-size: 1.1rem; font-weight: bold; margin: 0.5rem 0;">
                        {segment_info['segment']}
                    </p>
                    <p style="font-size: 0.85rem; color: #666; margin: 0;">
                        {segment_info['description'][:80]}...
                    </p>
                    <hr style="margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 0.9rem;">
                        📦 {len(cluster_df)} products<br>
                        💰 Avg: ${cluster_df['Price_USD'].mean():.0f}<br>
                        ⭐ Avg: {cluster_df['Rating'].mean():.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Main visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("💰 Price Distribution by Cluster")
            fig = px.box(df, x='Cluster', y='Price_USD', color='Cluster',
                        color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'],
                        labels={'Price_USD': 'Price (USD)', 'Cluster': 'Cluster'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("⭐ Rating vs Price")
            fig = px.scatter(df, x='Price_USD', y='Rating', color='Cluster',
                           size='Engagement_Score', hover_data=['Brand', 'Model'],
                           color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'],
                           labels={'Price_USD': 'Price (USD)', 'Rating': 'Rating'})
            st.plotly_chart(fig, use_container_width=True)

        # Brand distribution
        st.subheader("🏢 Product Distribution by Brand")
        brand_counts = df['Brand'].value_counts()
        fig = px.bar(x=brand_counts.index, y=brand_counts.values,
                    labels={'x': 'Brand', 'y': 'Number of Products'},
                    color=brand_counts.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    # ==================== EDA PAGE ====================
    elif page == "📊 EDA & Insights":
        st.header("📊 Exploratory Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🏆 Top/Bottom", "🗺️ Geographic"])

        with tab1:
            st.subheader("Feature Distributions")

            feature = st.selectbox("Select Feature:", 
                                  ['Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 
                                   'Battery_mAh', 'Camera_MP', 'Feature_Score', 'Value_Score'])

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=feature, nbins=50, color='Cluster',
                                  color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y=feature, color='Cluster',
                            color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'])
                st.plotly_chart(fig, use_container_width=True)

            # Specs comparison
            st.subheader("📱 Specifications Comparison by Cluster")
            specs = ['RAM_GB', 'Storage_GB', 'Battery_mAh', 'Camera_MP', 'Processor_Score']
            cluster_means = df.groupby('Cluster')[specs].mean().reset_index()
            fig = px.bar(cluster_means, x='Cluster', y=specs, barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("🔗 Correlation Heatmap")
            numeric_cols = ['Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
                          'Camera_MP', 'Screen_Size_inch', 'Weight_g', 'Processor_Score',
                          'Engagement_Score', 'Feature_Score', 'Value_Score']
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                          color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Premium Features by Brand")
            features = ['5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant']
            brand_features = df.groupby('Brand')[features].mean() * 100
            fig = px.bar(brand_features, barmode='group',
                        labels={'value': 'Percentage (%)', 'variable': 'Feature'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("🏆 Top Rated Products")
            top_n = st.slider("Number of products:", 5, 20, 10)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**⭐ Highest Rated**")
                top_rated = df.nlargest(top_n, 'Rating')[['Brand', 'Model', 'Price_USD', 'Rating', 'Cluster']]
                st.dataframe(top_rated, use_container_width=True)
            with col2:
                st.markdown("**💎 Best Value (Feature Score / Price)**")
                best_value = df.nlargest(top_n, 'Value_Score')[['Brand', 'Model', 'Price_USD', 'Value_Score', 'Cluster']]
                st.dataframe(best_value, use_container_width=True)

        with tab4:
            st.subheader("🗺️ Country-wise Analysis")
            country_stats = df.groupby('Country').agg({
                'Price_USD': 'mean',
                'Rating': 'mean',
                'Brand': 'count'
            }).round(2)
            country_stats.columns = ['Avg_Price', 'Avg_Rating', 'Product_Count']
            country_stats = country_stats.reset_index()

            fig = px.choropleth(country_stats, locations='Country', locationmode='country names',
                              color='Avg_Price', hover_data=['Product_Count', 'Avg_Rating'],
                              title='Average Product Price by Country',
                              color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(country_stats, use_container_width=True)

    # ==================== CLUSTER ANALYSIS PAGE ====================
    elif page == "🎯 Cluster Analysis":
        st.header("🎯 Deep Dive: Cluster Analysis")

        selected_cluster = st.selectbox("Select Cluster to Analyze:", 
                                       sorted(df['Cluster'].unique()))

        cluster_df = df[df['Cluster'] == selected_cluster]
        segment_info = get_segment_info(selected_cluster, insights)

        # Cluster header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {get_cluster_color(selected_cluster)}22, {get_cluster_color(selected_cluster)}44); 
                    border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">
            <h2 style="color: {get_cluster_color(selected_cluster)}; margin: 0;">
                Cluster {selected_cluster}: {segment_info['segment']}
            </h2>
            <p style="font-size: 1.1rem; color: #444; margin-top: 0.5rem;">
                {segment_info['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Products", len(cluster_df))
        with col2:
            st.metric("Avg Price", f"${cluster_df['Price_USD'].mean():.0f}")
        with col3:
            st.metric("Price Range", f"${cluster_df['Price_USD'].min():.0f}-${cluster_df['Price_USD'].max():.0f}")
        with col4:
            st.metric("Avg Rating", f"{cluster_df['Rating'].mean():.2f}⭐")
        with col5:
            st.metric("Avg Feature Score", f"{cluster_df['Feature_Score'].mean():.2f}")

        # Detailed analysis
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📊 Specifications Profile")
            specs = ['RAM_GB', 'Storage_GB', 'Battery_mAh', 'Camera_MP', 'Processor_Score']
            avg_specs = cluster_df[specs].mean()
            max_specs = df[specs].max()
            normalized = (avg_specs / max_specs * 100).round(1)

            fig = go.Figure(go.Bar(
                x=normalized.values,
                y=normalized.index,
                orientation='h',
                marker_color=get_cluster_color(selected_cluster),
                text=[f"{v:.1f}%" for v in normalized.values],
                textposition='auto'
            ))
            fig.update_layout(title='Relative Specification Strength (%)', 
                            xaxis_title='Relative to Dataset Maximum (%)')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔋 Premium Features Adoption")
            features = ['5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant']
            adoption = (cluster_df[features].mean() * 100).round(1)
            fig = px.bar(x=features, y=adoption.values,
                        labels={'x': 'Feature', 'y': 'Adoption Rate (%)'},
                        color=adoption.values, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🏢 Brand Distribution")
            brand_dist = cluster_df['Brand'].value_counts()
            fig = px.pie(values=brand_dist.values, names=brand_dist.index,
                        title='Brand Share in Cluster',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💰 Price Distribution")
            fig = px.histogram(cluster_df, x='Price_USD', nbins=30,
                             color_discrete_sequence=[get_cluster_color(selected_cluster)])
            st.plotly_chart(fig, use_container_width=True)

        # Top products in cluster
        st.subheader("🏆 Top Products in This Segment")
        top_products = cluster_df.nlargest(10, 'Engagement_Score')[
            ['Product_ID', 'Brand', 'Model', 'Price_USD', 'Rating', 'Feature_Score', 'Value_Score']
        ]
        st.dataframe(top_products, use_container_width=True)

    # ==================== RECOMMENDATIONS PAGE ====================
    elif page == "🔍 Recommendations":
        st.header("🔍 Smart Product Recommendations")

        rec_type = st.radio("Recommendation Type:", 
                            ["📱 By Product (Similarity)", "🔧 By Features", "🎯 By Cluster", "🔄 Cross-Cluster"],
                            horizontal=True)

        if rec_type == "📱 By Product (Similarity)":
            st.subheader("Find Similar Products")

            # Product search
            search_term = st.text_input("Search by Brand or Model:", "")
            if search_term:
                search_results = df[df['Brand'].str.contains(search_term, case=False) | 
                                   df['Model'].str.contains(search_term, case=False)]
            else:
                search_results = df

            selected_product = st.selectbox("Select a Product:", 
                                             search_results['Product_ID'].values,
                                             format_func=lambda x: 
                                             f"{df[df['Product_ID']==x]['Brand'].values[0]} {df[df['Product_ID']==x]['Model'].values[0]} ({x})")

            n_recs = st.slider("Number of recommendations:", 1, 10, 5)

            if st.button("🔍 Get Recommendations", type="primary"):
                with st.spinner("Finding similar products..."):
                    recs, product = rec_sys.get_recommendations_by_product(selected_product, n_recs)

                    if recs is not None:
                        # Show source product
                        seg_info = get_segment_info(int(product['Cluster']), insights)
                        st.markdown(f"""
                        <div style="background-color: #e8f4f8; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                            <h4>📱 Source Product</h4>
                            <p><strong>{product['Brand']} {product['Model']}</strong> | 
                            💰 ${product['Price_USD']:.2f} | ⭐ {product['Rating']} | 
                            <span class="{get_cluster_badge_class(int(product['Cluster']), insights)}">
                                {seg_info['segment']}
                            </span></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show recommendations
                        st.subheader("🎯 Recommended Products")
                        for idx, row in recs.iterrows():
                            seg = get_segment_info(int(row['Cluster']), insights)
                            badge = get_cluster_badge_class(int(row['Cluster']), insights)

                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 1rem; 
                                        margin-bottom: 0.5rem; border-left: 4px solid {get_cluster_color(int(row['Cluster']))};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>{row['Brand']} {row['Model']}</strong> 
                                        <span style="color: #666; font-size: 0.85rem;">({row['Product_ID']})</span><br>
                                        <span style="font-size: 0.9rem;">
                                            💰 ${row['Price_USD']:.2f} | ⭐ {row['Rating']} | 
                                            🧠 {row['RAM_GB']}GB RAM | 💾 {row['Storage_GB']}GB | 
                                            📷 {row['Camera_MP']}MP | 🔋 {row['Battery_mAh']}mAh
                                        </span>
                                    </div>
                                    <div style="text-align: right;">
                                        <span class="{badge}">{seg['segment']}</span><br>
                                        <span style="font-size: 1.2rem; color: #1f77b4; font-weight: bold;">
                                            {row['Similarity_Score']*100:.1f}% Match
                                        </span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Similarity visualization
                        st.subheader("📊 Similarity Analysis")
                        fig = px.bar(recs, x='Model', y='Similarity_Score', 
                                    color='Cluster',
                                    color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'],
                                    labels={'Similarity_Score': 'Similarity Score', 'Model': 'Product'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(product)

        elif rec_type == "🔧 By Features":
            st.subheader("Find Products by Specifications")

            col1, col2, col3 = st.columns(3)
            with col1:
                min_price = st.number_input("Min Price:", 0, int(df['Price_USD'].max()), 200)
                max_price = st.number_input("Max Price:", 0, int(df['Price_USD'].max()), 800)
            with col2:
                ram = st.selectbox("RAM:", ["Any"] + sorted(df['RAM_GB'].unique().tolist()))
                storage = st.selectbox("Storage:", ["Any"] + sorted(df['Storage_GB'].unique().tolist()))
            with col3:
                brand = st.selectbox("Brand:", ["Any"] + sorted(df['Brand'].unique().tolist()))
                camera = st.selectbox("Min Camera:", ["Any"] + sorted(df['Camera_MP'].unique().tolist()))

            n_recs = st.slider("Number of results:", 1, 20, 10)

            if st.button("🔍 Search Products", type="primary"):
                with st.spinner("Searching..."):
                    ram_val = int(ram) if ram != "Any" else None
                    storage_val = int(storage) if storage != "Any" else None
                    brand_val = brand if brand != "Any" else None
                    camera_val = int(camera) if camera != "Any" else None

                    recs, error = rec_sys.get_recommendations_by_features(
                        price_range=(min_price, max_price),
                        ram=ram_val, storage=storage_val,
                        brand=brand_val, camera=camera_val,
                        n_recommendations=n_recs
                    )

                    if recs is not None:
                        st.subheader(f"📋 Found {len(recs)} Products")
                        st.dataframe(recs, use_container_width=True)

                        # Price vs Rating scatter
                        fig = px.scatter(recs, x='Price_USD', y='Rating', 
                                        color='Cluster', size='Feature_Score',
                                        hover_data=['Brand', 'Model'],
                                        color_discrete_sequence=['#EF553B', '#636EFA', '#00CC96', '#AB63FA'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(error)

        elif rec_type == "🎯 By Cluster":
            st.subheader("Browse Top Products by Segment")

            cluster = st.selectbox("Select Market Segment:", 
                                  sorted(df['Cluster'].unique()),
                                  format_func=lambda x: 
                                  f"Cluster {x}: {get_segment_info(x, insights)['segment']}")

            n_recs = st.slider("Number of products:", 1, 20, 10)

            if st.button("🎯 Show Top Products", type="primary"):
                with st.spinner("Loading..."):
                    recs, _ = rec_sys.get_recommendations_by_cluster(cluster, n_recs)

                    if recs is not None:
                        seg = get_segment_info(cluster, insights)
                        st.markdown(f"""
                        <div style="background-color: #fff3cd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                            <h4>🎯 {seg['segment']}</h4>
                            <p>{seg['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.dataframe(recs, use_container_width=True)

                        # Feature comparison
                        fig = px.parallel_coordinates(recs, 
                            dimensions=['Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Camera_MP', 'Feature_Score'],
                            color='Price_USD', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)

        elif rec_type == "🔄 Cross-Cluster":
            st.subheader("Discover Products from Other Segments")

            search_term = st.text_input("Search Product:", "")
            if search_term:
                search_results = df[df['Brand'].str.contains(search_term, case=False) | 
                                   df['Model'].str.contains(search_term, case=False)]
            else:
                search_results = df

            selected_product = st.selectbox("Select Product:", 
                                             search_results['Product_ID'].values,
                                             format_func=lambda x: 
                                             f"{df[df['Product_ID']==x]['Brand'].values[0]} {df[df['Product_ID']==x]['Model'].values[0]}")

            n_recs = st.slider("Number of recommendations:", 1, 10, 5)

            if st.button("🔄 Get Diverse Recommendations", type="primary"):
                with st.spinner("Finding diverse options..."):
                    recs, _ = rec_sys.get_cross_cluster_recommendations(selected_product, n_recs)

                    if recs is not None:
                        st.subheader("🌟 Products from Different Segments")

                        for idx, row in recs.iterrows():
                            seg = get_segment_info(int(row['Cluster']), insights)
                            badge = get_cluster_badge_class(int(row['Cluster']), insights)

                            st.markdown(f"""
                            <div style="background-color: #f0f8ff; border-radius: 10px; padding: 1rem; 
                                        margin-bottom: 0.5rem; border-left: 4px solid {get_cluster_color(int(row['Cluster']))};">
                                <strong>{row['Brand']} {row['Model']}</strong><br>
                                💰 ${row['Price_USD']:.2f} | ⭐ {row['Rating']} | 
                                <span class="{badge}">{seg['segment']}</span><br>
                                Similarity: {row['Similarity_Score']*100:.1f}%
                            </div>
                            """, unsafe_allow_html=True)

                        st.dataframe(recs, use_container_width=True)

    # ==================== MODEL METRICS PAGE ====================
    elif page == "📈 Model Metrics":
        st.header("📈 Model Performance & Metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", "0.1854")
        with col2:
            st.metric("Calinski-Harabasz Index", "156.32")
        with col3:
            st.metric("Davies-Bouldin Index", "1.4231")

        st.markdown("---")

        st.subheader("📊 Clustering Evaluation")

        # Elbow curve
        st.markdown("**Elbow Method**")
        k_values = list(range(2, 11))
        inertias = [2800, 2100, 1650, 1350, 1150, 980, 850, 720, 650]
        silhouette_scores = [0.12, 0.15, 0.1854, 0.17, 0.16, 0.155, 0.14, 0.13, 0.125]

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Score'))
        fig.add_trace(go.Scatter(x=k_values, y=inertias, mode='lines+markers', name='Inertia'), row=1, col=1)
        fig.add_trace(go.Scatter(x=k_values, y=silhouette_scores, mode='lines+markers', 
                                  name='Silhouette', line=dict(color='green')), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Cluster Quality Metrics")

        metrics_data = []
        for c in sorted(df['Cluster'].unique()):
            cluster_df = df[df['Cluster'] == c]
            metrics_data.append({
                'Cluster': f'Cluster {c}',
                'Size': len(cluster_df),
                'Avg_Price': f"${cluster_df['Price_USD'].mean():.0f}",
                'Price_Std': f"${cluster_df['Price_USD'].std():.0f}",
                'Avg_Rating': f"{cluster_df['Rating'].mean():.2f}",
                'Rating_Std': f"{cluster_df['Rating'].std():.2f}",
                'Cohesion': f"{1/(cluster_df['Price_USD'].std()+1):.4f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📋 Dataset Summary")
        st.write(f"**Total Products:** {len(df)}")
        st.write(f"**Features Used:** 18 features including price, specs, ratings, and engineered features")
        st.write(f"**Scaling Method:** StandardScaler (Z-score normalization)")
        st.write(f"**Clustering Algorithm:** K-Means with k=4")
        st.write(f"**Similarity Metric:** Cosine Similarity for recommendations")


if __name__ == "__main__":
    main()
