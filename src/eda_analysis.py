"""
===============================================================================
Module: Exploratory Data Analysis (EDA)
Description: Comprehensive EDA with visualizations using Seaborn, Matplotlib, Plotly
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def generate_eda_report(df, output_dir="reports"):
    """Generate comprehensive EDA report with visualizations."""

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 70)

    # 1. Basic Statistics
    print("\n" + "=" * 70)
    print("1. BASIC STATISTICAL SUMMARY")
    print("=" * 70)
    print(df.describe())

    # 2. Product Distribution by Brand
    print("\n" + "=" * 70)
    print("2. PRODUCT DISTRIBUTION BY BRAND")
    print("=" * 70)
    brand_counts = df['Brand'].value_counts()
    print(brand_counts)

    fig = px.bar(
        x=brand_counts.index, 
        y=brand_counts.values,
        labels={'x': 'Brand', 'y': 'Number of Products'},
        title='Product Distribution by Brand',
        color=brand_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.write_html(f"{output_dir}/brand_distribution.html")
    try:
        fig.write_image(f"{output_dir}/brand_distribution.png", width=1000, height=600)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: brand_distribution.html & .png")

    # 3. Price Analysis
    print("\n" + "=" * 70)
    print("3. PRICE ANALYSIS")
    print("=" * 70)
    print(f"Price Statistics:")
    print(f"  Mean Price: ${df['Price_USD'].mean():.2f}")
    print(f"  Median Price: ${df['Price_USD'].median():.2f}")
    print(f"  Min Price: ${df['Price_USD'].min():.2f}")
    print(f"  Max Price: ${df['Price_USD'].max():.2f}")
    print(f"  Std Dev: ${df['Price_USD'].std():.2f}")

    # Price distribution
    fig = px.histogram(df, x='Price_USD', nbins=50,
                       title='Price Distribution of Mobile Products',
                       labels={'Price_USD': 'Price (USD)', 'count': 'Frequency'},
                       color_discrete_sequence=['#636EFA'])
    fig.write_html(f"{output_dir}/price_distribution.html")
    try:
        fig.write_image(f"{output_dir}/price_distribution.png", width=1000, height=600)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: price_distribution.html & .png")

    # 4. Rating Analysis
    print("\n" + "=" * 70)
    print("4. RATING ANALYSIS")
    print("=" * 70)
    print(f"Rating Statistics:")
    print(f"  Mean Rating: {df['Rating'].mean():.2f}")
    print(f"  Median Rating: {df['Rating'].median():.2f}")

    # Top rated products
    top_rated = df.nlargest(10, 'Rating')[['Brand', 'Model', 'Price_USD', 'Rating', 'Num_Reviews']]
    print(f"\nTop 10 Highest Rated Products:")
    print(top_rated.to_string(index=False))

    # Lowest rated products
    low_rated = df.nsmallest(10, 'Rating')[['Brand', 'Model', 'Price_USD', 'Rating', 'Num_Reviews']]
    print(f"\nTop 10 Lowest Rated Products:")
    print(low_rated.to_string(index=False))

    fig = px.box(df, x='Brand', y='Rating', 
                 title='Rating Distribution by Brand',
                 labels={'Rating': 'Product Rating', 'Brand': 'Brand'},
                 color='Brand')
    fig.write_html(f"{output_dir}/rating_by_brand.html")
    try:
        fig.write_image(f"{output_dir}/rating_by_brand.png", width=1200, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: rating_by_brand.html & .png")

    # 5. Price vs Rating Relationship
    print("\n" + "=" * 70)
    print("5. PRICE vs RATING RELATIONSHIP")
    print("=" * 70)

    correlation = df['Price_USD'].corr(df['Rating'])
    print(f"Correlation between Price and Rating: {correlation:.4f}")

    fig = px.scatter(df, x='Price_USD', y='Rating', 
                     color='Brand', size='Num_Reviews',
                     hover_data=['Model', 'RAM_GB', 'Storage_GB'],
                     title='Price vs Rating (Bubble size = Number of Reviews)',
                     labels={'Price_USD': 'Price (USD)', 'Rating': 'Product Rating'})
    fig.write_html(f"{output_dir}/price_vs_rating.html")
    try:
        fig.write_image(f"{output_dir}/price_vs_rating.png", width=1200, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: price_vs_rating.html & .png")

    # 6. Specifications Analysis
    print("\n" + "=" * 70)
    print("6. SPECIFICATIONS ANALYSIS")
    print("=" * 70)

    specs = ['RAM_GB', 'Storage_GB', 'Battery_mAh', 'Camera_MP', 'Screen_Size_inch']
    print("Average Specifications by Brand (Top 5 brands):")
    top5_brands = df['Brand'].value_counts().head(5).index
    brand_specs = df[df['Brand'].isin(top5_brands)].groupby('Brand')[specs].mean()
    print(brand_specs.round(2))

    # RAM distribution
    fig = px.pie(df, names='RAM_GB', title='RAM Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.write_html(f"{output_dir}/ram_distribution.html")
    try:
        fig.write_image(f"{output_dir}/ram_distribution.png", width=800, height=600)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: ram_distribution.html & .png")

    # 7. Country-wise Analysis
    print("\n" + "=" * 70)
    print("7. COUNTRY-WISE ANALYSIS")
    print("=" * 70)

    country_stats = df.groupby('Country').agg({
        'Price_USD': ['mean', 'count'],
        'Rating': 'mean',
        'Engagement_Score': 'mean'
    }).round(2)
    country_stats.columns = ['Avg_Price', 'Product_Count', 'Avg_Rating', 'Avg_Engagement']
    print(country_stats)

    fig = px.choropleth(
        country_stats.reset_index(),
        locations='Country',
        locationmode='country names',
        color='Avg_Price',
        hover_data=['Product_Count', 'Avg_Rating'],
        title='Average Product Price by Country',
        color_continuous_scale='Plasma'
    )
    fig.write_html(f"{output_dir}/country_price_map.html")
    try:
        fig.write_image(f"{output_dir}/country_price_map.png", width=1200, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: country_price_map.html & .png")

    # 8. Correlation Heatmap
    print("\n" + "=" * 70)
    print("8. CORRELATION ANALYSIS")
    print("=" * 70)

    numeric_cols = ['Price_USD', 'Rating', 'RAM_GB', 'Storage_GB', 'Battery_mAh',
                    'Camera_MP', 'Screen_Size_inch', 'Weight_g', 'Processor_Score',
                    'Engagement_Score', 'Feature_Score', 'Value_Score']
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.write_html(f"{output_dir}/correlation_heatmap.html")
    try:
        fig.write_image(f"{output_dir}/correlation_heatmap.png", width=1000, height=900)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: correlation_heatmap.html & .png")

    # 9. Feature Score Analysis
    print("\n" + "=" * 70)
    print("9. FEATURE SCORE & VALUE ANALYSIS")
    print("=" * 70)
    print(f"Feature Score - Mean: {df['Feature_Score'].mean():.2f}, Std: {df['Feature_Score'].std():.2f}")
    print(f"Value Score - Mean: {df['Value_Score'].mean():.2f}, Std: {df['Value_Score'].std():.2f}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Feature Score Distribution', 'Value Score Distribution'))
    fig.add_trace(go.Histogram(x=df['Feature_Score'], name='Feature Score', marker_color='#636EFA'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Value_Score'], name='Value Score', marker_color='#EF553B'), row=1, col=2)
    fig.update_layout(title_text='Feature & Value Score Distributions', showlegend=False)
    fig.write_html(f"{output_dir}/score_distributions.html")
    try:
        fig.write_image(f"{output_dir}/score_distributions.png", width=1200, height=500)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: score_distributions.html & .png")

    # 10. 5G and Premium Features Analysis
    print("\n" + "=" * 70)
    print("10. PREMIUM FEATURES ANALYSIS")
    print("=" * 70)

    features = ['5G_Support', 'Wireless_Charging', 'Fast_Charging', 'Water_Resistant']
    for feat in features:
        pct = df[feat].mean() * 100
        print(f"  {feat}: {pct:.1f}% of products have this feature")

    feature_by_brand = df.groupby('Brand')[features].mean() * 100
    fig = px.bar(feature_by_brand, barmode='group',
                 title='Premium Features by Brand (%)',
                 labels={'value': 'Percentage (%)', 'variable': 'Feature'})
    fig.write_html(f"{output_dir}/premium_features.html")
    try:
        fig.write_image(f"{output_dir}/premium_features.png", width=1200, height=700)
    except Exception:
        pass  # kaleido not available
    print(f"  -> Saved: premium_features.html & .png")

    print("\n" + "=" * 70)
    print("EDA REPORT GENERATION COMPLETE!")
    print("=" * 70)
    print(f"All visualizations saved to: {output_dir}/")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/mobile_products_processed.csv")
    generate_eda_report(df)
