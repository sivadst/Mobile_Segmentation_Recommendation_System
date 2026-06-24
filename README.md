# 📱 Mobile Product Segmentation & Recommendation System

## Project Overview
End-to-end Machine Learning pipeline for mobile phone market segmentation and intelligent product recommendations using Python, K-Means clustering, and Cosine Similarity.

## 🎯 Objectives
- Clean and preprocess mobile product data
- Perform EDA to identify trends and patterns
- Apply K-Means clustering for product segmentation
- Build similarity-based recommendation system
- Develop interactive Streamlit application

## 📁 Project Structure
```
Mobile_Segmentation_Recommendation_System/
├── data/
│   ├── mobile_products_raw.csv          # Original dataset
│   ├── mobile_products_processed.csv    # Cleaned & engineered data
│   └── mobile_products_clustered.csv    # Data with cluster labels
├── src/
│   ├── data_preprocessing.py            # Data cleaning & feature engineering
│   ├── eda_analysis.py                  # Exploratory data analysis
│   ├── clustering_model.py              # K-Means clustering implementation
│   └── recommendation_system.py         # Similarity-based recommendations
├── app/
│   └── streamlit_app.py                 # Interactive web application
├── models/
│   ├── kmeans_model.pkl                 # Trained clustering model
│   └── recommendation_system.pkl        # Trained recommendation system
├── reports/                             # Generated visualizations
├── notebooks/                           # Jupyter notebooks
├── main_pipeline.py                     # End-to-end pipeline runner
└── requirements.txt                     # Python dependencies
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main_pipeline.py
```

### Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## 📊 Key Features
| Feature | Description |
|---------|-------------|
| **4 Market Segments** | Premium Flagship, Upper Mid-Range, Mid-Range, Budget |
| **18 Features** | Price, specs, ratings, engineered features |
| **Cosine Similarity** | For product-to-product recommendations |
| **Interactive Visualizations** | Plotly, Seaborn, Matplotlib |
| **4 Recommendation Modes** | By Product, By Features, By Cluster, Cross-Cluster |

## 📈 Clustering Metrics
- Silhouette Score: 0.1854
- Calinski-Harabasz Index: 156.32
- Davies-Bouldin Index: 1.4231

## 🏆 Market Segments
| Cluster | Segment | Avg Price | Characteristics |
|---------|---------|-----------|-----------------|
| 0 | Premium Flagship | $1000+ | Top specs, premium features, excellent ratings |
| 1 | Upper Mid-Range | $600-900 | Balanced specs, good value, strong performance |
| 2 | Mid-Range | $300-600 | Decent specs, affordable, everyday use |
| 3 | Budget/Entry-Level | <$300 | Basic features, cost-effective, essential functions |

## 🛠️ Technical Stack
- **Python** - Core language
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - Machine Learning (K-Means, StandardScaler, Cosine Similarity)
- **Plotly/Seaborn/Matplotlib** - Data visualization
- **Streamlit** - Web application framework

## 📋 Deliverables
- [x] CSV dataset (raw & processed)
- [x] Python preprocessing scripts
- [x] K-Means clustering model
- [x] Cluster analysis report with insights
- [x] Similarity-based recommendation system
- [x] Interactive Streamlit application
- [x] EDA report with visualizations
- [x] Complete documentation

## 👤 Author
selvasiva


## 📄 License
This project is for educational purposes.
# Mobile_Segmentation_Recommendation_System
