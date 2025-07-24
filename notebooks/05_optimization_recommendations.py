"""
Generate optimal locations for new EV charging stations
Input: Trained model, NYC geographic data, existing stations
Output: Recommended coordinates and priority ranking
"""
import pandas as pd
import numpy as np
import folium
from folium import plugins
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and data
print("Loading trained models and data...")
best_model = joblib.load('../models/best_gap_score_model.pkl')
scaler = joblib.load('../models/feature_scaler.pkl')
model_results = joblib.load('../models/model_results.pkl')
zip_features = pd.read_csv('../data/processed_zip_features.csv')
df_stations = pd.read_csv('./data/NYC_EV_Fleet_Station_Network_20250709.csv')

print(f"Best model: {model_results['best_model_name']}")
print(f"ZIP codes available: {len(zip_features)}")

# 1. Identify High-Priority Areas
def identify_priority_areas(threshold_percentile=75):
    """
    Identify ZIP codes with highest predicted Gap Scores
    """
    # Use actual predictions from the model
    feature_cols = [col for col in zip_features.columns if col not in 
                   ['zipcode', 'borough', 'gap_score']]
    
    X_predict = zip_features[feature_cols]
    
    # Make predictions
    if model_results['best_model_name'] in ['SVR', 'Linear Regression', 'Ridge Regression']:
        X_scaled = scaler.transform(X_predict)
        predictions = best_model.predict(X_scaled)
    else:
        predictions = best_model.predict(X_predict)
    
    zip_features['predicted_gap_score'] = predictions
    
    # Define priority levels
    threshold = np.percentile(predictions, threshold_percentile)
    zip_features['priority'] = np.where(
        predictions >= threshold, 'HIGH',
        np.where(predictions >= np.percentile(predictions, 50), 'MEDIUM', 'LOW')
    )
    
    return zip_features.sort_values('predicted_gap_score', ascending=False)

priority_areas = identify_priority_areas()
high_priority = priority_areas[priority_areas['priority'] == 'HIGH']

print(f"\nPRIORITY ANALYSIS")
print(f"High priority ZIP codes: {len(high_priority)}")
print(f"Top 5 highest Gap Score predictions:")
print(high_priority[['zipcode', 'borough', 'predicted_gap_score']].head())

# 2. Generate Optimal Station Locations
def generate_station_coordinates(high_priority_zips, n_stations=20):
    # Simulate ZIP code centroids (in production, use real geographic data)
    np.random.seed(42)
    
    recommended_locations = []
    for _, zip_data in high_priority_zips.head(n_stations).iterrows():
        # Generate coordinates within ZIP code bounds (simplified)
        base_lat = 40.7128 + np.random.uniform(-0.2, 0.2)
        base_lon = -74.0060 + np.random.uniform(-0.2, 0.2)
        
        # Adjust by borough
        if zip_data['borough'] == 'Brooklyn':
            base_lat -= 0.05
            base_lon += 0.05
        elif zip_data['borough'] == 'Queens':
            base_lat -= 0.02
            base_lon += 0.15
        elif zip_data['borough'] == 'Bronx':
            base_lat += 0.12
            base_lon += 0.05
        elif zip_data['borough'] == 'Staten Island':
            base_lat -= 0.15
            base_lon -= 0.08
        
        # Find optimal spot within ZIP (simulate demand hotspots)
        optimal_lat = base_lat + np.random.uniform(-0.01, 0.01)
        optimal_lon = base_lon + np.random.uniform(-0.01, 0.01)
        
        recommended_locations.append({
            'zipcode': zip_data['zipcode'],
            'borough': zip_data['borough'],
            'predicted_gap_score': zip_data['predicted_gap_score'],
            'priority': zip_data['priority'],
            'recommended_lat': optimal_lat,
            'recommended_lon': optimal_lon,
            'station_type': 'Level 3 Fast Charger' if zip_data['predicted_gap_score'] > priority_areas['predicted_gap_score'].quantile(0.9) else 'Level 2 Charger'
        })
    
    return pd.DataFrame(recommended_locations)

recommended_stations = generate_station_coordinates(high_priority)
print(f"\nGenerated {len(recommended_stations)} recommended station locations")

# 3. Clustering Analysis
def perform_clustering_analysis(recommendations, n_clusters=5):
    """
    Cluster recommended locations for deployment planning
    """  
    coords = recommendations[['recommended_lat', 'recommended_lon']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    recommendations['cluster'] = kmeans.fit_predict(coords)
    
    # Calculate cluster statistics
    cluster_stats = recommendations.groupby('cluster').agg({
        'predicted_gap_score': ['mean', 'count'],
        'recommended_lat': 'mean',
        'recommended_lon': 'mean'
    }).round(4)
    
    cluster_stats.columns = ['avg_gap_score', 'station_count', 'center_lat', 'center_lon']
    cluster_stats = cluster_stats.sort_values('avg_gap_score', ascending=False)
    
    return recommendations, cluster_stats

clustered_stations, cluster_stats = perform_clustering_analysis(recommended_stations)

print(f"\n=== CLUSTERING ANALYSIS ===")
print("Cluster deployment priority (by average Gap Score):")
print(cluster_stats)

# 4. Create Recommendation Map
def create_recommendation_map(recommendations, existing_stations, cluster_stats):
    """
    Create interactive map showing recommended locations
    """
    # Initialize map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    
    # Color palette for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Add existing stations
    existing_clean = existing_stations.dropna(subset=['LATITUDE', 'LONGITUDE'])
    for _, station in existing_clean.iterrows():
        folium.CircleMarker(
            location=[station['LATITUDE'], station['LONGITUDE']],
            radius=2,
            popup=f"Existing: {station.get('STATION NAME', 'Unknown')}",
            color='gray',
            fill=True,
            fillOpacity=0.3
        ).add_to(m)
    
    # Add recommended stations with clustering
    for _, rec in recommendations.iterrows():
        cluster_color = colors[rec['cluster'] % len(colors)]
        
        folium.Marker(
            location=[rec['recommended_lat'], rec['recommended_lon']],
            popup=f"""
            <b>Recommended Station</b><br>
            ZIP: {rec['zipcode']}<br>
            Borough: {rec['borough']}<br>
            Gap Score: {rec['predicted_gap_score']:.2f}<br>
            Type: {rec['station_type']}<br>
            Priority: {rec['priority']}<br>
            Cluster: {rec['cluster']}
            """,
            icon=folium.Icon(color=cluster_color, icon='lightning', prefix='fa')
        ).add_to(m)
    
    # Add cluster centers
    for cluster_id, stats in cluster_stats.iterrows():
        folium.Marker(
            location=[stats['center_lat'], stats['center_lon']],
            popup=f"""
            <b>Cluster {cluster_id} Center</b><br>
            Avg Gap Score: {stats['avg_gap_score']:.2f}<br>
            Stations: {stats['station_count']}<br>
            Deployment Priority: {cluster_id + 1}
            """,
            icon=folium.Icon(color='black', icon='star')
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <b>NYC EV Station Recommendations</b><br>
    <i class="fa fa-lightning" style="color:red"></i> Cluster 1 (Highest Priority)<br>
    <i class="fa fa-lightning" style="color:blue"></i> Cluster 2<br>
    <i class="fa fa-lightning" style="color:green"></i> Cluster 3<br>
    <i class="fa fa-star" style="color:black"></i> Cluster Centers<br>
    <i class="fa fa-circle" style="color:gray"></i> Existing Stations<br>
    <br><b>Deployment Order:</b><br>
    Red → Blue → Green → Purple → Orange
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# Create and save map
recommendation_map = create_recommendation_map(clustered_stations, df_stations, cluster_stats)
recommendation_map.save('../outputs/station_recommendations_map.html')
print("Recommendation map saved as 'station_recommendations_map.html'")

# ===== 5. Business Impact Analysis =====
def calculate_business_impact(recommendations):
    """
    Calculate potential business impact of recommendations
    """
    # Estimate metrics (in production, use real data)
    total_gap_reduction = recommendations['predicted_gap_score'].sum()
    avg_stations_per_cluster = len(recommendations) / recommendations['cluster'].nunique()
    
    # Estimate costs (simplified)
    level3_cost = 50000  # Level 3 charger installation cost
    level2_cost = 15000  # Level 2 charger installation cost
    