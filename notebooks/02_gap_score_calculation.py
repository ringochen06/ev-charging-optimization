"""
Calculate Gap Scores to identify charging demand gaps
Input: NYC EV charging station data
Output: Gap scores by region with priority rankings
"""

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. DATA LOADING AND PREPROCESSING
def load_and_clean_data(file_path):
    """Load and clean the charging station data"""
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
    return df_clean

# 2. GAP SCORE COMPONENTS
def calculate_charging_density(df, area_type='borough'):
    """Calculate charging station density by borough"""
    borough_counts = df.groupby('BOROUGH').size()
    
    # NYC borough areas in square miles
    borough_areas = {
        'Manhattan': 23, 'Brooklyn': 70, 'Queens': 109,
        'Bronx': 42, 'Staten Island': 58
    }
    
    charging_density = {}
    for borough in borough_counts.index:
        if borough in borough_areas:
            density = borough_counts[borough] / borough_areas[borough]
            charging_density[borough] = density
    
    return pd.Series(charging_density, name='charging_density')

def get_ev_demand_proxy():
    """Estimate EV demand by borough based on demographics and income"""
    # Based on high income, young population, environmental consciousness
    borough_ev_demand = {
        'Manhattan': 8.5,      # High income, environmentally conscious
        'Brooklyn': 6.2,       # Young population
        'Queens': 5.8,         # Diverse communities  
        'Bronx': 3.4,          # Lower relative income
        'Staten Island': 4.1   # Suburban, but high vehicle dependency
    }
    return pd.Series(borough_ev_demand, name='ev_demand_proxy')

def get_traffic_intensity():
    """Estimate traffic intensity by borough"""
    borough_traffic = {
        'Manhattan': 9.2,      # Highest traffic density
        'Brooklyn': 6.8,       # High density residential
        'Queens': 6.1,         # Airport, highways
        'Bronx': 5.3,          # Medium density
        'Staten Island': 4.5   # Lowest density, but vehicle dependent
    }
    return pd.Series(borough_traffic, name='traffic_intensity')

# 3. GAP SCORE CALCULATION
def calculate_gap_score(ev_demand, traffic_intensity, charging_density, 
                       w1=0.5, w2=0.3, w3=0.2):
    """
    Calculate Gap Score using the formula:
    Gap_Score = w1 * EV_Demand + w2 * Traffic_Intensity - w3 * Charging_Density
    """
    common_boroughs = set(ev_demand.index) & set(traffic_intensity.index) & set(charging_density.index)
    
    gap_scores = {}
    for borough in common_boroughs:
        gap_score = (w1 * ev_demand[borough] + 
                    w2 * traffic_intensity[borough] - 
                    w3 * charging_density[borough])
        gap_scores[borough] = gap_score
    
    return pd.Series(gap_scores, name='gap_score').sort_values(ascending=False)

# 4. VISUALIZATION
def create_component_charts(ev_demand, traffic_intensity, charging_density, gap_scores):
    """Create comparison charts for all gap score components"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # EV Demand
    axes[0,0].bar(ev_demand.index, ev_demand.values, color='lightblue', alpha=0.7)
    axes[0,0].set_title('EV Demand Proxy by Borough')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Traffic Intensity
    axes[0,1].bar(traffic_intensity.index, traffic_intensity.values, color='lightgreen', alpha=0.7)
    axes[0,1].set_title('Traffic Intensity by Borough')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Charging Density
    axes[1,0].bar(charging_density.index, charging_density.values, color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Current Charging Density (stations/sq mi)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Gap Scores
    colors = ['red' if x > 6 else 'orange' if x > 4 else 'green' for x in gap_scores.values]
    axes[1,1].bar(gap_scores.index, gap_scores.values, color=colors, alpha=0.7)
    axes[1,1].set_title('Gap Scores by Borough')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_gap_score_map(df_stations, gap_scores, ev_demand, traffic_intensity, charging_density):
    """Create interactive map showing gap scores"""
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    
    borough_centers = {
        'Manhattan': [40.7831, -73.9712],
        'Brooklyn': [40.6782, -73.9442], 
        'Queens': [40.7282, -73.7949],
        'Bronx': [40.8448, -73.8648],
        'Staten Island': [40.5795, -74.1502]
    }
    
    # Add gap score markers
    for borough, score in gap_scores.items():
        if borough in borough_centers:
            color = 'red' if score > 6 else 'orange' if score > 4 else 'green'
            priority = 'HIGH' if score > 6 else 'MEDIUM' if score > 4 else 'LOW'
            
            folium.Marker(
                location=borough_centers[borough],
                popup=f"<b>{borough}</b><br>"
                      f"Gap Score: {score:.2f}<br>"
                      f"Priority: {priority}<br>"
                      f"EV Demand: {ev_demand.get(borough, 0):.1f}<br>"
                      f"Traffic: {traffic_intensity.get(borough, 0):.1f}<br>"
                      f"Charging Density: {charging_density.get(borough, 0):.1f}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
    
    # Add existing charging stations
    for _, station in df_stations.iterrows():
        folium.CircleMarker(
            location=[station['LATITUDE'], station['LONGITUDE']],
            radius=2,
            popup=station.get('STATION NAME', 'Unknown'),
            color='blue',
            fill=True,
            fillOpacity=0.3
        ).add_to(m)
    
    return m

def get_priority_recommendations(gap_scores):
    """Generate priority recommendations based on gap scores"""
    high_priority = gap_scores[gap_scores > 6].index.tolist()
    medium_priority = gap_scores[(gap_scores > 4) & (gap_scores <= 6)].index.tolist()
    low_priority = gap_scores[gap_scores <= 4].index.tolist()
    
    return {
        'high': high_priority,
        'medium': medium_priority, 
        'low': low_priority,
        'stats': {
            'mean': gap_scores.mean(),
            'std': gap_scores.std(),
            'min': gap_scores.min(),
            'max': gap_scores.max()
        }
    }

# 5. MAIN EXECUTION
def main():
    df_clean = load_and_clean_data("data/NYC_EV_Fleet_Station_Network_20250709.csv")
    
    # Calculate components
    charging_density = calculate_charging_density(df_clean)
    ev_demand = get_ev_demand_proxy()
    traffic_intensity = get_traffic_intensity()
    
    # Calculate gap scores
    gap_scores = calculate_gap_score(ev_demand, traffic_intensity, charging_density)
    
    # Display results
    print("Gap Scores by Borough:")
    for borough, score in gap_scores.items():
        print(f"{borough}: {score:.2f}")
    
    # Generate recommendations
    recommendations = get_priority_recommendations(gap_scores)
    print(f"\nHIGH PRIORITY: {recommendations['high']}")
    print(f"MEDIUM PRIORITY: {recommendations['medium']}")
    print(f"LOW PRIORITY: {recommendations['low']}")
    
    # Create visualizations
    fig = create_component_charts(ev_demand, traffic_intensity, charging_density, gap_scores)
    plt.show()
    
    # Create and save map
    gap_map = create_gap_score_map(df_clean, gap_scores, ev_demand, traffic_intensity, charging_density)
    gap_map.save('nyc_gap_score_map.html')
    
    # Save results
    gap_scores.to_csv('gap_scores_by_borough.csv')
    
    return gap_scores, recommendations

if __name__ == "__main__":
    gap_scores, recommendations = main()