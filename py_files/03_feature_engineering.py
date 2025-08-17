"""
Create independent features for ML modeling (excluding Gap Score components)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic

# Load data
df_stations = pd.read_csv("./data/NYC_EV_Fleet_Station_Network_20250709.csv")
df_clean = df_stations.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
print(f"Loaded {len(df_clean)} stations")

# 1. Create ZIP Code Features
def create_zip_features():
    """Simulate ZIP code demographic features (use real census data in production)"""
    np.random.seed(42)
    zip_codes = [z for z in range(10001, 10280) if z not in [10003, 10004, 10006]]
    
    features = []
    for zip_code in zip_codes:
        features.append({
            'zipcode': zip_code,
            'median_income': np.random.normal(65000, 25000),
            'renter_percentage': np.random.uniform(0.3, 0.8),
            'population_density': np.random.lognormal(8, 1),
            'transit_accessibility': np.random.uniform(0.2, 1.0),
            'land_use_mix': np.random.uniform(0.1, 0.9),
            'college_educated_pct': np.random.uniform(0.2, 0.7),
            'avg_commute_time': np.random.normal(30, 10),
            'parking_availability': np.random.uniform(0.1, 0.8)
        })
    return pd.DataFrame(features)

# 2. Calculate Distance to Nearest Charger
def calculate_distances(zip_features, stations_df):
    """Calculate distance from ZIP centroid to nearest charging station"""
    np.random.seed(42)
    distances = []
    
    for _, row in zip_features.iterrows():
        # Simulate ZIP centroid
        zip_lat = np.random.uniform(40.4774, 40.9176)
        zip_lon = np.random.uniform(-74.2591, -73.7004)
        
        # Find nearest station
        min_distance = float('inf')
        for _, station in stations_df.iterrows():
            dist = geodesic((zip_lat, zip_lon), 
                          (station['LATITUDE'], station['LONGITUDE'])).miles
            min_distance = min(min_distance, dist)
        distances.append(min_distance)
    
    return distances

# 3. Create Target Variable (Gap Score)
def calculate_gap_score(features_df):
    """Gap_Score = 0.5 * EV_per_capita + 0.3 * traffic_volume - 0.2 * charger_density"""
    np.random.seed(42)
    
    # Components (excluded from features)
    ev_per_capita = (features_df['median_income'] / 100000) * np.random.uniform(0.02, 0.08, len(features_df))
    traffic_volume = (features_df['population_density'] / 1000 + 
                     features_df['avg_commute_time'] / 10) * np.random.uniform(0.8, 1.2, len(features_df))
    charger_density = np.random.exponential(2, len(features_df))
    
    return 0.5 * ev_per_capita + 0.3 * traffic_volume - 0.2 * charger_density

# 4. Main Pipeline
print("Creating features...")
zip_features = create_zip_features()
zip_features['distance_to_nearest_charger'] = calculate_distances(zip_features, df_clean)

def assign_borough(zipcode):
    if 10001 <= zipcode <= 10282: return 'Manhattan'
    elif 11201 <= zipcode <= 11256: return 'Brooklyn'
    elif 11101 <= zipcode <= 11697: return 'Queens'
    elif 10451 <= zipcode <= 10475: return 'Bronx'
    else: return 'Staten Island'

zip_features['borough'] = zip_features['zipcode'].apply(assign_borough)
zip_features['gap_score'] = calculate_gap_score(zip_features)

# Encode and clean
le_borough = LabelEncoder()
zip_features['borough_encoded'] = le_borough.fit_transform(zip_features['borough'])
zip_features['median_income'] = np.clip(zip_features['median_income'], 20000, 200000)

# Feature columns (independent variables only)
feature_columns = [
    'median_income', 'renter_percentage', 'population_density',
    'transit_accessibility', 'land_use_mix', 'college_educated_pct',
    'avg_commute_time', 'parking_availability', 'distance_to_nearest_charger',
    'borough_encoded'
]

# Save data
X = zip_features[feature_columns]
y = zip_features['gap_score']

zip_features.to_csv('../data/processed_zip_features.csv', index=False)
X.to_csv('../data/X_features.csv', index=False)
y.to_csv('../data/y_target.csv', index=False)

print(f"Dataset: {X.shape}, Target range: {y.min():.2f} to {y.max():.2f}")
print("Files saved: processed_zip_features.csv, X_features.csv, y_target.csv")