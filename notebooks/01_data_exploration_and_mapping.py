import pandas as pd
import folium

# Load the dataset
df = pd.read_csv("./data/NYC_EV_Fleet_Station_Network_20250709.csv")
df.head()

# Quick check
print("Number of stations:", len(df))
print("Column names:", df.columns.tolist())

# Drop rows with missing latitude or longitude
df_clean = df.dropna(subset=["LATITUDE", "LONGITUDE"])

# Initialize base map centered on NYC
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Add stations as blue circles
for _, row in df_clean.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=3,
        popup=row.get("STATION NAME", ""),
        color="blue",
        fill=True,
        fill_opacity=0.6
    ).add_to(nyc_map)

# Display the map
nyc_map

