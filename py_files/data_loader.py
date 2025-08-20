"""
Data Loader for NYC EV Charging Station Optimization
"""

import pandas as pd
import numpy as np
import requests
import geopandas as gpd
from io import StringIO


class NYCDataLoader:
    def __init__(self):
        self.base_url = "https://data.cityofnewyork.us/resource/"

    def load_charging_stations(self):
        """Load existing EV charging stations"""
        try:
            df = pd.read_csv("./data/NYC_EV_Fleet_Station_Network_20250709.csv")
            df_clean = df.dropna(subset=["LATITUDE", "LONGITUDE"])
            print(f"Loaded {len(df_clean)} charging stations")
            return df_clean
        except Exception as e:
            print(f"Error loading charging stations: {e}")
            return None

    def load_traffic_data(self):
        """Load NYC traffic volume data"""
        # NYC Open Data: Automated Traffic Volume Counts
        url = f"{self.base_url}7ym2-wayt.json?$limit=10000"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                print(f"Loaded {len(df)} traffic records")
                return df
            else:
                print("Using fallback traffic data")
                return self._create_fallback_traffic_data()
        except Exception as e:
            print(f"Error loading traffic data: {e}, using fallback")
            return self._create_fallback_traffic_data()

    def load_demographic_data(self):
        """Load NYC demographic data by ZIP code"""
        # Simplified demographic data (in production, use ACS API)
        zip_codes = []

        # Manhattan ZIPs
        for zip_code in range(10001, 10040):
            zip_codes.append(
                {
                    "zipcode": zip_code,
                    "borough": "Manhattan",
                    "median_income": np.random.normal(85000, 30000),
                    "college_educated_pct": np.random.uniform(0.4, 0.8),
                    "population_density": np.random.uniform(50000, 100000),
                    "latitude": 40.7589 + np.random.uniform(-0.05, 0.05),
                    "longitude": -73.9851 + np.random.uniform(-0.03, 0.03),
                }
            )

        # Brooklyn ZIPs
        for zip_code in range(11201, 11240):
            zip_codes.append(
                {
                    "zipcode": zip_code,
                    "borough": "Brooklyn",
                    "median_income": np.random.normal(55000, 20000),
                    "college_educated_pct": np.random.uniform(0.25, 0.6),
                    "population_density": np.random.uniform(20000, 50000),
                    "latitude": 40.6892 + np.random.uniform(-0.08, 0.08),
                    "longitude": -73.9442 + np.random.uniform(-0.08, 0.08),
                }
            )

        # Queens ZIPs
        for zip_code in range(11101, 11140):
            zip_codes.append(
                {
                    "zipcode": zip_code,
                    "borough": "Queens",
                    "median_income": np.random.normal(65000, 25000),
                    "college_educated_pct": np.random.uniform(0.3, 0.65),
                    "population_density": np.random.uniform(15000, 40000),
                    "latitude": 40.7282 + np.random.uniform(-0.1, 0.1),
                    "longitude": -73.7949 + np.random.uniform(-0.1, 0.1),
                }
            )

        # Bronx ZIPs
        for zip_code in range(10451, 10475):
            zip_codes.append(
                {
                    "zipcode": zip_code,
                    "borough": "Bronx",
                    "median_income": np.random.normal(45000, 15000),
                    "college_educated_pct": np.random.uniform(0.2, 0.5),
                    "population_density": np.random.uniform(15000, 35000),
                    "latitude": 40.8448 + np.random.uniform(-0.05, 0.05),
                    "longitude": -73.8648 + np.random.uniform(-0.05, 0.05),
                }
            )

        # Staten Island ZIPs
        for zip_code in range(10301, 10315):
            zip_codes.append(
                {
                    "zipcode": zip_code,
                    "borough": "Staten Island",
                    "median_income": np.random.normal(70000, 20000),
                    "college_educated_pct": np.random.uniform(0.3, 0.6),
                    "population_density": np.random.uniform(5000, 15000),
                    "latitude": 40.5795 + np.random.uniform(-0.05, 0.05),
                    "longitude": -74.1502 + np.random.uniform(-0.05, 0.05),
                }
            )

        df = pd.DataFrame(zip_codes)
        df["median_income"] = np.clip(df["median_income"], 25000, 200000)
        print(f"Created demographic data for {len(df)} ZIP codes")
        return df

    def load_pluto_data(self):
        """Load NYC PLUTO land use data (simplified version)"""
        # In production: use actual PLUTO data
        # For now, create representative land use features

        zip_codes = []
        boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

        for borough in boroughs:
            if borough == "Manhattan":
                zip_range = range(10001, 10040)
                land_use_mix_range = (0.7, 0.9)  # High mixed use
                parking_availability_range = (0.1, 0.3)  # Low parking
            elif borough == "Brooklyn":
                zip_range = range(11201, 11240)
                land_use_mix_range = (0.4, 0.7)
                parking_availability_range = (0.2, 0.5)
            elif borough == "Queens":
                zip_range = range(11101, 11140)
                land_use_mix_range = (0.3, 0.6)
                parking_availability_range = (0.4, 0.7)
            elif borough == "Bronx":
                zip_range = range(10451, 10475)
                land_use_mix_range = (0.3, 0.6)
                parking_availability_range = (0.3, 0.6)
            else:  # Staten Island
                zip_range = range(10301, 10315)
                land_use_mix_range = (0.2, 0.4)  # More residential
                parking_availability_range = (0.6, 0.8)  # High parking

            for zip_code in zip_range:
                zip_codes.append(
                    {
                        "zipcode": zip_code,
                        "borough": borough,
                        "land_use_mix": np.random.uniform(*land_use_mix_range),
                        "parking_availability": np.random.uniform(
                            *parking_availability_range
                        ),
                        "commercial_area_pct": np.random.uniform(0.1, 0.8),
                        "residential_area_pct": np.random.uniform(0.2, 0.9),
                    }
                )

        df = pd.DataFrame(zip_codes)
        print(f"Created PLUTO land use data for {len(df)} ZIP codes")
        return df

    def load_transit_accessibility(self):
        """Load transit accessibility scores"""
        # Based on subway station density and bus routes
        zip_codes = []

        for borough in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
            if borough == "Manhattan":
                zip_range = range(10001, 10040)
                transit_range = (0.8, 1.0)  # Excellent transit
            elif borough == "Brooklyn":
                zip_range = range(11201, 11240)
                transit_range = (0.5, 0.8)  # Good transit
            elif borough == "Queens":
                zip_range = range(11101, 11140)
                transit_range = (0.4, 0.7)  # Moderate transit
            elif borough == "Bronx":
                zip_range = range(10451, 10475)
                transit_range = (0.4, 0.7)  # Moderate transit
            else:  # Staten Island
                zip_range = range(10301, 10315)
                transit_range = (0.2, 0.4)  # Limited transit

            for zip_code in zip_range:
                zip_codes.append(
                    {
                        "zipcode": zip_code,
                        "transit_accessibility": np.random.uniform(*transit_range),
                        "avg_commute_time": (
                            np.random.normal(35, 10)
                            if borough != "Manhattan"
                            else np.random.normal(25, 8)
                        ),
                    }
                )

        df = pd.DataFrame(zip_codes)
        df["avg_commute_time"] = np.clip(df["avg_commute_time"], 15, 60)
        print(f"Created transit data for {len(df)} ZIP codes")
        return df

    def _create_fallback_traffic_data(self):
        """Create fallback traffic data when API fails"""
        boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        traffic_data = []

        for borough in boroughs:
            if borough == "Manhattan":
                traffic_intensity = np.random.uniform(0.8, 1.0)
            elif borough == "Brooklyn":
                traffic_intensity = np.random.uniform(0.6, 0.8)
            elif borough == "Queens":
                traffic_intensity = np.random.uniform(0.5, 0.7)
            elif borough == "Bronx":
                traffic_intensity = np.random.uniform(0.4, 0.6)
            else:  # Staten Island
                traffic_intensity = np.random.uniform(0.3, 0.5)

            traffic_data.append(
                {"borough": borough, "traffic_intensity": traffic_intensity}
            )

        return pd.DataFrame(traffic_data)

    def integrate_all_data(self):
        """Integrate all data sources into a single dataset"""
        print("Loading and integrating all NYC data sources...")

        # Load all datasets
        demographic_data = self.load_demographic_data()
        pluto_data = self.load_pluto_data()
        transit_data = self.load_transit_accessibility()
        charging_stations = self.load_charging_stations()

        # Merge on zipcode
        integrated_data = demographic_data.merge(
            pluto_data, on=["zipcode", "borough"], how="inner"
        )
        integrated_data = integrated_data.merge(transit_data, on="zipcode", how="inner")

        # Calculate distance to nearest charger for each ZIP
        if charging_stations is not None:
            distances = []
            for _, zip_row in integrated_data.iterrows():
                min_dist = float("inf")
                for _, station in charging_stations.iterrows():
                    lat_diff = zip_row["latitude"] - station["LATITUDE"]
                    lon_diff = zip_row["longitude"] - station["LONGITUDE"]
                    dist = np.sqrt(lat_diff**2 + lon_diff**2) * 69  # Approximate miles
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)

            integrated_data["distance_to_nearest_charger"] = distances
        else:
            integrated_data["distance_to_nearest_charger"] = np.random.uniform(
                0.5, 5.0, len(integrated_data)
            )

        # Add renter percentage (correlated with income)
        integrated_data["renter_percentage"] = np.clip(
            0.8
            - (integrated_data["median_income"] - 40000) / 100000
            + np.random.normal(0, 0.1, len(integrated_data)),
            0.2,
            0.9,
        )

        print(f"Successfully integrated data for {len(integrated_data)} ZIP codes")
        return integrated_data


def load_nyc_data():
    """Main function to load all NYC data"""
    loader = NYCDataLoader()
    return loader.integrate_all_data()


if __name__ == "__main__":
    # Test the data loader
    data = load_nyc_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"\nSample data:")
    print(data.head())

    # Save the integrated dataset
    data.to_csv("./data/nyc_integrated_data.csv", index=False)
    print("\nSaved integrated data to './data/nyc_integrated_data.csv'")
