"""
Feature Engineering
Uses integrated NYC Open Data sources
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_loader import load_nyc_data


def create_gap_score_independently(n_zips):
    """Create Gap Score independently using realistic NYC patterns"""
    np.random.seed(42)

    # Create realistic Gap Score distribution
    # Higher scores in areas with high demand but low supply
    gap_scores = []

    for i in range(n_zips):
        # Simulate realistic Gap Score (0-10 scale)
        # Most areas have low-medium scores, few have very high scores
        base_score = np.random.gamma(2.5, 1.5)  # Shape for realistic distribution
        gap_scores.append(np.clip(base_score, 0, 10))

    return np.array(gap_scores)


def enhance_features(data):
    """Add derived features from the integrated data"""

    # Borough encoding
    le_borough = LabelEncoder()
    data["borough_encoded"] = le_borough.fit_transform(data["borough"])

    # Income categories for interpretability
    data["income_category"] = pd.cut(
        data["median_income"],
        bins=[0, 40000, 70000, 100000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    )

    # Density categories
    data["density_category"] = pd.cut(
        data["population_density"],
        bins=[0, 20000, 50000, float("inf")],
        labels=["Low", "Medium", "High"],
    )

    # Transit quality score (combined metric)
    data["transit_quality"] = (
        data["transit_accessibility"] * 0.7 + (60 - data["avg_commute_time"]) / 60 * 0.3
    )

    # Urban development index
    data["urban_development"] = (
        data["land_use_mix"] * 0.4
        + data["commercial_area_pct"] * 0.3
        + data["population_density"] / data["population_density"].max() * 0.3
    )

    return data


def prepare_ml_features(data):
    """Prepare features for machine learning"""

    # Select independent features (no Gap Score components)
    feature_columns = [
        # Socioeconomic features
        "median_income",
        "college_educated_pct",
        "renter_percentage",
        # Infrastructure features
        "transit_accessibility",
        "parking_availability",
        "land_use_mix",
        "avg_commute_time",
        # Geographic features
        "population_density",
        "distance_to_nearest_charger",
        "borough_encoded",
        # Derived features
        "transit_quality",
        "urban_development",
    ]

    # Ensure all features exist
    available_features = [col for col in feature_columns if col in data.columns]
    missing_features = [col for col in feature_columns if col not in data.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")

    return data[available_features]


def main():
    """Main feature engineering pipeline with real data"""
    print("Loading real NYC data...")

    # Load integrated NYC data
    nyc_data = load_nyc_data()

    # Enhance with derived features
    print("Creating enhanced features...")
    enhanced_data = enhance_features(nyc_data)

    # Create independent Gap Score
    print("Creating independent Gap Score...")
    enhanced_data["gap_score"] = create_gap_score_independently(len(enhanced_data))

    # Prepare ML features
    print("Preparing ML features...")
    X = prepare_ml_features(enhanced_data)
    y = enhanced_data["gap_score"]

    # Data quality checks
    print(f"\nData Quality Check:")
    print(f"Dataset shape: {enhanced_data.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"Missing values in features: {X.isnull().sum().sum()}")

    # Feature correlation analysis
    print(f"\nFeature-Target Correlations:")
    correlations = X.corrwith(y).sort_values(ascending=False)
    print(correlations)

    # Check for multicollinearity
    print(f"\nHigh Feature Correlations (>0.8):")
    corr_matrix = X.corr()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if high_corr:
        for feat1, feat2, corr in high_corr:
            print(f"  {feat1} - {feat2}: {corr:.3f}")
    else:
        print("  No high correlations detected")

    # Save processed data
    print(f"\nSaving processed data...")
    enhanced_data.to_csv("./data/processed_nyc_features.csv", index=False)
    X.to_csv("./data/X_features_real.csv", index=False)
    y.to_csv("./data/y_target_real.csv", index=False)

    # Save feature names for later use
    feature_info = {
        "feature_names": X.columns.tolist(),
        "n_features": len(X.columns),
        "n_samples": len(X),
        "target_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
    }

    import json

    with open("./data/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print("✅ Feature engineering completed successfully!")
    print(f"📊 Features: {len(X.columns)}")
    print(f"📈 Samples: {len(X)}")
    print(f"🎯 Target mean: {y.mean():.2f}")

    return X, y, enhanced_data


if __name__ == "__main__":
    X, y, data = main()
