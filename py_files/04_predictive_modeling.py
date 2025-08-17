"""
Train ML models to predict Gap Scores for unseen ZIP codes
Input: Processed features and target variables
Output: Trained models and performance metrics
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load processed data
print("Loading processed data...")
X = pd.read_csv('../data/X_features.csv')
y = pd.read_csv('../data/y_target.csv').squeeze()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 1. Data Preparation
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, labels=False)
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 2. Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0)
}

print("\nTraining models...")
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    # Use scaled data for SVR and linear models, original for tree-based
    if name in ['SVR', 'Linear Regression', 'Ridge Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Cross-validation on scaled data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        # Cross-validation on original data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'cv_rmse': cv_rmse,
        'predictions': y_pred
    }

# 3. Model Comparison
print("\n=== MODEL PERFORMANCE ===")
performance_df = pd.DataFrame({
    name: {
        'RMSE': results[name]['rmse'],
        'MAE': results[name]['mae'], 
        'R²': results[name]['r2'],
        'CV_RMSE': results[name]['cv_rmse']
    }
    for name in results.keys()
}).T

print(performance_df.round(4))

# Find best model
best_model_name = performance_df['R²'].idxmax()
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} (R² = {performance_df.loc[best_model_name, 'R²']:.4f})")

# 4. Hyperparameter Tuning for Best Model
print(f"\nTuning {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    tuned_model = RandomForestRegressor(random_state=42)
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    tuned_model = GradientBoostingRegressor(random_state=42)
    
else:
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    tuned_model = Ridge()

# Grid search
grid_search = GridSearchCV(
    tuned_model, param_grid, cv=5, 
    scoring='neg_mean_squared_error', n_jobs=-1
)

if best_model_name in ['SVR', 'Linear Regression', 'Ridge Regression']:
    grid_search.fit(X_train_scaled, y_train)
    final_predictions = grid_search.predict(X_test_scaled)
else:
    grid_search.fit(X_train, y_train)
    final_predictions = grid_search.predict(X_test)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {np.sqrt(-grid_search.best_score_):.4f}")

# Final model performance
final_r2 = r2_score(y_test, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"Final model - R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")

# 5. Feature Importance
if hasattr(grid_search.best_estimator_, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': grid_search.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== FEATURE IMPORTANCE ===")
    print(importance.round(4))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('../outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. Model Evaluation Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Predictions vs Actual
axes[0,0].scatter(y_test, final_predictions, alpha=0.6)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Gap Score')
axes[0,0].set_ylabel('Predicted Gap Score')
axes[0,0].set_title(f'Predictions vs Actual (R² = {final_r2:.3f})')

# Residuals
residuals = y_test - final_predictions
axes[0,1].scatter(final_predictions, residuals, alpha=0.6)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_xlabel('Predicted Gap Score')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residual Plot')

# Model comparison
model_names = list(performance_df.index)
r2_scores = performance_df['R²'].values
axes[1,0].bar(model_names, r2_scores, alpha=0.7)
axes[1,0].set_ylabel('R² Score')
axes[1,0].set_title('Model Comparison (R²)')
axes[1,0].tick_params(axis='x', rotation=45)

# Error distribution
axes[1,1].hist(residuals, bins=20, alpha=0.7)
axes[1,1].set_xlabel('Residuals')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('../outputs/model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Save Models
print("\nSaving models...")

# Save best model and scaler
joblib.dump(grid_search.best_estimator_, '../models/best_gap_score_model.pkl')
joblib.dump(scaler, '../models/feature_scaler.pkl')

# Save all results
model_results = {
    'performance_comparison': performance_df,
    'best_model_name': best_model_name,
    'best_params': grid_search.best_params_,
    'feature_importance': importance if 'importance' in locals() else None
}

joblib.dump(model_results, '../models/model_results.pkl')

print("Models saved successfully!")
print(f"Best model: {best_model_name}")
print(f"Final performance: R² = {final_r2:.4f}, RMSE = {final_rmse:.4f}")

# 8. Prediction Function 
def predict_gap_score(features_dict, model=grid_search.best_estimator_, scaler=scaler):
    features_df = pd.DataFrame([features_dict])
    
    if best_model_name in ['SVR', 'Linear Regression', 'Ridge Regression']:
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
    else:
        prediction = model.predict(features_df)[0]
    
    return prediction

# Example prediction
example_features = {
    'median_income': 75000,
    'renter_percentage': 0.6,
    'population_density': 15000,
    'transit_accessibility': 0.8,
    'land_use_mix': 0.7,
    'college_educated_pct': 0.5,
    'avg_commute_time': 35,
    'parking_availability': 0.3,
    'distance_to_nearest_charger': 2.5,
    'borough_encoded': 0  # Manhattan
}

example_prediction = predict_gap_score(example_features)
print(f"\nExample prediction: {example_prediction:.3f}")