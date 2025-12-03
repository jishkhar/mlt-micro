import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                               StackingRegressor, VotingRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_data(filepath='DWLR_Dataset_2023.csv'):
    """Loads groundwater data from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_advanced_features(df):
    """Creates comprehensive feature set with time series patterns."""
    df = df.copy()
    df = df.sort_values('Date')
    
    target_col = 'Water_Level_m'
    rainfall_col = 'Rainfall_mm'
    
    # Temporal features
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Cyclical encoding for seasonality
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Water level lag features (multiple time scales)
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]:
        df[f'WL_Lag_{lag}'] = df[target_col].shift(lag)
    
    # Rainfall lag features
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        df[f'Rain_Lag_{lag}'] = df[rainfall_col].shift(lag)
    
    # Rolling statistics for water level
    for window in [3, 7, 14, 30, 60, 90]:
        df[f'WL_Roll_Mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'WL_Roll_Std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'WL_Roll_Min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'WL_Roll_Max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
    
    # Rolling statistics for rainfall
    for window in [7, 14, 30, 60, 90]:
        df[f'Rain_Roll_Sum_{window}'] = df[rainfall_col].rolling(window=window).sum()
        df[f'Rain_Roll_Mean_{window}'] = df[rainfall_col].rolling(window=window).mean()
        df[f'Rain_Roll_Max_{window}'] = df[rainfall_col].rolling(window=window).max()
    
    # Exponential moving averages
    for span in [7, 14, 30]:
        df[f'WL_EMA_{span}'] = df[target_col].shift(1).ewm(span=span).mean()
        df[f'Rain_EMA_{span}'] = df[rainfall_col].ewm(span=span).mean()
    
    # Rate of change features
    df['WL_Change_1d'] = df[target_col].diff(1)
    df['WL_Change_7d'] = df[target_col].diff(7)
    df['WL_Change_30d'] = df[target_col].diff(30)
    
    # Interaction features
    df['Rain_x_WL_Lag1'] = df[rainfall_col] * df['WL_Lag_1']
    df['Rain_Sum7_x_WL_Lag7'] = df['Rain_Roll_Sum_7'] * df['WL_Lag_7']
    
    # Cumulative rainfall (recent moisture)
    df['Rain_Cumsum_7d'] = df[rainfall_col].rolling(window=7).sum()
    df['Rain_Cumsum_30d'] = df[rainfall_col].rolling(window=30).sum()
    
    # Drop rows with NaN values
    df = df.dropna()

    # One-hot encode Location if present
    if 'Location' in df.columns:
        df = pd.get_dummies(df, columns=['Location'], prefix='Loc', drop_first=False)
        # Ensure boolean columns are converted to int/float for models
        loc_cols = [c for c in df.columns if c.startswith('Loc_')]
        df[loc_cols] = df[loc_cols].astype(int)
    
    return df

def create_stacked_model():
    """Creates an advanced stacking ensemble model."""
    
    # Base models with different characteristics
    base_models = [
        ('xgb', XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )),
        ('catboost', CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            random_state=42,
            verbose=False
        )),
        ('gbr', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ]
    
    # Meta-learner (uses Ridge regression for stability)
    meta_model = Ridge(alpha=1.0)
    
    # Create stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_model

def create_voting_model():
    """Creates a voting ensemble as an alternative."""
    
    models = [
        ('xgb', XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            random_state=42,
            verbose=-1
        )),
        ('catboost', CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            random_state=42,
            verbose=False
        ))
    ]
    
    return VotingRegressor(estimators=models, n_jobs=-1)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluates model performance with multiple metrics."""
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    # MAPE (Mean Absolute Percentage Error)
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'train_preds': train_preds,
        'test_preds': test_preds
    }

def train_and_compare_models(df):
    """Trains multiple models and selects the best performer."""
    
    target_col = 'Water_Level_m'
    exclude_cols = ['Date', target_col]
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nTotal features: {len(features)}")
    
    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[features]
    y_train = train[target_col]
    X_test = test[features]
    y_test = test[target_col]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Dictionary to store models and results
    models = {
        'Stacking Ensemble': create_stacked_model(),
        'Voting Ensemble': create_voting_model(),
        'XGBoost': XGBRegressor(
            n_estimators=400,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=400,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=400,
            learning_rate=0.02,
            depth=6,
            random_state=42,
            verbose=False
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        print(f"\n{name} Results:")
        print(f"  Train RMSE: {results[name]['train_rmse']:.4f}")
        print(f"  Test RMSE:  {results[name]['test_rmse']:.4f}")
        print(f"  Test MAE:   {results[name]['test_mae']:.4f}")
        print(f"  Test R²:    {results[name]['test_r2']:.4f}")
        print(f"  Test MAPE:  {results[name]['test_mape']:.2f}%")
    
    # Select best model based on test RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    best_model = models[best_model_name]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    
    return best_model, results[best_model_name], test, best_model_name

def plot_comprehensive_results(test_df, predictions, actual, model_name):
    """Creates comprehensive visualization of results."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Time series comparison
    axes[0].plot(test_df['Date'], actual, label='Actual', color='blue', linewidth=2)
    axes[0].plot(test_df['Date'], predictions, label='Predicted', 
                 color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].set_title(f'Groundwater Level Prediction - {model_name}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Water Level (m)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals over time
    residuals = actual - predictions
    axes[1].scatter(test_df['Date'], residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Prediction Residuals Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Actual vs Predicted scatter
    axes[2].scatter(actual, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 
                 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[2].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Actual Water Level (m)', fontsize=11)
    axes[2].set_ylabel('Predicted Water Level (m)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'optimized_groundwater_prediction.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()

def main():
    print("="*60)
    print("OPTIMIZED GROUNDWATER LEVEL PREDICTION")
    print("="*60)
    
    print("\n[1/4] Loading data...")
    try:
        df = load_data()
        print(f"✓ Loaded {len(df)} rows of data")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    print("\n[2/4] Creating advanced features...")
    df_processed = create_advanced_features(df)
    print(f"✓ Feature engineering complete: {len(df_processed)} rows")
    
    print("\n[3/4] Training and comparing models...")
    best_model, results, test_df, model_name = train_and_compare_models(df_processed)
    
    print("\n[4/4] Generating visualizations...")
    plot_comprehensive_results(
        test_df, 
        results['test_preds'], 
        test_df['Water_Level_m'].values,
        model_name
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nKey Improvements:")
    print("  • Advanced feature engineering (150+ features)")
    print("  • Stacking ensemble with 5 diverse models")
    print("  • Multiple evaluation metrics")
    print("  • Enhanced visualizations")
    print("="*60)

if __name__ == "__main__":
    main()