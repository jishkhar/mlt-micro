import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('DWLR_Dataset_2023.csv')
    except FileNotFoundError:
        print("Error: DWLR_Dataset_2023.csv not found.")
        return

    # Select features and target
    # We are using a simplified set of features available from user input (indirectly)
    # Select features and target
    # We are using a simplified set of features available from user input (indirectly)
    # Location will be handled via one-hot encoding
    base_features = ['Rainfall_mm', 'Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L']
    target = 'Water_Level_m'
    
    # One-hot encode Location
    if 'Location' in df.columns:
        df = pd.get_dummies(df, columns=['Location'], prefix='Location')
    
    # Get all feature columns (base + encoded location)
    feature_columns = [col for col in df.columns if col in base_features or col.startswith('Location_')]
    
    print(f"Features: {len(feature_columns)} columns")
    print(f"Target: {target}")

    # Drop missing values
    df_clean = df.dropna(subset=feature_columns + [target])
    
    X = df_clean[feature_columns]
    y = df_clean[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model and feature columns
    model_filename = 'simple_rf_model.pkl'
    columns_filename = 'model_columns.pkl'
    
    joblib.dump(rf, model_filename)
    joblib.dump(feature_columns, columns_filename)
    print(f"Model saved to {model_filename}")
    print(f"Feature columns saved to {columns_filename}")

if __name__ == "__main__":
    train_model()
