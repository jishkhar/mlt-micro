import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def load_data(filepath='DWLR_Dataset_2023.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def train_prophet(df):
    # Prepare data for Prophet (ds, y)
    # We will predict Water_Level_m directly. Prophet handles trends well.
    prophet_df = df[['Date', 'Water_Level_m', 'Rainfall_mm']].rename(columns={'Date': 'ds', 'Water_Level_m': 'y'})
    
    # Split
    train_size = int(len(df) * 0.8)
    train = prophet_df.iloc[:train_size]
    test = prophet_df.iloc[train_size:]
    
    # Model
    # Adding Rainfall as a regressor might help
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.add_regressor('Rainfall_mm')
    
    model.fit(train)
    
    # Predict
    future = test[['ds', 'Rainfall_mm']]
    forecast = model.predict(future)
    
    # Metrics
    y_true = test['y'].values
    y_pred = forecast['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Prophet RMSE: {rmse:.4f}")
    print(f"Prophet MAE: {mae:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test['ds'], y_true, label='Actual', color='blue')
    plt.plot(test['ds'], y_pred, label='Predicted (Prophet)', color='green', linestyle='--')
    plt.title('Groundwater Prediction with Prophet')
    plt.legend()
    plt.savefig('groundwater_prediction_prophet.png')
    
    return rmse, mae

if __name__ == "__main__":
    df = load_data()
    train_prophet(df)
