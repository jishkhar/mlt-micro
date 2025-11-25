import pandas as pd
import matplotlib.pyplot as plt

def analyze_data():
    df = pd.read_csv('DWLR_Dataset_2023.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f"Train Range: {train['Water_Level_m'].min()} - {train['Water_Level_m'].max()}")
    print(f"Test Range: {test['Water_Level_m'].min()} - {test['Water_Level_m'].max()}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train['Date'], train['Water_Level_m'], label='Train')
    plt.plot(test['Date'], test['Water_Level_m'], label='Test')
    plt.legend()
    plt.title('Train vs Test Water Levels')
    plt.savefig('data_split_analysis.png')

if __name__ == "__main__":
    analyze_data()
