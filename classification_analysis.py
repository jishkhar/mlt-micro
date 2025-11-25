import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class Logger(object):
    def __init__(self, filename='classification_report.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def load_and_preprocess_data(filepath='DWLR_Dataset_2023.csv'):
    """Loads data and creates categorical target."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Create 'Situation' based on Water_Level_m quantiles
    # Low water level (deep depth) = Critical, High water level (shallow depth) = Safe
    # Note: Water_Level_m usually means depth below ground level. Higher value = worse situation.
    # Let's assume Water_Level_m is depth.
    
    q1 = df['Water_Level_m'].quantile(0.33)
    q2 = df['Water_Level_m'].quantile(0.66)
    
    def categorize(level):
        if level < q1:
            return 'SAFE' # Shallow depth
        elif level < q2:
            return 'SEMI-CRITICAL' # Medium depth
        else:
            return 'CRITICAL' # Deep depth
            
    df['Situation'] = df['Water_Level_m'].apply(categorize)
    
    print(f"Created 'Situation' categories based on thresholds: <{q1:.2f} (Safe), <{q2:.2f} (Semi), >={q2:.2f} (Critical)")
    print(df['Situation'].value_counts())
    
    # Plot Situation count
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Situation', data=df, palette='hls')
    plt.title('Distribution of Groundwater Situation')
    plt.savefig('situation_distribution.png')
    plt.close()
    print("Saved situation_distribution.png")
    
    return df

def prepare_features(df):
    """Prepares X and y for classification."""
    # Select features - using other available columns
    # We drop Water_Level_m as it was used to create the target (leakage)
    # We drop Date as it's not directly usable without extraction
    
    feature_cols = ['Rainfall_mm', 'Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L']
    
    # Drop NaNs in feature columns
    df_clean = df.dropna(subset=feature_cols)
    print(f"Dropped {len(df) - len(df_clean)} rows with missing values.")
    
    X = df_clean[feature_cols].values
    y = df_clean['Situation'].values
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le, feature_cols

def train_evaluate_models(X, y, le, feature_names):
    """Trains models and evaluates them."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n" + "="*40)
    print("LOGISTIC REGRESSION")
    print("="*40)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy: {acc_lr:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
    results['Logistic Regression'] = acc_lr
    
    # 2. Decision Tree
    print("\n" + "="*40)
    print("DECISION TREE CLASSIFIER")
    print("="*40)
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Accuracy: {acc_dt:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))
    results['Decision Tree'] = acc_dt
    
    # Plot Decision Tree Structure
    plt.figure(figsize=(25, 12))
    plot_tree(dt, feature_names=feature_names, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree Visualization')
    plt.savefig('decision_tree_viz.png')
    plt.close()
    print("Saved decision_tree_viz.png")
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names, y=dt.feature_importances_)
    plt.title('Decision Tree Feature Importance')
    plt.savefig('dt_feature_importance.png')
    plt.close()
    print("Saved dt_feature_importance.png")
    


    # 3. Random Forest
    print("\n" + "="*40)
    print("RANDOM FOREST CLASSIFIER")
    print("="*40)
    # Increased estimators to 300
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy: {acc_rf:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    results['Random Forest'] = acc_rf
    
    # 4. XGBoost
    print("\n" + "="*40)
    print("XGBOOST CLASSIFIER")
    print("="*40)
    # Increased estimators to 300
    xgb = XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"Accuracy: {acc_xgb:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    results['XGBoost'] = acc_xgb

    # 5. MLP Classifier (Neural Network)
    print("\n" + "="*40)
    print("MLP CLASSIFIER (NEURAL NETWORK)")
    print("="*40)
    # Training in epochs (max_iter)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"Accuracy: {acc_mlp:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))
    results['MLP (Neural Net)'] = acc_mlp
    
    # Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.savefig('model_comparison.png')
    plt.close()
    print("Saved model_comparison.png")
    
    return results

def main():
    # Redirect stdout to file
    sys.stdout = Logger()
    
    try:
        df = load_and_preprocess_data()
        X, y, le, feature_names = prepare_features(df)
        train_evaluate_models(X, y, le, feature_names)
        print("\nAnalysis Complete. Report saved to classification_report.txt")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
