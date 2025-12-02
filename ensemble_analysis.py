import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Import functions from the existing analysis file
# Assuming classification_analysis.py is in the same directory and importable
try:
    from classification_analysis import load_and_preprocess_data, prepare_features
except ImportError:
    print("Error: Could not import from classification_analysis.py. Make sure it is in the same directory.")
    sys.exit(1)

from sklearn.neural_network import MLPClassifier

def train_ensemble_model(X, y, le):
    """Trains Voting Classifier (RF + XGB + MLP) with various strategies."""
    
    print("\n" + "="*50)
    print("TRAINING ENSEMBLE MODELS (RF + XGB + MLP)")
    print("="*50)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Define individual models
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    xgb = XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
    
    # 1. Train Individual Models
    print("Training individual models...")
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"XGBoost Accuracy:       {xgb_acc:.4f}")
    print(f"MLP (Neural Net) Acc:   {mlp_acc:.4f}")
    
    results = {
        'Random Forest': rf_acc,
        'XGBoost': xgb_acc,
        'MLP': mlp_acc
    }

    # Define Estimators list
    estimators = [('rf', rf), ('xgb', xgb), ('mlp', mlp)]

    # 2. Soft Voting (Unweighted)
    print("\n--- 1. Soft Voting (Unweighted) ---")
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train, y_train)
    acc_soft = accuracy_score(y_test, voting_soft.predict(X_test))
    print(f"Accuracy: {acc_soft:.4f}")
    results['Voting (Soft)'] = acc_soft

    # 3. Weighted Voting (Soft)
    # Giving more weight to RF and MLP if they are better
    # Heuristic weights: RF=2, XGB=1, MLP=1 (adjust based on individual performance)
    print("\n--- 2. Weighted Voting (Soft) [2, 1, 1] ---")
    voting_weighted = VotingClassifier(estimators=estimators, voting='soft', weights=[2, 1, 1])
    voting_weighted.fit(X_train, y_train)
    acc_weighted = accuracy_score(y_test, voting_weighted.predict(X_test))
    print(f"Accuracy: {acc_weighted:.4f}")
    results['Voting (Weighted Soft)'] = acc_weighted
    
    # 4. Hard Voting
    # Majority rule voting
    print("\n--- 3. Hard Voting ---")
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)
    acc_hard = accuracy_score(y_test, voting_hard.predict(X_test))
    print(f"Accuracy: {acc_hard:.4f}")
    results['Voting (Hard)'] = acc_hard
    
    # 5. Stacking Classifier
    print("\n--- 4. Stacking Classifier ---")
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    acc_stack = accuracy_score(y_test, stacking_clf.predict(X_test))
    print(f"Accuracy: {acc_stack:.4f}")
    results['Stacking'] = acc_stack
    
    # Find Best Model
    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]
    print(f"\n" + "="*30)
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"ACCURACY: {best_acc:.4f}")
    print("="*30)
    
    # Determine which model object corresponds to the best name for plotting
    model_map = {
        'Random Forest': rf,
        'XGBoost': xgb,
        'MLP': mlp,
        'Voting (Soft)': voting_soft,
        'Voting (Weighted Soft)': voting_weighted,
        'Voting (Hard)': voting_hard,
        'Stacking': stacking_clf
    }
    
    best_model = model_map[best_model_name]
    
    # Re-predict with best model for plots
    # Note: If it's an individual model, we need to ensure it's fitted (it is)
    y_final_pred = best_model.predict(X_test)
        
    print(f"\nDetailed Report for: {best_model_name}")
    print(classification_report(y_test, y_final_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix ({best_model_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('ensemble_confusion_matrix.png')
    plt.close()
    print("Saved ensemble_confusion_matrix.png")

    # --- Actual vs Predicted Comparison Plot ---
    print("\nGenerating Actual vs Predicted Comparison Graph...")
    
    actual_counts = pd.Series(le.inverse_transform(y_test)).value_counts().sort_index()
    pred_counts = pd.Series(le.inverse_transform(y_final_pred)).value_counts().sort_index()
    
    df_plot = pd.DataFrame({
        'Situation': actual_counts.index,
        'Actual': actual_counts.values,
        'Predicted': pred_counts.values
    })
    
    df_melted = df_plot.melt(id_vars='Situation', var_name='Type', value_name='Count')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Situation', y='Count', hue='Type', data=df_melted, palette=['#1f77b4', '#ff7f0e'])
    plt.title(f'Actual vs Predicted Class Distribution ({best_model_name})')
    plt.xlabel('Groundwater Situation')
    plt.ylabel('Number of Instances')
    plt.legend(title='Type')
    plt.savefig('ensemble_actual_vs_predicted.png')
    plt.close()
    print("Saved ensemble_actual_vs_predicted.png")

    return best_acc

def main():
    try:
        # Load data using existing function
        df = load_and_preprocess_data()
        
        # Prepare features using existing function
        X, y, le, feature_names = prepare_features(df)
        
        # Train and evaluate ensemble
        train_ensemble_model(X, y, le)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
