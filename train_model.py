import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_model():
    """
    Trains an XGBoost model on the features.csv data.
    """
    print("Loading data...")
    try:
        df = pd.read_csv("features.csv", dtype={'user_id': str})
    except FileNotFoundError:
        print("features.csv not found. Run features.py first.")
        return

    # Create target variable based on simple logic for training purposes
    # High risk = 1 (Default), Low risk = 0 (Good)
    
    # Updated Risk Score Logic for Gen-Z features
    # Higher score = Higher risk
    risk_score = (
        (df['net_cashflow'] < 0.1).astype(int) * 2 +
        (df['income_stability'] > 0.3).astype(int) * 2 +
        (df['neg_balance_days'] > 5).astype(int) * 3 +
        (df['declined_txns'] > 2).astype(int) * 3 +
        (df['gambling_ratio'] > 0.05).astype(int) * 4 + # High weight
        (df['bnpl_failures'] > 0).astype(int) * 3 +
        (df['night_txns'] > 5).astype(int) * 2 +
        (df['weekend_ratio'] > 1.5).astype(int) * 1 +
        (df['loan_apps'] > 2).astype(int) * 3 +
        (df['parental_dependency'] > 0.5).astype(int) * 2
    )
    
    # Normalize risk score and assign labels
    # Threshold: if risk score > 8, then default = 1
    df['target'] = (risk_score > 8).astype(int)
    
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Drop non-feature columns
    cols_to_drop = ['user_id', 'target']
    if 'Unnamed: 0' in df.columns:
        cols_to_drop.append('Unnamed: 0')
        
    X = df.drop(cols_to_drop, axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Monotone constraints
    # 1 = increasing constraint (higher value -> higher risk)
    # -1 = decreasing constraint (higher value -> lower risk)
    
    feature_names = X.columns.tolist()
    constraints = []
    for feat in feature_names:
        if feat in ['net_cashflow', 'eom_balance', 'income_stability', 'upi_stability', 'sim_age', 'device_age', 'signup_tenure']:
            constraints.append(-1) # Higher is better (Lower Risk)
            # Note: income_stability in features.py is std/mean, so higher is actually worse (more volatile). 
            # Wait, let's check features.py: income_stability = std/mean. So higher is worse.
            # So income_stability should be 1.
        elif feat == 'income_stability':
             constraints.append(1)
        else:
            # Most other features (gambling, night txns, etc.) are riskier if higher
            constraints.append(1) 
            
    # Correction for specific features where higher is better
    # net_cashflow: Higher is better -> -1
    # eom_balance: Higher is better -> -1
    # income_stability: Higher is WORSE -> 1
    # upi_stability: Higher is WORSE (volatility) -> 1
    # sim_age: Higher is better -> -1
    # device_age: Higher is better -> -1
    # signup_tenure: Higher is better -> -1
    # address_stability: Higher is better (1=Stable) -> -1
    
    # Let's refine the constraints list logic
    constraints_dict = {
        'net_cashflow': -1,
        'eom_balance': -1,
        'sim_age': -1,
        'device_age': -1,
        'signup_tenure': -1,
        'upi_tenure': -1,
        'address_stability': -1,
        'income_stability': 1,
        'upi_stability': 1
    }
    
    final_constraints = []
    for feat in feature_names:
        final_constraints.append(constraints_dict.get(feat, 1)) # Default to 1 (Higher is Riskier)

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        monotone_constraints=tuple(final_constraints),
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model Accuracy: {acc:.4f}")
    print(f"Model AUC: {auc:.4f}")
    
    # Save model
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model to xgb_model.pkl")
    
    # Train SHAP explainer
    print("Generating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    with open("shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("Saved SHAP explainer to shap_explainer.pkl")

if __name__ == "__main__":
    train_model()
