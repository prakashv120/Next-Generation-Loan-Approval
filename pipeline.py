import pandas as pd
import pickle
import xgboost as xgb

class CreditPipeline:
    def __init__(self, model_path="xgb_model.pkl"):
        """
        Initializes the pipeline with the trained model.
        """
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            print(f"Model file {model_path} not found.")
            self.model = None

    def run_waterfall(self, user_features):
        """
        Runs the waterfall logic for a single user.
        
        Args:
            user_features (dict or pd.Series): Features for the user.
            
        Returns:
            dict: Decision result containing 'decision', 'reason', 'pd', 'gate'.
        """
        # Gate 1: Fraud Check (Mocked)
        # In a real system, this would check identity verification, etc.
        # For MVP, we assume everyone passes unless flagged explicitly (not implemented here)
        fraud_score = 0.0 # Low risk
        if fraud_score > 0.5:
            return {
                'decision': 'Reject',
                'reason': 'Fraud Check Failed',
                'pd': None,
                'gate': 1
            }
            
        # Gate 2: Cash Flow Model
        if self.model is None:
            return {'decision': 'Error', 'reason': 'Model not loaded', 'pd': None, 'gate': 2}
            
        # Prepare input for model
        # Ensure the order of columns matches training
        # We need to convert the single user features to a DataFrame
        # The model expects specific columns. We assume user_features has them.
        
        # Get feature names from the model if possible, or assume standard order
        # For safety, let's load the feature names from the model object if available
        # or just rely on the input being a dict with correct keys.
        
        # Convert to DataFrame
        input_df = pd.DataFrame([user_features])
        
        # Drop non-feature columns if present (like user_id)
        if 'user_id' in input_df.columns:
            input_df = input_df.drop('user_id', axis=1)
            
        # Predict Probability of Default (PD)
        # XGBoost predict_proba returns [prob_0, prob_1]
        pd_score = self.model.predict_proba(input_df)[0][1]
        
        if pd_score > 0.8:
            return {
                'decision': 'Reject',
                'reason': f'High Probability of Default ({pd_score:.2f})',
                'pd': pd_score,
                'gate': 2
            }
        elif pd_score < 0.1:
            return {
                'decision': 'Approve',
                'reason': f'Low Probability of Default ({pd_score:.2f})',
                'pd': pd_score,
                'gate': 2
            }
        else:
            # Gate 3: Bureau Referral / Manual Review
            return {
                'decision': 'Refer',
                'reason': f'Moderate Risk ({pd_score:.2f}) - Manual Review Required',
                'pd': pd_score,
                'gate': 3
            }

if __name__ == "__main__":
    # Test the pipeline
    try:
        df = pd.read_csv("features.csv", dtype={'user_id': str})
        pipeline = CreditPipeline()
        
        print("Testing pipeline on first 5 users:")
        for i in range(5):
            user_row = df.iloc[i]
            result = pipeline.run_waterfall(user_row)
            print(f"User {user_row['user_id']}: {result}")
            
    except FileNotFoundError:
        print("features.csv not found.")
