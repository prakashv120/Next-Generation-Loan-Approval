import pandas as pd
import numpy as np

class CashFlowFeatures:
    def __init__(self, transactions_df, users_df=None):
        """
        Initializes the feature engineering class.
        
        Args:
            transactions_df (pd.DataFrame): DataFrame containing raw transaction logs.
            users_df (pd.DataFrame): DataFrame containing static user profile data.
        """
        self.df = transactions_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['user_id', 'date'])
        
        # Handle time if present
        if 'time' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['date'].astype(str) + ' ' + self.df['time'].astype(str))
            self.df['hour'] = self.df['datetime'].dt.hour
        else:
            self.df['hour'] = 12 # Default to noon if missing
            
        self.users_df = users_df
        if self.users_df is not None:
            self.users_df = self.users_df.set_index('user_id')
        
    def calculate_features(self):
        """
        Calculates the specific features for each user.
        """
        features = []
        
        for user_id, group in self.df.groupby('user_id'):
            user_features = self._calculate_user_features(user_id, group)
            features.append(user_features)
            
        return pd.DataFrame(features).set_index('user_id')
    
    def _calculate_user_features(self, user_id, group):
        """
        Calculates features for a single user.
        """
        # Helper variables
        inflows = group[group['amount'] > 0]
        outflows = group[group['amount'] < 0]
        
        # Filter for successful transactions for most calcs
        success_group = group[group['status'] == 'Success'] if 'status' in group.columns else group
        success_inflows = success_group[success_group['amount'] > 0]
        success_outflows = success_group[success_group['amount'] < 0]
        
        total_inflow = success_inflows['amount'].sum()
        total_outflow = abs(success_outflows['amount'].sum())
        
        # --- 1. Cashflow Strength Features (35-40%) ---
        # Net monthly cashflow
        net_cashflow = total_inflow - total_outflow
        
        # Income stability (month-to-month)
        if len(success_inflows) > 1:
            monthly_inflow = success_inflows.set_index('date').resample('M')['amount'].sum()
            income_stability = monthly_inflow.std() / monthly_inflow.mean() if monthly_inflow.mean() > 0 else 1.0
        else:
            income_stability = 1.0 # High volatility
            
        # End-of-month balance (buffer) - Simulated
        group['balance'] = group['amount'].cumsum()
        eom_balance = group.set_index('date').resample('M')['balance'].last().mean()
        
        # Negative balance days
        neg_balance_days = len(group[group['balance'] < 0])
        
        # Frequent low-balance (<200)
        low_balance_days = len(group[group['balance'] < 200])
        
        # --- 2. Digital Payment Behavior (18-22%) ---
        # UPI declined transactions
        if 'status' in group.columns:
            declined_txns = len(group[group['status'] == 'Declined'])
        else:
            declined_txns = 0
            
        # UPI inflow stability (same as income stability but specifically for transfers)
        upi_inflows = success_inflows[success_inflows['category'].isin(['Transfer In', 'Parental Transfer'])]
        if len(upi_inflows) > 1:
            upi_stability = upi_inflows['amount'].std() / upi_inflows['amount'].mean()
        else:
            upi_stability = 0.0
            
        # Wallet transfers to friends (Outflows to Discretionary/Transfer)
        wallet_transfers = len(success_outflows[success_outflows['category'] == 'Discretionary']) # Proxy
        
        # --- 3. Spending Type Ratios (12-15%) ---
        def get_ratio(cat_list):
            spend = abs(success_outflows[success_outflows['category'].isin(cat_list)]['amount'].sum())
            return spend / total_inflow if total_inflow > 0 else 0.0
            
        essential_ratio = get_ratio(['Essential', 'Rent', 'Utilities', 'Grocery', 'Gas', 'Medical'])
        discretionary_ratio = get_ratio(['Discretionary', 'Shopping', 'Entertainment'])
        food_delivery_ratio = get_ratio(['Food Delivery'])
        gaming_ratio = get_ratio(['Gaming'])
        fashion_ratio = get_ratio(['Fashion'])
        
        # --- 4. Risky Merchant & BNPL Behavior (10-15%) ---
        gambling_crypto_spend = abs(success_outflows[success_outflows['category'] == 'Gambling/Crypto']['amount'].sum())
        gambling_ratio = gambling_crypto_spend / total_inflow if total_inflow > 0 else 0.0
        
        bnpl_spend = abs(success_outflows[success_outflows['category'] == 'BNPL']['amount'].sum())
        bnpl_ratio = bnpl_spend / total_inflow if total_inflow > 0 else 0.0
        
        # BNPL repayment delays (Failed BNPL txns)
        if 'status' in group.columns:
            bnpl_failures = len(group[(group['category'] == 'BNPL') & (group['status'] == 'Failed')])
        else:
            bnpl_failures = 0
            
        # --- 5. GenZ Behavioral Patterns (10-12%) ---
        # Night transactions (2am-5am)
        night_txns = len(success_outflows[(success_outflows['hour'] >= 2) & (success_outflows['hour'] <= 5)])
        
        # Weekend overspending
        weekend_spend = abs(success_outflows[success_outflows['date'].dt.weekday >= 5]['amount'].sum())
        weekday_spend = abs(success_outflows[success_outflows['date'].dt.weekday < 5]['amount'].sum())
        weekend_ratio = weekend_spend / (weekday_spend + 1.0)
        
        # Micro-spend count (20-200)
        micro_spends = len(success_outflows[(abs(success_outflows['amount']) >= 20) & (abs(success_outflows['amount']) <= 200)])
        
        # Refund-driven spending
        refunds = len(success_inflows[success_inflows['category'] == 'Refund'])
        
        # --- 6. Income Source Mix (8-10%) ---
        # Parental transfer dependency
        parental_income = success_inflows[success_inflows['category'] == 'Parental Transfer']['amount'].sum()
        parental_dependency = parental_income / total_inflow if total_inflow > 0 else 0.0
        
        # Gig income
        gig_income = success_inflows[success_inflows['category'] == 'Freelance Income']['amount'].sum()
        gig_ratio = gig_income / total_inflow if total_inflow > 0 else 0.0
        
        # --- 7. Subscription & Micro-Commitments (5-8%) ---
        # Failed subscriptions
        if 'status' in group.columns:
            failed_subs = len(group[(group['category'] == 'Subscription') & (group['status'] == 'Failed')])
        else:
            failed_subs = 0
            
        active_subs = success_outflows[success_outflows['category'] == 'Subscription']['merchant_name'].nunique()
        
        # --- 8. Device & App Behavior (4-6%) ---
        # From users_df
        if self.users_df is not None and user_id in self.users_df.index:
            u = self.users_df.loc[user_id]
            sim_age = u.get('sim_age_months', 0)
            device_age = u.get('device_age_months', 0)
            loan_apps = u.get('loan_apps_installed', 0)
            gaming_apps = u.get('gaming_apps_installed', 0)
            finance_apps = u.get('finance_apps_installed', 0)
            signup_tenure = u.get('signup_tenure_days', 0)
            upi_tenure = u.get('upi_id_tenure_days', 0)
            address_stability = u.get('address_stability_flag', 0)
        else:
            sim_age = 0
            device_age = 0
            loan_apps = 0
            gaming_apps = 0
            finance_apps = 0
            signup_tenure = 0
            upi_tenure = 0
            address_stability = 0
            
        # --- 9. Personal Stability Indicators (3-5%) ---
        # (Covered above with tenure vars)
        
        return {
            'user_id': user_id,
            'net_cashflow': net_cashflow,
            'income_stability': income_stability,
            'eom_balance': eom_balance,
            'neg_balance_days': neg_balance_days,
            'low_balance_days': low_balance_days,
            'declined_txns': declined_txns,
            'upi_stability': upi_stability,
            'wallet_transfers': wallet_transfers,
            'essential_ratio': essential_ratio,
            'discretionary_ratio': discretionary_ratio,
            'food_delivery_ratio': food_delivery_ratio,
            'gaming_ratio': gaming_ratio,
            'fashion_ratio': fashion_ratio,
            'gambling_ratio': gambling_ratio,
            'bnpl_ratio': bnpl_ratio,
            'bnpl_failures': bnpl_failures,
            'night_txns': night_txns,
            'weekend_ratio': weekend_ratio,
            'micro_spends': micro_spends,
            'refunds': refunds,
            'parental_dependency': parental_dependency,
            'gig_ratio': gig_ratio,
            'failed_subs': failed_subs,
            'active_subs': active_subs,
            'sim_age': sim_age,
            'device_age': device_age,
            'loan_apps': loan_apps,
            'gaming_apps': gaming_apps,
            'finance_apps': finance_apps,
            'signup_tenure': signup_tenure,
            'upi_tenure': upi_tenure,
            'address_stability': address_stability
        }

if __name__ == "__main__":
    try:
        df = pd.read_csv("transactions.csv", dtype={'user_id': str})
        try:
            users = pd.read_csv("users.csv", dtype={'user_id': str})
        except FileNotFoundError:
            print("users.csv not found, proceeding without it.")
            users = None
            
        features_engine = CashFlowFeatures(df, users)
        features_df = features_engine.calculate_features()
        print(features_df.head())
        features_df.to_csv("features.csv")
        print("Saved to features.csv")
    except FileNotFoundError:
        print("transactions.csv not found. Run data_gen.py first.")
