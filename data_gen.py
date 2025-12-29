import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_users=1000):
    """
    Generates a synthetic dataset of raw bank transactions and user profiles.
    
    Args:
        num_users (int): Number of users to generate data for.
        
    Returns:
        tuple: (transactions_df, users_df)
    """
    print(f"Generating data for {num_users} users...")
    
    data = []
    user_data = []
    
    # Define categories
    income_categories = ['Payroll', 'Direct Deposit', 'Freelance Income', 'Transfer In', 'Parental Transfer']
    risky_categories = ['DraftKings', 'FanDuel', 'Coinbase', 'Binance', 'Casino', 'PokerStars']
    bnpl_categories = ['Klarna', 'Affirm', 'Afterpay', 'Sezzle', 'Zip']
    essential_categories = ['Rent', 'Utilities', 'Grocery', 'Gas', 'Medical']
    discretionary_categories = ['Uber', 'McDonalds', 'Starbucks', 'Netflix', 'Amazon', 'Cinema', 'Bar']
    food_delivery_categories = ['UberEats', 'DoorDash', 'GrubHub', 'Zomato', 'Swiggy']
    gaming_categories = ['Steam', 'PlayStation', 'Xbox', 'Nintendo', 'Riot Games', 'Roblox']
    fashion_categories = ['Zara', 'H&M', 'Shein', 'Nike', 'Adidas', 'Myntra']
    subscription_categories = ['Netflix', 'Spotify', 'Apple Music', 'Prime', 'Disney+', 'Hulu']
    
    # Define user profiles
    # 0: Stable Salaried (Low risk)
    # 1: Volatile Gig Worker (Medium risk, high income)
    # 2: High BNPL/Gambler (High risk)
    # 3: Gen Z / Student (Parental support, gaming, night spend)
    
    user_types = np.random.choice([0, 1, 2, 3], size=num_users, p=[0.3, 0.2, 0.2, 0.3])
    
    start_date = datetime.now() - timedelta(days=90) # 3 months of history
    
    for user_id, user_type in enumerate(user_types):
        current_date = start_date
        
        # --- User Profile Data (Static) ---
        sim_age = np.random.randint(1, 60) # Months
        device_age = np.random.randint(1, 36) # Months
        loan_apps = 0
        gaming_apps = 0
        finance_apps = 0
        signup_tenure = np.random.randint(1, 1000) # Days
        upi_id_tenure = np.random.randint(1, 1000) # Days
        address_stability = 1 # 1 = Stable, 0 = Unstable
        
        if user_type == 0: # Stable
            income_freq = 14
            income_amount_base = 3000
            income_volatility = 0.05
            risky_prob = 0.01
            bnpl_prob = 0.02
            overdraft_prob = 0.0
            night_prob = 0.05
            weekend_spike = 1.2
            loan_apps = np.random.randint(0, 2)
            gaming_apps = np.random.randint(0, 3)
            finance_apps = np.random.randint(2, 5)
            
        elif user_type == 1: # Gig Worker
            income_freq = 3
            income_amount_base = 200
            income_volatility = 0.4
            risky_prob = 0.05
            bnpl_prob = 0.05
            overdraft_prob = 0.1
            night_prob = 0.2
            weekend_spike = 1.5
            loan_apps = np.random.randint(1, 4)
            gaming_apps = np.random.randint(1, 5)
            finance_apps = np.random.randint(1, 3)
            
        elif user_type == 2: # High Risk
            income_freq = 14
            income_amount_base = 2500
            income_volatility = 0.1
            risky_prob = 0.3
            bnpl_prob = 0.2
            overdraft_prob = 0.4
            night_prob = 0.3
            weekend_spike = 2.0
            loan_apps = np.random.randint(3, 8)
            gaming_apps = np.random.randint(2, 6)
            finance_apps = np.random.randint(0, 2)
            
        else: # Gen Z / Student
            income_freq = 30 # Monthly allowance/transfer
            income_amount_base = 1500
            income_volatility = 0.2
            risky_prob = 0.1 # Crypto/Skins
            bnpl_prob = 0.15
            overdraft_prob = 0.2
            night_prob = 0.4 # High night activity
            weekend_spike = 2.5
            loan_apps = np.random.randint(0, 3)
            gaming_apps = np.random.randint(5, 15)
            finance_apps = np.random.randint(1, 4)
            sim_age = np.random.randint(1, 12) # Newer SIMs
            device_age = np.random.randint(1, 12) # Newer devices
        
        user_data.append({
            'user_id': user_id,
            'user_type': user_type,
            'sim_age_months': sim_age,
            'device_age_months': device_age,
            'loan_apps_installed': loan_apps,
            'gaming_apps_installed': gaming_apps,
            'finance_apps_installed': finance_apps,
            'signup_tenure_days': signup_tenure,
            'upi_id_tenure_days': upi_id_tenure,
            'address_stability_flag': address_stability
        })
            
        # --- Transaction Generation ---
        days_since_pay = 0
        
        for day in range(90):
            date = start_date + timedelta(days=day)
            is_weekend = date.weekday() >= 5
            
            # Income
            is_payday = False
            if user_type == 1:
                if random.random() < 0.4:
                    is_payday = True
            elif user_type == 3: # Student - Parental Transfer
                 if day % 30 == 0:
                     is_payday = True
            else:
                if days_since_pay >= income_freq:
                    is_payday = True
                    days_since_pay = 0
                else:
                    days_since_pay += 1
            
            if is_payday:
                amount = np.random.normal(income_amount_base, income_amount_base * income_volatility)
                amount = max(amount, 50)
                
                cat = 'Payroll'
                merch = 'Employer'
                if user_type == 1:
                    cat = 'Freelance Income'
                    merch = 'Client'
                elif user_type == 3:
                    cat = 'Parental Transfer'
                    merch = 'Dad/Mom'
                
                data.append({
                    'user_id': user_id,
                    'date': date,
                    'time': '09:00:00', # Income usually morning
                    'amount': round(amount, 2),
                    'merchant_name': merch,
                    'category': cat,
                    'status': 'Success'
                })
            
            # Expenses
            num_txns = np.random.poisson(2)
            if is_weekend:
                num_txns = int(num_txns * weekend_spike)
            
            for _ in range(num_txns):
                # Time of day
                if random.random() < night_prob:
                    hour = random.randint(2, 5) # Night 2am-5am
                else:
                    hour = random.randint(8, 23)
                time_str = f"{hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                
                # Category & Merchant
                rand_val = random.random()
                status = 'Success'
                
                if rand_val < risky_prob:
                    cat = 'Gambling/Crypto'
                    merch = random.choice(risky_categories)
                    amt = random.uniform(20, 200)
                elif rand_val < risky_prob + bnpl_prob:
                    cat = 'BNPL'
                    merch = random.choice(bnpl_categories)
                    amt = random.uniform(20, 100)
                    # Simulate BNPL repayment delay/failure
                    if random.random() < 0.1:
                        status = 'Failed'
                elif rand_val < risky_prob + bnpl_prob + 0.15: # Food Delivery
                    cat = 'Food Delivery'
                    merch = random.choice(food_delivery_categories)
                    amt = random.uniform(15, 60)
                elif rand_val < risky_prob + bnpl_prob + 0.15 + 0.1: # Gaming
                    cat = 'Gaming'
                    merch = random.choice(gaming_categories)
                    amt = random.uniform(5, 100)
                elif rand_val < risky_prob + bnpl_prob + 0.15 + 0.1 + 0.1: # Fashion
                    cat = 'Fashion'
                    merch = random.choice(fashion_categories)
                    amt = random.uniform(30, 150)
                elif rand_val < risky_prob + bnpl_prob + 0.15 + 0.1 + 0.1 + 0.05: # Subscription
                    cat = 'Subscription'
                    merch = random.choice(subscription_categories)
                    amt = random.uniform(5, 20)
                    if random.random() < 0.1: # Failed sub
                        status = 'Failed'
                elif rand_val < risky_prob + bnpl_prob + 0.15 + 0.1 + 0.1 + 0.05 + 0.2: # Essential
                    cat = 'Essential'
                    merch = random.choice(essential_categories)
                    amt = random.uniform(50, 150)
                else:
                    cat = 'Discretionary'
                    merch = random.choice(discretionary_categories)
                    amt = random.uniform(5, 50)
                    # Micro-spend check
                    if amt < 200 and cat == 'Discretionary':
                        # Just a tag, we handle logic in features
                        pass
                
                # UPI Decline Simulation
                if status == 'Success' and random.random() < 0.05:
                    status = 'Declined'
                
                # Refund Simulation
                if status == 'Success' and random.random() < 0.02:
                    # Create the purchase
                    data.append({
                        'user_id': user_id,
                        'date': date,
                        'time': time_str,
                        'amount': round(-amt, 2),
                        'merchant_name': merch,
                        'category': cat,
                        'status': 'Success'
                    })
                    # Create the refund
                    data.append({
                        'user_id': user_id,
                        'date': date + timedelta(days=random.randint(1, 5)),
                        'time': '10:00:00',
                        'amount': round(amt, 2),
                        'merchant_name': merch,
                        'category': 'Refund',
                        'status': 'Success'
                    })
                    continue # Skip adding the purchase again
                
                # Add transaction
                data.append({
                    'user_id': user_id,
                    'date': date,
                    'time': time_str,
                    'amount': round(-amt, 2), # Expense is negative
                    'merchant_name': merch,
                    'category': cat,
                    'status': status
                })
                
            # Simulate Overdraft fees
            if random.random() < overdraft_prob and day % 10 == 0:
                 data.append({
                    'user_id': user_id,
                    'date': date,
                    'time': '08:00:00',
                    'amount': -35.00,
                    'merchant_name': 'Bank Fee',
                    'category': 'Fee',
                    'status': 'Success'
                })

    df_txns = pd.DataFrame(data)
    df_users = pd.DataFrame(user_data)
    
    print(f"Generated {len(df_txns)} transactions and {len(df_users)} user profiles.")
    return df_txns, df_users

if __name__ == "__main__":
    df_txns, df_users = generate_synthetic_data()
    df_txns.to_csv("transactions.csv", index=False)
    df_users.to_csv("users.csv", index=False)
    print("Saved to transactions.csv and users.csv")
