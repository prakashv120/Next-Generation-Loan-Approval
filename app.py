import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import altair as alt
import pickle
import numpy as np
from features import CashFlowFeatures
from pipeline import CreditPipeline

# Page Config
st.set_page_config(
    page_title="Gen-Z Credit Assistant",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for friendly look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1 {
        color: #4f46e5;
    }
    h2, h3 {
        color: #1f2937;
    }
    .friendly-text {
        font-size: 18px;
        color: #4b5563;
        line-height: 1.6;
    }
    .success-box {
        padding: 20px;
        background-color: #dcfce7;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        color: #14532d;
    }
    .fail-box {
        padding: 20px;
        background-color: #fee2e2;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        color: #7f1d1d;
    }
    .warn-box {
        padding: 20px;
        background-color: #fef3c7;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        color: #78350f;
    }
    .offer-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #6366f1;
        text-align: center;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.1);
    }
    .offer-value {
        font-size: 32px;
        font-weight: bold;
        color: #4338ca;
    }
    .offer-label {
        font-size: 16px;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper: Friendly Names Mapping ---
FRIENDLY_NAMES = {
    'net_cashflow': 'Monthly Savings Ratio',
    'income_stability': 'Income Volatility',
    'eom_balance': 'End-of-Month Buffer',
    'neg_balance_days': 'Days with Negative Balance',
    'low_balance_days': 'Days with Low Balance (<$200)',
    'declined_txns': 'Declined Transactions',
    'upi_stability': 'Transfer Inflow Stability',
    'wallet_transfers': 'Wallet Transfers to Friends',
    'essential_ratio': 'Essential Spending %',
    'discretionary_ratio': 'Discretionary Spending %',
    'food_delivery_ratio': 'Food Delivery Spend %',
    'gaming_ratio': 'Gaming Spend %',
    'fashion_ratio': 'Fashion Spend %',
    'gambling_ratio': 'Gambling/Crypto Spend %',
    'bnpl_ratio': 'BNPL Usage %',
    'bnpl_failures': 'Missed BNPL Payments',
    'night_txns': 'Late Night Transactions (2am-5am)',
    'weekend_ratio': 'Weekend Spending Spike',
    'micro_spends': 'Micro-transactions Count',
    'refunds': 'Refunds Count',
    'parental_dependency': 'Reliance on Parents',
    'gig_ratio': 'Gig Economy Income %',
    'failed_subs': 'Failed Subscriptions',
    'active_subs': 'Active Subscriptions',
    'sim_age': 'SIM Card Age (Months)',
    'device_age': 'Device Age (Months)',
    'loan_apps': 'Loan Apps Installed',
    'gaming_apps': 'Gaming Apps Installed',
    'finance_apps': 'Finance Apps Installed',
    'signup_tenure': 'App Usage Tenure',
    'upi_tenure': 'UPI ID Tenure',
    'address_stability': 'Address Stability'
}

# --- Load Resources ---
@st.cache_resource
def load_pipeline():
    return CreditPipeline()

@st.cache_resource
def load_explainer():
    try:
        with open("shap_explainer.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

pipeline = load_pipeline()
explainer = load_explainer()

# --- Header ---
st.title("🚀 Gen-Z Credit Scoring Engine")
st.markdown("""
<p class="friendly-text">
    Next-gen credit scoring based on cash flow, lifestyle, and behavioral patterns.
    Upload your bank statement file below.
</p>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Step 1: Upload File")
    uploaded_file = st.file_uploader("Choose transactions file", type="csv")
    st.info("ℹ️ Please use `transactions.csv`.")
    
    st.markdown("---")
    st.header("Step 2: User Data (Optional)")
    user_file = st.file_uploader("Choose user profile file", type="csv")
    st.info("ℹ️ Please use `users.csv` if available.")
    
    st.markdown("---")
    st.header("Step 3: Capital Allocation")
    total_capital = st.number_input("Total Capital Pool ($)", min_value=0, value=1000000, step=10000, format="%d")

# --- Main Logic ---
if uploaded_file is not None:
    try:
        # 1. Load Data
        transactions_df = pd.read_csv(uploaded_file, dtype={'user_id': str})
        
        users_df = None
        if user_file is not None:
            users_df = pd.read_csv(user_file, dtype={'user_id': str})
        
        # 2. Calculate Features
        with st.spinner("Analyzing financial DNA..."):
            features_engine = CashFlowFeatures(transactions_df, users_df)
            features_df = features_engine.calculate_features()
            
            results = []
            for user_id, row in features_df.iterrows():
                # Run Pipeline
                res = pipeline.run_waterfall(row)
                res['user_id'] = user_id
                
                # Score 0-100
                if res['pd'] is not None:
                    res['score'] = int((1 - res['pd']) * 100)
                else:
                    res['score'] = 0
                
                # --- FINANCIAL CALCULATIONS ---
                # Get user transactions
                user_txns = transactions_df[transactions_df['user_id'] == user_id]
                monthly_income = user_txns[user_txns['amount'] > 0]['amount'].sum() / 3
                
                # Loan Limit
                # Use net_cashflow (was n_fcf)
                n_fcf = max(0, row.get('net_cashflow', 0))
                loan_limit = monthly_income * n_fcf * 0.3 * 12
                loan_limit = round(loan_limit / 100) * 100
                
                # Interest Rate
                pd_val = res['pd'] if res['pd'] is not None else 1.0
                interest_rate = 8.0 + (pd_val * 20.0)
                if interest_rate > 36: interest_rate = 36.0
                
                res['loan_limit'] = loan_limit if res['decision'] == 'Approve' else 0
                res['interest_rate'] = interest_rate
                res['monthly_income'] = monthly_income
                
                results.append(res)
            
            results_df = pd.DataFrame(results)

        # --- Dashboard View ---
        st.markdown("---")
        st.header("📊 Portfolio Overview")
        
        # Capital Logic
        total_demand = results_df[results_df['decision'] == 'Approve']['loan_limit'].sum()
        approved_users = len(results_df[results_df['decision'] == 'Approve'])
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Capital", f"${total_capital:,.0f}")
        c2.metric("Loan Demand", f"${total_demand:,.0f}", delta=f"${total_capital - total_demand:,.0f} Remaining" if total_capital >= total_demand else f"Shortfall")
        c3.metric("Approved Users", approved_users)
        
        # Progress Bar
        utilization = min(1.0, total_demand / total_capital) if total_capital > 0 else 1.0
        st.progress(utilization)
        
        st.markdown("---")
        st.header("👤 Applicant Report")
        
        # Search Filter
        all_ids = sorted(results_df['user_id'].unique())
        selected_user_id = st.selectbox("Select Applicant ID", all_ids)
        
        if selected_user_id is not None:
            # Get Data
            user_res = results_df[results_df['user_id'] == selected_user_id].iloc[0]
            user_feats = features_df.loc[selected_user_id]
            user_txns = transactions_df[transactions_df['user_id'] == selected_user_id]
            
            # --- DECISION SECTION ---
            decision = user_res['decision']
            score = user_res['score']
            loan_limit = user_res['loan_limit']
            interest_rate = user_res['interest_rate']
            risk_rate = (user_res['pd'] * 100) if user_res['pd'] else 0
            
            if decision == 'Approve':
                st.markdown(f"""
                <div class="success-box">
                    <h2>🎉 Approved (Score: {score})</h2>
                    <p class="friendly-text">Low risk profile. Recommended for approval.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💰 Offer")
                o1, o2, o3 = st.columns(3)
                with o1:
                    st.metric("Max Loan Amount", f"${loan_limit:,.0f}")
                with o2:
                    st.metric("Interest Rate", f"{interest_rate:.1f}%")
                with o3:
                    st.metric("Risk Probability", f"{risk_rate:.1f}%")
                    
            elif decision == 'Reject':
                st.markdown(f"""
                <div class="fail-box">
                    <h2>❌ Rejected (Score: {score})</h2>
                    <p class="friendly-text">High risk factors detected.</p>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Risk Probability", f"{risk_rate:.1f}%")
                
            else:
                st.markdown(f"""
                <div class="warn-box">
                    <h2>⚠️ Refer (Score: {score})</h2>
                    <p class="friendly-text">Manual review required.</p>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Risk Probability", f"{risk_rate:.1f}%")
            
            st.markdown("---")
            
            # --- VISUALS SECTION ---
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.subheader("🧐 Risk Factors")
                
                if explainer:
                    # Calculate SHAP
                    shap_input = pd.DataFrame([user_feats])
                    # Drop extra columns if any
                    cols = explainer.feature_names
                    # Ensure columns match
                    shap_input = shap_input[cols] if cols else shap_input
                    
                    shap_values = explainer(shap_input)
                    
                    vals = shap_values.values[0]
                    names = shap_values.feature_names
                    
                    # Create DataFrame for Chart
                    impact_data = []
                    for n, v in zip(names, vals):
                        impact_data.append({
                            'Factor': FRIENDLY_NAMES.get(n, n),
                            'Impact': v,
                            'Type': 'Good' if v < 0 else 'Bad'
                        })
                    
                    impact_df = pd.DataFrame(impact_data)
                    impact_df['AbsImpact'] = impact_df['Impact'].abs()
                    impact_df = impact_df.sort_values('AbsImpact', ascending=False).head(10)
                    
                    chart = alt.Chart(impact_df).mark_bar().encode(
                        x=alt.X('Impact', title='Impact on Risk'),
                        y=alt.Y('Factor', sort='-x'),
                        color=alt.Color('Type', scale=alt.Scale(domain=['Good', 'Bad'], range=['#22c55e', '#ef4444']))
                    ).properties(height=350)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Model explainer not loaded.")

            with col_viz2:
                st.subheader("💸 Spending Habits")
                
                # Spending Breakdown
                expenses = user_txns[user_txns['amount'] < 0].copy()
                expenses['amount'] = expenses['amount'].abs()
                cat_spend = expenses.groupby('category')['amount'].sum().reset_index()
                
                donut = alt.Chart(cat_spend).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="amount", type="quantitative"),
                    color=alt.Color(field="category", type="nominal"),
                    tooltip=['category', 'amount']
                ).properties(height=350)
                
                st.altair_chart(donut, use_container_width=True)
                
            st.markdown("---")
            
            # --- NEW VISUALIZATIONS ---
            st.subheader("📈 Financial Health Deep Dive")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### Cash Flow Trend (90 Days)")
                # Calculate running balance
                user_txns_sorted = user_txns.sort_values('date')
                user_txns_sorted['balance'] = user_txns_sorted['amount'].cumsum()
                
                line_chart = alt.Chart(user_txns_sorted).mark_line(color='#6366f1').encode(
                    x='date:T',
                    y='balance:Q',
                    tooltip=['date', 'balance']
                ).properties(height=300)
                
                st.altair_chart(line_chart, use_container_width=True)
                
            with col_d2:
                st.markdown("#### Peer Comparison")
                # Compare user metrics to average
                avg_feats = features_df.mean(numeric_only=True)
                
                comparison_data = [
                    {'Metric': 'Savings Ratio', 'You': row['net_cashflow'] / 1000 if abs(row['net_cashflow']) > 1 else row['net_cashflow'], 'Avg': avg_feats['net_cashflow'] / 1000 if abs(avg_feats['net_cashflow']) > 1 else avg_feats['net_cashflow']},
                    {'Metric': 'Income Stability', 'You': row['income_stability'], 'Avg': avg_feats['income_stability']},
                    {'Metric': 'Gambling %', 'You': row['gambling_ratio'], 'Avg': avg_feats['gambling_ratio']},
                    {'Metric': 'BNPL %', 'You': row['bnpl_ratio'], 'Avg': avg_feats['bnpl_ratio']}
                ]
                comp_df = pd.DataFrame(comparison_data).melt('Metric', var_name='Type', value_name='Value')
                
                bar_chart = alt.Chart(comp_df).mark_bar().encode(
                    x=alt.X('Metric', axis=alt.Axis(labelAngle=0)),
                    y='Value',
                    color=alt.Color('Type', scale=alt.Scale(domain=['You', 'Avg'], range=['#6366f1', '#9ca3af'])),
                    xOffset='Type:N'
                ).properties(height=300)
                
                st.altair_chart(bar_chart, use_container_width=True)
            
            # Risk Gauge (HTML/CSS)
            st.markdown("#### Credit Risk Gauge")
            gauge_color = "#22c55e" if score > 75 else "#f59e0b" if score > 50 else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                <div style="position: relative; width: 200px; height: 100px; overflow: hidden;">
                    <div style="position: absolute; top: 0; left: 0; width: 200px; height: 200px; border-radius: 50%; background: #e5e7eb;"></div>
                    <div style="position: absolute; top: 0; left: 0; width: 200px; height: 200px; border-radius: 50%; background: conic-gradient({gauge_color} {score * 1.8}deg, transparent 0deg); transform: rotate(-90deg);"></div>
                    <div style="position: absolute; top: 20px; left: 20px; width: 160px; height: 160px; border-radius: 50%; background: white; display: flex; justify-content: center; align-items: center;">
                        <span style="font-size: 32px; font-weight: bold; color: {gauge_color}; margin-bottom: 50px;">{score}</span>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: -40px; color: #6b7280;">Credit Score</div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Please ensure you uploaded the correct files.")
else:
    st.info("👈 Upload `transactions.csv` to start.")
