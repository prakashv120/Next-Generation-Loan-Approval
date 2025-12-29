# Gen-Z Credit Scoring Engine - Standalone Run Folder

## 📁 What is this folder?

This is a **standalone folder** containing all necessary files to run the Gen-Z Credit Scoring Engine. You can run the entire application with a single command!

## 🚀 Quick Start

### Run the Application

Simply open a terminal in this folder and run:

```bash
python run.py
```

That's it! The script will:
1. ✅ Generate synthetic data (users and transactions)
2. ✅ Calculate credit features
3. ✅ Train the ML model (if needed)
4. ✅ Launch the Streamlit dashboard in your browser

## 📋 Requirements

Make sure you have the following Python packages installed:

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap altair matplotlib
```

## 📂 Files in this folder

- `run.py` - Main entry point (run this!)
- `app.py` - Streamlit dashboard UI
- `features.py` - Feature engineering logic
- `pipeline.py` - Credit decision pipeline
- `data_gen.py` - Synthetic data generator
- `train_model.py` - Model training script
- `users.csv` - User profiles (generated automatically)
- `transactions.csv` - Transaction history (generated automatically)
- `xgb_model.pkl` - Trained ML model (created on first run)
- `shap_explainer.pkl` - SHAP explainer for interpretability

## 💡 Usage Tips

### In VS Code
1. Open this `run` folder in VS Code
2. Open a new terminal (Terminal → New Terminal)
3. Run: `python run.py`

### In Command Prompt/PowerShell
1. Navigate to this folder: `cd path\to\run`
2. Run: `python run.py`

### Stop the Server
Press `Ctrl+C` in the terminal to stop the Streamlit server

## 🎯 What the dashboard does

The dashboard allows you to:
- 📊 View credit scores for all users
- 💰 See loan recommendations and interest rates
- 📈 Analyze spending patterns and cash flow
- 🎲 Understand risk factors with SHAP visualizations
- 💼 Manage capital allocation across applicants

## 🔄 Re-running

You can run `python run.py` as many times as you want:
- It will skip data generation if files already exist
- It will skip model training if the model is already trained
- It always launches the dashboard

## 🧹 Starting Fresh

To generate new data and retrain the model:
1. Delete: `users.csv`, `transactions.csv`, `xgb_model.pkl`, `shap_explainer.pkl`
2. Run: `python run.py`

## 📞 Need Help?

If you encounter any errors:
1. Make sure all required packages are installed
2. Check that you're using Python 3.7 or higher
3. Ensure you have write permissions in this folder

Happy credit scoring! 🎉
