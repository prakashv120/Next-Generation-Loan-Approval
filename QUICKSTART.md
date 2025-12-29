# Quick Start Guide - Gen-Z Credit Scoring Engine

## 🎯 What You Have

A complete, standalone folder (`run`) that contains everything needed to run the Gen-Z Credit Scoring Engine!

## 🚀 How to Run (3 Simple Steps)

### Step 1: Install Dependencies (One-time setup)

Open PowerShell in the `run` folder and run:

```powershell
.\install.ps1
```

Or manually install:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python run.py
```

### Step 3: Use the Dashboard

The Streamlit dashboard will automatically open in your browser at `http://localhost:8501`

## ✨ Features

- **Automatic Setup**: The script handles data generation, feature calculation, and model training
- **Smart Execution**: Skips steps that are already completed (data exists, model trained)
- **User-Friendly**: Clear progress messages and error handling
- **VS Code Compatible**: Works perfectly in VS Code terminals

## 📁 Folder Structure

```
run/
├── run.py              ← Run this file!
├── README.md           ← Detailed documentation
├── requirements.txt    ← Python dependencies
├── install.ps1         ← Install script
├── app.py              ← Streamlit dashboard
├── features.py         ← Feature engineering
├── pipeline.py         ← Credit decision logic
├── data_gen.py         ← Data generator
├── train_model.py      ← Model trainer
├── users.csv           ← User data (auto-generated)
├── transactions.csv    ← Transaction data (auto-generated)
├── xgb_model.pkl       ← ML model (auto-created)
└── shap_explainer.pkl  ← SHAP explainer (auto-created)
```

## 🎨 What the Dashboard Shows

1. **Portfolio Overview**: Total capital, loan demand, approved users
2. **Applicant Reports**: Individual credit scores and risk analysis
3. **Visualizations**: 
   - Risk factor breakdowns (SHAP)
   - Spending habits (pie charts)
   - Cash flow trends (line charts)
   - Peer comparisons (bar charts)
   - Risk gauges
4. **Financial Metrics**: Loan limits, interest rates, risk probabilities

## 🔄 Common Commands

| Command | Purpose |
|---------|---------|
| `python run.py` | Run the entire application |
| `.\install.ps1` | Install dependencies |
| `Ctrl+C` | Stop the server |

## 💡 Tips

- **First Run**: Takes a few minutes to generate data and train model
- **Subsequent Runs**: Much faster (uses existing data and model)
- **Fresh Start**: Delete CSV and PKL files, then run `python run.py`
- **In VS Code**: Open the `run` folder, then use the integrated terminal

## ✅ Verification

To verify everything is working:

1. Navigate to the `run` folder
2. Run: `python run.py`
3. Wait for the browser to open
4. Upload `transactions.csv` in the dashboard
5. View the credit scores and visualizations

## 🐛 Troubleshooting

**Error: Module not found**
- Solution: Run `pip install -r requirements.txt`

**Error: Permission denied**
- Solution: Run PowerShell as Administrator

**Browser doesn't open**
- Solution: Manually visit `http://localhost:8501`

---

**Ready to go!** Just type `python run.py` and you're all set! 🎉
