import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"⏳ {description}...")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error running: {description}")
        print(f"Command: {command}")
        sys.exit(1)
    print(f"✅ {description} completed successfully!\n")

def main():
    print("\n" + "="*60)
    print("🚀 Gen-Z Credit Scoring Engine")
    print("="*60)
    print("This will run the complete credit scoring pipeline:")
    print("  1. Generate synthetic data (if needed)")
    print("  2. Calculate features")
    print("  3. Train model (if needed)")
    print("  4. Launch Streamlit dashboard")
    print("="*60 + "\n")
    
    # Get Python executable
    python_exe = sys.executable
    
    # Step 1: Generate Data (if files don't exist)
    if not os.path.exists("users.csv") or not os.path.exists("transactions.csv"):
        run_command(f'"{python_exe}" data_gen.py', "Step 1: Generating Synthetic Data")
    else:
        print("\n✓ Data files already exist, skipping generation")
    
    # Step 2: Calculate Features
    run_command(f'"{python_exe}" features.py', "Step 2: Calculating Features")
    
    # Step 3: Train Model (if model doesn't exist)
    if not os.path.exists("xgb_model.pkl"):
        run_command(f'"{python_exe}" train_model.py', "Step 3: Training Model")
    else:
        print("\n✓ Model already exists, skipping training")
    
    # Step 4: Launch UI
    print("\n" + "="*60)
    print("🎯 Launching Streamlit Dashboard...")
    print("="*60)
    print("\n📌 The dashboard will open in your browser automatically")
    print("📌 To stop the server, press Ctrl+C")
    print("\n" + "="*60 + "\n")
    
    cmd = f'"{python_exe}" -m streamlit run app.py'
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        print("Goodbye!\n")

if __name__ == "__main__":
    main()
