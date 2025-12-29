"""
Test script to verify all components are working
"""
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    errors = []
    
    modules = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'streamlit',
        'shap',
        'altair',
        'matplotlib'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module} - NOT FOUND")
            errors.append(module)
    
    return errors

def test_files():
    """Test if all required files exist"""
    print("\nTesting files...")
    errors = []
    
    files = [
        'app.py',
        'features.py',
        'pipeline.py',
        'data_gen.py',
        'train_model.py',
        'users.csv',
        'transactions.csv',
        'xgb_model.pkl',
        'shap_explainer.pkl'
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  [OK] {file} ({size:,} bytes)")
        else:
            print(f"  [FAIL] {file} - NOT FOUND")
            errors.append(file)
    
    return errors

def test_components():
    """Test if custom modules can be imported"""
    print("\nTesting custom components...")
    errors = []
    
    try:
        from features import CashFlowFeatures
        print("  [OK] CashFlowFeatures imported")
    except Exception as e:
        print(f"  [FAIL] CashFlowFeatures - {e}")
        errors.append('CashFlowFeatures')
    
    try:
        from pipeline import CreditPipeline
        print("  [OK] CreditPipeline imported")
    except Exception as e:
        print(f"  [FAIL] CreditPipeline - {e}")
        errors.append('CreditPipeline')
    
    return errors

def main():
    print("="*60)
    print("Gen-Z Credit Scoring - Setup Verification")
    print("="*60)
    print()
    
    import_errors = test_imports()
    file_errors = test_files()
    component_errors = test_components()
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    if not import_errors and not file_errors and not component_errors:
        print("[SUCCESS] ALL CHECKS PASSED!")
        print("\nYou're ready to run the application:")
        print("    python run.py")
    else:
        print("[ERROR] SOME CHECKS FAILED")
        
        if import_errors:
            print(f"\n[WARNING] Missing packages: {', '.join(import_errors)}")
            print("   Run: pip install -r requirements.txt")
        
        if file_errors:
            print(f"\n[WARNING] Missing files: {', '.join(file_errors)}")
            print("   These files should have been copied")
        
        if component_errors:
            print(f"\n[WARNING] Component errors: {', '.join(component_errors)}")
    
    print()

if __name__ == "__main__":
    main()
