"""
================================================================================
UIDAI HACKATHON 2026 - MASTER ANALYSIS SCRIPT
================================================================================
This script links and runs ALL analysis modules in sequence.
State: ODISHA

Files Linked:
  1. UIDAI.py          - Data Cleaning & Preparation
  2. enrolment.py      - Enrollment Analysis
  3. demographics.py   - Demographic Update Analysis
  4. biometric.py      - Biometric Update Analysis
  5. integrated_analysis.py - Combined Analysis
  6. advanced_ml_analysis.py - ML Algorithms

Run: python master_analysis.py
================================================================================
"""

import os
import sys
import subprocess
from datetime import datetime

# Configuration
DATA_DIR = r"c:\Users\rohan\Aadharcard"
PYTHON_ENV = "PYTHONIOENCODING=utf-8"

# All analysis scripts in execution order
SCRIPTS = [
    ("enrolment.py", "Enrollment Analysis"),
    ("demographics.py", "Demographic Analysis"),
    ("biometric.py", "Biometric Analysis"),
    ("integrated_analysis.py", "Integrated Analysis"),
    ("advanced_ml_analysis.py", "Advanced ML Analysis"),
]

def print_header():
    print("\n" + "="*70)
    print(" UIDAI HACKATHON 2026 - MASTER ANALYSIS PIPELINE")
    print("="*70)
    print(f" State: ODISHA")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def print_section(title, index, total):
    print(f"\n" + "-"*70)
    print(f" [{index}/{total}] {title}")
    print("-"*70)

def run_script(script_name):
    """Run a Python script and return success status"""
    script_path = os.path.join(DATA_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_name}")
        return False
    
    try:
        # Set environment and run
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=DATA_DIR,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"  [SUCCESS] {script_name} completed")
            return True
        else:
            print(f"  [ERROR] {script_name} failed")
            print(f"  Error: {result.stderr[:500] if result.stderr else 'Unknown'}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Failed to run {script_name}: {str(e)}")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "odisha_enrolment_clean.csv",
        "odisha_demographic_clean.csv",
        "odisha_biometric_clean.csv"
    ]
    
    print("\n[1] CHECKING DATA FILES...")
    all_exist = True
    
    for file in required_files:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file}")
            all_exist = False
    
    return all_exist

def list_output_files():
    """List all generated output files"""
    output_patterns = ['.png', '_analysis.csv', '_pincodes.csv', '_scores.csv', '_summary.csv']
    
    print("\n" + "="*70)
    print(" GENERATED OUTPUT FILES")
    print("="*70)
    
    for file in sorted(os.listdir(DATA_DIR)):
        for pattern in output_patterns:
            if pattern in file:
                size = os.path.getsize(os.path.join(DATA_DIR, file))
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                print(f"  {file:<45} {size_str:>10}")
                break

def main():
    print_header()
    
    # Check data files
    if not check_data_files():
        print("\n[ERROR] Missing data files. Run UIDAI.py first to prepare data.")
        return
    
    # Run all analysis scripts
    print("\n[2] RUNNING ANALYSIS PIPELINE...")
    
    results = []
    total = len(SCRIPTS)
    
    for i, (script, description) in enumerate(SCRIPTS, 1):
        print_section(description, i, total)
        success = run_script(script)
        results.append((script, description, success))
    
    # Summary
    print("\n" + "="*70)
    print(" PIPELINE SUMMARY")
    print("="*70)
    
    success_count = sum(1 for _, _, s in results if s)
    print(f"\n  Scripts Run: {total}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {total - success_count}")
    
    print("\n  RESULTS:")
    for script, desc, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"    {status} {desc}")
    
    # List outputs
    list_output_files()
    
    # Dashboard instructions
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("""
  To run the interactive dashboard:
    python -m streamlit run dashboard.py

  Or open in browser: http://localhost:8501
""")
    
    print("="*70)
    print(" MASTER ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
