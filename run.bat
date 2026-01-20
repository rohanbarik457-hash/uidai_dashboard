@echo off
echo ====================================================
echo UIDAI HACKATHON 2026 - ODISHA ANALYSIS
echo ====================================================
echo.

set PYTHONIOENCODING=utf-8

echo [1/5] Running Enrollment Analysis...
python enrolment.py
echo.

echo [2/5] Running Demographic Analysis...
python demographics.py
echo.

echo [3/5] Running Biometric Analysis...
python biometric.py
echo.

echo [4/5] Running Integrated Analysis...
python integrated_analysis.py
echo.

echo [5/5] Running Advanced ML Analysis...
python advanced_ml_analysis.py
echo.

echo ====================================================
echo ALL ANALYSIS COMPLETE!
echo ====================================================
echo.
echo To run dashboard: python -m streamlit run dashboard.py
echo.
pause
