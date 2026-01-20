# UIDAI Hackathon 2026 - Odisha Aadhaar Analysis

## 📁 Project Structure

```
Aadharcard/
│
├── 📊 dashboard_unified.py    ← Main Interactive Dashboard
├── 📄 README.md               ← This File
├── 📦 requirements.txt        ← Python Dependencies
├── 🖥️ run.bat                 ← Quick Start Script
│
├── 📂 data/
│   ├── raw/                   ← Original API Data Files
│   │   ├── api_data_aadhar_*.csv (12 files)
│   │   └── aadhar_*_merged.csv (3 files)
│   │
│   └── processed/             ← Cleaned Odisha Data
│       ├── odisha_enrolment_clean.csv
│       ├── odisha_demographic_clean.csv
│       └── odisha_biometric_clean.csv
│
├── 📂 output/
│   ├── charts/                ← Generated Visualizations
│   │   ├── advanced_ml_analysis.png
│   │   ├── odisha_enrollment_analysis_charts.png
│   │   ├── odisha_demographic_analysis_charts.png
│   │   ├── odisha_biometric_analysis_charts.png
│   │   └── odisha_integrated_analysis.png
│   │
│   └── reports/               ← Analysis CSV Reports
│       ├── district_inclusion_scores.csv
│       ├── ml_pincode_analysis.csv
│       └── *.csv (various analysis outputs)
│
└── 📂 scripts/                ← Analysis Python Scripts
    ├── UIDAI.py               ← Data Cleaning
    ├── enrolment.py           ← Enrollment Analysis
    ├── demographics.py        ← Demographics Analysis
    ├── biometric.py           ← Biometric Analysis
    ├── integrated_analysis.py ← Combined Analysis
    ├── advanced_ml_analysis.py ← ML Algorithms
    └── master_analysis.py     ← Run All Scripts
```

## 🚀 Quick Start

```powershell
# Run Interactive Dashboard
python -m streamlit run dashboard_unified.py
```

Open http://localhost:8501 in browser

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| 📋 Dataset Selector | Switch between Enrollment, Demographics, Biometrics |
| 📍 Pincode Filter | Filter by specific pincode |
| 🔮 Future Prediction | 3-month forecast using Linear Regression |
| ⚡ Spike Analysis | Detects spikes with reasons |
| 🤖 ML Insights | K-Means clustering, Anomaly detection |
| 📥 Data Download | Export pincode data as CSV |

## 🤖 ML Algorithms Used

1. **K-Means Clustering** - Pincode segmentation
2. **Linear Regression** - Future prediction
3. **Isolation Forest** - Anomaly detection
4. **Z-Score Analysis** - Spike detection

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| Total Enrollments | 120,454 |
| Demo Updates | 852,642 |
| Bio Updates | 2,422,010 |
| Districts | 30 |
| Pincodes | 945 |

## 📦 Requirements

```
pip install -r requirements.txt
```

## 👥 Team

UIDAI Hackathon 2026 Participant

---

**State Analyzed:** Odisha, India
