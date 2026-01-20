# 🆔 UIDAI Hackathon 2026 - Unified Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

> **भारतीय विशिष्ट पहचान प्राधिकरण | Unique Identification Authority of India**

A comprehensive data analytics dashboard for Aadhaar enrollment and update analysis for the state of **Odisha**.

![Dashboard Preview](output/charts/odisha_enrollment_analysis_charts.png)

---

## 🎯 Problem Statement

Analysis of Aadhaar data to identify:
- **Enrollment Patterns**: New Aadhaar registrations across age groups
- **Demographic Updates**: Address, mobile, and name changes
- **Biometric Updates**: Fingerprint, iris, and face revalidation patterns
- **Service Gaps**: Underserved areas and high-load pincodes

---

## 📊 Features

### 📋 Enrollment Analysis
- **Bal Aadhaar (0-5 years)**: Child enrollment tracking
- **Youth (5-17 years)**: School-age registrations
- **Adults (18+)**: Adult first-time enrollments
- District-wise and pincode-wise breakdown

### 📝 Demographic Updates
- Address change tracking (migration patterns)
- Mobile number updates
- Name correction analysis
- Age group wise update patterns

### � Biometric Updates
- **Mandatory Updates**: Children at age 5, 10, 15
- **Revalidation**: Adults with fingerprint wear
- Iris and face photo updates
- Authentication failure corrections

### 🤖 Advanced Analytics
- **K-Means Clustering**: Pincode segmentation
- **Anomaly Detection**: Outlier identification using Isolation Forest
- **Trend Prediction**: Linear regression forecasting
- **Statistical Analysis**: Univariate, bivariate, trivariate analysis

---

## 🚀 Live Demo

**Streamlit Cloud**: [Coming Soon]

---

## 💻 Local Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/rohanbarik457-hash/uidai_dashboard.git
cd uidai_dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashborad.py
```

### Access
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

---

## 📁 Project Structure

```
uidai_dashboard/
├── dashborad.py              # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore file
├── run.bat                   # Windows batch runner
│
├── data/
│   └── processed/            # Cleaned datasets
│       ├── odisha_enrolment_clean.csv
│       ├── odisha_demographic_clean.csv
│       └── odisha_biometric_clean.csv
│
├── output/
│   ├── charts/               # Generated visualizations
│   └── reports/              # Analysis reports (CSV)
│
└── scripts/                  # Analysis scripts
    ├── enrolment.py
    ├── demographics.py
    ├── biometric.py
    └── integrated-analysis.py
```

---

## � Data Overview

| Dataset | Records | Description |
|---------|---------|-------------|
| Enrollment | 120,454+ | New Aadhaar registrations |
| Demographic | 150,000+ | Address/Mobile/Name updates |
| Biometric | 180,000+ | Fingerprint/Iris/Face updates |

### Districts Covered: **30** (All districts of Odisha)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web Dashboard Framework |
| **Pandas** | Data Manipulation |
| **Plotly** | Interactive Visualizations |
| **Scikit-learn** | Machine Learning (K-Means, Isolation Forest) |
| **SciPy** | Statistical Analysis |
| **NumPy** | Numerical Computing |

---

## 📊 Dashboard Sections

1. **📊 Dashboard Overview** - Summary metrics and pie charts
2. **📈 Trends & Prediction** - Time series with 3-month forecast
3. **📉 Statistical Analysis** - Univariate, bivariate, trivariate
4. **🔄 District Comparison** - Side-by-side district analysis
5. **📍 Pincode Analysis** - Detailed pincode breakdown
6. **� Age Analysis** - Age group distribution
7. **⚡ Gap Analysis** - Service gaps and recommendations
8. **🤖 ML Insights** - Clustering and anomaly detection

---

## 🎨 Design Theme

- **Indian Tricolor Theme**: Saffron, White, Green accents
- **Aadhaar Branding**: Official UIDAI colors
- **Government Style**: Professional government dashboard look

---

## 👥 Team

**UIDAI Hackathon 2026 - Odisha Analysis Team**

---

## 📜 License

This project is created for UIDAI Hackathon 2026.

---

## 🙏 Acknowledgments

- **UIDAI** - Unique Identification Authority of India
- **Government of India** - Ministry of Electronics & IT
- **Odisha State Government**

---

<div align="center">

**Built with ❤️ for Digital India | डिजिटल भारत के लिए**

🇮🇳 **जय हिंद** 🇮🇳

</div>
