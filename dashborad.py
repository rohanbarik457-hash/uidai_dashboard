"""
UIDAI HACKATHON 2026 - UNIFIED DASHBOARD
=========================================
Complete Analysis Dashboard for Odisha
Includes: Enrollment + Demographics + Biometrics
Each dataset shows its specific relevant content
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="UIDAI Hackathon 2026 - Unified Dashboard",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling with Indian Tricolor Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    /* Indian Tricolor Theme */
    :root {
        --saffron: #FF9933;
        --white: #FFFFFF;
        --green: #138808;
        --navy: #000080;
        --aadhaar-blue: #2E3192;
        --aadhaar-orange: #ED1C24;
    }
    
    /* Tricolor Header Bar */
    .tricolor-bar {
        height: 8px;
        background: linear-gradient(90deg, #FF9933 33%, #FFFFFF 33%, #FFFFFF 66%, #138808 66%);
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    
    /* Main Header with Aadhaar Colors */
    .mainHeader {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    .enrollHeader { 
        background: linear-gradient(135deg, #FF9933 0%, #2E3192 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .demoHeader { 
        background: linear-gradient(135deg, #ED1C24 0%, #2E3192 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .bioHeader { 
        background: linear-gradient(135deg, #138808 0%, #2E3192 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subHeader {
        font-size: 1.2rem;
        color: #2E3192;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Government Branding Header */
    .govt-header {
        background: linear-gradient(135deg, #2E3192 0%, #1a1a5e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .govt-header h2 {
        margin: 0;
        font-size: 1.2rem;
    }
    
    .govt-header p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Logo Container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Insight Boxes with Tricolor */
    .insightBox {
        background: linear-gradient(135deg, #fff8f0 0%, #ffffff 100%);
        border-left: 4px solid #FF9933;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .insightBoxGreen {
        background: linear-gradient(135deg, #f0fff4 0%, #ffffff 100%);
        border-left: 4px solid #138808;
    }
    
    .spikeBox {
        background: linear-gradient(135deg, #fff3e6 0%, #ffffff 100%);
        border-left: 4px solid #FF9933;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .predictionBox {
        background: linear-gradient(135deg, #e8f0ff 0%, #ffffff 100%);
        border-left: 4px solid #2E3192;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Metric Cards with Tricolor Accent */
    .stMetric {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-top: 3px solid;
        border-image: linear-gradient(90deg, #FF9933, #FFFFFF, #138808) 1;
        padding: 1rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E3192 0%, #1a1a5e 100%);
    }
    
    /* Footer with Government Style */
    .govt-footer {
        background: linear-gradient(135deg, #2E3192 0%, #1a1a5e 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .govt-footer p {
        margin: 0.3rem 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f0f0 100%);
        border-radius: 8px 8px 0 0;
        border: 1px solid #ddd;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #FF9933 0%, #e68a2e 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading Functions
@st.cache_data
def loadEnrollmentData():
    dataFrame = pd.read_csv("data/processed/odisha_enrolment_clean.csv")
    dataFrame["date"] = pd.to_datetime(dataFrame["date"], format="%d-%m-%Y")
    dataFrame["month"] = dataFrame["date"].dt.month
    dataFrame["monthName"] = dataFrame["date"].dt.month_name()
    dataFrame["totalEnrollments"] = dataFrame["age_0_5"] + dataFrame["age_5_17"] + dataFrame["age_18_greater"]
    dataFrame["balAadhaarCount"] = dataFrame["age_0_5"]
    dataFrame["youthCount"] = dataFrame["age_5_17"]
    dataFrame["adultCount"] = dataFrame["age_18_greater"]
    return dataFrame

@st.cache_data
def loadDemographicData():
    dataFrame = pd.read_csv("data/processed/odisha_demographic_clean.csv")
    dataFrame["date"] = pd.to_datetime(dataFrame["date"], format="%d-%m-%Y")
    dataFrame["month"] = dataFrame["date"].dt.month
    dataFrame["monthName"] = dataFrame["date"].dt.month_name()
    dataFrame["totalDemoUpdates"] = dataFrame["demo_age_5_17"] + dataFrame["demo_age_17_"]
    dataFrame["youthDemoUpdates"] = dataFrame["demo_age_5_17"]
    dataFrame["adultDemoUpdates"] = dataFrame["demo_age_17_"]
    return dataFrame

@st.cache_data
def loadBiometricData():
    dataFrame = pd.read_csv("data/processed/odisha_biometric_clean.csv")
    dataFrame["date"] = pd.to_datetime(dataFrame["date"], format="%d-%m-%Y")
    dataFrame["month"] = dataFrame["date"].dt.month
    dataFrame["monthName"] = dataFrame["date"].dt.month_name()
    dataFrame["totalBioUpdates"] = dataFrame["bio_age_5_17"] + dataFrame["bio_age_17_"]
    dataFrame["youthBioUpdates"] = dataFrame["bio_age_5_17"]
    dataFrame["adultBioUpdates"] = dataFrame["bio_age_17_"]
    return dataFrame

# Load all data
enrollmentData = loadEnrollmentData()
demographicData = loadDemographicData()
biometricData = loadBiometricData()

# Sidebar Configuration
# Styled Text Logo (no image dependency)
st.sidebar.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #2E3192 0%, #1a1a5e 100%); border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
    <div style='font-size: 2.5rem; font-weight: bold;'>
        <span style='color: #FF9933;'>आ</span><span style='color: white;'>धा</span><span style='color: #138808;'>र</span>
    </div>
    <div style='font-size: 1.8rem; font-weight: bold; letter-spacing: 3px;'>
        <span style='color: #FF9933;'>AA</span><span style='color: white;'>DH</span><span style='color: #138808;'>AAR</span>
    </div>
    <p style='color: #ccc; margin: 10px 0 0 0; font-size: 0.75rem;'>भारतीय विशिष्ट पहचान प्राधिकरण</p>
    <p style='color: #FF9933; margin: 3px 0 0 0; font-size: 0.7rem;'>UIDAI - Govt. of India</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🆔 UIDAI Hackathon 2026")
st.sidebar.markdown("---")

# Analysis Section Dropdown
st.sidebar.markdown("### 📊 Analysis Options")
analysisSection = st.sidebar.selectbox(
    "🔍 Select Analysis Type",
    ["📊 Dashboard Overview", "📈 Trends & Prediction", "📉 Statistical Analysis", 
     "🔄 District Comparison", "📍 Pincode Analysis", "👥 Age Analysis", 
     "⚡ Gap Analysis", "🤖 ML Insights"],
    index=0
)

st.sidebar.markdown("---")

# Dataset Selection
datasetChoice = st.sidebar.selectbox(
    "� Select Dataset",
    ["📋 Enrollment Data", "📝 Demographic Updates", "🔐 Biometric Updates"],
    index=0
)

# Set active dataset based on selection
if datasetChoice == "📋 Enrollment Data":
    activeData = enrollmentData.copy()
    totalColumn = "totalEnrollments"
    headerClass = "enrollHeader"
    headerText = "📋 Aadhaar Enrollment Analysis"
    subText = "New Aadhaar Enrollments - Bal Aadhaar, Youth & Adult Registration in Odisha"
    colorScheme = "Blues"
    primaryColor = "#667eea"
    
elif datasetChoice == "📝 Demographic Updates":
    activeData = demographicData.copy()
    totalColumn = "totalDemoUpdates"
    headerClass = "demoHeader"
    headerText = "📝 Demographic Update Analysis"
    subText = "Address, Mobile Number & Name Updates - Who Updated What & Where in Odisha"
    colorScheme = "Reds"
    primaryColor = "#E74C3C"
    
else:
    activeData = biometricData.copy()
    totalColumn = "totalBioUpdates"
    headerClass = "bioHeader"
    headerText = "🔐 Biometric Update Analysis"
    subText = "Fingerprint, Iris & Face Updates - Who Updated Biometrics & Where in Odisha"
    colorScheme = "Greens"
    primaryColor = "#27AE60"

st.sidebar.markdown("---")

# District Filter
districtList = ["All Districts"] + sorted(activeData["district"].unique().tolist())
selectedDistrict = st.sidebar.selectbox("🏢 Select District", districtList)

# Pincode Filter
if selectedDistrict != "All Districts":
    pincodeList = ["All Pincodes"] + sorted(activeData[activeData["district"] == selectedDistrict]["pincode"].unique().astype(str).tolist())
else:
    pincodeList = ["All Pincodes"]
selectedPincode = st.sidebar.selectbox("📍 Select Pincode", pincodeList)

# Month Filter
monthList = ["All Months"] + sorted(activeData["monthName"].unique().tolist(), 
    key=lambda x: ["January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"].index(x) 
    if x in ["January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"] else 12)
selectedMonth = st.sidebar.selectbox("📅 Select Month", monthList)

# Apply Filters
filteredData = activeData.copy()
if selectedDistrict != "All Districts":
    filteredData = filteredData[filteredData["district"] == selectedDistrict]
if selectedPincode != "All Pincodes":
    filteredData = filteredData[filteredData["pincode"] == int(selectedPincode)]
if selectedMonth != "All Months":
    filteredData = filteredData[filteredData["monthName"] == selectedMonth]

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.info(f"📊 Records: {len(filteredData):,}")
st.sidebar.info(f"📍 Pincodes: {filteredData['pincode'].nunique()}")
st.sidebar.info(f"🏢 Districts: {filteredData['district'].nunique()}")

# Main Header with Government Branding
# Tricolor Bar
st.markdown('<div class="tricolor-bar"></div>', unsafe_allow_html=True)

# Government Header
st.markdown("""
<div class="govt-header">
    <h2>🇮🇳 भारत सरकार | Government of India</h2>
    <p>इलेक्ट्रॉनिक्स और सूचना प्रौद्योगिकी मंत्रालय | Ministry of Electronics & Information Technology</p>
    <p style='color: #FF9933; font-weight: bold;'>भारतीय विशिष्ट पहचान प्राधिकरण | Unique Identification Authority of India (UIDAI)</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<h1 class="mainHeader {headerClass}">{headerText}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subHeader">{subText}</p>', unsafe_allow_html=True)

# Calculate values based on dataset type
totalValue = filteredData[totalColumn].sum()
districtsCount = filteredData["district"].nunique()
pincodesCount = filteredData["pincode"].nunique()

# Dataset-specific metrics
if datasetChoice == "📋 Enrollment Data":
    balAadhaar = filteredData["balAadhaarCount"].sum()
    youthValue = filteredData["youthCount"].sum()
    adultValue = filteredData["adultCount"].sum()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📋 Total New Enrollments", f"{totalValue:,}")
    col2.metric("👶 Bal Aadhaar (Age 0-5)", f"{balAadhaar:,}", f"{balAadhaar/totalValue*100:.1f}%")
    col3.metric("🎒 Youth Enrollment (Age 5-17)", f"{youthValue:,}", f"{youthValue/totalValue*100:.1f}%")
    col4.metric("👨 Adult Enrollment (Age 18+)", f"{adultValue:,}", f"{adultValue/totalValue*100:.1f}%")
    col5.metric("🏢 Districts Covered", districtsCount)

elif datasetChoice == "📝 Demographic Updates":
    youthValue = filteredData["youthDemoUpdates"].sum()
    adultValue = filteredData["adultDemoUpdates"].sum()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📝 Total Address/Mobile/Name Updates", f"{totalValue:,}")
    col2.metric("🎒 Youth Updates (Age 5-17)", f"{youthValue:,}", f"{youthValue/totalValue*100:.1f}%")
    col3.metric("👨 Adult Updates (Age 17+)", f"{adultValue:,}", f"{adultValue/totalValue*100:.1f}%")
    col4.metric("🏢 Districts with Updates", districtsCount)
    col5.metric("📍 Pincodes with Updates", pincodesCount)

else:
    youthValue = filteredData["youthBioUpdates"].sum()
    adultValue = filteredData["adultBioUpdates"].sum()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔐 Total Fingerprint/Iris/Face Updates", f"{totalValue:,}")
    col2.metric("🎒 Youth Biometric (Age 5-17)", f"{youthValue:,}", f"{youthValue/totalValue*100:.1f}%")
    col3.metric("👨 Adult Biometric (Age 17+)", f"{adultValue:,}", f"{adultValue/totalValue*100:.1f}%")
    col4.metric("🏢 Districts with Updates", districtsCount)
    col5.metric("📍 Pincodes with Updates", pincodesCount)

st.markdown("---")

# Content based on sidebar Analysis Section selection
# SECTION 1: DASHBOARD OVERVIEW
if analysisSection == "📊 Dashboard Overview":
    if datasetChoice == "📋 Enrollment Data":
        st.subheader("📋 Enrollment Overview - New Aadhaar Registrations")
        st.markdown("""
        **What this shows:** New Aadhaar card registrations including:
        - 👶 **Bal Aadhaar (0-5)**: First-time enrollment for children
        - 🎒 **Youth (5-17)**: School-age first enrollments
        - 👨 **Adults (18+)**: Adult first-time Aadhaar registration
        """)
        
        pieLabels = ["Bal Aadhaar (0-5 years)", "Youth (5-17 years)", "Adults (18+ years)"]
        pieValues = [balAadhaar, youthValue, adultValue]
        pieColors = ["#667eea", "#764ba2", "#f093fb"]
        
    elif datasetChoice == "📝 Demographic Updates":
        st.subheader("📝 Demographic Updates - Address, Mobile & Name Changes")
        st.markdown("""
        **What this shows:** Existing Aadhaar holders updating their details:
        - 📍 **Address Updates**: People who changed their residence address
        - 📱 **Mobile Updates**: People who updated their phone number
        - ✏️ **Name Corrections**: People who corrected their name spelling
        """)
        
        pieLabels = ["Youth Updates (5-17 years)", "Adult Updates (17+ years)"]
        pieValues = [youthValue, adultValue]
        pieColors = ["#E74C3C", "#2C3E50"]
        
    else:
        st.subheader("🔐 Biometric Updates - Fingerprint, Iris & Face")
        st.markdown("""
        **What this shows:** Biometric revalidation and updates:
        - 👆 **Fingerprint Updates**: Re-capture of fingerprints (wear, aging)
        - 👁️ **Iris Updates**: Re-capture of iris patterns
        - 😊 **Face Updates**: Updated facial photograph
        
        **Why people update biometrics:**
        - Children grow and features change (mandatory at age 5, 10, 15)
        - Manual laborers with worn fingerprints
        - Authentication failures requiring revalidation
        """)
        
        pieLabels = ["Youth Mandatory (5-17 years)", "Adult Revalidation (17+ years)"]
        pieValues = [youthValue, adultValue]
        pieColors = ["#27AE60", "#1ABC9C"]
    
    chartCol1, chartCol2 = st.columns(2)
    
    with chartCol1:
        pieChart = go.Figure(data=[go.Pie(
            labels=pieLabels,
            values=pieValues,
            hole=0.5,
            marker_colors=pieColors,
            textinfo="label+percent",
            textposition="outside"
        )])
        pieChart.add_annotation(x=0.5, y=0.5, text=f"<b>{totalValue:,}</b><br>Total", 
                               font=dict(size=16), showarrow=False)
        pieChart.update_layout(title="Age Group Distribution", height=450)
        st.plotly_chart(pieChart, use_container_width=True)
    
    with chartCol2:
        districtTotals = filteredData.groupby("district")[totalColumn].sum().nlargest(10)
        
        barChart = go.Figure(data=[go.Bar(
            x=districtTotals.values,
            y=districtTotals.index,
            orientation="h",
            marker=dict(color=districtTotals.values, colorscale=colorScheme),
            text=[f"{v:,}" for v in districtTotals.values],
            textposition="outside"
        )])
        barChart.update_layout(
            title="Top 10 Districts",
            yaxis={"categoryorder": "total ascending"},
            height=450
        )
        st.plotly_chart(barChart, use_container_width=True)

# SECTION 2: TRENDS & PREDICTION
elif analysisSection == "📈 Trends & Prediction":
    st.subheader("📈 Time Trends & Future Prediction")
    
    monthlyData = filteredData.groupby("monthName")[totalColumn].sum().reset_index()
    monthOrder = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    monthlyData["sortOrder"] = monthlyData["monthName"].apply(
        lambda x: monthOrder.index(x) if x in monthOrder else 12)
    monthlyData = monthlyData.sort_values("sortOrder")
    
    # Trend Line Chart
    trendChart = go.Figure()
    trendChart.add_trace(go.Scatter(
        x=monthlyData["monthName"],
        y=monthlyData[totalColumn],
        mode="lines+markers",
        name="Actual",
        line=dict(color=primaryColor, width=3),
        marker=dict(size=12)
    ))
    
    # Future Prediction using Linear Regression
    if len(monthlyData) >= 3:
        xValues = np.arange(len(monthlyData)).reshape(-1, 1)
        yValues = monthlyData[totalColumn].values
        
        model = LinearRegression()
        model.fit(xValues, yValues)
        
        # Predict next 3 months
        futureX = np.arange(len(monthlyData), len(monthlyData) + 3).reshape(-1, 1)
        futureY = model.predict(futureX)
        futureMonths = ["Jan (Forecast)", "Feb (Forecast)", "Mar (Forecast)"]
        
        trendChart.add_trace(go.Scatter(
            x=futureMonths,
            y=futureY,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="#FFC107", width=3, dash="dash"),
            marker=dict(size=12, symbol="diamond")
        ))
        
        st.markdown(f"""
        <div class="predictionBox">
            <h4>🔮 3-Month Forecast</h4>
            <p>Based on current trends, predicted values for next 3 months:</p>
            <ul>
                <li>Month +1: <b>{futureY[0]:,.0f}</b></li>
                <li>Month +2: <b>{futureY[1]:,.0f}</b></li>
                <li>Month +3: <b>{futureY[2]:,.0f}</b></li>
            </ul>
            <small>Trend Direction: {"📈 Increasing" if model.coef_[0] > 0 else "📉 Decreasing"} ({model.coef_[0]:+,.0f} per month)</small>
        </div>
        """, unsafe_allow_html=True)
    
    trendChart.update_layout(title="Monthly Trend with Forecast", height=400)
    st.plotly_chart(trendChart, use_container_width=True)
    
    # Spike Detection and Analysis
    st.subheader("⚡ Spike Detection & Analysis")
    
    if len(monthlyData) > 2:
        monthlyData["zScore"] = stats.zscore(monthlyData[totalColumn])
        spikes = monthlyData[abs(monthlyData["zScore"]) > 1.5]
        
        if len(spikes) > 0:
            for idx, spike in spikes.iterrows():
                spikeMonth = spike["monthName"]
                spikeValue = spike[totalColumn]
                spikeZ = spike["zScore"]
                
                if spikeZ > 0:
                    spikeType = "📈 HIGH SPIKE"
                    
                    if datasetChoice == "📋 Enrollment Data":
                        reasons = [
                            "🏫 School admission season requiring Aadhaar",
                            "💰 Government scheme deadline for DBT",
                            "📢 Mass enrollment camps organized",
                            "📋 Deadline for scholarship applications"
                        ]
                    elif datasetChoice == "📝 Demographic Updates":
                        reasons = [
                            "🏠 Post-migration address updates",
                            "📱 Mobile number portability drives",
                            "🎓 Students updating for exam/college admission",
                            "💼 Job requirement for updated Aadhaar"
                        ]
                    else:
                        reasons = [
                            "🎒 Mandatory 5-year child biometric update",
                            "🏦 Bank account linking deadline",
                            "💰 DBT/Subsidy schemes requiring fresh biometrics",
                            "🔄 Authentication failure corrections"
                        ]
                else:
                    spikeType = "📉 LOW DIP"
                    reasons = [
                        "🌧️ Monsoon/weather affecting center visits",
                        "🎉 Festival/holiday period",
                        "🏛️ Government centers closed",
                        "📅 No active enrollment drives"
                    ]
                
                st.markdown(f"""
                <div class="spikeBox">
                    <h4>{spikeType} in {spikeMonth}</h4>
                    <p><b>Value:</b> {spikeValue:,} (Z-Score: {spikeZ:+.2f})</p>
                    <p><b>Possible Reasons:</b></p>
                    <ul>
                        {"".join([f"<li>{r}</li>" for r in reasons[:3]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant spikes detected. Data shows stable pattern.")
    
    # Growth Rate Chart
    if len(monthlyData) > 1:
        monthlyData["growthRate"] = monthlyData[totalColumn].pct_change() * 100
        
        growthChart = go.Figure(data=[go.Bar(
            x=monthlyData["monthName"][1:],
            y=monthlyData["growthRate"][1:],
            marker_color=["#27AE60" if g > 0 else "#E74C3C" for g in monthlyData["growthRate"][1:]],
            text=[f"{g:+.1f}%" for g in monthlyData["growthRate"][1:]],
            textposition="outside"
        )])
        growthChart.update_layout(title="Month-on-Month Growth Rate", height=350)
        st.plotly_chart(growthChart, use_container_width=True)

# SECTION 3: STATISTICAL ANALYSIS
elif analysisSection == "📉 Statistical Analysis":
    st.subheader("📉 Statistical Analysis - Univariate, Bivariate & Trivariate")
    
    st.markdown("""
    <div class="insightBox">
        <h4>📊 Analysis Types</h4>
        <ul>
            <li><b>Univariate:</b> Single variable analysis - distribution, mean, median, std</li>
            <li><b>Bivariate:</b> Two variable correlation - relationship between variables</li>
            <li><b>Trivariate:</b> Three variable 3D analysis - complex patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Univariate Analysis
    st.markdown("### 📊 Univariate Analysis - Single Variable Statistics")
    
    if datasetChoice == "📋 Enrollment Data":
        univariateData = {
            "Variable": ["Total Enrollments", "Bal Aadhaar (0-5)", "Youth (5-17)", "Adults (18+)"],
            "Sum": [filteredData["totalEnrollments"].sum(), filteredData["balAadhaarCount"].sum(), 
                   filteredData["youthCount"].sum(), filteredData["adultCount"].sum()],
            "Mean": [filteredData["totalEnrollments"].mean(), filteredData["balAadhaarCount"].mean(),
                    filteredData["youthCount"].mean(), filteredData["adultCount"].mean()],
            "Median": [filteredData["totalEnrollments"].median(), filteredData["balAadhaarCount"].median(),
                      filteredData["youthCount"].median(), filteredData["adultCount"].median()],
            "Std Dev": [filteredData["totalEnrollments"].std(), filteredData["balAadhaarCount"].std(),
                       filteredData["youthCount"].std(), filteredData["adultCount"].std()],
            "Min": [filteredData["totalEnrollments"].min(), filteredData["balAadhaarCount"].min(),
                   filteredData["youthCount"].min(), filteredData["adultCount"].min()],
            "Max": [filteredData["totalEnrollments"].max(), filteredData["balAadhaarCount"].max(),
                   filteredData["youthCount"].max(), filteredData["adultCount"].max()]
        }
        numericCols = ["totalEnrollments", "balAadhaarCount", "youthCount", "adultCount"]
    elif datasetChoice == "📝 Demographic Updates":
        univariateData = {
            "Variable": ["Total Updates", "Youth Updates (5-17)", "Adult Updates (17+)"],
            "Sum": [filteredData["totalDemoUpdates"].sum(), filteredData["youthDemoUpdates"].sum(), 
                   filteredData["adultDemoUpdates"].sum()],
            "Mean": [filteredData["totalDemoUpdates"].mean(), filteredData["youthDemoUpdates"].mean(),
                    filteredData["adultDemoUpdates"].mean()],
            "Median": [filteredData["totalDemoUpdates"].median(), filteredData["youthDemoUpdates"].median(),
                      filteredData["adultDemoUpdates"].median()],
            "Std Dev": [filteredData["totalDemoUpdates"].std(), filteredData["youthDemoUpdates"].std(),
                       filteredData["adultDemoUpdates"].std()],
            "Min": [filteredData["totalDemoUpdates"].min(), filteredData["youthDemoUpdates"].min(),
                   filteredData["adultDemoUpdates"].min()],
            "Max": [filteredData["totalDemoUpdates"].max(), filteredData["youthDemoUpdates"].max(),
                   filteredData["adultDemoUpdates"].max()]
        }
        numericCols = ["totalDemoUpdates", "youthDemoUpdates", "adultDemoUpdates"]
    else:
        univariateData = {
            "Variable": ["Total Bio Updates", "Youth Mandatory (5-17)", "Adult Revalidation (17+)"],
            "Sum": [filteredData["totalBioUpdates"].sum(), filteredData["youthBioUpdates"].sum(), 
                   filteredData["adultBioUpdates"].sum()],
            "Mean": [filteredData["totalBioUpdates"].mean(), filteredData["youthBioUpdates"].mean(),
                    filteredData["adultBioUpdates"].mean()],
            "Median": [filteredData["totalBioUpdates"].median(), filteredData["youthBioUpdates"].median(),
                      filteredData["adultBioUpdates"].median()],
            "Std Dev": [filteredData["totalBioUpdates"].std(), filteredData["youthBioUpdates"].std(),
                       filteredData["adultBioUpdates"].std()],
            "Min": [filteredData["totalBioUpdates"].min(), filteredData["youthBioUpdates"].min(),
                   filteredData["adultBioUpdates"].min()],
            "Max": [filteredData["totalBioUpdates"].max(), filteredData["youthBioUpdates"].max(),
                   filteredData["adultBioUpdates"].max()]
        }
        numericCols = ["totalBioUpdates", "youthBioUpdates", "adultBioUpdates"]
    
    univariateDf = pd.DataFrame(univariateData)
    st.dataframe(univariateDf.style.format({"Sum": "{:,.0f}", "Mean": "{:,.2f}", "Median": "{:,.2f}", 
                                            "Std Dev": "{:,.2f}", "Min": "{:,.0f}", "Max": "{:,.0f}"}), 
                 use_container_width=True)
    
    # Distribution Histogram
    uniCol1, uniCol2 = st.columns(2)
    with uniCol1:
        histFig = px.histogram(filteredData, x=totalColumn, nbins=30, 
                               title=f"Distribution of {totalColumn}", 
                               color_discrete_sequence=["#FF9933"])
        histFig.update_layout(height=350)
        st.plotly_chart(histFig, use_container_width=True)
    
    with uniCol2:
        boxFig = px.box(filteredData, y=totalColumn, title=f"Box Plot - {totalColumn}",
                       color_discrete_sequence=["#138808"])
        boxFig.update_layout(height=350)
        st.plotly_chart(boxFig, use_container_width=True)
    
    # Bivariate Analysis
    st.markdown("### 📈 Bivariate Analysis - Two Variable Correlation")
    
    corrMatrix = filteredData[numericCols].corr()
    
    biCol1, biCol2 = st.columns(2)
    with biCol1:
        corrFig = px.imshow(corrMatrix, text_auto=True, aspect="auto",
                           title="Correlation Matrix Heatmap",
                           color_continuous_scale="RdYlGn")
        corrFig.update_layout(height=400)
        st.plotly_chart(corrFig, use_container_width=True)
    
    with biCol2:
        if len(numericCols) >= 2:
            scatterFig = px.scatter(filteredData, x=numericCols[1], y=numericCols[0],
                                   color="district" if len(filteredData["district"].unique()) <= 10 else None,
                                   title=f"Scatter: {numericCols[0]} vs {numericCols[1]}")
            scatterFig.update_layout(height=400)
            st.plotly_chart(scatterFig, use_container_width=True)
    
    # Trivariate Analysis
    st.markdown("### 🔷 Trivariate Analysis - 3D Visualization")
    
    if len(numericCols) >= 3:
        trivariateFig = px.scatter_3d(filteredData, x=numericCols[0], y=numericCols[1], z=numericCols[2],
                                      color="district" if len(filteredData["district"].unique()) <= 15 else None,
                                      title="3D Scatter: Three Variable Analysis",
                                      hover_data=["pincode"])
        trivariateFig.update_layout(height=600)
        st.plotly_chart(trivariateFig, use_container_width=True)
    else:
        st.info("Trivariate analysis requires 3 numeric variables")

# SECTION 4: DISTRICT COMPARISON
elif analysisSection == "🔄 District Comparison":
    st.subheader("🔄 District vs District Comparison")
    
    st.markdown("""
    <div class="insightBox">
        <h4>📊 Compare Two Districts</h4>
        <p>Select two districts to compare their performance side-by-side with detailed insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # District Selection
    allDistricts = sorted(activeData["district"].unique().tolist())
    
    compCol1, compCol2 = st.columns(2)
    with compCol1:
        district1 = st.selectbox("🏢 Select District 1", allDistricts, index=0, key="comp_dist1")
    with compCol2:
        district2 = st.selectbox("🏢 Select District 2", allDistricts, index=min(1, len(allDistricts)-1), key="comp_dist2")
    
    # Get data for both districts
    dist1Data = activeData[activeData["district"] == district1]
    dist2Data = activeData[activeData["district"] == district2]
    
    # Comparison Metrics
    st.markdown("### 📊 Comparison Metrics")
    
    if datasetChoice == "📋 Enrollment Data":
        metrics1 = {
            "Total Enrollments": dist1Data["totalEnrollments"].sum(),
            "Bal Aadhaar (0-5)": dist1Data["balAadhaarCount"].sum(),
            "Youth (5-17)": dist1Data["youthCount"].sum(),
            "Adults (18+)": dist1Data["adultCount"].sum(),
            "Pincodes": dist1Data["pincode"].nunique(),
            "Avg per Pincode": dist1Data["totalEnrollments"].sum() / max(dist1Data["pincode"].nunique(), 1)
        }
        metrics2 = {
            "Total Enrollments": dist2Data["totalEnrollments"].sum(),
            "Bal Aadhaar (0-5)": dist2Data["balAadhaarCount"].sum(),
            "Youth (5-17)": dist2Data["youthCount"].sum(),
            "Adults (18+)": dist2Data["adultCount"].sum(),
            "Pincodes": dist2Data["pincode"].nunique(),
            "Avg per Pincode": dist2Data["totalEnrollments"].sum() / max(dist2Data["pincode"].nunique(), 1)
        }
    elif datasetChoice == "📝 Demographic Updates":
        metrics1 = {
            "Total Updates": dist1Data["totalDemoUpdates"].sum(),
            "Youth Updates (5-17)": dist1Data["youthDemoUpdates"].sum(),
            "Adult Updates (17+)": dist1Data["adultDemoUpdates"].sum(),
            "Pincodes": dist1Data["pincode"].nunique(),
            "Avg per Pincode": dist1Data["totalDemoUpdates"].sum() / max(dist1Data["pincode"].nunique(), 1)
        }
        metrics2 = {
            "Total Updates": dist2Data["totalDemoUpdates"].sum(),
            "Youth Updates (5-17)": dist2Data["youthDemoUpdates"].sum(),
            "Adult Updates (17+)": dist2Data["adultDemoUpdates"].sum(),
            "Pincodes": dist2Data["pincode"].nunique(),
            "Avg per Pincode": dist2Data["totalDemoUpdates"].sum() / max(dist2Data["pincode"].nunique(), 1)
        }
    else:
        metrics1 = {
            "Total Bio Updates": dist1Data["totalBioUpdates"].sum(),
            "Youth Mandatory (5-17)": dist1Data["youthBioUpdates"].sum(),
            "Adult Revalidation (17+)": dist1Data["adultBioUpdates"].sum(),
            "Pincodes": dist1Data["pincode"].nunique(),
            "Avg per Pincode": dist1Data["totalBioUpdates"].sum() / max(dist1Data["pincode"].nunique(), 1)
        }
        metrics2 = {
            "Total Bio Updates": dist2Data["totalBioUpdates"].sum(),
            "Youth Mandatory (5-17)": dist2Data["youthBioUpdates"].sum(),
            "Adult Revalidation (17+)": dist2Data["adultBioUpdates"].sum(),
            "Pincodes": dist2Data["pincode"].nunique(),
            "Avg per Pincode": dist2Data["totalBioUpdates"].sum() / max(dist2Data["pincode"].nunique(), 1)
        }
    
    # Comparison Table
    comparisonDf = pd.DataFrame({
        "Metric": list(metrics1.keys()),
        f"🏢 {district1}": list(metrics1.values()),
        f"🏢 {district2}": list(metrics2.values())
    })
    comparisonDf["Difference"] = comparisonDf[f"🏢 {district1}"] - comparisonDf[f"🏢 {district2}"]
    comparisonDf["Winner"] = comparisonDf.apply(
        lambda x: f"✅ {district1}" if x["Difference"] > 0 else (f"✅ {district2}" if x["Difference"] < 0 else "🟰 Tie"), axis=1
    )
    
    st.dataframe(comparisonDf.style.format({f"🏢 {district1}": "{:,.0f}", f"🏢 {district2}": "{:,.0f}", 
                                            "Difference": "{:+,.0f}"}), use_container_width=True)
    
    # Visual Comparison
    st.markdown("### 📊 Visual Comparison")
    
    vizCol1, vizCol2 = st.columns(2)
    
    with vizCol1:
        compBarFig = go.Figure(data=[
            go.Bar(name=district1, x=list(metrics1.keys())[:4], y=list(metrics1.values())[:4], marker_color="#FF9933"),
            go.Bar(name=district2, x=list(metrics2.keys())[:4], y=list(metrics2.values())[:4], marker_color="#138808")
        ])
        compBarFig.update_layout(barmode="group", title="Side-by-Side Comparison", height=400)
        st.plotly_chart(compBarFig, use_container_width=True)
    
    with vizCol2:
        # Radar Chart Comparison
        categories = list(metrics1.keys())[:4]
        radarFig = go.Figure()
        radarFig.add_trace(go.Scatterpolar(
            r=[metrics1[c] for c in categories],
            theta=categories,
            fill="toself",
            name=district1,
            line_color="#FF9933"
        ))
        radarFig.add_trace(go.Scatterpolar(
            r=[metrics2[c] for c in categories],
            theta=categories,
            fill="toself",
            name=district2,
            line_color="#138808"
        ))
        radarFig.update_layout(title="Radar Comparison", height=400)
        st.plotly_chart(radarFig, use_container_width=True)
    
    # Insights
    st.markdown("### 💡 Comparison Insights")
    
    totalKey = list(metrics1.keys())[0]
    diff = metrics1[totalKey] - metrics2[totalKey]
    diffPercent = abs(diff) / max(metrics2[totalKey], 1) * 100
    
    if diff > 0:
        st.success(f"🏆 **{district1}** leads with **{diff:,.0f}** more records ({diffPercent:.1f}% higher)")
    elif diff < 0:
        st.success(f"🏆 **{district2}** leads with **{abs(diff):,.0f}** more records ({diffPercent:.1f}% higher)")
    else:
        st.info("🟰 Both districts have equal performance")
    
    # Density Comparison
    density1 = list(metrics1.values())[-1]
    density2 = list(metrics2.values())[-1]
    
    if density1 > density2:
        st.info(f"📍 **{district1}** has higher density ({density1:,.0f} per pincode vs {density2:,.0f})")
    else:
        st.info(f"📍 **{district2}** has higher density ({density2:,.0f} per pincode vs {density1:,.0f})")

# SECTION 5: PINCODE ANALYSIS
elif analysisSection == "📍 Pincode Analysis":
    st.subheader("📍 Pincode-wise Detailed Analysis")
    
    if datasetChoice == "📋 Enrollment Data":
        st.markdown("**Pincode-wise New Aadhaar Enrollments** - Where are people getting new Aadhaar cards?")
        pincodeAgg = filteredData.groupby(["pincode", "district"]).agg({
            "totalEnrollments": "sum",
            "balAadhaarCount": "sum",
            "youthCount": "sum",
            "adultCount": "sum"
        }).reset_index()
        pincodeAgg.columns = ["Pincode", "District", "Total Enrollments", "Bal Aadhaar (0-5)", "Youth (5-17)", "Adults (18+)"]
        
    elif datasetChoice == "📝 Demographic Updates":
        st.markdown("**Pincode-wise Address/Mobile/Name Updates** - Where are people updating their details?")
        pincodeAgg = filteredData.groupby(["pincode", "district"]).agg({
            "totalDemoUpdates": "sum",
            "youthDemoUpdates": "sum",
            "adultDemoUpdates": "sum"
        }).reset_index()
        pincodeAgg.columns = ["Pincode", "District", "Total Updates", "Youth Updates (5-17)", "Adult Updates (17+)"]
        
    else:
        st.markdown("**Pincode-wise Fingerprint/Iris/Face Updates** - Where are biometric updates happening?")
        pincodeAgg = filteredData.groupby(["pincode", "district"]).agg({
            "totalBioUpdates": "sum",
            "youthBioUpdates": "sum",
            "adultBioUpdates": "sum"
        }).reset_index()
        pincodeAgg.columns = ["Pincode", "District", "Total Bio Updates", "Youth Mandatory (5-17)", "Adult Revalidation (17+)"]
    
    valueColumn = pincodeAgg.columns[2]
    pincodeAgg = pincodeAgg.sort_values(valueColumn, ascending=False)
    
    # Top Pincodes Chart
    top15 = pincodeAgg.head(15)
    
    pinCol1, pinCol2 = st.columns(2)
    
    with pinCol1:
        topChart = go.Figure(data=[go.Bar(
            x=top15[valueColumn],
            y=[f"{p} - {d[:10]}" for p, d in zip(top15["Pincode"], top15["District"])],
            orientation="h",
            marker=dict(color=top15[valueColumn], colorscale=colorScheme),
            text=[f"{v:,}" for v in top15[valueColumn]],
            textposition="outside"
        )])
        topChart.update_layout(
            title="Top 15 Pincodes",
            yaxis={"categoryorder": "total ascending"},
            height=500
        )
        st.plotly_chart(topChart, use_container_width=True)
    
    with pinCol2:
        # District-wise pincode count
        districtPincodes = pincodeAgg.groupby("District").agg({
            valueColumn: "sum",
            "Pincode": "count"
        }).rename(columns={"Pincode": "Pincode Count"})
        districtPincodes["Avg per Pincode"] = (districtPincodes[valueColumn] / districtPincodes["Pincode Count"]).round(0)
        districtPincodes = districtPincodes.sort_values(valueColumn, ascending=False).head(15)
        
        densityChart = go.Figure(data=[go.Bar(
            x=districtPincodes.index,
            y=districtPincodes["Avg per Pincode"],
            marker=dict(color=districtPincodes["Avg per Pincode"], colorscale=colorScheme),
            text=[f"{v:,.0f}" for v in districtPincodes["Avg per Pincode"]],
            textposition="outside"
        )])
        densityChart.update_layout(
            title="Average per Pincode by District",
            xaxis_tickangle=-45,
            height=500
        )
        st.plotly_chart(densityChart, use_container_width=True)
    
    # Detailed Pincode Table
    st.subheader("📋 Complete Pincode Data Table")
    st.dataframe(
        pincodeAgg.head(50).style.format({v: "{:,.0f}" for v in pincodeAgg.columns[2:]}),
        use_container_width=True
    )
    
    # Download option
    csvData = pincodeAgg.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Pincode Data",
        data=csvData,
        file_name=f"pincode_analysis_{datasetChoice.split()[1].lower()}.csv",
        mime="text/csv"
    )

# SECTION 6: AGE ANALYSIS
elif analysisSection == "👥 Age Analysis":
    if datasetChoice == "📋 Enrollment Data":
        st.subheader("👥 Age-wise New Enrollment Analysis")
        st.markdown("""
        **Who is getting new Aadhaar cards?**
        - 👶 **Bal Aadhaar (0-5)**: New birth registrations, child identity
        - 🎒 **Youth (5-17)**: School admission requirements
        - 👨 **Adults (18+)**: New arrivals, missed earlier enrollment
        """)
        
        ageLabels = ["Bal Aadhaar (0-5 years)", "Youth (5-17 years)", "Adults (18+ years)"]
        ageValues = [balAadhaar, youthValue, adultValue]
        ageColors = ["#667eea", "#764ba2", "#f093fb"]
        
        youthCol = "youthCount"
        adultCol = "adultCount"
        extraCol = "balAadhaarCount"
        
    elif datasetChoice == "📝 Demographic Updates":
        st.subheader("👥 Age-wise Address/Mobile/Name Update Analysis")
        st.markdown("""
        **Who is updating their Aadhaar details?**
        - 🎒 **Youth (5-17)**: Parents updating on behalf, school record corrections
        - 👨 **Adults (17+)**: 
            - 🏠 Address change due to migration, job relocation
            - 📱 Mobile number updates
            - ✏️ Name spelling corrections
        """)
        
        ageLabels = ["Youth Updates (5-17)", "Adult Updates (17+)"]
        ageValues = [youthValue, adultValue]
        ageColors = ["#E74C3C", "#2C3E50"]
        
        youthCol = "youthDemoUpdates"
        adultCol = "adultDemoUpdates"
        extraCol = None
        
    else:
        st.subheader("👥 Age-wise Biometric Update Analysis")
        st.markdown("""
        **Who is updating their biometrics?**
        - 🎒 **Youth (5-17) - MANDATORY**: 
            - Children must update biometrics at age 5, 10, 15
            - Features change as they grow
        - 👨 **Adults (17+) - REVALIDATION**:
            - 👆 Fingerprint wear (manual laborers, farmers)
            - 👁️ Iris changes due to age/medical conditions
            - 😊 Face photo update for aging
            - 🔄 Authentication failure corrections
        """)
        
        ageLabels = ["Youth Mandatory (5-17)", "Adult Revalidation (17+)"]
        ageValues = [youthValue, adultValue]
        ageColors = ["#27AE60", "#1ABC9C"]
        
        youthCol = "youthBioUpdates"
        adultCol = "adultBioUpdates"
        extraCol = None
    
    ageCol1, ageCol2 = st.columns(2)
    
    with ageCol1:
        gaugeChart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=youthValue / totalValue * 100,
            title={"text": "Youth Share (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": primaryColor},
                "steps": [
                    {"range": [0, 30], "color": "#FFEEEE"},
                    {"range": [30, 60], "color": "#FFFFEE"},
                    {"range": [60, 100], "color": "#EEFFEE"}
                ]
            }
        ))
        gaugeChart.update_layout(height=300)
        st.plotly_chart(gaugeChart, use_container_width=True)
    
    with ageCol2:
        ageBarChart = go.Figure(data=[go.Bar(
            x=ageLabels,
            y=ageValues,
            marker_color=ageColors,
            text=[f"{v:,}" for v in ageValues],
            textposition="outside"
        )])
        ageBarChart.update_layout(title="Age Group Distribution", height=300)
        st.plotly_chart(ageBarChart, use_container_width=True)
    
    # District-wise Age Breakdown
    st.subheader("District-wise Age Breakdown")
    
    if extraCol:
        ageByDistrict = filteredData.groupby("district").agg({
            extraCol: "sum", youthCol: "sum", adultCol: "sum"
        }).nlargest(15, youthCol)
        
        ageDistChart = go.Figure()
        ageDistChart.add_trace(go.Bar(name="0-5", x=ageByDistrict.index, y=ageByDistrict[extraCol], marker_color="#667eea"))
        ageDistChart.add_trace(go.Bar(name="5-17", x=ageByDistrict.index, y=ageByDistrict[youthCol], marker_color="#764ba2"))
        ageDistChart.add_trace(go.Bar(name="18+", x=ageByDistrict.index, y=ageByDistrict[adultCol], marker_color="#f093fb"))
    else:
        ageByDistrict = filteredData.groupby("district").agg({youthCol: "sum", adultCol: "sum"}).nlargest(15, youthCol)
        
        ageDistChart = go.Figure()
        ageDistChart.add_trace(go.Bar(name="Youth", x=ageByDistrict.index, y=ageByDistrict[youthCol], marker_color=primaryColor))
        ageDistChart.add_trace(go.Bar(name="Adult", x=ageByDistrict.index, y=ageByDistrict[adultCol], marker_color="#2C3E50"))
    
    ageDistChart.update_layout(barmode="group", height=400, xaxis_tickangle=-45)
    st.plotly_chart(ageDistChart, use_container_width=True)

# SECTION 7: GAP ANALYSIS
elif analysisSection == "⚡ Gap Analysis":
    st.subheader("⚡ Service Gap & Stress Analysis")
    
    pincodeStats = filteredData.groupby(["pincode", "district"])[totalColumn].sum().reset_index()
    
    highThreshold = pincodeStats[totalColumn].quantile(0.75)
    lowThreshold = pincodeStats[totalColumn].quantile(0.25)
    
    highLoadPincodes = pincodeStats[pincodeStats[totalColumn] > highThreshold]
    lowServicePincodes = pincodeStats[pincodeStats[totalColumn] < lowThreshold]
    
    gapCol1, gapCol2 = st.columns(2)
    
    with gapCol1:
        st.error(f"⚠️ **{len(highLoadPincodes)} High-Load Pincodes**")
        st.markdown(f"Pincodes with > {highThreshold:,.0f} records need additional resources")
        
        highByDistrict = highLoadPincodes.groupby("district").size().nlargest(10)
        if len(highByDistrict) > 0:
            highChart = px.bar(x=highByDistrict.index, y=highByDistrict.values,
                              title="Districts with High-Load Pincodes",
                              color=highByDistrict.values, color_continuous_scale="Reds")
            st.plotly_chart(highChart, use_container_width=True)
    
    with gapCol2:
        st.info(f"📍 **{len(lowServicePincodes)} Underserved Pincodes**")
        st.markdown(f"Pincodes with < {lowThreshold:,.0f} records need outreach")
        
        lowByDistrict = lowServicePincodes.groupby("district").size().nlargest(10)
        if len(lowByDistrict) > 0:
            lowChart = px.bar(x=lowByDistrict.index, y=lowByDistrict.values,
                             title="Districts with Underserved Pincodes",
                             color=lowByDistrict.values, color_continuous_scale="Blues")
            st.plotly_chart(lowChart, use_container_width=True)
    
    # Recommendations based on dataset
    st.markdown("### 🎯 Policy Recommendations")
    
    if datasetChoice == "📋 Enrollment Data":
        st.markdown("""
        **For Enrollment Gaps:**
        - 🚐 Deploy mobile enrollment vans to low-coverage pincodes
        - 🏫 Partner with schools for Bal Aadhaar camps
        - 📢 Awareness campaigns in tribal/remote areas
        """)
    elif datasetChoice == "📝 Demographic Updates":
        st.markdown("""
        **For Demographic Update Gaps:**
        - 📱 Enable online self-service for mobile/address updates
        - 🏛️ Extend center timings in high-demand areas
        - 👨‍👩‍👧 Family update camps for communities
        """)
    else:
        st.markdown("""
        **For Biometric Update Gaps:**
        - 🔧 Upgrade biometric devices in high-load centers
        - 🏫 School-based biometric camps for children
        - 👴 Special drives for elderly with fingerprint issues
        """)

# SECTION 8: ML INSIGHTS
elif analysisSection == "🤖 ML Insights":
    st.subheader("🤖 Machine Learning Analysis")
    
    pincodeML = filteredData.groupby(["pincode", "district"])[totalColumn].sum().reset_index()
    
    if len(pincodeML) > 20:
        mlFeatures = pincodeML[[totalColumn]].copy()
        
        scaler = StandardScaler()
        scaledFeatures = scaler.fit_transform(mlFeatures)
        
        mlCol1, mlCol2 = st.columns(2)
        
        with mlCol1:
            clusterCount = st.slider("Number of Clusters", 2, 6, 3)
            kmeansModel = KMeans(n_clusters=clusterCount, random_state=42, n_init=10)
            pincodeML["cluster"] = kmeansModel.fit_predict(scaledFeatures)
            
            clusterChart = px.scatter(
                pincodeML, 
                x=pincodeML.index, 
                y=totalColumn,
                color="cluster",
                hover_data=["pincode", "district"],
                title="K-Means Clustering",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(clusterChart, use_container_width=True)
        
        with mlCol2:
            clusterStats = pincodeML.groupby("cluster").agg({totalColumn: ["mean", "count"]}).round(0)
            clusterStats.columns = ["Average", "Count"]
            
            st.markdown("### Cluster Statistics")
            st.dataframe(clusterStats, use_container_width=True)
        
        # Anomaly Detection
        st.subheader("🔍 Anomaly Detection")
        
        isoForest = IsolationForest(contamination=0.05, random_state=42)
        pincodeML["anomaly"] = isoForest.fit_predict(scaledFeatures)
        anomalies = pincodeML[pincodeML["anomaly"] == -1]
        
        st.warning(f"**{len(anomalies)} Anomalous Pincodes Detected** - Require Investigation")
        st.dataframe(anomalies[["pincode", "district", totalColumn]].nlargest(10, totalColumn), use_container_width=True)
    else:
        st.info("Select 'All Districts' for complete ML analysis")

# Footer with Government Branding
st.markdown('<div class="tricolor-bar"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="govt-footer">
    <p style='font-size: 1.2rem; font-weight: bold;'>🆔 UIDAI Hackathon 2026</p>
    <p>भारतीय विशिष्ट पहचान प्राधिकरण | Unique Identification Authority of India</p>
    <p style='color: #FF9933;'>📍 State Analysis: ODISHA | राज्य विश्लेषण: ओडिशा</p>
    <p style='font-size: 0.8rem; opacity: 0.8; margin-top: 1rem;'>Built with ❤️ for Digital India | डिजिटल भारत के लिए</p>
</div>
""", unsafe_allow_html=True)

