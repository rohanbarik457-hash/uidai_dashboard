"""
UIDAI HACKATHON 2026 - INTEGRATED DISTRICT-LEVEL ANALYSIS
Combines: Enrollment + Demographic + Biometric Datasets
State: ODISHA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
data_dir = r"c:\Users\rohan\Aadharcard"

print("\n UIDAI HACKATHON 2026 - INTEGRATED ODISHA ANALYSIS")
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================
print("STEP 1: DATA LOADING & CLEANING")

# Load all datasets
enroll_df = pd.read_csv(f"{data_dir}/odisha_enrolment_clean.csv")
demo_df = pd.read_csv(f"{data_dir}/odisha_demographic_clean.csv")
bio_df = pd.read_csv(f"{data_dir}/odisha_biometric_clean.csv")

print(f"  Enrollment: {len(enroll_df):,} records")
print(f"  Demographic: {len(demo_df):,} records")
print(f"  Biometric: {len(bio_df):,} records")

# Convert dates and extract features
for df in [enroll_df, demo_df, bio_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()

# Remove duplicates
enroll_df = enroll_df.drop_duplicates()
demo_df = demo_df.drop_duplicates()
bio_df = bio_df.drop_duplicates()

# Fill nulls with 0
enroll_df = enroll_df.fillna(0)
demo_df = demo_df.fillna(0)
bio_df = bio_df.fillna(0)

print(f"  Districts: {enroll_df['district'].nunique()}")
print(f"  Date range: {enroll_df['date'].min().strftime('%d-%m-%Y')} to {enroll_df['date'].max().strftime('%d-%m-%Y')}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\nSTEP 2: FEATURE ENGINEERING")

# Enrollment totals
enroll_df['Total_Enrollments'] = enroll_df['age_0_5'] + enroll_df['age_5_17'] + enroll_df['age_18_greater']

# Demographic totals
demo_df['Total_Demo_Updates'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']

# Biometric totals (Note: modality breakdown not available, using age-wise)
bio_df['Total_Biometric'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']

print("  Created: Total_Enrollments, Total_Demo_Updates, Total_Biometric")

# ============================================================
# STEP 3: RATIO & NORMALIZATION ANALYSIS
# ============================================================
print("\nSTEP 3: RATIO & NORMALIZATION ANALYSIS")

# Aggregate by district
enroll_dist = enroll_df.groupby('district')['Total_Enrollments'].sum()
demo_dist = demo_df.groupby('district')['Total_Demo_Updates'].sum()
bio_dist = bio_df.groupby('district')['Total_Biometric'].sum()

# Create merged district summary
district_summary = pd.DataFrame({
    'Enrollments': enroll_dist,
    'Demo_Updates': demo_dist,
    'Bio_Updates': bio_dist
}).fillna(0)

# Calculate rates
district_summary['Demo_Rate'] = (district_summary['Demo_Updates'] / district_summary['Enrollments']).round(2)
district_summary['Bio_Rate'] = (district_summary['Bio_Updates'] / district_summary['Enrollments']).round(2)
district_summary = district_summary.sort_values('Enrollments', ascending=False)

print("\n  DISTRICT-LEVEL RATES (Demo & Bio Updates per Enrollment):")
for district in district_summary.head(10).index:
    row = district_summary.loc[district]
    print(f"    {district}: Demo Rate {row['Demo_Rate']:.2f} | Bio Rate {row['Bio_Rate']:.2f}")

# Interpretation
high_demo = district_summary[district_summary['Demo_Rate'] > district_summary['Demo_Rate'].median() * 1.5].index.tolist()
high_bio = district_summary[district_summary['Bio_Rate'] > district_summary['Bio_Rate'].median() * 1.5].index.tolist()

print(f"\n  High Demo Update Rate Districts: {', '.join(high_demo[:5])}")
print(f"  High Bio Update Rate Districts: {', '.join(high_bio[:5])}")

# ============================================================
# STEP 4: AGE COHORT ANALYSIS
# ============================================================
print("\nSTEP 4: AGE COHORT (LIFE-CYCLE) ANALYSIS")

# Enrollment by age
enroll_age = {
    'Age 0-5 (Early Inclusion)': enroll_df['age_0_5'].sum(),
    'Age 5-17 (Education)': enroll_df['age_5_17'].sum(),
    'Age 18+ (Adult)': enroll_df['age_18_greater'].sum()
}
total_enroll = sum(enroll_age.values())

print("\n  ENROLLMENT BY AGE COHORT:")
for cohort, count in enroll_age.items():
    print(f"    {cohort}: {count:,} ({count/total_enroll*100:.1f}%)")

# Demo updates by age
demo_youth = demo_df['demo_age_5_17'].sum()
demo_adult = demo_df['demo_age_17_'].sum()
total_demo = demo_youth + demo_adult

print("\n  DEMOGRAPHIC UPDATES BY AGE:")
print(f"    Youth (5-17): {demo_youth:,} ({demo_youth/total_demo*100:.1f}%)")
print(f"    Adults (17+): {demo_adult:,} ({demo_adult/total_demo*100:.1f}%)")

# Bio updates by age
bio_youth = bio_df['bio_age_5_17'].sum()
bio_adult = bio_df['bio_age_17_'].sum()
total_bio = bio_youth + bio_adult

print("\n  BIOMETRIC UPDATES BY AGE:")
print(f"    Youth (5-17): {bio_youth:,} ({bio_youth/total_bio*100:.1f}%)")
print(f"    Adults (17+): {bio_adult:,} ({bio_adult/total_bio*100:.1f}%)")

# ============================================================
# STEP 5: TIME-SERIES ANALYSIS
# ============================================================
print("\nSTEP 5: TIME-SERIES ANALYSIS")

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

enroll_monthly = enroll_df.groupby('month_name')['Total_Enrollments'].sum()
demo_monthly = demo_df.groupby('month_name')['Total_Demo_Updates'].sum()
bio_monthly = bio_df.groupby('month_name')['Total_Biometric'].sum()

# Reindex by month order
enroll_monthly = enroll_monthly.reindex([m for m in month_order if m in enroll_monthly.index])
demo_monthly = demo_monthly.reindex([m for m in month_order if m in demo_monthly.index])
bio_monthly = bio_monthly.reindex([m for m in month_order if m in bio_monthly.index])

print("\n  MONTHLY TRENDS:")
print(f"  {'Month':<12} {'Enrollment':>12} {'Demo':>12} {'Bio':>12}")
for month in enroll_monthly.index:
    e = enroll_monthly.get(month, 0)
    d = demo_monthly.get(month, 0)
    b = bio_monthly.get(month, 0)
    print(f"  {month:<12} {e:>12,.0f} {d:>12,.0f} {b:>12,.0f}")

# Peak months
print(f"\n  Peak Enrollment Month: {enroll_monthly.idxmax()}")
print(f"  Peak Demo Update Month: {demo_monthly.idxmax()}")
print(f"  Peak Bio Update Month: {bio_monthly.idxmax()}")

# ============================================================
# STEP 6: PINCODE ANALYSIS
# ============================================================
print("\nSTEP 6: GEOGRAPHIC (PINCODE) ANALYSIS")

# Aggregate by pincode
enroll_pin = enroll_df.groupby(['pincode', 'district'])['Total_Enrollments'].sum().reset_index()
demo_pin = demo_df.groupby(['pincode', 'district'])['Total_Demo_Updates'].sum().reset_index()
bio_pin = bio_df.groupby(['pincode', 'district'])['Total_Biometric'].sum().reset_index()

# Merge pincode data
pin_merged = enroll_pin.merge(demo_pin, on=['pincode', 'district'], how='outer')
pin_merged = pin_merged.merge(bio_pin, on=['pincode', 'district'], how='outer').fillna(0)

# Calculate rates
pin_merged['Demo_Rate'] = np.where(pin_merged['Total_Enrollments'] > 0,
                                    pin_merged['Total_Demo_Updates'] / pin_merged['Total_Enrollments'], 0)
pin_merged['Bio_Rate'] = np.where(pin_merged['Total_Enrollments'] > 0,
                                   pin_merged['Total_Biometric'] / pin_merged['Total_Enrollments'], 0)

print(f"\n  Total Pincodes Analyzed: {len(pin_merged)}")

# High-load pincodes
high_load = pin_merged.nlargest(10, 'Total_Biometric')
print("\n  TOP 10 HIGH-LOAD PINCODES (by Biometric Updates):")
for _, row in high_load.iterrows():
    print(f"    {int(row['pincode'])} ({row['district'][:12]}): Bio={row['Total_Biometric']:,.0f}")

# Low-service pincodes
low_service = pin_merged[pin_merged['Total_Enrollments'] < pin_merged['Total_Enrollments'].quantile(0.25)]
print(f"\n  Low-Service (Exclusion) Pincodes: {len(low_service)}")

# ============================================================
# STEP 7: ANOMALY DETECTION
# ============================================================
print("\nSTEP 7: ANOMALY DETECTION")

# Z-score for biometric updates
pin_merged['Bio_ZScore'] = (pin_merged['Total_Biometric'] - pin_merged['Total_Biometric'].mean()) / pin_merged['Total_Biometric'].std()

# Flag anomalies (|Z| > 2)
anomalies = pin_merged[abs(pin_merged['Bio_ZScore']) > 2]
print(f"  Pincodes with abnormal biometric activity (|Z| > 2): {len(anomalies)}")

if len(anomalies) > 0:
    print("\n  TOP ANOMALOUS PINCODES:")
    for _, row in anomalies.nlargest(5, 'Bio_ZScore').iterrows():
        print(f"    {int(row['pincode'])} ({row['district'][:12]}): Z={row['Bio_ZScore']:.2f}")

# ============================================================
# STEP 8: STABILITY ANALYSIS
# ============================================================
print("\nSTEP 8: STABILITY & VOLATILITY ANALYSIS")

# Calculate CV by district
enroll_cv = enroll_df.groupby('district')['Total_Enrollments'].agg(['mean', 'std'])
enroll_cv['CV'] = (enroll_cv['std'] / enroll_cv['mean'] * 100).round(1)

bio_cv = bio_df.groupby('district')['Total_Biometric'].agg(['mean', 'std'])
bio_cv['CV'] = (bio_cv['std'] / bio_cv['mean'] * 100).round(1)

print("\n  ENROLLMENT VOLATILITY (Top 5 Volatile Districts):")
for district in enroll_cv.nlargest(5, 'CV').index:
    print(f"    {district}: CV = {enroll_cv.loc[district, 'CV']:.1f}%")

print("\n  BIOMETRIC VOLATILITY (Top 5 Volatile Districts):")
for district in bio_cv.nlargest(5, 'CV').index:
    print(f"    {district}: CV = {bio_cv.loc[district, 'CV']:.1f}%")

# ============================================================
# STEP 9: PINCODE SEGMENTATION
# ============================================================
print("\nSTEP 9: PINCODE SEGMENTATION")

# Define thresholds
enroll_median = pin_merged['Total_Enrollments'].median()
bio_median = pin_merged['Total_Biometric'].median()

# Segment pincodes
def segment_pincode(row):
    high_enroll = row['Total_Enrollments'] > enroll_median
    high_bio = row['Total_Biometric'] > bio_median
    
    if high_enroll and not high_bio:
        return 'Stable Zone'
    elif high_bio:
        return 'Stress Zone'
    else:
        return 'Exclusion Zone'

pin_merged['Segment'] = pin_merged.apply(segment_pincode, axis=1)

segment_counts = pin_merged['Segment'].value_counts()
print("\n  PINCODE SEGMENTATION:")
for seg, count in segment_counts.items():
    pct = count / len(pin_merged) * 100
    print(f"    {seg}: {count} pincodes ({pct:.1f}%)")

# Districts with most exclusion zones
exclusion = pin_merged[pin_merged['Segment'] == 'Exclusion Zone']
exclusion_by_dist = exclusion.groupby('district').size().sort_values(ascending=False)
print("\n  DISTRICTS WITH MOST EXCLUSION ZONES:")
for district, count in exclusion_by_dist.head(5).items():
    print(f"    {district}: {count} pincodes")

# ============================================================
# STEP 10: QUALITY STRESS INDEX
# ============================================================
print("\nSTEP 10: QUALITY STRESS INDEX")

# Normalize rates to 0-1
pin_merged['Demo_Rate_Norm'] = pin_merged['Demo_Rate'] / pin_merged['Demo_Rate'].max()
pin_merged['Bio_Rate_Norm'] = pin_merged['Bio_Rate'] / pin_merged['Bio_Rate'].max()

# Enrollment volatility by pincode (using CV proxy)
pin_merged['Enroll_Vol'] = abs(pin_merged['Total_Enrollments'] - enroll_median) / enroll_median
pin_merged['Enroll_Vol_Norm'] = pin_merged['Enroll_Vol'] / pin_merged['Enroll_Vol'].max()

# Quality Stress Index
pin_merged['Stress_Index'] = (
    pin_merged['Bio_Rate_Norm'] * 0.5 +
    pin_merged['Demo_Rate_Norm'] * 0.3 +
    pin_merged['Enroll_Vol_Norm'] * 0.2
).round(3)

# Top stressed pincodes
print("\n  TOP 10 STRESSED PINCODES (Quality Stress Index):")
stressed = pin_merged.nlargest(10, 'Stress_Index')
for _, row in stressed.iterrows():
    print(f"    {int(row['pincode'])} ({row['district'][:12]}): Index = {row['Stress_Index']:.3f}")

# District stress ranking
district_stress = pin_merged.groupby('district')['Stress_Index'].mean().sort_values(ascending=False)
print("\n  TOP 5 STRESSED DISTRICTS:")
for district, stress in district_stress.head(5).items():
    print(f"    {district}: Avg Stress = {stress:.3f}")

# ============================================================
# STEP 11: VISUALIZATIONS
# ============================================================
print("\nSTEP 11: CREATING VISUALIZATIONS")

fig = plt.figure(figsize=(24, 28))
fig.suptitle('UIDAI HACKATHON 2026 - INTEGRATED ODISHA ANALYSIS', fontsize=18, fontweight='bold', y=0.98)

# Chart 1: Monthly Trends (All 3 datasets) - Use common months
ax1 = fig.add_subplot(4, 2, 1)
common_months = [m for m in enroll_monthly.index if m in demo_monthly.index and m in bio_monthly.index]
x = range(len(common_months))
ax1.plot(x, [enroll_monthly[m] for m in common_months], marker='o', label='Enrollment', color='#2E86AB', linewidth=2)
ax1.plot(x, [demo_monthly[m] for m in common_months], marker='s', label='Demo Updates', color='#E74C3C', linewidth=2)
ax1.plot(x, [bio_monthly[m] for m in common_months], marker='^', label='Bio Updates', color='#27AE60', linewidth=2)
ax1.set_xticks(x)
ax1.set_xticklabels(common_months, rotation=45, ha='right')
ax1.set_title('Monthly Trends: Enrollment vs Demo vs Bio', fontweight='bold')
ax1.legend()
ax1.set_ylabel('Count')

# Chart 2: Age Cohort Comparison
ax2 = fig.add_subplot(4, 2, 2)
cohorts = ['0-5', '5-17', '17+']
enroll_ages = [enroll_df['age_0_5'].sum(), enroll_df['age_5_17'].sum(), enroll_df['age_18_greater'].sum()]
demo_ages = [0, demo_youth, demo_adult]
bio_ages = [0, bio_youth, bio_adult]
x = np.arange(len(cohorts))
width = 0.25
ax2.bar(x - width, enroll_ages, width, label='Enrollment', color='#2E86AB')
ax2.bar(x, demo_ages, width, label='Demo', color='#E74C3C')
ax2.bar(x + width, bio_ages, width, label='Bio', color='#27AE60')
ax2.set_xticks(x)
ax2.set_xticklabels(cohorts)
ax2.set_title('Age Cohort Comparison', fontweight='bold')
ax2.legend()

# Chart 3: District Rankings
ax3 = fig.add_subplot(4, 2, 3)
top10 = district_summary.head(10)
x = range(len(top10))
ax3.barh(x, top10['Enrollments'], label='Enrollment', color='#2E86AB', alpha=0.8)
ax3.set_yticks(x)
ax3.set_yticklabels(top10.index)
ax3.set_title('Top 10 Districts by Enrollment', fontweight='bold')
ax3.invert_yaxis()

# Chart 4: Update Rates
ax4 = fig.add_subplot(4, 2, 4)
top10_rates = district_summary.head(10)
x = np.arange(len(top10_rates))
width = 0.35
ax4.bar(x - width/2, top10_rates['Demo_Rate'], width, label='Demo Rate', color='#E74C3C')
ax4.bar(x + width/2, top10_rates['Bio_Rate'], width, label='Bio Rate', color='#27AE60')
ax4.set_xticks(x)
ax4.set_xticklabels(top10_rates.index, rotation=45, ha='right')
ax4.set_title('Update Rates by District', fontweight='bold')
ax4.legend()

# Chart 5: Segment Distribution
ax5 = fig.add_subplot(4, 2, 5)
colors = {'Stable Zone': '#27AE60', 'Stress Zone': '#E74C3C', 'Exclusion Zone': '#95A5A6'}
ax5.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
        colors=[colors[s] for s in segment_counts.index], startangle=90)
ax5.set_title('Pincode Segmentation', fontweight='bold')

# Chart 6: Stress Index Heatmap
ax6 = fig.add_subplot(4, 2, 6)
stress_by_dist = pin_merged.groupby('district')['Stress_Index'].mean().sort_values(ascending=False).head(15)
ax6.barh(range(len(stress_by_dist)), stress_by_dist.values, color=plt.cm.Reds(stress_by_dist.values / stress_by_dist.max()))
ax6.set_yticks(range(len(stress_by_dist)))
ax6.set_yticklabels(stress_by_dist.index)
ax6.set_title('Quality Stress Index by District', fontweight='bold')
ax6.set_xlabel('Stress Index')
ax6.invert_yaxis()

# Chart 7: Exclusion Zone Map
ax7 = fig.add_subplot(4, 2, 7)
ax7.barh(range(len(exclusion_by_dist.head(10))), exclusion_by_dist.head(10).values, color='#95A5A6')
ax7.set_yticks(range(len(exclusion_by_dist.head(10))))
ax7.set_yticklabels(exclusion_by_dist.head(10).index)
ax7.set_title('Exclusion Zones by District', fontweight='bold')
ax7.set_xlabel('Number of Pincodes')
ax7.invert_yaxis()

# Chart 8: Volatility Comparison
ax8 = fig.add_subplot(4, 2, 8)
vol_top10 = enroll_cv.nlargest(10, 'CV')
ax8.barh(range(len(vol_top10)), vol_top10['CV'].values, color='#F39C12')
ax8.set_yticks(range(len(vol_top10)))
ax8.set_yticklabels(vol_top10.index)
ax8.set_title('Enrollment Volatility (CV%) by District', fontweight='bold')
ax8.set_xlabel('Coefficient of Variation (%)')
ax8.invert_yaxis()

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(f"{data_dir}/odisha_integrated_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Charts saved: odisha_integrated_analysis.png")

# ============================================================
# STEP 12: EXECUTIVE SUMMARY
# ============================================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"""
  STATE: Odisha
  ANALYSIS PERIOD: {enroll_df['date'].min().strftime('%d-%m-%Y')} to {enroll_df['date'].max().strftime('%d-%m-%Y')}
  
  COVERAGE (Enrollment):
    Total Enrollments: {total_enroll:,}
    Bal Aadhaar (0-5): {enroll_age['Age 0-5 (Early Inclusion)']:,} ({enroll_age['Age 0-5 (Early Inclusion)']/total_enroll*100:.1f}%)
    Youth (5-17): {enroll_age['Age 5-17 (Education)']:,} ({enroll_age['Age 5-17 (Education)']/total_enroll*100:.1f}%)
    Adults (18+): {enroll_age['Age 18+ (Adult)']:,} ({enroll_age['Age 18+ (Adult)']/total_enroll*100:.1f}%)
  
  DATA ACCURACY (Demographic Updates):
    Total Updates: {total_demo:,}
    Avg Demo Rate: {district_summary['Demo_Rate'].mean():.2f} updates per enrollment
  
  AUTHENTICATION RELIABILITY (Biometric Updates):
    Total Updates: {total_bio:,}
    Avg Bio Rate: {district_summary['Bio_Rate'].mean():.2f} updates per enrollment
  
  GEOGRAPHIC ANALYSIS:
    Districts: {len(district_summary)}
    Pincodes: {len(pin_merged)}
    Exclusion Zone Pincodes: {len(exclusion)} ({len(exclusion)/len(pin_merged)*100:.1f}%)
    Stress Zone Pincodes: {segment_counts.get('Stress Zone', 0)}
    
  QUALITY INDICATORS:
    Anomalous Pincodes: {len(anomalies)}
    Most Stressed District: {district_stress.idxmax()}
""")

# ============================================================
# STEP 13: POLICY RECOMMENDATIONS
# ============================================================
print("="*60)
print("POLICY RECOMMENDATIONS")
print("="*60)
print(f"""
  1. COVERAGE GAPS:
     - Deploy mobile enrollment in {len(exclusion)} exclusion zone pincodes
     - Focus districts: {', '.join(exclusion_by_dist.head(3).index)}
  
  2. DATA ACCURACY:
     - School integration for youth demographic updates
     - Target districts with high demo rate: {', '.join(high_demo[:3])}
  
  3. AUTHENTICATION RELIABILITY:
     - Device upgrades in high-stress pincodes
     - Calibration program for: {', '.join(high_bio[:3])}
  
  4. OPERATIONAL STABILITY:
     - Reduce volatility in: {', '.join(enroll_cv.nlargest(3, 'CV').index)}
     - Establish permanent centers vs. campaign-driven drives
  
  5. INCLUSION PRIORITY:
     - Top exclusion districts: {', '.join(exclusion_by_dist.head(3).index)}
     - Mobile Aadhaar vans for remote pincodes
""")

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("="*60)
print("OUTPUT FILES")
print("="*60)

district_summary.to_csv(f"{data_dir}/integrated_district_summary.csv")
pin_merged.to_csv(f"{data_dir}/integrated_pincode_analysis.csv", index=False)
exclusion.to_csv(f"{data_dir}/exclusion_zone_pincodes.csv", index=False)
stressed.to_csv(f"{data_dir}/high_stress_pincodes.csv", index=False)

print(f"""
  - odisha_integrated_analysis.png (8 charts)
  - integrated_district_summary.csv
  - integrated_pincode_analysis.csv
  - exclusion_zone_pincodes.csv
  - high_stress_pincodes.csv
""")

print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
