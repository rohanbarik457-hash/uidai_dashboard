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

enroll_df = pd.read_csv(f"{data_dir}/odisha_enrolment_clean.csv")
demo_df = pd.read_csv(f"{data_dir}/odisha_demographic_clean.csv")
bio_df = pd.read_csv(f"{data_dir}/odisha_biometric_clean.csv")

print(f"  Enrollment: {len(enroll_df):,} records")
print(f"  Demographic: {len(demo_df):,} records")
print(f"  Biometric: {len(bio_df):,} records")

for df in [enroll_df, demo_df, bio_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()

enroll_df = enroll_df.drop_duplicates().fillna(0)
demo_df = demo_df.drop_duplicates().fillna(0)
bio_df = bio_df.drop_duplicates().fillna(0)

print(f"  Districts: {enroll_df['district'].nunique()}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\nSTEP 2: FEATURE ENGINEERING")

enroll_df['Total_Enrollments'] = enroll_df['age_0_5'] + enroll_df['age_5_17'] + enroll_df['age_18_greater']
demo_df['Total_Demo_Updates'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
bio_df['Total_Biometric'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']

print("  Created: Total_Enrollments, Total_Demo_Updates, Total_Biometric")

# ============================================================
# STEP 3: RATIO ANALYSIS
# ============================================================
print("\nSTEP 3: RATIO ANALYSIS")

enroll_dist = enroll_df.groupby('district')['Total_Enrollments'].sum()
demo_dist = demo_df.groupby('district')['Total_Demo_Updates'].sum()
bio_dist = bio_df.groupby('district')['Total_Biometric'].sum()

district_summary = pd.DataFrame({
    'Enrollments': enroll_dist,
    'Demo_Updates': demo_dist,
    'Bio_Updates': bio_dist
}).fillna(0)

district_summary['Demo_Rate'] = (district_summary['Demo_Updates'] / district_summary['Enrollments']).round(2)
district_summary['Bio_Rate'] = (district_summary['Bio_Updates'] / district_summary['Enrollments']).round(2)
district_summary = district_summary.sort_values('Enrollments', ascending=False)

print("\n  DISTRICT UPDATE RATES:")
for district in district_summary.head(10).index:
    row = district_summary.loc[district]
    print(f"    {district}: Demo {row['Demo_Rate']:.1f}x | Bio {row['Bio_Rate']:.1f}x")

# ============================================================
# STEP 4: AGE COHORT ANALYSIS
# ============================================================
print("\nSTEP 4: AGE COHORT ANALYSIS")

enroll_age = {
    '0-5': enroll_df['age_0_5'].sum(),
    '5-17': enroll_df['age_5_17'].sum(),
    '18+': enroll_df['age_18_greater'].sum()
}
total_enroll = sum(enroll_age.values())

demo_youth = demo_df['demo_age_5_17'].sum()
demo_adult = demo_df['demo_age_17_'].sum()
total_demo = demo_youth + demo_adult

bio_youth = bio_df['bio_age_5_17'].sum()
bio_adult = bio_df['bio_age_17_'].sum()
total_bio = bio_youth + bio_adult

print(f"\n  ENROLLMENT: 0-5 ({enroll_age['0-5']/total_enroll*100:.1f}%) | 5-17 ({enroll_age['5-17']/total_enroll*100:.1f}%) | 18+ ({enroll_age['18+']/total_enroll*100:.1f}%)")
print(f"  DEMO: Youth ({demo_youth/total_demo*100:.1f}%) | Adult ({demo_adult/total_demo*100:.1f}%)")
print(f"  BIO: Youth ({bio_youth/total_bio*100:.1f}%) | Adult ({bio_adult/total_bio*100:.1f}%)")

# ============================================================
# STEP 5: TIME-SERIES ANALYSIS
# ============================================================
print("\nSTEP 5: TIME-SERIES ANALYSIS")

month_order = ['March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

enroll_monthly = enroll_df.groupby('month_name')['Total_Enrollments'].sum()
demo_monthly = demo_df.groupby('month_name')['Total_Demo_Updates'].sum()
bio_monthly = bio_df.groupby('month_name')['Total_Biometric'].sum()

print(f"\n  Peak Enrollment: {enroll_monthly.idxmax()} ({enroll_monthly.max():,})")
print(f"  Peak Demo: {demo_monthly.idxmax()} ({demo_monthly.max():,})")
print(f"  Peak Bio: {bio_monthly.idxmax()} ({bio_monthly.max():,})")

# ============================================================
# STEP 6: PINCODE ANALYSIS
# ============================================================
print("\nSTEP 6: PINCODE ANALYSIS")

enroll_pin = enroll_df.groupby(['pincode', 'district'])['Total_Enrollments'].sum().reset_index()
demo_pin = demo_df.groupby(['pincode', 'district'])['Total_Demo_Updates'].sum().reset_index()
bio_pin = bio_df.groupby(['pincode', 'district'])['Total_Biometric'].sum().reset_index()

pin_merged = enroll_pin.merge(demo_pin, on=['pincode', 'district'], how='outer')
pin_merged = pin_merged.merge(bio_pin, on=['pincode', 'district'], how='outer').fillna(0)

pin_merged['Demo_Rate'] = np.where(pin_merged['Total_Enrollments'] > 0,
                                    pin_merged['Total_Demo_Updates'] / pin_merged['Total_Enrollments'], 0)
pin_merged['Bio_Rate'] = np.where(pin_merged['Total_Enrollments'] > 0,
                                   pin_merged['Total_Biometric'] / pin_merged['Total_Enrollments'], 0)

print(f"  Total Pincodes: {len(pin_merged)}")

# ============================================================
# STEP 7: ANOMALY DETECTION
# ============================================================
print("\nSTEP 7: ANOMALY DETECTION")

pin_merged['Bio_ZScore'] = (pin_merged['Total_Biometric'] - pin_merged['Total_Biometric'].mean()) / pin_merged['Total_Biometric'].std()
anomalies = pin_merged[abs(pin_merged['Bio_ZScore']) > 2]
print(f"  Anomalous pincodes (|Z| > 2): {len(anomalies)}")

# ============================================================
# STEP 8: STABILITY ANALYSIS
# ============================================================
print("\nSTEP 8: STABILITY ANALYSIS")

enroll_cv = enroll_df.groupby('district')['Total_Enrollments'].agg(['mean', 'std'])
enroll_cv['CV'] = (enroll_cv['std'] / enroll_cv['mean'] * 100).round(1)

print("\n  Most Volatile Districts:")
for district in enroll_cv.nlargest(5, 'CV').index:
    print(f"    {district}: CV = {enroll_cv.loc[district, 'CV']:.1f}%")

# ============================================================
# STEP 9: PINCODE SEGMENTATION
# ============================================================
print("\nSTEP 9: PINCODE SEGMENTATION")

enroll_median = pin_merged['Total_Enrollments'].median()
bio_median = pin_merged['Total_Biometric'].median()

def segment(row):
    if row['Total_Enrollments'] > enroll_median and row['Total_Biometric'] <= bio_median:
        return 'Stable Zone'
    elif row['Total_Biometric'] > bio_median:
        return 'Stress Zone'
    else:
        return 'Exclusion Zone'

pin_merged['Segment'] = pin_merged.apply(segment, axis=1)
segment_counts = pin_merged['Segment'].value_counts()

print("\n  SEGMENTATION:")
for seg, count in segment_counts.items():
    print(f"    {seg}: {count} ({count/len(pin_merged)*100:.1f}%)")

exclusion = pin_merged[pin_merged['Segment'] == 'Exclusion Zone']
exclusion_by_dist = exclusion.groupby('district').size().sort_values(ascending=False)

# ============================================================
# STEP 10: QUALITY STRESS INDEX
# ============================================================
print("\nSTEP 10: QUALITY STRESS INDEX")

pin_merged['Demo_Rate_Norm'] = pin_merged['Demo_Rate'] / (pin_merged['Demo_Rate'].max() + 0.001)
pin_merged['Bio_Rate_Norm'] = pin_merged['Bio_Rate'] / (pin_merged['Bio_Rate'].max() + 0.001)
pin_merged['Enroll_Vol'] = abs(pin_merged['Total_Enrollments'] - enroll_median) / (enroll_median + 0.001)
pin_merged['Enroll_Vol_Norm'] = pin_merged['Enroll_Vol'] / (pin_merged['Enroll_Vol'].max() + 0.001)

pin_merged['Stress_Index'] = (
    pin_merged['Bio_Rate_Norm'] * 0.5 +
    pin_merged['Demo_Rate_Norm'] * 0.3 +
    pin_merged['Enroll_Vol_Norm'] * 0.2
).round(3)

district_stress = pin_merged.groupby('district')['Stress_Index'].mean().sort_values(ascending=False)

print("\n  TOP 5 STRESSED DISTRICTS:")
for district, stress in district_stress.head(5).items():
    print(f"    {district}: {stress:.3f}")

# ============================================================
# STEP 11: VISUALIZATIONS
# ============================================================
print("\nSTEP 11: CREATING VISUALIZATIONS")

fig = plt.figure(figsize=(24, 28))
fig.suptitle('UIDAI HACKATHON 2026 - INTEGRATED ODISHA ANALYSIS', fontsize=18, fontweight='bold', y=0.98)

# Chart 1: Age Cohort Comparison
ax1 = fig.add_subplot(4, 2, 1)
cohorts = ['0-5', '5-17', '17+']
enroll_ages = [enroll_df['age_0_5'].sum(), enroll_df['age_5_17'].sum(), enroll_df['age_18_greater'].sum()]
demo_ages = [0, demo_youth, demo_adult]
bio_ages = [0, bio_youth, bio_adult]
x = np.arange(len(cohorts))
width = 0.25
ax1.bar(x - width, enroll_ages, width, label='Enrollment', color='#2E86AB')
ax1.bar(x, demo_ages, width, label='Demo', color='#E74C3C')
ax1.bar(x + width, bio_ages, width, label='Bio', color='#27AE60')
ax1.set_xticks(x)
ax1.set_xticklabels(cohorts)
ax1.set_title('Age Cohort Comparison', fontweight='bold')
ax1.legend()

# Chart 2: District Rankings
ax2 = fig.add_subplot(4, 2, 2)
top10 = district_summary.head(10)
ax2.barh(range(len(top10)), top10['Enrollments'], color='#2E86AB')
ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels(top10.index)
ax2.set_title('Top 10 Districts by Enrollment', fontweight='bold')
ax2.invert_yaxis()

# Chart 3: Update Rates
ax3 = fig.add_subplot(4, 2, 3)
top10_rates = district_summary.head(10)
x = np.arange(len(top10_rates))
ax3.bar(x - 0.2, top10_rates['Demo_Rate'], 0.4, label='Demo Rate', color='#E74C3C')
ax3.bar(x + 0.2, top10_rates['Bio_Rate'], 0.4, label='Bio Rate', color='#27AE60')
ax3.set_xticks(x)
ax3.set_xticklabels(top10_rates.index, rotation=45, ha='right')
ax3.set_title('Update Rates by District', fontweight='bold')
ax3.legend()

# Chart 4: Segment Distribution
ax4 = fig.add_subplot(4, 2, 4)
colors = {'Stable Zone': '#27AE60', 'Stress Zone': '#E74C3C', 'Exclusion Zone': '#95A5A6'}
ax4.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
        colors=[colors[s] for s in segment_counts.index], startangle=90)
ax4.set_title('Pincode Segmentation', fontweight='bold')

# Chart 5: Stress Index
ax5 = fig.add_subplot(4, 2, 5)
stress_top = district_stress.head(15)
ax5.barh(range(len(stress_top)), stress_top.values, color=plt.cm.Reds(stress_top.values / stress_top.max()))
ax5.set_yticks(range(len(stress_top)))
ax5.set_yticklabels(stress_top.index)
ax5.set_title('Quality Stress Index by District', fontweight='bold')
ax5.invert_yaxis()

# Chart 6: Exclusion Zones
ax6 = fig.add_subplot(4, 2, 6)
excl_top = exclusion_by_dist.head(10)
ax6.barh(range(len(excl_top)), excl_top.values, color='#95A5A6')
ax6.set_yticks(range(len(excl_top)))
ax6.set_yticklabels(excl_top.index)
ax6.set_title('Exclusion Zones by District', fontweight='bold')
ax6.invert_yaxis()

# Chart 7: Volatility
ax7 = fig.add_subplot(4, 2, 7)
vol_top = enroll_cv.nlargest(10, 'CV')
ax7.barh(range(len(vol_top)), vol_top['CV'].values, color='#F39C12')
ax7.set_yticks(range(len(vol_top)))
ax7.set_yticklabels(vol_top.index)
ax7.set_title('Enrollment Volatility (CV%)', fontweight='bold')
ax7.invert_yaxis()

# Chart 8: Summary Stats
ax8 = fig.add_subplot(4, 2, 8)
summary_data = [total_enroll, total_demo, total_bio]
ax8.bar(['Enrollments', 'Demo Updates', 'Bio Updates'], summary_data, color=['#2E86AB', '#E74C3C', '#27AE60'])
ax8.set_title('Total Counts Summary', fontweight='bold')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(f"{data_dir}/odisha_integrated_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Charts saved: odisha_integrated_analysis.png")

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"""
  COVERAGE: {total_enroll:,} enrollments
    - Bal Aadhaar (0-5): {enroll_age['0-5']/total_enroll*100:.1f}%
    - Youth (5-17): {enroll_age['5-17']/total_enroll*100:.1f}%
    - Adults (18+): {enroll_age['18+']/total_enroll*100:.1f}%
  
  DATA ACCURACY: {total_demo:,} demographic updates
    - Avg Rate: {district_summary['Demo_Rate'].mean():.1f}x per enrollment
  
  AUTH RELIABILITY: {total_bio:,} biometric updates
    - Avg Rate: {district_summary['Bio_Rate'].mean():.1f}x per enrollment
  
  GEOGRAPHIC:
    - Districts: {len(district_summary)}
    - Pincodes: {len(pin_merged)}
    - Exclusion Zones: {len(exclusion)} ({len(exclusion)/len(pin_merged)*100:.1f}%)
    - Stress Zones: {segment_counts.get('Stress Zone', 0)}
    - Anomalous: {len(anomalies)}
""")

# ============================================================
# POLICY RECOMMENDATIONS
# ============================================================
print("="*60)
print("POLICY RECOMMENDATIONS")
print("="*60)
print(f"""
  1. COVERAGE: Mobile enrollment in {len(exclusion)} exclusion pincodes
  2. DATA ACCURACY: School integration for youth updates
  3. AUTH RELIABILITY: Device upgrades in stress zones
  4. STABILITY: Reduce volatility in {enroll_cv.nlargest(1, 'CV').index[0]}
  5. PRIORITY DISTRICTS: {', '.join(exclusion_by_dist.head(3).index)}
""")

# Save outputs
district_summary.to_csv(f"{data_dir}/integrated_district_summary.csv")
pin_merged.to_csv(f"{data_dir}/integrated_pincode_analysis.csv", index=False)
exclusion.to_csv(f"{data_dir}/exclusion_zone_pincodes.csv", index=False)

print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
