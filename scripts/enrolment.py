"""
UIDAI HACKATHON 2026 - ODISHA AADHAAR ENROLLMENT ANALYSIS
Dataset: odisha_enrolment_clean.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

data_dir = r"c:\Users\rohan\Aadharcard"

print("\n UIDAI HACKATHON 2026 - ODISHA ENROLLMENT ANALYSIS")
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# STEP 1: DATA PREPARATION
print("STEP 1: DATA PREPARATION")
df = pd.read_csv(f"{data_dir}/odisha_enrolment_clean.csv")
print(f"  Records loaded: {len(df):,}")

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_name'] = df['date'].dt.month_name()
df['month_year'] = df['date'].dt.to_period('M')

df['age_0_5'] = df['age_0_5'].fillna(0)
df['age_5_17'] = df['age_5_17'].fillna(0)
df['age_18_greater'] = df['age_18_greater'].fillna(0)

before_dedup = len(df)
df = df.drop_duplicates()
print(f"  Duplicates removed: {before_dedup - len(df):,}")
print(f"  Date range: {df['date'].min().strftime('%d-%m-%Y')} to {df['date'].max().strftime('%d-%m-%Y')}")
print(f"  Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}")

# STEP 2: FEATURE ENGINEERING
print("\nSTEP 2: FEATURE ENGINEERING")
df['Total_Enrollments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
df['Age_0_5_Percent'] = np.where(df['Total_Enrollments'] > 0, (df['age_0_5'] / df['Total_Enrollments']) * 100, 0)
df['Age_5_17_Percent'] = np.where(df['Total_Enrollments'] > 0, (df['age_5_17'] / df['Total_Enrollments']) * 100, 0)
df['Age_18_Plus_Percent'] = np.where(df['Total_Enrollments'] > 0, (df['age_18_greater'] / df['Total_Enrollments']) * 100, 0)
print("  Created: Total_Enrollments, Age percentages")

# STEP 3: DESCRIPTIVE STATISTICS
print("\nSTEP 3: DESCRIPTIVE STATISTICS")
total_enrollments = df['Total_Enrollments'].sum()
total_0_5 = df['age_0_5'].sum()
total_5_17 = df['age_5_17'].sum()
total_18_plus = df['age_18_greater'].sum()

print(f"\n  TOTAL ENROLLMENTS: {total_enrollments:,}")
print(f"  Bal Aadhaar (0-5):  {total_0_5:,} ({total_0_5/total_enrollments*100:.1f}%)")
print(f"  Youth (5-17):       {total_5_17:,} ({total_5_17/total_enrollments*100:.1f}%)")
print(f"  Adults (18+):       {total_18_plus:,} ({total_18_plus/total_enrollments*100:.1f}%)")

district_stats = df.groupby('district')['Total_Enrollments'].sum().sort_values(ascending=False)
print(f"\n  TOP 10 DISTRICTS:")
for i, (district, total) in enumerate(district_stats.head(10).items(), 1):
    print(f"    {i}. {district}: {total:,}")

# STEP 4: TIME-SERIES ANALYSIS
print("\nSTEP 4: TIME-SERIES ANALYSIS")
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly = df.groupby('month_name')['Total_Enrollments'].sum()
monthly = monthly.reindex([m for m in month_order if m in monthly.index])

peak_month = monthly.idxmax()
low_month = monthly.idxmin()
print(f"  Peak month: {peak_month} ({monthly[peak_month]:,})")
print(f"  Low month: {low_month} ({monthly[low_month]:,})")

print(f"\n  MONTHLY TREND:")
for month, value in monthly.items():
    print(f"    {month}: {value:,}")

# STEP 5: AGE-WISE ANALYSIS
print("\nSTEP 5: AGE-WISE ANALYSIS")
district_age = df.groupby('district').agg({
    'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum', 'Total_Enrollments': 'sum'
})
district_age['Bal_Pct'] = (district_age['age_0_5'] / district_age['Total_Enrollments'] * 100).round(1)
district_age = district_age.sort_values('Total_Enrollments', ascending=False)

print(f"\n  DISTRICT AGE DISTRIBUTION (Top 10):")
for district in district_age.head(10).index:
    pct = district_age.loc[district, 'Bal_Pct']
    print(f"    {district}: {pct:.1f}% Bal Aadhaar")

# STEP 6: PINCODE ANALYSIS
print("\nSTEP 6: PINCODE ANALYSIS")
pincode_stats = df.groupby(['pincode', 'district'])['Total_Enrollments'].sum().reset_index()
pincode_stats = pincode_stats.sort_values('Total_Enrollments', ascending=False)

print(f"\n  TOP 10 PINCODES:")
for i, (_, row) in enumerate(pincode_stats.head(10).iterrows(), 1):
    print(f"    {i}. {row['pincode']} ({row['district']}): {row['Total_Enrollments']:,}")

# STEP 7: GAP ANALYSIS
print("\nSTEP 7: GAP ANALYSIS")
threshold_25 = pincode_stats['Total_Enrollments'].quantile(0.25)
low_enrollment_pincodes = pincode_stats[pincode_stats['Total_Enrollments'] < threshold_25]
print(f"  Low-enrollment pincodes (< {threshold_25:.0f}): {len(low_enrollment_pincodes)}")

low_by_district = low_enrollment_pincodes.groupby('district').size().sort_values(ascending=False)
print(f"\n  DISTRICTS WITH MOST LOW-ENROLLMENT PINCODES:")
for district, count in low_by_district.head(5).items():
    print(f"    {district}: {count}")

# STEP 8: OPERATIONAL ANALYSIS
print("\nSTEP 8: OPERATIONAL ANALYSIS")
district_cv = df.groupby('district')['Total_Enrollments'].agg(['mean', 'std'])
district_cv['CV'] = (district_cv['std'] / district_cv['mean'] * 100).round(1)

print(f"\n  MOST VOLATILE DISTRICTS (High CV):")
for district in district_cv.nlargest(5, 'CV').index:
    print(f"    {district}: CV = {district_cv.loc[district, 'CV']:.1f}%")

# STEP 9: VISUALIZATIONS
print("\nSTEP 9: CREATING VISUALIZATIONS")
fig = plt.figure(figsize=(20, 24))
fig.suptitle('ODISHA AADHAAR ENROLLMENT ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

# Chart 1: Monthly Trend
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, color='#2E86AB')
ax1.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color='#2E86AB')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly.index, rotation=45, ha='right')
ax1.set_title('Monthly Enrollment Trend', fontweight='bold')
ax1.set_ylabel('Enrollments')

# Chart 2: Age Distribution by District
ax2 = fig.add_subplot(3, 2, 2)
top10 = district_age.head(10)
x = range(len(top10))
ax2.bar(x, top10['age_0_5'], label='0-5', color='#E8D21D')
ax2.bar(x, top10['age_5_17'], bottom=top10['age_0_5'], label='5-17', color='#2E86AB')
ax2.bar(x, top10['age_18_greater'], bottom=top10['age_0_5']+top10['age_5_17'], label='18+', color='#A23B72')
ax2.set_xticks(x)
ax2.set_xticklabels(top10.index, rotation=45, ha='right')
ax2.set_title('Age Distribution by District', fontweight='bold')
ax2.legend()

# Chart 3: Top Pincodes
ax3 = fig.add_subplot(3, 2, 3)
top15 = pincode_stats.head(15)
ax3.barh(range(15), top15['Total_Enrollments'].values, color=plt.cm.viridis(np.linspace(0.2, 0.8, 15)))
ax3.set_yticks(range(15))
ax3.set_yticklabels([f"{r['pincode']}" for _, r in top15.iterrows()])
ax3.set_title('Top 15 Pincodes', fontweight='bold')
ax3.invert_yaxis()

# Chart 4: Age Group Pie
ax4 = fig.add_subplot(3, 2, 4)
ax4.pie([total_0_5, total_5_17, total_18_plus], labels=['0-5', '5-17', '18+'], 
        colors=['#E8D21D', '#2E86AB', '#A23B72'], autopct='%1.1f%%', startangle=90)
ax4.set_title('Age Group Distribution', fontweight='bold')

# Chart 5: Heatmap
ax5 = fig.add_subplot(3, 2, 5)
heatmap_data = df.groupby(['district', 'month_name'])['Total_Enrollments'].sum().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(columns=[m for m in month_order if m in heatmap_data.columns])
heatmap_data = heatmap_data.loc[district_age.head(15).index]
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax5)
ax5.set_title('District vs Month Heatmap', fontweight='bold')

# Chart 6: District Rankings
ax6 = fig.add_subplot(3, 2, 6)
top15_dist = district_stats.head(15)
ax6.barh(range(15), top15_dist.values, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, 15))[::-1])
ax6.set_yticks(range(15))
ax6.set_yticklabels(top15_dist.index)
ax6.set_title('Top 15 Districts', fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(f"{data_dir}/odisha_enrollment_analysis_charts.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Charts saved: odisha_enrollment_analysis_charts.png")

# STEP 10: INSIGHTS & RECOMMENDATIONS
print("\nSTEP 10: KEY INSIGHTS & RECOMMENDATIONS")
print(f"""
  EXECUTIVE SUMMARY:
  - Total Enrollments: {total_enrollments:,}
  - Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}
  - Peak Month: {peak_month} | Low Month: {low_month}
  
  KEY FINDINGS:
  - Bal Aadhaar (0-5): {total_0_5/total_enrollments*100:.1f}%
  - Youth (5-17): {total_5_17/total_enrollments*100:.1f}%
  - Adults (18+): {total_18_plus/total_enrollments*100:.1f}%
  - Low-enrollment pincodes: {len(low_enrollment_pincodes)}
  
  POLICY RECOMMENDATIONS:
  1. Mobile enrollment camps for {len(low_enrollment_pincodes)} low-enrollment pincodes
  2. School-based Bal Aadhaar programs
  3. Adult awareness drives in underperforming districts
  4. Seasonal enrollment drives during {low_month}
""")

# Save outputs
district_age.to_csv(f"{data_dir}/odisha_district_analysis.csv")
pincode_stats.to_csv(f"{data_dir}/odisha_pincode_analysis.csv", index=False)
low_enrollment_pincodes.to_csv(f"{data_dir}/odisha_low_enrollment_pincodes.csv", index=False)

print("\n  OUTPUT FILES:")
print("    - odisha_enrollment_analysis_charts.png")
print("    - odisha_district_analysis.csv")
print("    - odisha_pincode_analysis.csv")
print("    - odisha_low_enrollment_pincodes.csv")

print(f"\n ANALYSIS COMPLETE!")
print(f" Records: {len(df):,} | Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}\n")
