"""
UIDAI HACKATHON 2026 - ODISHA BIOMETRIC UPDATE ANALYSIS
Dataset: odisha_biometric_clean.csv
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

print("\n UIDAI HACKATHON 2026 - ODISHA BIOMETRIC UPDATE ANALYSIS")
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# STEP 1: DATA PREPARATION
print("STEP 1: DATA PREPARATION")
df = pd.read_csv(f"{data_dir}/odisha_biometric_clean.csv")
print(f"  Records loaded: {len(df):,}")

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_name'] = df['date'].dt.month_name()
df['month_year'] = df['date'].dt.to_period('M')

df['bio_age_5_17'] = df['bio_age_5_17'].fillna(0)
df['bio_age_17_'] = df['bio_age_17_'].fillna(0)

before_dedup = len(df)
df = df.drop_duplicates()
print(f"  Duplicates removed: {before_dedup - len(df):,}")
print(f"  Date range: {df['date'].min().strftime('%d-%m-%Y')} to {df['date'].max().strftime('%d-%m-%Y')}")
print(f"  Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}")

# STEP 2: FEATURE ENGINEERING
print("\nSTEP 2: FEATURE ENGINEERING")
df['Total_Bio'] = df['bio_age_5_17'] + df['bio_age_17_']
df['Youth_Percent'] = np.where(df['Total_Bio'] > 0, (df['bio_age_5_17'] / df['Total_Bio']) * 100, 0)
df['Adult_Percent'] = np.where(df['Total_Bio'] > 0, (df['bio_age_17_'] / df['Total_Bio']) * 100, 0)
print("  Created: Total_Bio, Youth_Percent, Adult_Percent")

# STEP 3: DESCRIPTIVE STATISTICS
print("\nSTEP 3: DESCRIPTIVE STATISTICS")
total_bio = df['Total_Bio'].sum()
total_5_17 = df['bio_age_5_17'].sum()
total_17_plus = df['bio_age_17_'].sum()

print(f"\n  TOTAL BIOMETRIC UPDATES: {total_bio:,}")
print(f"  Youth (5-17) - Mandatory:     {total_5_17:,} ({total_5_17/total_bio*100:.1f}%)")
print(f"  Adults (17+) - Revalidation:  {total_17_plus:,} ({total_17_plus/total_bio*100:.1f}%)")

district_stats = df.groupby('district')['Total_Bio'].sum().sort_values(ascending=False)
print(f"\n  TOP 10 DISTRICTS:")
for i, (district, total) in enumerate(district_stats.head(10).items(), 1):
    print(f"    {i}. {district}: {total:,}")

# STEP 4: AGE-GROUP ANALYSIS
print("\nSTEP 4: AGE-GROUP ANALYSIS")
district_age = df.groupby('district').agg({
    'bio_age_5_17': 'sum', 'bio_age_17_': 'sum', 'Total_Bio': 'sum'
})
district_age['Youth_Pct'] = (district_age['bio_age_5_17'] / district_age['Total_Bio'] * 100).round(1)
district_age['Adult_Pct'] = (district_age['bio_age_17_'] / district_age['Total_Bio'] * 100).round(1)
district_age = district_age.sort_values('Total_Bio', ascending=False)

print(f"\n  DISTRICT AGE DISTRIBUTION:")
for district in district_age.head(10).index:
    youth = district_age.loc[district, 'Youth_Pct']
    adult = district_age.loc[district, 'Adult_Pct']
    dominant = "Youth" if youth > adult else "Adult"
    print(f"    {district}: Youth {youth:.1f}% | Adult {adult:.1f}% [{dominant}]")

youth_heavy = district_age[district_age['Youth_Pct'] > 50].index.tolist()
adult_heavy = district_age[district_age['Adult_Pct'] > 50].index.tolist()
print(f"\n  Youth-heavy districts: {len(youth_heavy)}")
print(f"  Adult-heavy districts: {len(adult_heavy)}")

# STEP 5: TIME-SERIES ANALYSIS
print("\nSTEP 5: TIME-SERIES ANALYSIS")
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly = df.groupby('month_name')['Total_Bio'].sum()
monthly = monthly.reindex([m for m in month_order if m in monthly.index])

peak_month = monthly.idxmax()
low_month = monthly.idxmin()
print(f"  Peak month: {peak_month} ({monthly[peak_month]:,})")
print(f"  Low month: {low_month} ({monthly[low_month]:,})")

monthly_age = df.groupby('month_name').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'})
monthly_age = monthly_age.reindex([m for m in month_order if m in monthly_age.index])

print(f"\n  MONTHLY TREND:")
for month in monthly_age.index:
    youth = monthly_age.loc[month, 'bio_age_5_17']
    adult = monthly_age.loc[month, 'bio_age_17_']
    print(f"    {month}: Youth {youth:,} | Adult {adult:,}")

# STEP 6: PINCODE ANALYSIS
print("\nSTEP 6: PINCODE ANALYSIS")
pincode_stats = df.groupby(['pincode', 'district']).agg({
    'Total_Bio': 'sum', 'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'
}).reset_index()
pincode_stats['Youth_Pct'] = (pincode_stats['bio_age_5_17'] / pincode_stats['Total_Bio'] * 100).round(1)
pincode_stats = pincode_stats.sort_values('Total_Bio', ascending=False)

print(f"\n  TOP 10 PINCODES:")
for i, (_, row) in enumerate(pincode_stats.head(10).iterrows(), 1):
    print(f"    {i}. {row['pincode']} ({row['district']}): {row['Total_Bio']:,}")

district_density = df.groupby('district').agg({'Total_Bio': 'sum', 'pincode': 'nunique'})
district_density['Density'] = (district_density['Total_Bio'] / district_density['pincode']).round(0)
district_density = district_density.sort_values('Density', ascending=False)

print(f"\n  TOP 5 DISTRICTS BY UPDATE DENSITY:")
for district in district_density.head(5).index:
    print(f"    {district}: {district_density.loc[district, 'Density']:.0f} updates/pincode")

# STEP 7: QUALITY ANALYSIS
print("\nSTEP 7: QUALITY & STRESS ANALYSIS")
threshold_75 = pincode_stats['Total_Bio'].quantile(0.75)
threshold_25 = pincode_stats['Total_Bio'].quantile(0.25)

high_update_pincodes = pincode_stats[pincode_stats['Total_Bio'] > threshold_75]
low_update_pincodes = pincode_stats[pincode_stats['Total_Bio'] < threshold_25]

print(f"  High-stress pincodes (> {threshold_75:.0f}): {len(high_update_pincodes)}")
print(f"  Low-update pincodes (< {threshold_25:.0f}): {len(low_update_pincodes)}")

high_by_district = high_update_pincodes.groupby('district').size().sort_values(ascending=False)
low_by_district = low_update_pincodes.groupby('district').size().sort_values(ascending=False)

print(f"\n  DISTRICTS WITH MOST HIGH-STRESS PINCODES:")
for district, count in high_by_district.head(5).items():
    print(f"    {district}: {count}")

print(f"\n  DISTRICTS WITH MOST LOW-UPDATE PINCODES:")
for district, count in low_by_district.head(5).items():
    print(f"    {district}: {count}")

# STEP 8: OPERATIONAL ANALYSIS
print("\nSTEP 8: OPERATIONAL ANALYSIS")
district_cv = df.groupby('district')['Total_Bio'].agg(['mean', 'std'])
district_cv['CV'] = (district_cv['std'] / district_cv['mean'] * 100).round(1)

print(f"\n  MOST VOLATILE DISTRICTS:")
for district in district_cv.nlargest(5, 'CV').index:
    print(f"    {district}: CV = {district_cv.loc[district, 'CV']:.1f}%")

print(f"\n  MOST STABLE DISTRICTS:")
for district in district_cv.nsmallest(5, 'CV').index:
    print(f"    {district}: CV = {district_cv.loc[district, 'CV']:.1f}%")

# STEP 9: VISUALIZATIONS
print("\nSTEP 9: CREATING VISUALIZATIONS")
fig = plt.figure(figsize=(20, 24))
fig.suptitle('ODISHA BIOMETRIC UPDATE ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

# Chart 1: Monthly Youth vs Adult Trend
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(range(len(monthly_age)), monthly_age['bio_age_5_17'].values, marker='o', linewidth=2, color='#27AE60', label='Youth')
ax1.plot(range(len(monthly_age)), monthly_age['bio_age_17_'].values, marker='s', linewidth=2, color='#8E44AD', label='Adult')
ax1.set_xticks(range(len(monthly_age)))
ax1.set_xticklabels(monthly_age.index, rotation=45, ha='right')
ax1.set_title('Monthly Trend: Youth vs Adult', fontweight='bold')
ax1.legend()

# Chart 2: Age Distribution by District
ax2 = fig.add_subplot(3, 2, 2)
top10 = district_age.head(10)
x = range(len(top10))
ax2.bar(x, top10['bio_age_5_17'], label='Youth', color='#27AE60')
ax2.bar(x, top10['bio_age_17_'], bottom=top10['bio_age_5_17'], label='Adult', color='#8E44AD')
ax2.set_xticks(x)
ax2.set_xticklabels(top10.index, rotation=45, ha='right')
ax2.set_title('Age Distribution by District', fontweight='bold')
ax2.legend()

# Chart 3: Top Pincodes
ax3 = fig.add_subplot(3, 2, 3)
top15 = pincode_stats.head(15)
ax3.barh(range(15), top15['Total_Bio'].values, color=plt.cm.Greens(np.linspace(0.3, 0.9, 15)))
ax3.set_yticks(range(15))
ax3.set_yticklabels([f"{r['pincode']}" for _, r in top15.iterrows()])
ax3.set_title('Top 15 Pincodes', fontweight='bold')
ax3.invert_yaxis()

# Chart 4: Age Group Pie
ax4 = fig.add_subplot(3, 2, 4)
ax4.pie([total_5_17, total_17_plus], labels=['Youth (Mandatory)', 'Adult (Revalidation)'], 
        colors=['#27AE60', '#8E44AD'], autopct='%1.1f%%', startangle=90)
ax4.set_title('Age Group Distribution', fontweight='bold')

# Chart 5: Heatmap
ax5 = fig.add_subplot(3, 2, 5)
heatmap_data = df.groupby(['district', 'month_name'])['Total_Bio'].sum().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(columns=[m for m in month_order if m in heatmap_data.columns])
heatmap_data = heatmap_data.loc[district_age.head(15).index]
sns.heatmap(heatmap_data, cmap='Greens', ax=ax5)
ax5.set_title('District vs Month Heatmap', fontweight='bold')

# Chart 6: District Rankings
ax6 = fig.add_subplot(3, 2, 6)
top15_dist = district_stats.head(15)
ax6.barh(range(15), top15_dist.values, color=plt.cm.BuGn(np.linspace(0.3, 0.9, 15))[::-1])
ax6.set_yticks(range(15))
ax6.set_yticklabels(top15_dist.index)
ax6.set_title('Top 15 Districts', fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(f"{data_dir}/odisha_biometric_analysis_charts.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Charts saved: odisha_biometric_analysis_charts.png")

# STEP 10: INSIGHTS & RECOMMENDATIONS
print("\nSTEP 10: KEY INSIGHTS & RECOMMENDATIONS")
print(f"""
  EXECUTIVE SUMMARY:
  - Total Biometric Updates: {total_bio:,}
  - Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}
  - Peak Month: {peak_month} | Low Month: {low_month}
  
  KEY FINDINGS:
  - Youth (5-17) Mandatory: {total_5_17/total_bio*100:.1f}%
  - Adults (17+) Revalidation: {total_17_plus/total_bio*100:.1f}%
  - High-stress pincodes: {len(high_update_pincodes)}
  - Low-update pincodes: {len(low_update_pincodes)}
  
  POLICY RECOMMENDATIONS:
  1. Device upgrades in {len(high_update_pincodes)} high-stress pincodes
  2. School-based biometric camps for youth compliance
  3. Mobile units for {len(low_update_pincodes)} low-update pincodes
  4. Adult revalidation awareness campaigns
  5. Device calibration in volatile districts
""")

# Save outputs
district_age.to_csv(f"{data_dir}/biometric_district_analysis.csv")
pincode_stats.to_csv(f"{data_dir}/biometric_pincode_analysis.csv", index=False)
high_update_pincodes.to_csv(f"{data_dir}/biometric_high_update_pincodes.csv", index=False)
low_update_pincodes.to_csv(f"{data_dir}/biometric_low_update_pincodes.csv", index=False)

print("\n  OUTPUT FILES:")
print("    - odisha_biometric_analysis_charts.png")
print("    - biometric_district_analysis.csv")
print("    - biometric_pincode_analysis.csv")
print("    - biometric_high_update_pincodes.csv")
print("    - biometric_low_update_pincodes.csv")

print(f"\n ANALYSIS COMPLETE!")
print(f" Records: {len(df):,} | Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}\n")
