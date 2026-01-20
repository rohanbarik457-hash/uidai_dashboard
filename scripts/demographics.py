"""
UIDAI HACKATHON 2026 - ODISHA DEMOGRAPHIC UPDATE ANALYSIS
Dataset: odisha_demographic_clean.csv
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

print("\n UIDAI HACKATHON 2026 - ODISHA DEMOGRAPHIC UPDATE ANALYSIS")
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# STEP 1: DATA PREPARATION
print("STEP 1: DATA PREPARATION")
df = pd.read_csv(f"{data_dir}/odisha_demographic_clean.csv")
print(f"  Records loaded: {len(df):,}")

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_name'] = df['date'].dt.month_name()
df['month_year'] = df['date'].dt.to_period('M')

df['demo_age_5_17'] = df['demo_age_5_17'].fillna(0)
df['demo_age_17_'] = df['demo_age_17_'].fillna(0)

before_dedup = len(df)
df = df.drop_duplicates()
print(f"  Duplicates removed: {before_dedup - len(df):,}")
print(f"  Date range: {df['date'].min().strftime('%d-%m-%Y')} to {df['date'].max().strftime('%d-%m-%Y')}")
print(f"  Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}")

# STEP 2: FEATURE ENGINEERING
print("\nSTEP 2: FEATURE ENGINEERING")
df['Total_Updates'] = df['demo_age_5_17'] + df['demo_age_17_']
df['Youth_Percent'] = np.where(df['Total_Updates'] > 0, (df['demo_age_5_17'] / df['Total_Updates']) * 100, 0)
df['Adult_Percent'] = np.where(df['Total_Updates'] > 0, (df['demo_age_17_'] / df['Total_Updates']) * 100, 0)
print("  Created: Total_Updates, Youth_Percent, Adult_Percent")

# STEP 3: DESCRIPTIVE STATISTICS
print("\nSTEP 3: DESCRIPTIVE STATISTICS")
total_updates = df['Total_Updates'].sum()
total_5_17 = df['demo_age_5_17'].sum()
total_17_plus = df['demo_age_17_'].sum()

print(f"\n  TOTAL DEMOGRAPHIC UPDATES: {total_updates:,}")
print(f"  Youth (5-17):   {total_5_17:,} ({total_5_17/total_updates*100:.1f}%)")
print(f"  Adults (17+):   {total_17_plus:,} ({total_17_plus/total_updates*100:.1f}%)")

district_stats = df.groupby('district')['Total_Updates'].sum().sort_values(ascending=False)
print(f"\n  TOP 10 DISTRICTS:")
for i, (district, total) in enumerate(district_stats.head(10).items(), 1):
    print(f"    {i}. {district}: {total:,}")

# STEP 4: TIME-SERIES ANALYSIS
print("\nSTEP 4: TIME-SERIES ANALYSIS")
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly = df.groupby('month_name')['Total_Updates'].sum()
monthly = monthly.reindex([m for m in month_order if m in monthly.index])

peak_month = monthly.idxmax()
low_month = monthly.idxmin()
print(f"  Peak month: {peak_month} ({monthly[peak_month]:,})")
print(f"  Low month: {low_month} ({monthly[low_month]:,})")

print(f"\n  MONTHLY TREND:")
for month, value in monthly.items():
    print(f"    {month}: {value:,}")

# STEP 5: AGE-GROUP ANALYSIS
print("\nSTEP 5: AGE-GROUP ANALYSIS")
district_age = df.groupby('district').agg({
    'demo_age_5_17': 'sum', 'demo_age_17_': 'sum', 'Total_Updates': 'sum'
})
district_age['Youth_Pct'] = (district_age['demo_age_5_17'] / district_age['Total_Updates'] * 100).round(1)
district_age['Adult_Pct'] = (district_age['demo_age_17_'] / district_age['Total_Updates'] * 100).round(1)
district_age = district_age.sort_values('Total_Updates', ascending=False)

print(f"\n  DISTRICT AGE DISTRIBUTION:")
for district in district_age.head(10).index:
    youth = district_age.loc[district, 'Youth_Pct']
    adult = district_age.loc[district, 'Adult_Pct']
    print(f"    {district}: Youth {youth:.1f}% | Adult {adult:.1f}%")

dominant_group = "Adults (17+)" if total_17_plus > total_5_17 else "Youth (5-17)"
print(f"\n  Dominant group: {dominant_group}")
print(f"  Interpretation: {'Address/mobile changes prevalent' if total_17_plus > total_5_17 else 'School record corrections active'}")

# STEP 6: PINCODE ANALYSIS
print("\nSTEP 6: PINCODE ANALYSIS")
pincode_stats = df.groupby(['pincode', 'district'])['Total_Updates'].sum().reset_index()
pincode_stats = pincode_stats.sort_values('Total_Updates', ascending=False)

print(f"\n  TOP 10 PINCODES:")
for i, (_, row) in enumerate(pincode_stats.head(10).iterrows(), 1):
    print(f"    {i}. {row['pincode']} ({row['district']}): {row['Total_Updates']:,}")

district_density = df.groupby('district').agg({'Total_Updates': 'sum', 'pincode': 'nunique'})
district_density['Density'] = (district_density['Total_Updates'] / district_density['pincode']).round(0)
district_density = district_density.sort_values('Density', ascending=False)

print(f"\n  TOP 5 DISTRICTS BY UPDATE DENSITY:")
for district in district_density.head(5).index:
    print(f"    {district}: {district_density.loc[district, 'Density']:.0f} updates/pincode")

# STEP 7: GAP ANALYSIS
print("\nSTEP 7: GAP ANALYSIS")
threshold_25 = pincode_stats['Total_Updates'].quantile(0.25)
low_update_pincodes = pincode_stats[pincode_stats['Total_Updates'] < threshold_25]
print(f"  Low-update pincodes (< {threshold_25:.0f}): {len(low_update_pincodes)}")

low_by_district = low_update_pincodes.groupby('district').size().sort_values(ascending=False)
print(f"\n  DISTRICTS WITH MOST LOW-UPDATE PINCODES:")
for district, count in low_by_district.head(5).items():
    print(f"    {district}: {count}")

# STEP 8: OPERATIONAL ANALYSIS
print("\nSTEP 8: OPERATIONAL ANALYSIS")
district_cv = df.groupby('district')['Total_Updates'].agg(['mean', 'std'])
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
fig.suptitle('ODISHA DEMOGRAPHIC UPDATE ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

# Chart 1: Monthly Trend
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, color='#E74C3C')
ax1.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color='#E74C3C')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly.index, rotation=45, ha='right')
ax1.set_title('Monthly Update Trend', fontweight='bold')
ax1.set_ylabel('Updates')

# Chart 2: Age Distribution by District
ax2 = fig.add_subplot(3, 2, 2)
top10 = district_age.head(10)
x = range(len(top10))
ax2.bar(x, top10['demo_age_5_17'], label='5-17', color='#3498DB')
ax2.bar(x, top10['demo_age_17_'], bottom=top10['demo_age_5_17'], label='17+', color='#9B59B6')
ax2.set_xticks(x)
ax2.set_xticklabels(top10.index, rotation=45, ha='right')
ax2.set_title('Age Distribution by District', fontweight='bold')
ax2.legend()

# Chart 3: Top Pincodes
ax3 = fig.add_subplot(3, 2, 3)
top15 = pincode_stats.head(15)
ax3.barh(range(15), top15['Total_Updates'].values, color=plt.cm.Reds(np.linspace(0.3, 0.9, 15)))
ax3.set_yticks(range(15))
ax3.set_yticklabels([f"{r['pincode']}" for _, r in top15.iterrows()])
ax3.set_title('Top 15 Pincodes', fontweight='bold')
ax3.invert_yaxis()

# Chart 4: Age Group Pie
ax4 = fig.add_subplot(3, 2, 4)
ax4.pie([total_5_17, total_17_plus], labels=['5-17', '17+'], 
        colors=['#3498DB', '#9B59B6'], autopct='%1.1f%%', startangle=90)
ax4.set_title('Age Group Distribution', fontweight='bold')

# Chart 5: Heatmap
ax5 = fig.add_subplot(3, 2, 5)
heatmap_data = df.groupby(['district', 'month_name'])['Total_Updates'].sum().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(columns=[m for m in month_order if m in heatmap_data.columns])
heatmap_data = heatmap_data.loc[district_age.head(15).index]
sns.heatmap(heatmap_data, cmap='Reds', ax=ax5)
ax5.set_title('District vs Month Heatmap', fontweight='bold')

# Chart 6: District Rankings
ax6 = fig.add_subplot(3, 2, 6)
top15_dist = district_stats.head(15)
ax6.barh(range(15), top15_dist.values, color=plt.cm.RdPu(np.linspace(0.3, 0.9, 15))[::-1])
ax6.set_yticks(range(15))
ax6.set_yticklabels(top15_dist.index)
ax6.set_title('Top 15 Districts', fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(f"{data_dir}/odisha_demographic_analysis_charts.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Charts saved: odisha_demographic_analysis_charts.png")

# STEP 10: INSIGHTS & RECOMMENDATIONS
print("\nSTEP 10: KEY INSIGHTS & RECOMMENDATIONS")
print(f"""
  EXECUTIVE SUMMARY:
  - Total Updates: {total_updates:,}
  - Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}
  - Peak Month: {peak_month} | Low Month: {low_month}
  
  KEY FINDINGS:
  - Youth (5-17): {total_5_17/total_updates*100:.1f}%
  - Adults (17+): {total_17_plus/total_updates*100:.1f}%
  - Low-update pincodes: {len(low_update_pincodes)}
  
  POLICY RECOMMENDATIONS:
  1. Awareness campaigns in {len(low_update_pincodes)} low-update pincodes
  2. School integration for youth demographic updates
  3. Mobile update units for remote areas
  4. Seasonal drives during {low_month}
""")

# Save outputs
district_age.to_csv(f"{data_dir}/demographic_district_analysis.csv")
pincode_stats.to_csv(f"{data_dir}/demographic_pincode_analysis.csv", index=False)
low_update_pincodes.to_csv(f"{data_dir}/demographic_low_update_pincodes.csv", index=False)

print("\n  OUTPUT FILES:")
print("    - odisha_demographic_analysis_charts.png")
print("    - demographic_district_analysis.csv")
print("    - demographic_pincode_analysis.csv")
print("    - demographic_low_update_pincodes.csv")

print(f"\n ANALYSIS COMPLETE!")
print(f" Records: {len(df):,} | Districts: {df['district'].nunique()} | Pincodes: {df['pincode'].nunique()}\n")
