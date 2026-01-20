"""
UIDAI HACKATHON 2026 - ADVANCED ML ANALYSIS
============================================
Includes: All ML Algorithms + Statistical Analysis + Custom Indices
State: ODISHA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, classification_report, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
data_dir = r"c:\Users\rohan\Aadharcard"

print("\n" + "="*70)
print(" UIDAI HACKATHON 2026 - ADVANCED ML ANALYSIS (ODISHA)")
print("="*70)
print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================
# DATA LOADING
# ============================================================
print("\n[1] LOADING DATA...")

enroll_df = pd.read_csv(f"{data_dir}/odisha_enrolment_clean.csv")
demo_df = pd.read_csv(f"{data_dir}/odisha_demographic_clean.csv")
bio_df = pd.read_csv(f"{data_dir}/odisha_biometric_clean.csv")

for df in [enroll_df, demo_df, bio_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['month'] = df['date'].dt.month

enroll_df = enroll_df.drop_duplicates().fillna(0)
demo_df = demo_df.drop_duplicates().fillna(0)
bio_df = bio_df.drop_duplicates().fillna(0)

# Feature Engineering
enroll_df['Total_Enrollments'] = enroll_df['age_0_5'] + enroll_df['age_5_17'] + enroll_df['age_18_greater']
demo_df['Total_Demo'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
bio_df['Total_Bio'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']

print(f"  Enrollment: {len(enroll_df):,} | Demo: {len(demo_df):,} | Bio: {len(bio_df):,}")

# ============================================================
# CREATE MERGED PINCODE DATASET
# ============================================================
print("\n[2] CREATING MERGED DATASET...")

enroll_pin = enroll_df.groupby(['pincode', 'district']).agg({
    'Total_Enrollments': 'sum', 'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
}).reset_index()

demo_pin = demo_df.groupby(['pincode', 'district']).agg({
    'Total_Demo': 'sum', 'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'
}).reset_index()

bio_pin = bio_df.groupby(['pincode', 'district']).agg({
    'Total_Bio': 'sum', 'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'
}).reset_index()

# Merge all
df = enroll_pin.merge(demo_pin, on=['pincode', 'district'], how='outer')
df = df.merge(bio_pin, on=['pincode', 'district'], how='outer').fillna(0)

# Calculate rates
df['Demo_Rate'] = np.where(df['Total_Enrollments'] > 0, df['Total_Demo'] / df['Total_Enrollments'], 0)
df['Bio_Rate'] = np.where(df['Total_Enrollments'] > 0, df['Total_Bio'] / df['Total_Enrollments'], 0)
df['Youth_Enrollment_Pct'] = np.where(df['Total_Enrollments'] > 0, 
                                       (df['age_0_5'] + df['age_5_17']) / df['Total_Enrollments'] * 100, 0)

print(f"  Merged Pincodes: {len(df)}")

# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================
print("\n[3] UNIVARIATE ANALYSIS")

for col in ['Total_Enrollments', 'Total_Demo', 'Total_Bio']:
    print(f"\n  {col}:")
    print(f"    Mean: {df[col].mean():,.1f} | Median: {df[col].median():,.1f}")
    print(f"    Std: {df[col].std():,.1f} | Skewness: {df[col].skew():.2f}")
    print(f"    Min: {df[col].min():,.0f} | Max: {df[col].max():,.0f}")

# ============================================================
# BIVARIATE ANALYSIS - CORRELATION
# ============================================================
print("\n[4] BIVARIATE ANALYSIS - CORRELATION MATRIX")

numeric_cols = ['Total_Enrollments', 'Total_Demo', 'Total_Bio', 'age_0_5', 'age_5_17', 
                'age_18_greater', 'Demo_Rate', 'Bio_Rate']
corr_matrix = df[numeric_cols].corr()

print("\n  Key Correlations:")
print(f"    Enrollment ↔ Demo: {corr_matrix.loc['Total_Enrollments', 'Total_Demo']:.3f}")
print(f"    Enrollment ↔ Bio:  {corr_matrix.loc['Total_Enrollments', 'Total_Bio']:.3f}")
print(f"    Demo ↔ Bio:        {corr_matrix.loc['Total_Demo', 'Total_Bio']:.3f}")

# Statistical significance
r, p = stats.pearsonr(df['Total_Enrollments'], df['Total_Bio'])
print(f"\n  Pearson Correlation (Enroll vs Bio): r={r:.3f}, p-value={p:.2e}")

# ============================================================
# TRIVARIATE ANALYSIS
# ============================================================
print("\n[5] TRIVARIATE ANALYSIS")

# District-wise aggregation for trivariate
district_df = df.groupby('district').agg({
    'Total_Enrollments': 'sum',
    'Total_Demo': 'sum',
    'Total_Bio': 'sum',
    'pincode': 'count'
}).rename(columns={'pincode': 'Pincodes'})

district_df['Demo_Rate'] = district_df['Total_Demo'] / district_df['Total_Enrollments']
district_df['Bio_Rate'] = district_df['Total_Bio'] / district_df['Total_Enrollments']

print("  Enrollment × Demo Rate × Bio Rate (by District):")
for district in district_df.nlargest(5, 'Total_Enrollments').index:
    row = district_df.loc[district]
    print(f"    {district}: Enroll={row['Total_Enrollments']:,.0f} | Demo={row['Demo_Rate']:.1f}x | Bio={row['Bio_Rate']:.1f}x")

# ============================================================
# ML ALGORITHM 1: K-MEANS CLUSTERING
# ============================================================
print("\n[6] K-MEANS CLUSTERING")

# Prepare features
features = ['Total_Enrollments', 'Total_Demo', 'Total_Bio', 'Demo_Rate', 'Bio_Rate']
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K using elbow method
inertias = []
silhouettes = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_k = K_range[np.argmax(silhouettes)]
print(f"  Optimal K (by silhouette): {optimal_k}")

# Final clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n  CLUSTER DISTRIBUTION:")
for cluster in sorted(df['Cluster'].unique()):
    count = len(df[df['Cluster'] == cluster])
    avg_enroll = df[df['Cluster'] == cluster]['Total_Enrollments'].mean()
    avg_bio = df[df['Cluster'] == cluster]['Bio_Rate'].mean()
    print(f"    Cluster {cluster}: {count} pincodes | Avg Enroll: {avg_enroll:.0f} | Avg Bio Rate: {avg_bio:.1f}x")

# Name clusters based on characteristics
cluster_labels = {}
for c in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == c]
    avg_enroll = cluster_data['Total_Enrollments'].mean()
    avg_bio_rate = cluster_data['Bio_Rate'].mean()
    
    if avg_enroll > df['Total_Enrollments'].median() and avg_bio_rate < df['Bio_Rate'].median():
        cluster_labels[c] = 'Stable'
    elif avg_bio_rate > df['Bio_Rate'].median() * 1.5:
        cluster_labels[c] = 'High-Stress'
    elif avg_enroll < df['Total_Enrollments'].quantile(0.25):
        cluster_labels[c] = 'Exclusion'
    else:
        cluster_labels[c] = 'Moderate'

df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
print(f"\n  Cluster Labels: {cluster_labels}")

# ============================================================
# ML ALGORITHM 2: HIERARCHICAL CLUSTERING
# ============================================================
print("\n[7] HIERARCHICAL CLUSTERING")

# Sample for hierarchical (too slow for full dataset)
sample_df = df.sample(min(200, len(df)), random_state=42)
X_sample = scaler.fit_transform(sample_df[features])

linkage_matrix = linkage(X_sample, method='ward')
hc_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')

print(f"  Hierarchical clusters created on {len(sample_df)} samples")
print(f"  Cluster distribution: {np.bincount(hc_labels)[1:]}")

# ============================================================
# ML ALGORITHM 3: RANDOM FOREST CLASSIFICATION
# ============================================================
print("\n[8] RANDOM FOREST CLASSIFICATION")

# Predict cluster membership
X_rf = df[features]
y_rf = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
rf_accuracy = (y_pred == y_test).mean()

print(f"  Model Accuracy: {rf_accuracy:.2%}")
print(f"\n  Feature Importance:")
for feat, imp in sorted(zip(features, rf_model.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat}: {imp:.3f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_rf, y_rf, cv=5)
print(f"\n  Cross-Validation Score: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# ============================================================
# ML ALGORITHM 4: LINEAR REGRESSION (Enrollment Prediction)
# ============================================================
print("\n[9] LINEAR REGRESSION - ENROLLMENT PREDICTION")

# Predict enrollments from demo and bio
X_reg = df[['Total_Demo', 'Total_Bio']].values
y_reg = df['Total_Enrollments'].values

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_reg = lr_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"  R-squared: {r2:.3f}")
print(f"  RMSE: {np.sqrt(mse):,.1f}")
print(f"\n  Coefficients:")
print(f"    Demo: {lr_model.coef_[0]:.4f}")
print(f"    Bio: {lr_model.coef_[1]:.4f}")
print(f"    Intercept: {lr_model.intercept_:.4f}")

# ============================================================
# ML ALGORITHM 5: GRADIENT BOOSTING REGRESSION
# ============================================================
print("\n[10] GRADIENT BOOSTING REGRESSION")

gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train_reg, y_train_reg)
y_pred_gb = gb_model.predict(X_test_reg)

gb_r2 = r2_score(y_test_reg, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_gb))

print(f"  R-squared: {gb_r2:.3f}")
print(f"  RMSE: {gb_rmse:,.1f}")

# ============================================================
# ML ALGORITHM 6: ISOLATION FOREST (Anomaly Detection)
# ============================================================
print("\n[11] ISOLATION FOREST - ANOMALY DETECTION")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(X_scaled)
anomalies = df[df['Anomaly'] == -1]

print(f"  Anomalies Detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
print(f"\n  Top Anomalous Pincodes:")
for _, row in anomalies.nlargest(5, 'Total_Bio').iterrows():
    print(f"    {int(row['pincode'])} ({row['district'][:12]}): Bio={row['Total_Bio']:,.0f}")

# ============================================================
# ML ALGORITHM 7: PCA (Dimensionality Reduction)
# ============================================================
print("\n[12] PCA - DIMENSIONALITY REDUCTION")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"  Explained Variance Ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {var:.2%}")
print(f"  Total Explained: {sum(pca.explained_variance_ratio_):.2%}")

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# ============================================================
# CUSTOM INDEX: AADHAAR INCLUSION SCORE
# ============================================================
print("\n[13] AADHAAR INCLUSION SCORE (Custom Index)")

# Normalize metrics to 0-100
scaler_mm = MinMaxScaler(feature_range=(0, 100))

df['Coverage_Score'] = scaler_mm.fit_transform(df[['Total_Enrollments']])
df['Accuracy_Score'] = 100 - scaler_mm.fit_transform(df[['Demo_Rate']])  # Lower is better
df['Reliability_Score'] = 100 - scaler_mm.fit_transform(df[['Bio_Rate']])  # Lower is better

# Composite Index (weighted)
df['Inclusion_Score'] = (
    df['Coverage_Score'] * 0.4 +
    df['Accuracy_Score'] * 0.3 +
    df['Reliability_Score'] * 0.3
).round(2)

# District-level inclusion score
district_inclusion = df.groupby('district')['Inclusion_Score'].mean().sort_values(ascending=False)

print("\n  TOP 10 DISTRICTS BY INCLUSION SCORE:")
for i, (district, score) in enumerate(district_inclusion.head(10).items(), 1):
    print(f"    {i}. {district}: {score:.1f}")

print("\n  BOTTOM 5 DISTRICTS (Need Intervention):")
for district, score in district_inclusion.tail(5).items():
    print(f"    {district}: {score:.1f}")

# ============================================================
# TIME-SERIES FORECASTING (Simple Moving Average)
# ============================================================
print("\n[14] TIME-SERIES FORECASTING")

monthly_enroll = enroll_df.groupby('month')['Total_Enrollments'].sum()
monthly_enroll = monthly_enroll.sort_index()

# Simple forecast using growth rate
if len(monthly_enroll) > 1:
    growth_rate = monthly_enroll.pct_change().mean()
    last_value = monthly_enroll.iloc[-1]
    forecast_3m = [last_value * (1 + growth_rate) ** i for i in range(1, 4)]
    
    print(f"  Average Monthly Growth: {growth_rate:.1%}")
    print(f"  3-Month Forecast:")
    for i, val in enumerate(forecast_3m, 1):
        print(f"    Month +{i}: {val:,.0f}")

# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n[15] STATISTICAL SIGNIFICANCE TESTS")

# Chi-square test for district vs cluster
contingency = pd.crosstab(df['district'], df['Cluster'])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
print(f"  Chi-Square (District vs Cluster): χ²={chi2:.2f}, p={p_val:.2e}")

# ANOVA: Enrollment across clusters
cluster_groups = [df[df['Cluster'] == c]['Total_Enrollments'] for c in df['Cluster'].unique()]
f_stat, anova_p = stats.f_oneway(*cluster_groups)
print(f"  ANOVA (Enrollment by Cluster): F={f_stat:.2f}, p={anova_p:.2e}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n[16] CREATING VISUALIZATIONS...")

fig = plt.figure(figsize=(28, 32))
fig.suptitle('UIDAI HACKATHON 2026 - ADVANCED ML ANALYSIS (ODISHA)', fontsize=20, fontweight='bold', y=0.99)

# 1. Correlation Heatmap
ax1 = fig.add_subplot(4, 3, 1)
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.2f', ax=ax1, square=True)
ax1.set_title('Correlation Matrix', fontweight='bold')

# 2. K-Means Clusters
ax2 = fig.add_subplot(4, 3, 2)
scatter = ax2.scatter(df['Total_Enrollments'], df['Total_Bio'], c=df['Cluster'], cmap='viridis', alpha=0.6)
ax2.set_xlabel('Total Enrollments')
ax2.set_ylabel('Total Biometric')
ax2.set_title('K-Means Clustering', fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Cluster')

# 3. Feature Importance
ax3 = fig.add_subplot(4, 3, 3)
importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
importance_df = importance_df.sort_values('Importance', ascending=True)
ax3.barh(importance_df['Feature'], importance_df['Importance'], color='#2E86AB')
ax3.set_title('Random Forest Feature Importance', fontweight='bold')

# 4. PCA Visualization
ax4 = fig.add_subplot(4, 3, 4)
scatter = ax4.scatter(df['PC1'], df['PC2'], c=df['Cluster'], cmap='viridis', alpha=0.6)
ax4.set_xlabel('Principal Component 1')
ax4.set_ylabel('Principal Component 2')
ax4.set_title('PCA Visualization', fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Cluster')

# 5. Inclusion Score by District
ax5 = fig.add_subplot(4, 3, 5)
top15_inclusion = district_inclusion.head(15)
colors = plt.cm.RdYlGn(top15_inclusion.values / 100)
ax5.barh(range(len(top15_inclusion)), top15_inclusion.values, color=colors)
ax5.set_yticks(range(len(top15_inclusion)))
ax5.set_yticklabels(top15_inclusion.index)
ax5.set_title('Aadhaar Inclusion Score', fontweight='bold')
ax5.set_xlabel('Score (0-100)')
ax5.invert_yaxis()

# 6. Cluster Distribution
ax6 = fig.add_subplot(4, 3, 6)
cluster_counts = df['Cluster_Label'].value_counts()
colors = {'Stable': '#27AE60', 'High-Stress': '#E74C3C', 'Exclusion': '#95A5A6', 'Moderate': '#F39C12'}
ax6.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%',
        colors=[colors.get(c, '#3498DB') for c in cluster_counts.index])
ax6.set_title('Pincode Cluster Distribution', fontweight='bold')

# 7. Regression: Actual vs Predicted
ax7 = fig.add_subplot(4, 3, 7)
ax7.scatter(y_test_reg, y_pred_gb, alpha=0.5, color='#2E86AB')
ax7.plot([0, max(y_test_reg)], [0, max(y_test_reg)], 'r--', lw=2)
ax7.set_xlabel('Actual Enrollments')
ax7.set_ylabel('Predicted Enrollments')
ax7.set_title(f'Gradient Boosting (R²={gb_r2:.3f})', fontweight='bold')

# 8. Anomaly Visualization
ax8 = fig.add_subplot(4, 3, 8)
normal = df[df['Anomaly'] == 1]
anomaly = df[df['Anomaly'] == -1]
ax8.scatter(normal['Total_Enrollments'], normal['Total_Bio'], alpha=0.5, label='Normal', color='#27AE60')
ax8.scatter(anomaly['Total_Enrollments'], anomaly['Total_Bio'], alpha=0.8, label='Anomaly', color='#E74C3C', s=100)
ax8.set_xlabel('Total Enrollments')
ax8.set_ylabel('Total Biometric')
ax8.set_title('Isolation Forest Anomalies', fontweight='bold')
ax8.legend()

# 9. Elbow Curve
ax9 = fig.add_subplot(4, 3, 9)
ax9.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax9.set_xlabel('Number of Clusters (K)')
ax9.set_ylabel('Inertia')
ax9.set_title('K-Means Elbow Curve', fontweight='bold')

# 10. Silhouette Scores
ax10 = fig.add_subplot(4, 3, 10)
ax10.bar(K_range, silhouettes, color='#27AE60')
ax10.set_xlabel('Number of Clusters (K)')
ax10.set_ylabel('Silhouette Score')
ax10.set_title('Silhouette Score by K', fontweight='bold')

# 11. District Trivariate
ax11 = fig.add_subplot(4, 3, 11)
top10_dist = district_df.nlargest(10, 'Total_Enrollments')
scatter = ax11.scatter(top10_dist['Demo_Rate'], top10_dist['Bio_Rate'], 
                        s=top10_dist['Total_Enrollments']/100, c=top10_dist['Total_Enrollments'],
                        cmap='YlOrRd', alpha=0.7, edgecolors='black')
for idx, row in top10_dist.iterrows():
    ax11.annotate(idx[:6], (row['Demo_Rate'], row['Bio_Rate']), fontsize=8)
ax11.set_xlabel('Demo Rate')
ax11.set_ylabel('Bio Rate')
ax11.set_title('Trivariate: Demo × Bio × Enrollment', fontweight='bold')
plt.colorbar(scatter, ax=ax11, label='Enrollments')

# 12. Model Comparison
ax12 = fig.add_subplot(4, 3, 12)
models = ['Linear Reg', 'Gradient Boost']
r2_scores = [r2, gb_r2]
ax12.bar(models, r2_scores, color=['#3498DB', '#27AE60'])
ax12.set_ylabel('R² Score')
ax12.set_title('Regression Model Comparison', fontweight='bold')
ax12.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax12.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(f"{data_dir}/advanced_ml_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: advanced_ml_analysis.png")

# ============================================================
# SUMMARY REPORT
# ============================================================
print("\n" + "="*70)
print(" EXECUTIVE SUMMARY")
print("="*70)
print(f"""
  DATA ANALYZED:
    Pincodes: {len(df)} | Districts: {df['district'].nunique()}
    Enrollments: {df['Total_Enrollments'].sum():,.0f}
    Demo Updates: {df['Total_Demo'].sum():,.0f}
    Bio Updates: {df['Total_Bio'].sum():,.0f}

  ML MODELS TRAINED:
    1. K-Means Clustering (K={optimal_k}, Silhouette={silhouettes[optimal_k-2]:.3f})
    2. Hierarchical Clustering (4 clusters)
    3. Random Forest (Accuracy: {rf_accuracy:.2%}, CV: {cv_scores.mean():.2%})
    4. Linear Regression (R²: {r2:.3f})
    5. Gradient Boosting (R²: {gb_r2:.3f})
    6. Isolation Forest (Anomalies: {len(anomalies)})
    7. PCA (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})

  KEY FINDINGS:
    - Strongest Correlation: Enrollment ↔ Bio ({corr_matrix.loc['Total_Enrollments', 'Total_Bio']:.3f})
    - Most Important Feature: {features[np.argmax(rf_model.feature_importances_)]}
    - Cluster Distribution: {dict(df['Cluster_Label'].value_counts())}
    
  TOP INCLUSION DISTRICTS: {', '.join(district_inclusion.head(3).index)}
  PRIORITY INTERVENTION: {', '.join(district_inclusion.tail(3).index)}
""")

print("="*70)
print(" POLICY RECOMMENDATIONS")
print("="*70)
print(f"""
  1. EXCLUSION CLUSTERS: Focus mobile camps on {len(df[df['Cluster_Label']=='Exclusion'])} pincodes
  2. HIGH-STRESS AREAS: Device upgrades in {len(df[df['Cluster_Label']=='High-Stress'])} pincodes
  3. ANOMALIES: Investigate {len(anomalies)} anomalous pincodes
  4. PRIORITY DISTRICTS: {', '.join(district_inclusion.tail(5).index)}
  5. FORECAST: Expected {growth_rate:.1%} monthly growth if trends continue
""")

# Save outputs
df.to_csv(f"{data_dir}/ml_pincode_analysis.csv", index=False)
district_inclusion.to_frame('Inclusion_Score').to_csv(f"{data_dir}/district_inclusion_scores.csv")

print("\n" + "="*70)
print(" OUTPUT FILES")
print("="*70)
print("""
  - advanced_ml_analysis.png (12 visualizations)
  - ml_pincode_analysis.csv
  - district_inclusion_scores.csv
""")
print("="*70)
print(" ANALYSIS COMPLETE!")
print("="*70)
