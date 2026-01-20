import pandas as pd
import os
import re

# Directory containing the files
data_dir = r"c:\Users\rohan\Aadharcard"

# Define file groups to merge
file_groups = {
    "aadhar_biometric_merged.csv": [
        "api_data_aadhar_biometric_0_500000.csv",
        "api_data_aadhar_biometric_500000_1000000.csv",
        "api_data_aadhar_biometric_1000000_1500000.csv",
        "api_data_aadhar_biometric_1500000_1861108.csv",
    ],
    "aadhar_demographic_merged.csv": [
        "api_data_aadhar_demographic_0_500000.csv",
        "api_data_aadhar_demographic_500000_1000000.csv",
        "api_data_aadhar_demographic_1000000_1500000.csv",
        "api_data_aadhar_demographic_1500000_2000000.csv",
        "api_data_aadhar_demographic_2000000_2071700.csv",
    ],
    "aadhar_enrolment_merged.csv": [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv",
    ],
}

# Official 30 Odisha districts (as specified by user)
ODISHA_DISTRICTS = [
    "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack",
    "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur",
    "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Keonjhar", "Khordha",
    "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada",
    "Puri", "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh"
]

# District name mapping for common misspellings/variations
DISTRICT_MAPPING = {
    # Keonjhar variations
    "kendujhar": "Keonjhar",
    "keonjhar(kendujhar)": "Keonjhar",
    "kendujhar(keonjhar)": "Keonjhar",
    "kendujhar (keonjhar)": "Keonjhar",
    "keonjhar (kendujhar)": "Keonjhar",
    # Subarnapur variations
    "sonepur": "Subarnapur",
    "subarnapur(sonepur)": "Subarnapur",
    "subarnapur (sonepur)": "Subarnapur",
    "sonepur(subarnapur)": "Subarnapur",
    "sonepur (subarnapur)": "Subarnapur",
    "sonapur": "Subarnapur",
    # Other variations
    "bolangir": "Balangir",
    "baleswar": "Balasore",
    "balesore": "Balasore",
    "baleshwar": "Balasore",
    "bargah": "Bargarh",
    "baudh": "Boudh",
    "cutak": "Cuttack",
    "cuttak": "Cuttack",
    "deogarh(debgarh)": "Deogarh",
    "debgarh": "Deogarh",
    "debagarh": "Deogarh",
    "denkanal": "Dhenkanal",
    "dhenknal": "Dhenkanal",
    "jagatsinghpur(jagatsinghapur)": "Jagatsinghpur",
    "jagatsinghapur": "Jagatsinghpur",
    "jharsugada": "Jharsuguda",
    "phulbani": "Kandhamal",
    "kandhamal(phulbani)": "Kandhamal",
    "phulbani(kandhamal)": "Kandhamal",
    "koraput(jeypore)": "Koraput",
    "jeypore": "Koraput",
    "malkangir": "Malkangiri",
    "mayurbhanja": "Mayurbhanj",
    "nabarangapur": "Nabarangpur",
    "nayagarh(nayagad)": "Nayagarh",
    "nayagad": "Nayagarh",
    "nowrangpur": "Nabarangpur",
    "noapara": "Nuapada",
    "nowapara": "Nuapada",
    "raygarh": "Rayagada",
    "raygada": "Rayagada",
    "sambalpur(hirakud)": "Sambalpur",
    "hirakud": "Sambalpur",
    "sundargarh(sundergarh)": "Sundargarh",
    "sundergarh": "Sundargarh",
    # Additional mappings found in data
    "khorda": "Khordha",
    "khordha  *": "Khordha",
    "jajapur": "Jajpur",
    "jajapur  *": "Jajpur",
    "anugul": "Angul",
    "anugal": "Angul",
    "anugul  *": "Angul",
    "kendrapara *": "Kendrapara",
    "bhadrak(r)": "Bhadrak",
    "balianta": "Khordha",
}

# State name variations to standardize
STATE_MAPPING = {
    "odisha": "Odisha",
    "orissa": "Odisha",
    "ODISHA": "Odisha",
    "ORISSA": "Odisha",
}


def clean_state(state):
    """Standardize state name"""
    if pd.isna(state):
        return state
    state_lower = str(state).strip().lower()
    return STATE_MAPPING.get(state_lower, str(state).strip().title())


def clean_district(district):
    """Clean and standardize district name"""
    if pd.isna(district):
        return district
    district_clean = str(district).strip().lower()
    # Check mapping first
    if district_clean in DISTRICT_MAPPING:
        return DISTRICT_MAPPING[district_clean]
    # Convert to title case
    return str(district).strip().title()


def clean_pincode(pincode):
    """Validate and clean pincode (must be 6 digits)"""
    if pd.isna(pincode):
        return None
    pincode_str = str(pincode).strip()
    # Remove any non-digit characters
    pincode_str = re.sub(r'\D', '', pincode_str)
    # Check if valid 6-digit pincode
    if len(pincode_str) == 6 and pincode_str.isdigit():
        return int(pincode_str)
    return None


def is_odisha_pincode(pincode):
    """Check if pincode is in Odisha range (75xxxx, 76xxxx, 77xxxx)"""
    if pd.isna(pincode):
        return False
    pincode_str = str(int(pincode))
    return pincode_str.startswith(('75', '76', '77'))


def build_pincode_district_map(df):
    """
    Build pincode-to-district mapping using majority voting.
    For each pincode, assign the district with the highest count.
    """
    # Group by pincode and district, count occurrences
    pincode_district_counts = df.groupby(['pincode', 'district']).size().reset_index(name='count')
    
    # For each pincode, get the district with highest count (majority voting)
    idx = pincode_district_counts.groupby('pincode')['count'].idxmax()
    pincode_map = pincode_district_counts.loc[idx][['pincode', 'district']]
    
    return dict(zip(pincode_map['pincode'], pincode_map['district']))


def apply_pincode_district_mapping(df, pincode_map):
    """Apply pincode-to-district mapping to standardize districts"""
    df['district'] = df['pincode'].map(pincode_map)
    return df


# ========================================
# STEP 1: Merge and deduplicate files
# ========================================
print("=" * 70)
print(" UIDAI DATA CLEANING & ODISHA ANALYSIS SCRIPT")
print("=" * 70)

merged_files = {}

for output_file, input_files in file_groups.items():
    print(f"\n{'=' * 60}")
    print(f"Processing: {output_file}")
    print('=' * 60)

    dfs = []
    total_rows = 0

    for file in input_files:
        file_path = os.path.join(data_dir, file)
        print(f"  Reading: {file}...", end=" ")
        df = pd.read_csv(file_path)
        rows = len(df)
        total_rows += rows
        print(f"({rows:,} rows)")
        dfs.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows after merge: {len(merged_df):,}")

    # Remove duplicates
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    after_dedup = len(merged_df)
    duplicates_removed = before_dedup - after_dedup
    print(f"  Duplicates removed: {duplicates_removed:,}")
    print(f"  Rows after deduplication: {after_dedup:,}")

    # Save merged file
    output_path = os.path.join(data_dir, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f"  [OK] Saved to: {output_file}")

    merged_files[output_file] = merged_df

# ========================================
# STEP 2: Clean and filter for Odisha
# ========================================
print(f"\n{'=' * 70}")
print(" CLEANING DATA & FILTERING FOR ODISHA")
print("=" * 70)

odisha_files = {
    "aadhar_biometric_merged.csv": "odisha_biometric_clean.csv",
    "aadhar_demographic_merged.csv": "odisha_demographic_clean.csv",
    "aadhar_enrolment_merged.csv": "odisha_enrolment_clean.csv"
}

# First pass: collect all Odisha data to build pincode-district mapping
print("\n[STEP 2a] Building Pincode-District Mapping using Majority Voting...")
all_odisha_data = []

for merged_file in odisha_files.keys():
    df = merged_files[merged_file].copy()
    df['state'] = df['state'].apply(clean_state)
    odisha_df = df[df['state'].str.lower() == 'odisha'].copy()
    odisha_df['district'] = odisha_df['district'].apply(clean_district)
    odisha_df['pincode'] = odisha_df['pincode'].apply(clean_pincode)
    odisha_df = odisha_df.dropna(subset=['pincode'])
    odisha_df['pincode'] = odisha_df['pincode'].astype(int)
    all_odisha_data.append(odisha_df[['pincode', 'district']])

# Combine all data for pincode-district mapping
combined_df = pd.concat(all_odisha_data, ignore_index=True)
pincode_district_map = build_pincode_district_map(combined_df)
print(f"  Built mapping for {len(pincode_district_map):,} unique pincodes")

# Check how many pincodes had conflicts
pincode_counts = combined_df.groupby('pincode')['district'].nunique()
conflicts = pincode_counts[pincode_counts > 1]
print(f"  Pincodes with multiple districts (resolved via majority voting): {len(conflicts):,}")

# Second pass: apply the mapping and save cleaned files
print("\n[STEP 2b] Applying Pincode-District Mapping and Saving Cleaned Files...")

for merged_file, odisha_output in odisha_files.items():
    print(f"\n{'-' * 50}")
    print(f"Processing: {merged_file}")
    print('-' * 50)

    df = merged_files[merged_file].copy()

    # Clean state names
    print("  Cleaning state names...")
    df['state'] = df['state'].apply(clean_state)

    # Filter for Odisha
    odisha_df = df[df['state'].str.lower() == 'odisha'].copy()
    print(f"  Rows for Odisha: {len(odisha_df):,}")

    if len(odisha_df) == 0:
        print("  [WARN] No data found for Odisha!")
        continue

    # Clean pincodes first (needed for mapping)
    print("  Cleaning pincodes...")
    odisha_df['pincode'] = odisha_df['pincode'].apply(clean_pincode)

    # Remove rows with invalid pincodes
    before_pincode = len(odisha_df)
    odisha_df = odisha_df.dropna(subset=['pincode'])
    odisha_df['pincode'] = odisha_df['pincode'].astype(int)
    print(f"  Rows with valid pincodes: {len(odisha_df):,} (removed {before_pincode - len(odisha_df):,})")

    # Apply pincode-district mapping (this fixes one-to-many issue)
    print("  Applying pincode-district mapping (majority voting)...")
    odisha_df = apply_pincode_district_mapping(odisha_df, pincode_district_map)

    # Remove rows where pincode wasn't in map (shouldn't happen, but safety check)
    odisha_df = odisha_df.dropna(subset=['district'])

    # Verify: check if any pincodes still have multiple districts
    verify_conflicts = odisha_df.groupby('pincode')['district'].nunique()
    remaining_conflicts = verify_conflicts[verify_conflicts > 1]
    if len(remaining_conflicts) > 0:
        print(f"  [WARN] {len(remaining_conflicts)} pincodes still have multiple districts")
    else:
        print("  [OK] All pincodes now map to exactly one district")

    # Get unique districts
    unique_districts = sorted(odisha_df['district'].unique())
    print(f"\n  Unique districts found: {len(unique_districts)}")

    # Check for unmapped districts
    unmapped = [d for d in unique_districts if d not in ODISHA_DISTRICTS]
    if unmapped:
        print(f"  [WARN] Unmapped districts: {unmapped}")
    else:
        print("  [OK] All districts are official Odisha districts")

    # District distribution
    print("\n  District Distribution (Top 10):")
    district_counts = odisha_df['district'].value_counts()
    for district in list(district_counts.index)[:10]:
        print(f"    {district}: {district_counts[district]:,}")
    if len(district_counts) > 10:
        print(f"    ... and {len(district_counts) - 10} more districts")

    # Save cleaned Odisha data
    output_path = os.path.join(data_dir, odisha_output)
    odisha_df.to_csv(output_path, index=False)
    print(f"\n  [OK] Saved cleaned Odisha data to: {odisha_output}")

# ========================================
# STEP 3: Save Pincode-District Lookup Table
# ========================================
print(f"\n{'=' * 70}")
print(" SAVING PINCODE-DISTRICT LOOKUP TABLE")
print("=" * 70)

lookup_df = pd.DataFrame(list(pincode_district_map.items()), columns=['pincode', 'district'])
lookup_df = lookup_df.sort_values('pincode')
lookup_path = os.path.join(data_dir, "odisha_pincode_district_lookup.csv")
lookup_df.to_csv(lookup_path, index=False)
print(f"  [OK] Saved lookup table with {len(lookup_df):,} pincodes to: odisha_pincode_district_lookup.csv")

# Show district distribution in lookup
print("\n  Districts in lookup table:")
lookup_dist = lookup_df['district'].value_counts()
for district in ODISHA_DISTRICTS:
    count = lookup_dist.get(district, 0)
    print(f"    {district}: {count} pincodes")

# ========================================
# STEP 4: Summary
# ========================================
print(f"\n{'=' * 70}")
print(" SUMMARY")
print("=" * 70)
print("\n Files created:")
print("   - aadhar_biometric_merged.csv (all data merged)")
print("   - aadhar_demographic_merged.csv (all data merged)")
print("   - aadhar_enrolment_merged.csv (all data merged)")
print("   - odisha_biometric_clean.csv (Odisha only, cleaned)")
print("   - odisha_demographic_clean.csv (Odisha only, cleaned)")
print("   - odisha_enrolment_clean.csv (Odisha only, cleaned)")
print("   - odisha_pincode_district_lookup.csv (Pincode-District mapping)")
print("\n Key improvements:")
print("   - Removed duplicates from merged data")
print("   - Standardized state and district names")
print("   - Used MAJORITY VOTING to fix pincode-district conflicts")
print("   - Each pincode now maps to exactly ONE district")
print("\n [OK] All processing complete!")
print("=" * 70)
