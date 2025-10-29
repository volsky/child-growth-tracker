"""
Update WHO data from official WHO Excel files
This script processes the official WHO data files and creates CSV files
that match the format needed by the app.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def process_who_file(filename, output_filename):
    """Process a WHO Excel file and create a CSV with needed columns"""
    print(f"\nProcessing {filename}...")

    df = pd.read_excel(filename)

    # The WHO files have columns: Month, L, M, S, StDev, P01, P1, P3, P5, P10, P15, P25, P50, P75, P85, P90, P95, P97, P99, P999
    # We need: age_months, L, M, S, p3, p15, p50, p85, p97, mean, sd

    output_df = pd.DataFrame()
    output_df['age_months'] = df['Month']
    output_df['L'] = df['L']
    output_df['M'] = df['M']
    output_df['S'] = df['S']
    output_df['p3'] = df['P3']
    output_df['p15'] = df['P15']
    output_df['p50'] = df['P50']
    output_df['p85'] = df['P85']
    output_df['p97'] = df['P97']
    output_df['mean'] = df['M']  # M is the median/mean
    output_df['sd'] = df['StDev']  # Standard deviation

    # Save to CSV
    output_df.to_csv(output_filename, index=False)
    print(f"Saved to {output_filename}")
    print(f"Age range: {output_df['age_months'].min()}-{output_df['age_months'].max()} months")
    print(f"Total rows: {len(output_df)}")

    return output_df

def merge_0_5_and_5_19_data(file_0_5, file_5_19, output_file):
    """Merge 0-5 years and 5-19 years data"""
    print(f"\nMerging {file_0_5} and {file_5_19}...")

    df_0_5 = pd.read_csv(file_0_5)
    df_5_19 = pd.read_csv(file_5_19)

    # Remove overlapping ages (keep 0-5 data for months < 61, 5-19 data for >= 61)
    df_0_5_filtered = df_0_5[df_0_5['age_months'] < 61]

    # Combine
    combined = pd.concat([df_0_5_filtered, df_5_19], ignore_index=True)
    combined = combined.sort_values('age_months')

    # Save
    combined.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")
    print(f"Total age range: {combined['age_months'].min()}-{combined['age_months'].max()} months")
    print(f"Total rows: {len(combined)}")

    return combined

if __name__ == "__main__":
    print("="*60)
    print("Processing Official WHO Data Files")
    print("="*60)

    # Process 5-19 years data
    boys_height_5_19 = process_who_file(
        'who_boys_height_official.xlsx',
        'who_data/boys_height_5_19_official.csv'
    )

    girls_height_5_19 = process_who_file(
        'who_girls_height_official.xlsx',
        'who_data/girls_height_5_19_official.csv'
    )

    # For weight, we need to check if files exist
    try:
        boys_weight_5_10 = process_who_file(
            'who_boys_weight_official.xlsx',
            'who_data/boys_weight_5_10_official.csv'
        )
    except Exception as e:
        print(f"\nWarning: Could not process boys weight file: {e}")

    try:
        girls_weight_5_10 = process_who_file(
            'who_girls_weight_official.xlsx',
            'who_data/girls_weight_5_10_official.csv'
        )
    except Exception as e:
        print(f"\nWarning: Could not process girls weight file: {e}")

    print("\n" + "="*60)
    print("Merging with 0-5 years data (if available)")
    print("="*60)

    # Now merge with existing 0-5 data if we have it
    # For now, let's update the full files by replacing the 61+ month data
    # We'll keep the existing 0-60 month data from our current files

    print("\n" + "="*60)
    print("Verification - Check 204 months (17 years) for boys:")
    print("="*60)
    row_204 = boys_height_5_19[boys_height_5_19['age_months'] == 204].iloc[0]
    print(f"3rd percentile: {row_204['p3']:.1f} cm")
    print(f"10th percentile: calculated from LMS...")
    print(f"15th percentile: {row_204['p15']:.1f} cm")
    print(f"50th percentile: {row_204['p50']:.1f} cm")
    print(f"97th percentile: {row_204['p97']:.1f} cm")
    print(f"Mean: {row_204['mean']:.1f} cm")
    print(f"SD: {row_204['sd']:.2f} cm")

    # Calculate 10th percentile from LMS
    L, M, S = row_204['L'], row_204['M'], row_204['S']
    z_10 = norm.ppf(0.10)
    if L != 0:
        p10 = M * (1 + L * S * z_10) ** (1/L)
    else:
        p10 = M * np.exp(S * z_10)
    print(f"10th percentile (calculated): {p10:.1f} cm")

    print("\n✓ WHO data processing complete!")
