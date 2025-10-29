"""
Expand WHO weight data using LMS interpolation
WHO only provides weight-for-age up to 120 months (10 years)
This script fills in missing months using LMS interpolation
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.stats import norm

def calculate_percentile_from_lms(L, M, S, percentile):
    """Calculate percentile value from LMS parameters"""
    z = norm.ppf(percentile / 100.0)
    if L != 0:
        value = M * (1 + L * S * z) ** (1 / L)
    else:
        value = M * np.exp(S * z)
    return value

def expand_weight_data(input_file, output_file):
    """Expand weight data to include all months 0-120"""
    print(f"\nProcessing {input_file}...")

    df = pd.read_csv(input_file)
    print(f"Current data: {len(df)} rows, months {df['age_months'].min()}-{df['age_months'].max()}")

    # Create complete month range 0-120
    all_months = list(range(0, 121))

    # Interpolate L, M, S for all months
    f_L = interpolate.interp1d(df['age_months'], df['L'], kind='linear', fill_value='extrapolate')
    f_M = interpolate.interp1d(df['age_months'], df['M'], kind='linear', fill_value='extrapolate')
    f_S = interpolate.interp1d(df['age_months'], df['S'], kind='linear', fill_value='extrapolate')

    # Create new dataframe with all months
    expanded_data = []

    for month in all_months:
        L = float(f_L(month))
        M = float(f_M(month))
        S = float(f_S(month))

        # Calculate percentiles from LMS
        p3 = calculate_percentile_from_lms(L, M, S, 3)
        p15 = calculate_percentile_from_lms(L, M, S, 15)
        p50 = M  # 50th percentile is M
        p85 = calculate_percentile_from_lms(L, M, S, 85)
        p97 = calculate_percentile_from_lms(L, M, S, 97)

        # Calculate mean and SD
        mean = M
        sd = M * S  # Approximate SD

        expanded_data.append({
            'age_months': month,
            'L': L,
            'M': M,
            'S': S,
            'p3': p3,
            'p15': p15,
            'p50': p50,
            'p85': p85,
            'p97': p97,
            'mean': mean,
            'sd': sd
        })

    expanded_df = pd.DataFrame(expanded_data)
    expanded_df.to_csv(output_file, index=False)

    print(f"Expanded to {len(expanded_df)} rows covering months 0-120")
    print(f"Saved to {output_file}")

    # Verify a sample point
    sample_month = 60
    row = expanded_df[expanded_df['age_months'] == sample_month].iloc[0]
    print(f"\nSample at {sample_month} months (5 years):")
    print(f"  3rd percentile: {row['p3']:.2f} kg")
    print(f"  50th percentile: {row['p50']:.2f} kg")
    print(f"  97th percentile: {row['p97']:.2f} kg")

    return expanded_df

if __name__ == "__main__":
    print("=" * 60)
    print("Expanding WHO Weight Data (0-120 months / 0-10 years)")
    print("=" * 60)
    print("\nNote: WHO only provides weight-for-age data up to 10 years.")
    print("For children over 10, WHO recommends using BMI-for-age instead.")

    # Expand boys weight data
    boys_weight = expand_weight_data(
        'who_data/boys_weight_full.csv',
        'who_data/boys_weight_full_expanded.csv'
    )

    # Expand girls weight data
    girls_weight = expand_weight_data(
        'who_data/girls_weight_full.csv',
        'who_data/girls_weight_full_expanded.csv'
    )

    print("\n✓ Weight data expansion complete!")
    print("\nIMPORTANT: Weight-for-age charts should only be used up to 120 months (10 years)")
