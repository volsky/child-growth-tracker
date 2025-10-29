"""
CDC Growth Charts Data Processing
Converts CDC growth chart data to the same format as WHO data
CDC data is for ages 2-20 years (24-240 months)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def process_cdc_height_data():
    """Process CDC stature-for-age data"""
    # Read the CDC data
    df = pd.read_csv('cdc_statage.csv')

    # Separate boys (Sex=1) and girls (Sex=2)
    boys_df = df[df['Sex'] == 1].copy()
    girls_df = df[df['Sex'] == 2].copy()

    # Rename columns to match WHO format
    boys_df = boys_df.rename(columns={
        'Agemos': 'age_months',
        'P3': 'p3',
        'P50': 'p50',
        'P97': 'p97'
    })

    girls_df = girls_df.rename(columns={
        'Agemos': 'age_months',
        'P3': 'p3',
        'P50': 'p50',
        'P97': 'p97'
    })

    # Calculate p15 and p85 from LMS parameters
    for df_gender in [boys_df, girls_df]:
        p15_values = []
        p85_values = []
        mean_values = []
        sd_values = []

        for _, row in df_gender.iterrows():
            L, M, S = row['L'], row['M'], row['S']

            # Calculate 15th and 85th percentiles using LMS method
            z15 = norm.ppf(0.15)
            z85 = norm.ppf(0.85)

            if L != 0:
                p15 = M * (1 + L * S * z15) ** (1/L)
                p85 = M * (1 + L * S * z85) ** (1/L)
            else:
                p15 = M * np.exp(S * z15)
                p85 = M * np.exp(S * z85)

            p15_values.append(p15)
            p85_values.append(p85)
            mean_values.append(M)

            # Calculate SD from LMS (approximation)
            sd = M * S
            sd_values.append(sd)

        df_gender['p15'] = p15_values
        df_gender['p85'] = p85_values
        df_gender['mean'] = mean_values
        df_gender['sd'] = sd_values

    # Select only the columns we need
    output_cols = ['age_months', 'L', 'M', 'S', 'p3', 'p15', 'p50', 'p85', 'p97', 'mean', 'sd']

    boys_output = boys_df[output_cols]
    girls_output = girls_df[output_cols]

    # Save to CSV
    boys_output.to_csv('cdc_data/boys_height_cdc.csv', index=False)
    girls_output.to_csv('cdc_data/girls_height_cdc.csv', index=False)

    print(f"Boys height data: {len(boys_output)} rows saved")
    print(f"Girls height data: {len(girls_output)} rows saved")
    print(f"Age range: {boys_output['age_months'].min()}-{boys_output['age_months'].max()} months")

    # Check 17-year-old boy data
    boys_204 = boys_output[boys_output['age_months'] == 204]
    if not boys_204.empty:
        print(f"\n17-year-old boy (204 months) - CDC data:")
        print(f"  3rd percentile: {boys_204['p3'].values[0]:.1f} cm")
        print(f"  50th percentile: {boys_204['p50'].values[0]:.1f} cm")
        print(f"  97th percentile: {boys_204['p97'].values[0]:.1f} cm")

def process_cdc_weight_data():
    """Process CDC weight-for-age data"""
    # Read the CDC data
    df = pd.read_csv('cdc_wtage.csv')

    # Separate boys (Sex=1) and girls (Sex=2)
    boys_df = df[df['Sex'] == 1].copy()
    girls_df = df[df['Sex'] == 2].copy()

    # Rename columns to match WHO format
    boys_df = boys_df.rename(columns={
        'Agemos': 'age_months',
        'P3': 'p3',
        'P50': 'p50',
        'P97': 'p97'
    })

    girls_df = girls_df.rename(columns={
        'Agemos': 'age_months',
        'P3': 'p3',
        'P50': 'p50',
        'P97': 'p97'
    })

    # Calculate p15 and p85 from LMS parameters
    for df_gender in [boys_df, girls_df]:
        p15_values = []
        p85_values = []
        mean_values = []
        sd_values = []

        for _, row in df_gender.iterrows():
            L, M, S = row['L'], row['M'], row['S']

            # Calculate 15th and 85th percentiles using LMS method
            z15 = norm.ppf(0.15)
            z85 = norm.ppf(0.85)

            if L != 0:
                p15 = M * (1 + L * S * z15) ** (1/L)
                p85 = M * (1 + L * S * z85) ** (1/L)
            else:
                p15 = M * np.exp(S * z15)
                p85 = M * np.exp(S * z85)

            p15_values.append(p15)
            p85_values.append(p85)
            mean_values.append(M)

            # Calculate SD from LMS (approximation)
            sd = M * S
            sd_values.append(sd)

        df_gender['p15'] = p15_values
        df_gender['p85'] = p85_values
        df_gender['mean'] = mean_values
        df_gender['sd'] = sd_values

    # Select only the columns we need
    output_cols = ['age_months', 'L', 'M', 'S', 'p3', 'p15', 'p50', 'p85', 'p97', 'mean', 'sd']

    boys_output = boys_df[output_cols]
    girls_output = girls_df[output_cols]

    # Save to CSV
    boys_output.to_csv('cdc_data/boys_weight_cdc.csv', index=False)
    girls_output.to_csv('cdc_data/girls_weight_cdc.csv', index=False)

    print(f"\nBoys weight data: {len(boys_output)} rows saved")
    print(f"Girls weight data: {len(girls_output)} rows saved")
    print(f"Age range: {boys_output['age_months'].min()}-{boys_output['age_months'].max()} months")

if __name__ == "__main__":
    import os

    # Create cdc_data directory if it doesn't exist
    if not os.path.exists('cdc_data'):
        os.makedirs('cdc_data')

    print("Processing CDC Growth Charts data...")
    print("="*50)

    process_cdc_height_data()
    process_cdc_weight_data()

    print("\n" + "="*50)
    print("CDC data processing complete!")
    print("Files created in cdc_data/ directory")
