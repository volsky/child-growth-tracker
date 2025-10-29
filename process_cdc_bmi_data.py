#!/usr/bin/env python3
"""
Process CDC BMI-for-age data from CSV to separate boys and girls files
CDC data covers ages 24-240 months (2-20 years)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_percentile_from_lms(L, M, S, percentile):
    """Calculate percentile value from LMS parameters"""
    z = norm.ppf(percentile / 100.0)
    if L != 0:
        value = M * (1 + L * S * z) ** (1 / L)
    else:
        value = M * np.exp(S * z)
    return value

def process_cdc_bmi_file(input_csv, output_boys_csv, output_girls_csv):
    """
    Process CDC BMI CSV file to separate boys and girls CSV files

    CDC file format:
    - Sex: 1 = Male, 2 = Female
    - Agemos: Age in months
    - L, M, S: LMS parameters
    - P3, P5, P10, P25, P50, P75, P85, P90, P95, P97: Percentiles
    Note: CDC doesn't have P15, so we calculate it from LMS parameters
    """
    # Read CDC BMI data
    df = pd.read_csv(input_csv)

    # Remove any duplicate header rows (where Sex == 'Sex')
    df = df[df['Sex'] != 'Sex'].copy()

    # Convert Sex to string for comparison
    df['Sex'] = df['Sex'].astype(str)

    # Convert numeric columns to float
    numeric_cols = ['Agemos', 'L', 'M', 'S', 'P3', 'P5', 'P10', 'P25', 'P50', 'P75', 'P85', 'P90', 'P95', 'P97']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Process boys data (Sex = '1')
    boys_df = df[df['Sex'] == '1'].copy()
    boys_output = pd.DataFrame()
    boys_output['age_months'] = boys_df['Agemos']
    boys_output['L'] = boys_df['L']
    boys_output['M'] = boys_df['M']
    boys_output['S'] = boys_df['S']
    boys_output['p3'] = boys_df['P3']
    # Calculate P15 from LMS parameters since CDC doesn't provide it
    boys_output['p15'] = boys_df.apply(lambda row: calculate_percentile_from_lms(row['L'], row['M'], row['S'], 15), axis=1)
    boys_output['p50'] = boys_df['P50']
    boys_output['p85'] = boys_df['P85']
    boys_output['p97'] = boys_df['P97']
    boys_output['mean'] = boys_df['M']
    boys_output['sd'] = boys_df['M'] * boys_df['S']

    # Save boys data
    boys_output.to_csv(output_boys_csv, index=False)
    print(f"Boys BMI data saved to {output_boys_csv}")
    print(f"  Rows: {len(boys_output)}")
    print(f"  Age range: {boys_output['age_months'].min()}-{boys_output['age_months'].max()} months")

    # Process girls data (Sex = '2')
    girls_df = df[df['Sex'] == '2'].copy()
    girls_output = pd.DataFrame()
    girls_output['age_months'] = girls_df['Agemos']
    girls_output['L'] = girls_df['L']
    girls_output['M'] = girls_df['M']
    girls_output['S'] = girls_df['S']
    girls_output['p3'] = girls_df['P3']
    # Calculate P15 from LMS parameters since CDC doesn't provide it
    girls_output['p15'] = girls_df.apply(lambda row: calculate_percentile_from_lms(row['L'], row['M'], row['S'], 15), axis=1)
    girls_output['p50'] = girls_df['P50']
    girls_output['p85'] = girls_df['P85']
    girls_output['p97'] = girls_df['P97']
    girls_output['mean'] = girls_df['M']
    girls_output['sd'] = girls_df['M'] * girls_df['S']

    # Save girls data
    girls_output.to_csv(output_girls_csv, index=False)
    print(f"Girls BMI data saved to {output_girls_csv}")
    print(f"  Rows: {len(girls_output)}")
    print(f"  Age range: {girls_output['age_months'].min()}-{girls_output['age_months'].max()} months")

if __name__ == "__main__":
    # Process CDC BMI data
    process_cdc_bmi_file(
        'cdc_data/bmiagerev.csv',
        'cdc_data/boys_bmi_cdc.csv',
        'cdc_data/girls_bmi_cdc.csv'
    )

    print("\nCDC BMI data processing complete!")
