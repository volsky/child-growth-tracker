"""
Process WHO BMI-for-age Excel files into CSV format
WHO BMI-for-age data covers 61-228 months (5-19 years)
"""

import pandas as pd
import numpy as np

def process_bmi_file(excel_file, output_csv):
    """Process WHO BMI Excel file to CSV with needed columns"""
    print(f"\nProcessing {excel_file}...")

    # Read the Excel file
    df = pd.read_excel(excel_file)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Age range: {df['Month'].min()}-{df['Month'].max()} months")

    # Create output dataframe with needed columns
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

    # Calculate mean and SD
    # For BMI, mean ≈ M and SD ≈ M * S
    output_df['mean'] = df['M']
    output_df['sd'] = df['M'] * df['S']

    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    # Show sample
    print(f"\nSample at 120 months (10 years):")
    sample = output_df[output_df['age_months'] == 120].iloc[0]
    print(f"  3rd percentile: {sample['p3']:.2f} kg/m²")
    print(f"  50th percentile: {sample['p50']:.2f} kg/m²")
    print(f"  97th percentile: {sample['p97']:.2f} kg/m²")

    return output_df

if __name__ == "__main__":
    print("=" * 60)
    print("Processing WHO BMI-for-age Data (61-228 months / 5-19 years)")
    print("=" * 60)

    # Process boys BMI data
    boys_bmi = process_bmi_file(
        'who_data/bmi-boys-who2007.xlsx',
        'who_data/boys_bmi_full.csv'
    )

    # Process girls BMI data
    girls_bmi = process_bmi_file(
        'who_data/bmi-girls-who2007.xlsx',
        'who_data/girls_bmi_full.csv'
    )

    print("\n✓ BMI data processing complete!")
    print("\nBMI-for-age covers ages 5-19 years (61-228 months)")
