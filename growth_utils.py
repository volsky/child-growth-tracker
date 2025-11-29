"""
Utility functions for child growth calculations.

This module contains pure utility functions that can be imported and tested
independently of the Streamlit application.
"""

import pandas as pd
import os
from scipy import interpolate
from scipy.stats import norm


def load_growth_data_from_csv(filename, data_source='WHO'):
    """Load growth data from CSV file"""
    folder = 'who_data' if data_source == 'WHO' else 'cdc_data'
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame()


def get_height_data(gender, data_source='WHO'):
    """
    Returns growth percentiles for height-for-age (in cm)
    WHO: 0-228 months (0-19 years)
    CDC: 24-240 months (2-20 years)
    """
    if data_source == 'WHO':
        if gender == "Male":
            df = load_growth_data_from_csv('boys_height_full.csv', 'WHO')
        else:  # Female
            df = load_growth_data_from_csv('girls_height_full.csv', 'WHO')
    else:  # CDC
        if gender == "Male":
            df = load_growth_data_from_csv('boys_height_cdc.csv', 'CDC')
        else:  # Female
            df = load_growth_data_from_csv('girls_height_cdc.csv', 'CDC')

    if df.empty:
        return df

    # Return relevant columns
    return df[['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']]


def get_weight_data(gender, data_source='WHO'):
    """
    Returns growth percentiles for weight-for-age (in kg)
    WHO: 0-120 months (0-10 years)
    CDC: 24-240 months (2-20 years)
    """
    if data_source == 'WHO':
        if gender == "Male":
            df = load_growth_data_from_csv('boys_weight_full.csv', 'WHO')
        else:  # Female
            df = load_growth_data_from_csv('girls_weight_full.csv', 'WHO')
    else:  # CDC
        if gender == "Male":
            df = load_growth_data_from_csv('boys_weight_cdc.csv', 'CDC')
        else:  # Female
            df = load_growth_data_from_csv('girls_weight_cdc.csv', 'CDC')

    if df.empty:
        return df

    # Return relevant columns
    return df[['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']]


def get_bmi_data(gender, data_source='WHO'):
    """
    Returns growth percentiles for BMI-for-age (in kg/m²)
    WHO: 61-228 months (5-19 years)
    CDC: 24-240 months (2-20 years)
    """
    if data_source == 'WHO':
        if gender == "Male":
            df = load_growth_data_from_csv('boys_bmi_full.csv', 'WHO')
        else:  # Female
            df = load_growth_data_from_csv('girls_bmi_full.csv', 'WHO')
    else:  # CDC
        if gender == "Male":
            df = load_growth_data_from_csv('boys_bmi_cdc.csv', 'CDC')
        else:  # Female
            df = load_growth_data_from_csv('girls_bmi_cdc.csv', 'CDC')

    if df.empty:
        return df

    # Return relevant columns
    return df[['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']]


def get_growth_statistics(gender, data_source='WHO'):
    """
    Returns growth statistics (mean and SD) for SDS calculation
    This includes height, weight, and BMI statistics loaded from CSV
    """
    # Load height, weight, and BMI data
    if data_source == 'WHO':
        if gender == "Male":
            height_df = load_growth_data_from_csv('boys_height_full.csv', 'WHO')
            weight_df = load_growth_data_from_csv('boys_weight_full.csv', 'WHO')
            bmi_df = load_growth_data_from_csv('boys_bmi_full.csv', 'WHO')
        else:
            height_df = load_growth_data_from_csv('girls_height_full.csv', 'WHO')
            weight_df = load_growth_data_from_csv('girls_weight_full.csv', 'WHO')
            bmi_df = load_growth_data_from_csv('girls_bmi_full.csv', 'WHO')
    else:  # CDC
        if gender == "Male":
            height_df = load_growth_data_from_csv('boys_height_cdc.csv', 'CDC')
            weight_df = load_growth_data_from_csv('boys_weight_cdc.csv', 'CDC')
            bmi_df = load_growth_data_from_csv('boys_bmi_cdc.csv', 'CDC')
        else:
            height_df = load_growth_data_from_csv('girls_height_cdc.csv', 'CDC')
            weight_df = load_growth_data_from_csv('girls_weight_cdc.csv', 'CDC')
            bmi_df = load_growth_data_from_csv('girls_bmi_cdc.csv', 'CDC')

    if height_df.empty or weight_df.empty:
        return pd.DataFrame()

    # Merge height and weight data
    stats = pd.merge(
        height_df[['age_months', 'mean', 'sd']].rename(columns={'mean': 'height_mean', 'sd': 'height_sd'}),
        weight_df[['age_months', 'mean', 'sd']].rename(columns={'mean': 'weight_mean', 'sd': 'weight_sd'}),
        on='age_months',
        how='outer'
    )

    # Merge BMI data if available
    if not bmi_df.empty:
        stats = pd.merge(
            stats,
            bmi_df[['age_months', 'mean', 'sd']].rename(columns={'mean': 'bmi_mean', 'sd': 'bmi_sd'}),
            on='age_months',
            how='outer'
        )

    stats = stats.sort_values('age_months')
    return stats


def calculate_z_score(age_months, measurement, measurement_type, gender, data_source='WHO'):
    """
    Calculate SDS for a given measurement
    measurement_type: 'height', 'weight', or 'bmi'
    data_source: 'WHO' or 'CDC'
    """
    stats = get_growth_statistics(gender, data_source)

    # Check if the measurement type column exists
    mean_col = f'{measurement_type}_mean'
    sd_col = f'{measurement_type}_sd'

    if mean_col not in stats.columns or sd_col not in stats.columns:
        return None, None, None, None

    # Remove rows with NaN values for the columns we're interpolating
    # This is crucial because outer merge in get_growth_statistics can create NaN rows
    valid_rows = stats['age_months'].notna() & stats[mean_col].notna() & stats[sd_col].notna()
    stats_clean = stats[valid_rows].copy()

    if stats_clean.empty:
        return None, None, None, None

    # Interpolate to get mean and SD for exact age
    f_mean = interpolate.interp1d(stats_clean['age_months'], stats_clean[mean_col],
                                   kind='linear', fill_value='extrapolate')
    f_sd = interpolate.interp1d(stats_clean['age_months'], stats_clean[sd_col],
                                 kind='linear', fill_value='extrapolate')

    mean = float(f_mean(age_months))
    sd = float(f_sd(age_months))

    # Check for NaN or invalid values
    if pd.isna(mean) or pd.isna(sd) or sd == 0:
        return None, None, None, None

    z_score = (measurement - mean) / sd

    # Check if z_score is NaN
    if pd.isna(z_score):
        return None, None, None, None

    # Calculate percentile
    percentile = norm.cdf(z_score) * 100

    return z_score, percentile, mean, sd


def interpret_z_score(z_score, measurement_type):
    """
    Provide interpretation of SDS based on WHO guidelines
    """
    if measurement_type == 'height':
        if z_score < -3:
            return "⚠️ Severely stunted", "danger"
        elif z_score < -2:
            return "⚠️ Stunted", "warning"
        elif z_score <= 2:
            return "✅ Normal", "success"
        elif z_score <= 3:
            return "⚠️ Tall", "warning"
        else:
            return "⚠️ Very tall", "danger"
    elif measurement_type == 'weight':
        if z_score < -3:
            return "⚠️ Severely underweight", "danger"
        elif z_score < -2:
            return "⚠️ Underweight", "warning"
        elif z_score <= 2:
            return "✅ Normal", "success"
        elif z_score <= 3:
            return "⚠️ Overweight", "warning"
        else:
            return "⚠️ Obese", "danger"
    else:  # bmi
        if z_score < -3:
            return "⚠️ Severely wasted", "danger"
        elif z_score < -2:
            return "⚠️ Wasted", "warning"
        elif z_score <= 1:
            return "✅ Normal", "success"
        elif z_score <= 2:
            return "⚠️ Overweight", "warning"
        else:
            return "⚠️ Obese", "danger"


def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height (cm) and weight (kg)"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi


def calculate_age_in_months(birth_date, measurement_date):
    """Calculate age in months between two dates"""
    years = measurement_date.year - birth_date.year
    months = measurement_date.month - birth_date.month
    days = measurement_date.day - birth_date.day

    total_months = years * 12 + months
    if days < 0:
        total_months -= 1

    return total_months
