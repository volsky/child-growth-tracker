import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from scipy import interpolate
from scipy.stats import norm
import os
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Growth Standards from CSV files
# Data sources:
# - WHO Child Growth Standards (2006) and Growth Reference (2007) - Global/multi-ethnic
# - CDC Growth Charts (2000) - US population

@st.cache_data
def load_growth_data_from_csv(filename, data_source='WHO'):
    """Load growth data from CSV file with caching"""
    folder = 'who_data' if data_source == 'WHO' else 'cdc_data'
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        st.error(f"{data_source} data file not found: {filename}")
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

def get_default_measurements(age_months, gender, data_source='WHO'):
    """Get 50th percentile (median) values for height and weight based on age and gender"""
    # Get height data
    height_data = get_height_data(gender, data_source)
    weight_data = get_weight_data(gender, data_source)

    # Default fallback values
    default_height = 75.0
    default_weight = 10.0

    if not height_data.empty and age_months >= height_data['age_months'].min() and age_months <= height_data['age_months'].max():
        # Interpolate to get 50th percentile for exact age
        f_height = interpolate.interp1d(height_data['age_months'], height_data['p50'],
                                        kind='linear', fill_value='extrapolate')
        default_height = float(f_height(age_months))

    if not weight_data.empty and age_months >= weight_data['age_months'].min() and age_months <= weight_data['age_months'].max():
        # Interpolate to get 50th percentile for exact age
        f_weight = interpolate.interp1d(weight_data['age_months'], weight_data['p50'],
                                        kind='linear', fill_value='extrapolate')
        default_weight = float(f_weight(age_months))

    return round(default_height, 1), round(default_weight, 1)

def generate_pdf_report(child_info, today_measurement, data_points, height_fig, weight_fig, bmi_fig=None, data_source='WHO', measurements_table=None):
    """Generate PDF report with SDS, measurements table, and charts"""
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Summary and SDS
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Child Growth Report', fontsize=16, fontweight='bold')

        # Add child information
        info_text = f"""
Child Information:
─────────────────────────────────
Gender: {child_info['gender']}
Birth Date: {child_info['birth_date'].strftime('%Y-%m-%d')}
"""
        plt.text(0.1, 0.9, info_text, fontsize=12, verticalalignment='top',
                fontfamily='monospace', transform=fig.transFigure)

        if today_measurement:
            # Calculate SDS
            height_z, height_perc, height_mean, height_sd = calculate_z_score(
                today_measurement['age_months'], today_measurement['height'], 'height', today_measurement['gender'], data_source
            )
            weight_z, weight_perc, weight_mean, weight_sd = calculate_z_score(
                today_measurement['age_months'], today_measurement['weight'], 'weight', today_measurement['gender'], data_source
            )

            # Build height analysis text
            height_text = f"  Measurement: {today_measurement['height']:.1f} cm\n"
            if height_z is not None:
                height_interp, _ = interpret_z_score(height_z, 'height')
                height_text += f"""  SDS: {height_z:.2f}
  Percentile: {height_perc:.1f}%
  Interpretation: {height_interp}
  Expected mean: {height_mean:.1f} cm
  Standard deviation: {height_sd:.2f} cm"""
            else:
                height_text += "  SDS: Not available for this age/source"

            # Build weight analysis text
            weight_text = f"  Measurement: {today_measurement['weight']:.1f} kg\n"
            if weight_z is not None:
                weight_interp, _ = interpret_z_score(weight_z, 'weight')
                weight_text += f"""  SDS: {weight_z:.2f}
  Percentile: {weight_perc:.1f}%
  Interpretation: {weight_interp}
  Expected mean: {weight_mean:.1f} kg
  Standard deviation: {weight_sd:.2f} kg"""
            else:
                weight_text += "  SDS: Not available for this age/source"

            # Check if BMI data is available and calculate BMI SDS
            bmi_text = ""
            if 'bmi' in today_measurement and today_measurement['bmi'] is not None:
                bmi_available = False
                if data_source == 'WHO' and today_measurement['age_months'] >= 61:
                    bmi_available = True
                elif data_source == 'CDC' and today_measurement['age_months'] >= 24:
                    bmi_available = True

                if bmi_available:
                    bmi_z, bmi_perc, bmi_mean, bmi_sd = calculate_z_score(
                        today_measurement['age_months'], today_measurement['bmi'], 'bmi', today_measurement['gender'], data_source
                    )
                    if bmi_z is not None:
                        bmi_interp, _ = interpret_z_score(bmi_z, 'bmi')
                        bmi_text = f"""
BMI Analysis:
  Measurement: {today_measurement['bmi']:.2f} kg/m²
  SDS: {bmi_z:.2f}
  Percentile: {bmi_perc:.1f}%
  Interpretation: {bmi_interp}
  Expected mean: {bmi_mean:.2f} kg/m²
  Standard deviation: {bmi_sd:.2f} kg/m²
"""

            # Add today's measurements
            today_text = f"""
Today's Measurement ({today_measurement['date'].strftime('%Y-%m-%d')}):
─────────────────────────────────
Age: {today_measurement['age_months']} months ({today_measurement['age_months'] // 12} years, {today_measurement['age_months'] % 12} months)

Height Analysis:
{height_text}

Weight Analysis:
{weight_text}
{bmi_text}"""
            plt.text(0.1, 0.72, today_text, fontsize=10, verticalalignment='top',
                    fontfamily='monospace', transform=fig.transFigure)

        # Add historical data summary
        if data_points:
            try:
                hist_text = f"\nHistorical Measurements:\n{'─'*37}\n"
                hist_text += f"Total measurements: {len(data_points)}\n"
                df = pd.DataFrame(data_points)
                hist_text += f"Age range: {df['age'].min()}-{df['age'].max()} months\n"
                # Handle height range with possible None values
                valid_heights = df['height'].dropna()
                if not valid_heights.empty:
                    hist_text += f"Height range: {valid_heights.min():.1f}-{valid_heights.max():.1f} cm\n"
                else:
                    hist_text += "Height range: No height data available\n"
                # Handle weight range with possible None values
                valid_weights = df['weight'].dropna()
                if not valid_weights.empty:
                    hist_text += f"Weight range: {valid_weights.min():.1f}-{valid_weights.max():.1f} kg\n"
                else:
                    hist_text += "Weight range: No weight data available\n"
                # Add BMI range if available
                if 'bmi' in df.columns and df['bmi'].notna().any():
                    hist_text += f"BMI range: {df['bmi'].min():.2f}-{df['bmi'].max():.2f} kg/m²\n"

                plt.text(0.1, 0.25, hist_text, fontsize=10, verticalalignment='top',
                        fontfamily='monospace', transform=fig.transFigure)
            except Exception as e:
                logger.error(f"Error generating historical data summary in PDF: {e}")

        # Add footer
        data_source_text = "WHO Child Growth Standards (2006) and Growth Reference (2007)" if data_source == "WHO" else "CDC Growth Charts (2000) for US population"
        footer = f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nData source: {data_source_text}"
        plt.text(0.1, 0.05, footer, fontsize=8, verticalalignment='bottom',
                style='italic', transform=fig.transFigure)

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Measurements Table with SDS
        if measurements_table is not None and not measurements_table.empty:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('All Measurements with SDS', fontsize=14, fontweight='bold')

            # Remove the 'Today' emoji column for PDF
            table_for_pdf = measurements_table.copy()
            if 'Today' in table_for_pdf.columns:
                table_for_pdf = table_for_pdf.drop('Today', axis=1)

            # Create table
            ax = fig.add_subplot(111)
            ax.axis('off')

            # Format the table data
            table_data = []
            headers = list(table_for_pdf.columns)
            table_data.append(headers)

            for _, row in table_for_pdf.iterrows():
                formatted_row = []
                for col in headers:
                    val = row[col]
                    if pd.isna(val):
                        formatted_row.append('-')
                    elif col == 'Date':
                        formatted_row.append(str(val))
                    elif col == 'Age (months)':
                        formatted_row.append(str(int(val)))
                    elif 'SDS' in col or 'BMI' == col:
                        formatted_row.append(f'{val:.2f}' if not pd.isna(val) else '-')
                    elif '%ile' in col:
                        formatted_row.append(f'{val:.1f}' if not pd.isna(val) else '-')
                    else:
                        formatted_row.append(f'{val:.1f}' if not pd.isna(val) else '-')
                table_data.append(formatted_row)

            # Create matplotlib table
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 2)

            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

            # Add note
            note_text = f"Data source: {data_source}\nSDS calculated using {data_source} growth standards"
            plt.text(0.5, 0.02, note_text, fontsize=8, ha='center',
                    style='italic', transform=fig.transFigure)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 3+: Charts
        # Export plotly charts to images
        if height_fig:
            img_bytes = height_fig.to_image(format="png", width=1000, height=700)
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(BytesIO(img_bytes))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Height-for-Age Chart', fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        if weight_fig:
            img_bytes = weight_fig.to_image(format="png", width=1000, height=700)
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(BytesIO(img_bytes))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Weight-for-Age Chart', fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 4: BMI Chart (if available)
        if bmi_fig:
            img_bytes = bmi_fig.to_image(format="png", width=1000, height=700)
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(BytesIO(img_bytes))
            plt.imshow(img)
            plt.axis('off')
            plt.title('BMI-for-Age Chart', fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    buffer.seek(0)
    return buffer

# Streamlit App
st.set_page_config(
    page_title="Child Growth Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Child Growth Tracker with WHO Standards (0-19 years)"
    }
)
st.title("👶 Child Growth Tracker")

st.markdown("""
Track child growth with WHO or CDC growth standards and reference data.
Charts automatically switch between age ranges (0-5 years or 5-19 years) based on your child's age for better visualization.
**Mobile-friendly interface for easy tracking on the go!**
""")

# Initialize session state for storing data points
if 'data_points' not in st.session_state:
    st.session_state.data_points = []
if 'child_info' not in st.session_state:
    st.session_state.child_info = None
if 'today_measurement' not in st.session_state:
    st.session_state.today_measurement = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'WHO'

# Main content - Child Information and Data Entry
st.header("👤 Child Information")

col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])

with col1:
    child_gender = st.selectbox("Gender", ["Male", "Female"], key="child_gender")

with col2:
    child_birth_date = st.date_input("Birth Date",
                                      value=date.today().replace(year=date.today().year - 2),
                                      min_value=date.today().replace(year=date.today().year - 20),
                                      max_value=date.today(),
                                      key="birth_date")

with col3:
    # Calculate recommended data source based on child's age
    recommended_source = "WHO"  # Default
    force_cdc = False  # Flag to force CDC and disable selection

    if st.session_state.child_info:
        child_age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], date.today())

        # Force CDC if WHO data is not available (weight over 10 years)
        if child_age_months > 120:  # Over 10 years - WHO weight data not available
            force_cdc = True
            recommended_source = "CDC"

    # Get current selection or use recommended
    if force_cdc:
        default_index = 1  # CDC
        st.session_state.data_source = "CDC"
        data_source = st.selectbox(
            "Growth Charts",
            ["WHO", "CDC"],
            index=default_index,
            disabled=True,
            key="data_source_select",
            help="Data source locked to CDC because WHO doesn't provide weight data beyond 10 years"
        )
    else:
        # Normal selection
        if 'data_source' not in st.session_state or st.session_state.child_info:
            default_index = 0 if recommended_source == "WHO" else 1
        else:
            default_index = 0 if st.session_state.data_source == "WHO" else 1

        data_source = st.selectbox(
            "Growth Charts",
            ["WHO", "CDC"],
            index=default_index,
            key="data_source_select",
            help="WHO: Global multi-ethnic population (0-19 years)\nCDC: US population (2-20 years)"
        )
        st.session_state.data_source = data_source

with col4:
    st.markdown("&nbsp;")  # Spacer to align button
    st.markdown("&nbsp;")  # Spacer to align button
    if st.button("💾 Save", use_container_width=True, type="primary"):
        st.session_state.child_info = {
            'gender': child_gender,
            'birth_date': child_birth_date
        }
        st.success("Child info saved!")
        st.rerun()

st.divider()

# Today's Measurement Section
st.header("📅 Today's Measurement")

if st.session_state.child_info:
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])

    with col1:
        measurement_date = st.date_input("Date",
                                        value=date.today(),
                                        max_value=date.today(),
                                        key="measurement_date")

    # Calculate age and get default values for height and weight
    age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], measurement_date)
    default_height, default_weight = get_default_measurements(age_months, st.session_state.child_info['gender'], st.session_state.data_source)

    with col2:
        measurement_height = st.number_input("Height (cm)",
                                            min_value=0.0,
                                            max_value=250.0,
                                            value=default_height,
                                            step=0.1,
                                            key="measurement_height")

    with col3:
        measurement_weight = st.number_input("Weight (kg)",
                                            min_value=0.0,
                                            max_value=200.0,
                                            value=default_weight,
                                            step=0.1,
                                            key="measurement_weight")

    with col4:
        st.markdown("&nbsp;")  # Spacer to align button
        st.markdown("&nbsp;")  # Spacer to align button
        if st.button("➕ Add", use_container_width=True, type="primary"):
            # Calculate BMI
            bmi = calculate_bmi(measurement_height, measurement_weight)

            # Save today's measurement
            st.session_state.today_measurement = {
                'date': measurement_date,
                'age_months': age_months,
                'height': measurement_height,
                'weight': measurement_weight,
                'bmi': bmi,
                'gender': st.session_state.child_info['gender']
            }

            st.success("Measurement added!")
            st.rerun()

    # Display current age info
    st.info(f"Age at measurement: {age_months} months ({age_months // 12} years, {age_months % 12} months)")
else:
    st.warning("Please save child info first")

st.divider()

# Import Data Section - Available anytime
st.header("📥 Import Data")
uploaded_file = st.file_uploader("Import growth data from file", type=['txt'], key="import_file_early")
if uploaded_file is not None:
    # Use file ID to prevent re-processing the same file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # Check if this file has already been processed
    if 'last_imported_file' not in st.session_state or st.session_state.last_imported_file != file_id:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            import_data = yaml.safe_load(file_content)

            # Load child info
            if 'child_info' in import_data:
                st.session_state.child_info = {
                    'gender': import_data['child_info']['gender'],
                    'birth_date': datetime.strptime(import_data['child_info']['birth_date'], '%Y-%m-%d').date()
                }

            # Load data points
            if 'data_points' in import_data:
                st.session_state.data_points = []
                for point in import_data['data_points']:
                    point_copy = point.copy()
                    if 'date' in point_copy:
                        point_copy['date'] = datetime.strptime(point_copy['date'], '%Y-%m-%d').date()
                    st.session_state.data_points.append(point_copy)

            # Handle today's measurement based on date
            if 'today_measurement' in import_data and import_data['today_measurement'] is not None:
                today_copy = import_data['today_measurement'].copy()
                if 'date' in today_copy:
                    today_copy['date'] = datetime.strptime(today_copy['date'], '%Y-%m-%d').date()

                    # Check if the saved "today's measurement" is from a previous date
                    if today_copy['date'] < date.today():
                        # Add it to historical data points
                        st.session_state.data_points.append({
                            'date': today_copy['date'],
                            'gender': today_copy['gender'],
                            'age': today_copy['age_months'],
                            'height': today_copy['height'],
                            'weight': today_copy['weight'],
                            'bmi': today_copy.get('bmi')
                        })
                        st.session_state.today_measurement = None
                    else:
                        # Keep it as today's measurement if it's from today
                        st.session_state.today_measurement = today_copy
            else:
                st.session_state.today_measurement = None

            # Mark this file as processed
            st.session_state.last_imported_file = file_id

            st.success("✅ Data imported successfully! Child info and historical measurements loaded.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error importing file: {str(e)}")

st.divider()

# Main content - Today's Measurement SDS
if st.session_state.today_measurement:
    st.header("📈 Today's Measurement Analysis")

    today = st.session_state.today_measurement
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📏 Height Analysis")
        height_z, height_perc, height_mean, height_sd = calculate_z_score(
            today['age_months'], today['height'], 'height', today['gender'], st.session_state.data_source
        )

        st.metric("Height", f"{today['height']:.1f} cm")

        if height_z is not None:
            height_interp, height_status = interpret_z_score(height_z, 'height')
            st.metric("SDS", f"{height_z:.2f}")
            st.metric("Percentile", f"{height_perc:.1f}%")

            if height_status == "success":
                st.success(height_interp)
            elif height_status == "warning":
                st.warning(height_interp)
            else:
                st.error(height_interp)

            with st.expander("ℹ️ Details"):
                st.write(f"**Expected mean:** {height_mean:.1f} cm")
                st.write(f"**Standard deviation:** {height_sd:.2f} cm")
                st.write(f"**Age:** {today['age_months']} months")
        else:
            st.warning("⚠️ SDS not available")
            st.info(f"Height data not available for {today['age_months']} months in {st.session_state.data_source} database")

    with col2:
        st.subheader("⚖️ Weight Analysis")

        # Check if weight data is available for this age and data source
        # Automatically switch to CDC if WHO doesn't have data for this age
        weight_data_source = st.session_state.data_source
        if st.session_state.data_source == 'WHO' and today['age_months'] > 120:
            weight_data_source = 'CDC'
            st.info(f"ℹ️ Automatically using CDC data (WHO weight data only available up to 10 years)")

        if weight_data_source == 'CDC' and today['age_months'] < 24:
            st.warning("⚠️ CDC weight-for-age data is only available from 2 years (24 months)")
            st.info("Please use WHO data source for children under 2 years")
        else:
            weight_z, weight_perc, weight_mean, weight_sd = calculate_z_score(
                today['age_months'], today['weight'], 'weight', today['gender'], weight_data_source
            )

            st.metric("Weight", f"{today['weight']:.1f} kg")

            if weight_z is not None:
                weight_interp, weight_status = interpret_z_score(weight_z, 'weight')
                st.metric("SDS", f"{weight_z:.2f}")
                st.metric("Percentile", f"{weight_perc:.1f}%")

                if weight_status == "success":
                    st.success(weight_interp)
                elif weight_status == "warning":
                    st.warning(weight_interp)
                else:
                    st.error(weight_interp)

                with st.expander("ℹ️ Details"):
                    st.write(f"**Expected mean:** {weight_mean:.1f} kg")
                    st.write(f"**Standard deviation:** {weight_sd:.2f} kg")
                    st.write(f"**Age:** {today['age_months']} months")
            else:
                st.warning("⚠️ SDS not available")
                st.info(f"Weight data not available for {today['age_months']} months in {weight_data_source} database")

    with col3:
        st.subheader("📊 BMI Analysis")

        # BMI availability depends on data source and age
        # WHO: 61-228 months (5-19 years)
        # CDC: 24-240 months (2-20 years)
        if 'bmi' in today and today['bmi'] is not None:
            bmi_available = False
            if st.session_state.data_source == 'WHO' and today['age_months'] >= 61:
                bmi_available = True
            elif st.session_state.data_source == 'CDC' and today['age_months'] >= 24:
                bmi_available = True

            if bmi_available:
                bmi_z, bmi_perc, bmi_mean, bmi_sd = calculate_z_score(
                    today['age_months'], today['bmi'], 'bmi', today['gender'], st.session_state.data_source
                )

                if bmi_z is not None:
                    bmi_interp, bmi_status = interpret_z_score(bmi_z, 'bmi')

                    st.metric("BMI", f"{today['bmi']:.2f} kg/m²")
                    st.metric("SDS", f"{bmi_z:.2f}")
                    st.metric("Percentile", f"{bmi_perc:.1f}%")

                    if bmi_status == "success":
                        st.success(bmi_interp)
                    elif bmi_status == "warning":
                        st.warning(bmi_interp)
                    else:
                        st.error(bmi_interp)

                    with st.expander("ℹ️ Details"):
                        st.write(f"**Expected mean:** {bmi_mean:.2f} kg/m²")
                        st.write(f"**Standard deviation:** {bmi_sd:.2f} kg/m²")
                        st.write(f"**Age:** {today['age_months']} months")
                else:
                    st.warning("⚠️ BMI data not available for this age")

    st.divider()

# Editable Measurements Table with SDS
if st.session_state.child_info:
    st.header("📋 Measurements Table")

    # Import/Export section
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Export to YAML - Prepare data for download
        export_data = {
            'child_info': {
                'gender': st.session_state.child_info['gender'],
                'birth_date': st.session_state.child_info['birth_date'].strftime('%Y-%m-%d')
            },
            'data_points': [],
            'today_measurement': None
        }

        # Export historical data points
        for point in st.session_state.data_points:
            export_point = point.copy()
            if 'date' in export_point:
                export_point['date'] = export_point['date'].strftime('%Y-%m-%d')
            export_data['data_points'].append(export_point)

        # Export today's measurement
        if st.session_state.today_measurement:
            today_export = st.session_state.today_measurement.copy()
            if 'date' in today_export:
                today_export['date'] = today_export['date'].strftime('%Y-%m-%d')
            export_data['today_measurement'] = today_export

        yaml_str = yaml.dump(export_data, default_flow_style=False, sort_keys=False)
        st.download_button(
            label="📤 Download Data",
            data=yaml_str,
            file_name=f"growth_data_{st.session_state.child_info['birth_date'].strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        # Import from file
        uploaded_file = st.file_uploader("📥 Import Data", type=['txt'], key="import_file")
        if uploaded_file is not None:
            # Use file ID to prevent re-processing the same file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Check if this file has already been processed
            if 'last_imported_file' not in st.session_state or st.session_state.last_imported_file != file_id:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    import_data = yaml.safe_load(file_content)

                    # Load child info
                    if 'child_info' in import_data:
                        st.session_state.child_info = {
                            'gender': import_data['child_info']['gender'],
                            'birth_date': datetime.strptime(import_data['child_info']['birth_date'], '%Y-%m-%d').date()
                        }

                    # Load data points
                    if 'data_points' in import_data:
                        st.session_state.data_points = []
                        for point in import_data['data_points']:
                            point_copy = point.copy()
                            if 'date' in point_copy:
                                point_copy['date'] = datetime.strptime(point_copy['date'], '%Y-%m-%d').date()
                            st.session_state.data_points.append(point_copy)

                    # Handle today's measurement based on date
                    if 'today_measurement' in import_data and import_data['today_measurement'] is not None:
                        today_copy = import_data['today_measurement'].copy()
                        if 'date' in today_copy:
                            today_copy['date'] = datetime.strptime(today_copy['date'], '%Y-%m-%d').date()

                            # Check if the saved "today's measurement" is from a previous date
                            if today_copy['date'] < date.today():
                                # Add it to historical data points
                                st.session_state.data_points.append({
                                    'date': today_copy['date'],
                                    'gender': today_copy['gender'],
                                    'age': today_copy['age_months'],
                                    'height': today_copy['height'],
                                    'weight': today_copy['weight'],
                                    'bmi': today_copy.get('bmi')
                                })
                                st.session_state.today_measurement = None
                            else:
                                # Keep it as today's measurement if it's from today
                                st.session_state.today_measurement = today_copy
                    else:
                        st.session_state.today_measurement = None

                    # Mark this file as processed
                    st.session_state.last_imported_file = file_id

                    st.success("✅ Data imported successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error importing file: {str(e)}")

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.data_points = []
            st.success("All data cleared!")
            st.rerun()

    # Create editable dataframe with z-scores
    if st.session_state.data_points or st.session_state.today_measurement:
        # Combine today's measurement with historical data
        all_measurements = []

        if st.session_state.today_measurement:
            today = st.session_state.today_measurement.copy()
            today['is_today'] = True
            all_measurements.append(today)

        for point in st.session_state.data_points:
            point_copy = point.copy()
            point_copy['is_today'] = False
            all_measurements.append(point_copy)

        # Sort by date
        all_measurements.sort(key=lambda x: x['date'])

        # Create dataframe with z-scores
        table_data = []
        for measurement in all_measurements:
            try:
                # Get age in months (handle both 'age' and 'age_months' keys)
                age_months = measurement.get('age_months', measurement.get('age', 0))

                # Get height and weight values, handling missing data gracefully
                height_value = measurement.get('height')
                weight_value = measurement.get('weight')

                # Calculate z-scores only if values exist
                height_z, height_perc = None, None
                weight_z, weight_perc = None, None

                if height_value is not None:
                    try:
                        height_z, height_perc, _, _ = calculate_z_score(
                            age_months, height_value, 'height',
                            measurement['gender'], st.session_state.data_source
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating height z-score for age {age_months}: {e}")

                if weight_value is not None:
                    try:
                        weight_z, weight_perc, _, _ = calculate_z_score(
                            age_months, weight_value, 'weight',
                            measurement['gender'], st.session_state.data_source
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating weight z-score for age {age_months}: {e}")

                bmi_z = None
                bmi_perc = None
                if 'bmi' in measurement and measurement['bmi'] is not None:
                    bmi_available = False
                    if st.session_state.data_source == 'WHO' and age_months >= 61:
                        bmi_available = True
                    elif st.session_state.data_source == 'CDC' and age_months >= 24:
                        bmi_available = True

                    if bmi_available:
                        try:
                            bmi_z, bmi_perc, _, _ = calculate_z_score(
                                age_months, measurement['bmi'], 'bmi',
                                measurement['gender'], st.session_state.data_source
                            )
                        except Exception as e:
                            logger.warning(f"Error calculating BMI z-score for age {age_months}: {e}")

                row = {
                    'Date': measurement['date'].strftime('%Y-%m-%d'),
                    'Height (cm)': round(height_value, 1) if height_value is not None else None,
                    'Weight (kg)': round(weight_value, 1) if weight_value is not None else None,
                    'Age (months)': age_months,
                    'BMI': round(measurement['bmi'], 2) if 'bmi' in measurement and measurement['bmi'] is not None else None,
                    'Height SDS': round(height_z, 2) if height_z is not None else None,
                    'Height %ile': round(height_perc, 1) if height_perc is not None else None,
                    'Weight SDS': round(weight_z, 2) if weight_z is not None else None,
                    'Weight %ile': round(weight_perc, 1) if weight_perc is not None else None,
                    'BMI SDS': round(bmi_z, 2) if bmi_z is not None else None,
                    'BMI %ile': round(bmi_perc, 1) if bmi_perc is not None else None,
                    'Today': '🔸' if measurement.get('is_today', False) else ''
                }
                table_data.append(row)
            except Exception as e:
                logger.error(f"Error processing measurement: {e}")
                # Continue processing other measurements even if one fails
                continue

        df_table = pd.DataFrame(table_data)

        # Store table in session state for CSV export
        st.session_state.table_with_zscores = df_table.copy()

        edited_df = st.data_editor(
            df_table,
            use_container_width=True,
            num_rows="dynamic",
            disabled=["Age (months)", "Height SDS", "Height %ile", "Weight SDS", "Weight %ile", "BMI", "BMI SDS", "BMI %ile", "Today"],
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Age (months)": st.column_config.NumberColumn("Age (months)", width="small"),
                "Height (cm)": st.column_config.NumberColumn("Height (cm)", width="small", format="%.1f"),
                "Height SDS": st.column_config.NumberColumn("Height SDS", width="small", format="%.2f"),
                "Height %ile": st.column_config.NumberColumn("Height %", width="small", format="%.1f"),
                "Weight (kg)": st.column_config.NumberColumn("Weight (kg)", width="small", format="%.1f"),
                "Weight SDS": st.column_config.NumberColumn("Weight SDS", width="small", format="%.2f"),
                "Weight %ile": st.column_config.NumberColumn("Weight %", width="small", format="%.1f"),
                "BMI": st.column_config.NumberColumn("BMI", width="small", format="%.2f"),
                "BMI SDS": st.column_config.NumberColumn("BMI SDS", width="small", format="%.2f"),
                "BMI %ile": st.column_config.NumberColumn("BMI %", width="small", format="%.1f"),
                "Today": st.column_config.TextColumn("📍", width="small")
            },
            hide_index=True,
            key="measurements_table"
        )

        # Update session state with edited data
        if st.button("💾 Save Table Changes", use_container_width=False, type="primary"):
            try:
                # Clear existing data points
                st.session_state.data_points = []
                rows_processed = 0
                rows_skipped = 0

                # Process each row
                for idx, row in edited_df.iterrows():
                    try:
                        # A row needs at least a date and at least one measurement (height or weight)
                        has_date = pd.notna(row['Date'])
                        has_height = pd.notna(row['Height (cm)'])
                        has_weight = pd.notna(row['Weight (kg)'])

                        if not has_date:
                            logger.debug(f"Row {idx}: Skipping row without date")
                            rows_skipped += 1
                            continue

                        if not has_height and not has_weight:
                            logger.warning(f"Row {idx}: Skipping row with date but no height or weight")
                            rows_skipped += 1
                            continue

                        try:
                            measurement_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                        except ValueError as e:
                            logger.error(f"Row {idx}: Invalid date format '{row['Date']}': {e}")
                            rows_skipped += 1
                            continue

                        # Calculate age from birth_date and measurement_date
                        age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], measurement_date)

                        # Handle missing height or weight gracefully
                        height = float(row['Height (cm)']) if has_height else None
                        weight = float(row['Weight (kg)']) if has_weight else None

                        # Calculate BMI only if both height and weight are available
                        bmi = None
                        if height is not None and weight is not None and height > 0 and weight > 0:
                            bmi = calculate_bmi(height, weight)

                        # Check if this is today's measurement
                        is_today_row = row.get('Today', '') == '🔸'

                        if is_today_row:
                            st.session_state.today_measurement = {
                                'date': measurement_date,
                                'age_months': age_months,
                                'height': height,
                                'weight': weight,
                                'bmi': bmi,
                                'gender': st.session_state.child_info['gender']
                            }
                        else:
                            st.session_state.data_points.append({
                                'date': measurement_date,
                                'gender': st.session_state.child_info['gender'],
                                'age': age_months,
                                'height': height,
                                'weight': weight,
                                'bmi': bmi
                            })
                        rows_processed += 1
                        logger.info(f"Row {idx}: Successfully processed measurement from {measurement_date}")

                    except Exception as row_error:
                        logger.error(f"Row {idx}: Error processing row: {row_error}")
                        rows_skipped += 1
                        continue

                if rows_skipped > 0:
                    st.warning(f"⚠️ {rows_skipped} row(s) were skipped due to missing or invalid data. Check logs for details.")
                st.success(f"✅ {rows_processed} measurement(s) saved successfully!")
                st.rerun()
            except Exception as e:
                logger.error(f"Error saving table changes: {e}")
                st.error(f"❌ Error saving changes: {str(e)}")

    st.divider()

# Main content - Visualizations
if st.session_state.child_info:
    selected_gender = st.session_state.child_info['gender']
    growth_height = get_height_data(selected_gender, st.session_state.data_source)
    growth_weight = get_weight_data(selected_gender, st.session_state.data_source)

    # Calculate child's current age for dynamic chart scaling
    if st.session_state.today_measurement:
        current_age = st.session_state.today_measurement['age_months']
    else:
        current_age = calculate_age_in_months(st.session_state.child_info['birth_date'], date.today())

    # Collect all age values from data points
    all_ages = []
    if st.session_state.data_points:
        df_all = pd.DataFrame(st.session_state.data_points)
        all_ages.extend(df_all['age'].tolist())
    if st.session_state.today_measurement:
        all_ages.append(st.session_state.today_measurement['age_months'])

    # Dynamic X-axis range based on age and data source
    # CDC data: 24-240 months (2-20 years)
    # WHO data: 0-228 months (0-19 years)
    if st.session_state.data_source == 'CDC':
        # CDC covers 2-20 years - use dynamic ranges
        if current_age <= 60:  # 2-5 years
            x_range = [24, 60]  # Show 2-5 years
            age_group = "2-5 years"
        elif current_age <= 120:  # 5-10 years
            x_range = [60, 120]  # Show 5-10 years
            age_group = "5-10 years"
        else:  # 10+ years
            x_range = [120, 240]  # Show 10-20 years
            age_group = "10-20 years"
    else:  # WHO
        # Use dynamic age range based on child's current age
        if current_age <= 24:  # 0-2 years
            x_range = [0, 30]  # Show 0-2.5 years (add padding beyond current age)
            age_group = "0-2 years"
        elif current_age <= 60:  # 2-5 years
            x_range = [0, 60]  # Show 0-5 years
            age_group = "0-5 years"
        else:  # 5+ years
            x_range = [60, 228]  # Show 5-19 years
            age_group = "5-19 years"

    # Extend x_range to accommodate all data points if they fall outside
    if all_ages:
        min_age = min(all_ages)
        max_age = max(all_ages)

        # Extend lower bound if needed
        if min_age < x_range[0]:
            x_range[0] = max(0 if st.session_state.data_source == 'WHO' else 24, min_age - 2)  # Add small buffer

        # Extend upper bound if needed
        if max_age > x_range[1]:
            # Add 10% buffer or at least 6 months
            buffer = max(6, int((max_age - x_range[0]) * 0.1))
            x_range[1] = min(228 if st.session_state.data_source == 'WHO' else 240, max_age + buffer)

    st.header(f"📊 Growth Charts - {selected_gender} ({st.session_state.data_source}) - {age_group}")

    # Height-for-Age Chart
    fig_height = go.Figure()

    # Add WHO percentile lines (in reverse order for legend display)
    colors = {'p3': 'lightcoral', 'p15': 'lightblue', 'p50': 'green',
             'p85': 'lightblue', 'p97': 'lightcoral'}
    names = {'p3': '3rd percentile', 'p15': '15th percentile',
            'p50': '50th percentile (median)', 'p85': '85th percentile',
            'p97': '97th percentile'}

    for percentile in ['p97', 'p85', 'p50', 'p15', 'p3']:
        fig_height.add_trace(go.Scatter(
            x=growth_height['age_months'],
            y=growth_height[percentile],
            mode='lines',
            name=names[percentile],
            line=dict(color=colors[percentile], width=2 if percentile == 'p50' else 1),
            opacity=0.7
        ))

    # Add historical data points
    if st.session_state.data_points:
        df = pd.DataFrame(st.session_state.data_points)
        # Filter to only include rows with valid height values and sort by age
        df_with_height = df[df['height'].notna()].sort_values('age')
        if not df_with_height.empty:
            fig_height.add_trace(go.Scatter(
                x=df_with_height['age'],
                y=df_with_height['height'],
                mode='markers+lines',
                name='Historical Measurements',
                marker=dict(size=10, color='blue', symbol='circle'),
                line=dict(color='blue', width=2, dash='dash')
            ))

    # Add today's measurement
    if st.session_state.today_measurement and st.session_state.today_measurement.get('height') is not None:
        today = st.session_state.today_measurement
        fig_height.add_trace(go.Scatter(
            x=[today['age_months']],
            y=[today['height']],
            mode='markers',
            name="Today's Measurement",
            marker=dict(size=15, color='orange', symbol='circle', line=dict(color='darkorange', width=2))
        ))

    # Calculate Y-axis range to show all percentiles properly
    # Get min from 3rd percentile and max from 97th percentile in visible range
    visible_data = growth_height[growth_height['age_months'].between(x_range[0], x_range[1])]
    if not visible_data.empty:
        y_min = visible_data['p3'].min() * 0.95  # Add 5% padding below
        y_max = visible_data['p97'].max() * 1.02  # Add 2% padding above

        # Also check actual data points and extend y-axis if needed
        actual_heights = []
        if st.session_state.data_points:
            df = pd.DataFrame(st.session_state.data_points)
            # Only include non-null height values
            valid_heights = df['height'].dropna().tolist()
            actual_heights.extend(valid_heights)
        if st.session_state.today_measurement and st.session_state.today_measurement.get('height') is not None:
            actual_heights.append(st.session_state.today_measurement['height'])

        if actual_heights:
            min_height = min(actual_heights)
            max_height = max(actual_heights)
            # Extend y_min if data point is below
            if min_height < y_min:
                y_min = min_height * 0.95
            # Extend y_max if data point is above
            if max_height > y_max:
                y_max = max_height * 1.05
    else:
        y_min = None
        y_max = None

    fig_height.update_layout(
        title=f"Height-for-Age ({selected_gender}) - {age_group}",
        xaxis_title="Age (months)",
        yaxis_title="Height (cm)",
        hovermode='closest',
        showlegend=True,
        height=500,
        xaxis=dict(range=x_range),
        yaxis=dict(range=[y_min, y_max] if y_min else None),
        # Mobile optimization
        font=dict(size=12),
        margin=dict(l=50, r=20, t=50, b=50)
    )

    st.plotly_chart(fig_height, use_container_width=True)

    # Weight-for-Age Chart
    # Automatically switch to CDC if WHO doesn't have weight data for this age
    chart_weight_source = st.session_state.data_source
    if st.session_state.data_source == 'WHO' and current_age > 120:
        chart_weight_source = 'CDC'
        st.info("ℹ️ Automatically displaying CDC weight chart (WHO weight data only available up to 10 years)")

    # Check if CDC data is available for this age
    if chart_weight_source == 'CDC' and current_age < 24:
        st.warning("⚠️ CDC Weight-for-Age Chart Not Available")
        st.info("CDC data is only available from 2 years (24 months). Please use WHO data source for children under 2 years.")
    else:
        # Load appropriate weight data
        growth_weight_chart = get_weight_data(selected_gender, chart_weight_source)
        fig_weight = go.Figure()

        # Add percentile lines (in reverse order for legend display)
        for percentile in ['p97', 'p85', 'p50', 'p15', 'p3']:
            fig_weight.add_trace(go.Scatter(
                x=growth_weight_chart['age_months'],
                y=growth_weight_chart[percentile],
                mode='lines',
                name=names[percentile],
                line=dict(color=colors[percentile], width=2 if percentile == 'p50' else 1),
                opacity=0.7
            ))

        # Add historical data points
        if st.session_state.data_points:
            df = pd.DataFrame(st.session_state.data_points)
            # Filter to only include rows with valid weight values and sort by age
            df_with_weight = df[df['weight'].notna()].sort_values('age')
            if not df_with_weight.empty:
                fig_weight.add_trace(go.Scatter(
                    x=df_with_weight['age'],
                    y=df_with_weight['weight'],
                    mode='markers+lines',
                    name='Historical Measurements',
                    marker=dict(size=10, color='blue', symbol='circle'),
                    line=dict(color='blue', width=2, dash='dash')
                ))

        # Add today's measurement
        if st.session_state.today_measurement and st.session_state.today_measurement.get('weight') is not None:
            today = st.session_state.today_measurement
            fig_weight.add_trace(go.Scatter(
                x=[today['age_months']],
                y=[today['weight']],
                mode='markers',
                name="Today's Measurement",
                marker=dict(size=15, color='orange', symbol='circle', line=dict(color='darkorange', width=2))
            ))

        # Calculate Y-axis range for weight chart
        visible_weight_data = growth_weight_chart[growth_weight_chart['age_months'].between(x_range[0], x_range[1])]
        if not visible_weight_data.empty:
            y_min_weight = visible_weight_data['p3'].min() * 0.90  # Add 10% padding below
            y_max_weight = visible_weight_data['p97'].max() * 1.05  # Add 5% padding above

            # Also check actual data points and extend y-axis if needed
            actual_weights = []
            if st.session_state.data_points:
                df = pd.DataFrame(st.session_state.data_points)
                # Only include non-null weight values
                valid_weights = df['weight'].dropna().tolist()
                actual_weights.extend(valid_weights)
            if st.session_state.today_measurement and st.session_state.today_measurement.get('weight') is not None:
                actual_weights.append(st.session_state.today_measurement['weight'])

            if actual_weights:
                min_weight = min(actual_weights)
                max_weight = max(actual_weights)
                # Extend y_min if data point is below
                if min_weight < y_min_weight:
                    y_min_weight = min_weight * 0.90
                # Extend y_max if data point is above
                if max_weight > y_max_weight:
                    y_max_weight = max_weight * 1.10
        else:
            y_min_weight = None
            y_max_weight = None

        # Determine age range text for title
        if chart_weight_source == 'CDC':
            weight_age_range = "2-20 years"
        else:  # WHO
            weight_age_range = "0-10 years"  # WHO weight-for-age covers 0-10 years

        fig_weight.update_layout(
            title=f"Weight-for-Age ({selected_gender}) - {chart_weight_source} - {weight_age_range}",
            xaxis_title="Age (months)",
            yaxis_title="Weight (kg)",
            hovermode='closest',
            showlegend=True,
            height=500,
            xaxis=dict(range=x_range),
            yaxis=dict(range=[y_min_weight, y_max_weight] if y_min_weight else None),
            # Mobile optimization
            font=dict(size=12),
            margin=dict(l=50, r=20, t=50, b=50)
        )

        st.plotly_chart(fig_weight, use_container_width=True)

    # BMI-for-Age Chart
    # WHO: ages 5-19 (61-228 months)
    # CDC: ages 2-20 (24-240 months)
    fig_bmi = None  # Initialize to None, will be set if BMI chart is shown
    show_bmi_chart = False
    if st.session_state.data_source == 'WHO' and current_age >= 61:
        show_bmi_chart = True
    elif st.session_state.data_source == 'CDC' and current_age >= 24:
        show_bmi_chart = True

    if show_bmi_chart:
        st.divider()
        st.subheader("📊 BMI-for-Age Chart")

        growth_bmi = get_bmi_data(selected_gender, st.session_state.data_source)

        if not growth_bmi.empty:
            fig_bmi = go.Figure()

            colors = {'p3': 'lightcoral', 'p15': 'lightblue', 'p50': 'green',
                     'p85': 'lightblue', 'p97': 'lightcoral'}
            names = {'p3': '3rd percentile', 'p15': '15th percentile',
                    'p50': '50th percentile (median)', 'p85': '85th percentile',
                    'p97': '97th percentile'}

            # Add percentile lines (in reverse order for legend display)
            for percentile in ['p97', 'p85', 'p50', 'p15', 'p3']:
                fig_bmi.add_trace(go.Scatter(
                    x=growth_bmi['age_months'],
                    y=growth_bmi[percentile],
                    mode='lines',
                    name=names[percentile],
                    line=dict(color=colors[percentile], width=2 if percentile == 'p50' else 1),
                    opacity=0.7
                ))

            # Add historical data points if they have BMI
            if st.session_state.data_points:
                df = pd.DataFrame(st.session_state.data_points)
                if 'bmi' in df.columns:
                    # Filter to only include rows with valid BMI values and sort by age
                    df_with_bmi = df[df['bmi'].notna()].sort_values('age')
                    if not df_with_bmi.empty:
                        fig_bmi.add_trace(go.Scatter(
                            x=df_with_bmi['age'],
                            y=df_with_bmi['bmi'],
                            mode='markers+lines',
                            name='Historical Measurements',
                            marker=dict(size=10, color='blue', symbol='circle'),
                            line=dict(color='blue', width=2, dash='dash')
                        ))

            # Add today's measurement
            if st.session_state.today_measurement and 'bmi' in st.session_state.today_measurement and st.session_state.today_measurement['bmi'] is not None:
                today = st.session_state.today_measurement
                fig_bmi.add_trace(go.Scatter(
                    x=[today['age_months']],
                    y=[today['bmi']],
                    mode='markers',
                    name="Today's Measurement",
                    marker=dict(size=15, color='orange', symbol='circle', line=dict(color='darkorange', width=2))
                ))

            # Calculate Y-axis range for BMI chart
            visible_bmi_data = growth_bmi[growth_bmi['age_months'].between(x_range[0], x_range[1])]
            if not visible_bmi_data.empty:
                y_min_bmi = visible_bmi_data['p3'].min() * 0.90  # Add 10% padding below
                y_max_bmi = visible_bmi_data['p97'].max() * 1.05  # Add 5% padding above

                # Also check actual data points and extend y-axis if needed
                actual_bmis = []
                if st.session_state.data_points:
                    df = pd.DataFrame(st.session_state.data_points)
                    if 'bmi' in df.columns:
                        actual_bmis.extend(df[df['bmi'].notna()]['bmi'].tolist())
                if st.session_state.today_measurement and 'bmi' in st.session_state.today_measurement and st.session_state.today_measurement['bmi'] is not None:
                    actual_bmis.append(st.session_state.today_measurement['bmi'])

                if actual_bmis:
                    min_bmi = min(actual_bmis)
                    max_bmi = max(actual_bmis)
                    # Extend y_min if data point is below
                    if min_bmi < y_min_bmi:
                        y_min_bmi = min_bmi * 0.90
                    # Extend y_max if data point is above
                    if max_bmi > y_max_bmi:
                        y_max_bmi = max_bmi * 1.10
            else:
                y_min_bmi = None
                y_max_bmi = None

            # Set X-axis range based on data source - start with defaults
            if st.session_state.data_source == 'WHO':
                bmi_x_range = [61, 228]  # WHO: 5-19 years
            else:  # CDC
                bmi_x_range = [24, 240]  # CDC: 2-20 years

            # Extend BMI x-axis to accommodate all BMI data points if needed
            if all_ages:
                min_age = min(all_ages)
                max_age = max(all_ages)
                if min_age < bmi_x_range[0]:
                    bmi_x_range[0] = max(0, min_age - 2)
                if max_age > bmi_x_range[1]:
                    buffer = max(6, int((max_age - bmi_x_range[1]) * 0.1))
                    bmi_x_range[1] = min(228 if st.session_state.data_source == 'WHO' else 240, max_age + buffer)

            # Determine BMI age range based on data source
            if st.session_state.data_source == 'WHO':
                bmi_age_range = "5-19 years"  # WHO BMI-for-age covers 5-19 years
            else:  # CDC
                bmi_age_range = "2-20 years"  # CDC BMI-for-age covers 2-20 years

            fig_bmi.update_layout(
                title=f"BMI-for-Age ({selected_gender}) - {st.session_state.data_source} - {bmi_age_range}",
                xaxis_title="Age (months)",
                yaxis_title="BMI (kg/m²)",
                hovermode='closest',
                showlegend=True,
                height=500,
                xaxis=dict(range=bmi_x_range),
                yaxis=dict(range=[y_min_bmi, y_max_bmi] if y_min_bmi else None),
                font=dict(size=12),
                margin=dict(l=50, r=20, t=50, b=50)
            )

            st.plotly_chart(fig_bmi, use_container_width=True)

    # PDF Export Button
    st.divider()
    st.subheader("📄 Export Report")

    # Generate PDF data
    try:
        # Get measurements table with SDS if available
        measurements_table = st.session_state.get('table_with_zscores', None)

        pdf_buffer = generate_pdf_report(
            st.session_state.child_info,
            st.session_state.today_measurement,
            st.session_state.data_points,
            fig_height,
            fig_weight,
            fig_bmi,
            st.session_state.data_source,
            measurements_table
        )

        # Prepare filename
        filename = f"growth_report_{st.session_state.child_info['birth_date'].strftime('%Y%m%d')}.pdf"

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

    except Exception as e:
        st.error(f"⚠️ Error generating PDF: {str(e)}")

else:
    pass  # No child info saved yet

    # Show sample charts
    st.subheader(f"Example: {st.session_state.data_source} Growth Standards")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Height-for-Age Chart (Male)**")
        sample_height = get_height_data("Male", st.session_state.data_source)
        fig_sample = go.Figure()
        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_sample.add_trace(go.Scatter(
                x=sample_height['age_months'],
                y=sample_height[percentile],
                mode='lines',
                name=percentile.upper(),
                opacity=0.7
            ))
        fig_sample.update_layout(
            xaxis_title="Age (months)",
            yaxis_title="Height (cm)",
            height=400
        )
        st.plotly_chart(fig_sample, use_container_width=True)

    with col2:
        st.markdown("**Weight-for-Age Chart (Male)**")
        sample_weight = get_weight_data("Male", st.session_state.data_source)
        fig_weight_sample = go.Figure()
        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_weight_sample.add_trace(go.Scatter(
                x=sample_weight['age_months'],
                y=sample_weight[percentile],
                mode='lines',
                name=percentile.upper(),
                opacity=0.7
            ))
        fig_weight_sample.update_layout(
            xaxis_title="Age (months)",
            yaxis_title="Weight (kg)",
            height=400
        )
        st.plotly_chart(fig_weight_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Data Sources:**
- **WHO:** WHO Child Growth Standards (2006, 0-5 years) and WHO Growth Reference (2007, 5-19 years) based on global multi-ethnic populations
  - Height-for-age: 0-228 months (0-19 years)
  - Weight-for-age: 0-120 months (0-10 years only)
  - *Note: For children over 10 years, WHO recommends using BMI-for-age instead of weight-for-age*
- **CDC:** CDC Growth Charts (2000, 2-20 years) based on US population data
  - Both height and weight data available for full age range

Both datasets use LMS (Lambda-Mu-Sigma) parameters for percentile calculations.
""")
