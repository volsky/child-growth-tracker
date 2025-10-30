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
    Returns growth statistics (mean and SD) for Z-score calculation
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
    Calculate Z-score for a given measurement
    measurement_type: 'height', 'weight', or 'bmi'
    data_source: 'WHO' or 'CDC'
    """
    stats = get_growth_statistics(gender, data_source)

    # Check if the measurement type column exists
    mean_col = f'{measurement_type}_mean'
    sd_col = f'{measurement_type}_sd'

    if mean_col not in stats.columns or sd_col not in stats.columns:
        return None, None, None, None

    # Interpolate to get mean and SD for exact age
    f_mean = interpolate.interp1d(stats['age_months'], stats[mean_col],
                                   kind='linear', fill_value='extrapolate')
    f_sd = interpolate.interp1d(stats['age_months'], stats[sd_col],
                                 kind='linear', fill_value='extrapolate')

    mean = float(f_mean(age_months))
    sd = float(f_sd(age_months))

    z_score = (measurement - mean) / sd

    # Calculate percentile
    percentile = norm.cdf(z_score) * 100

    return z_score, percentile, mean, sd

def interpret_z_score(z_score, measurement_type):
    """
    Provide interpretation of Z-score based on WHO guidelines
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

def generate_pdf_report(child_info, today_measurement, data_points, height_fig, weight_fig, bmi_fig=None, data_source='WHO'):
    """Generate PDF report with Z-scores and charts"""
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Summary and Z-scores
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
            # Calculate Z-scores
            height_z, height_perc, height_mean, height_sd = calculate_z_score(
                today_measurement['age_months'], today_measurement['height'], 'height', today_measurement['gender'], data_source
            )
            weight_z, weight_perc, weight_mean, weight_sd = calculate_z_score(
                today_measurement['age_months'], today_measurement['weight'], 'weight', today_measurement['gender'], data_source
            )

            height_interp, _ = interpret_z_score(height_z, 'height')
            weight_interp, _ = interpret_z_score(weight_z, 'weight')

            # Check if BMI data is available and calculate BMI Z-score
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
  Z-score: {bmi_z:.2f}
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
  Measurement: {today_measurement['height']:.1f} cm
  Z-score: {height_z:.2f}
  Percentile: {height_perc:.1f}%
  Interpretation: {height_interp}
  Expected mean: {height_mean:.1f} cm
  Standard deviation: {height_sd:.2f} cm

Weight Analysis:
  Measurement: {today_measurement['weight']:.1f} kg
  Z-score: {weight_z:.2f}
  Percentile: {weight_perc:.1f}%
  Interpretation: {weight_interp}
  Expected mean: {weight_mean:.1f} kg
  Standard deviation: {weight_sd:.2f} kg
{bmi_text}"""
            plt.text(0.1, 0.72, today_text, fontsize=10, verticalalignment='top',
                    fontfamily='monospace', transform=fig.transFigure)

        # Add historical data summary
        if data_points:
            hist_text = f"\nHistorical Measurements:\n{'─'*37}\n"
            hist_text += f"Total measurements: {len(data_points)}\n"
            df = pd.DataFrame(data_points)
            hist_text += f"Age range: {df['age'].min()}-{df['age'].max()} months\n"
            hist_text += f"Height range: {df['height'].min():.1f}-{df['height'].max():.1f} cm\n"
            hist_text += f"Weight range: {df['weight'].min():.1f}-{df['weight'].max():.1f} kg\n"
            # Add BMI range if available
            if 'bmi' in df.columns and df['bmi'].notna().any():
                hist_text += f"BMI range: {df['bmi'].min():.2f}-{df['bmi'].max():.2f} kg/m²\n"

            plt.text(0.1, 0.25, hist_text, fontsize=10, verticalalignment='top',
                    fontfamily='monospace', transform=fig.transFigure)

        # Add footer
        data_source_text = "WHO Child Growth Standards (2006) and Growth Reference (2007)" if data_source == "WHO" else "CDC Growth Charts (2000) for US population"
        footer = f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nData source: {data_source_text}"
        plt.text(0.1, 0.05, footer, fontsize=8, verticalalignment='bottom',
                style='italic', transform=fig.transFigure)

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2 & 3: Charts
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

# Sidebar for Child Information
with st.sidebar:
    st.header("👤 Child Information")

    child_gender = st.selectbox("Gender", ["Male", "Female"], key="child_gender")
    child_birth_date = st.date_input("Birth Date",
                                      value=date.today().replace(year=date.today().year - 2),
                                      min_value=date.today().replace(year=date.today().year - 20),
                                      max_value=date.today(),
                                      key="birth_date")

    if st.button("Save Child Info", use_container_width=True):
        st.session_state.child_info = {
            'gender': child_gender,
            'birth_date': child_birth_date
        }
        st.success("Child info saved!")

    st.divider()

    # Data Source Selection - moved below child info
    st.header("⚙️ Data Source")

    # Calculate recommended data source based on child's age
    recommended_source = "WHO"  # Default
    recommendation_reason = ""
    force_cdc = False  # Flag to force CDC and disable selection

    if st.session_state.child_info:
        child_age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], date.today())

        # Force CDC if WHO data is not available (weight over 10 years)
        if child_age_months > 120:  # Over 10 years - WHO weight data not available
            force_cdc = True
            recommended_source = "CDC"
            recommendation_reason = "🔒 Automatically using CDC (WHO weight data only available up to 10 years)"
        # Recommendation logic based on age
        elif child_age_months < 24:  # Under 2 years
            recommended_source = "WHO"
            recommendation_reason = "Recommended: WHO is best for children under 2 years"
        elif child_age_months >= 24 and child_age_months <= 120:  # 2-10 years
            recommended_source = "WHO"
            recommendation_reason = "Recommended: WHO provides comprehensive data for this age range"
        else:  # Over 19 years
            recommended_source = "CDC"
            recommendation_reason = "Recommended: CDC covers up to 20 years"

        # Show recommendation
        if force_cdc:
            st.warning(recommendation_reason)
        else:
            st.info(f"💡 {recommendation_reason}")

    # Get current selection or use recommended
    if force_cdc:
        # Force CDC and disable selection
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

    if data_source == "WHO":
        st.caption("📊 WHO Child Growth Standards (2006) and Growth Reference (2007)")
    else:
        st.caption("📊 CDC Growth Charts (2000) for US population")

    st.divider()

    # Today's Measurement Section
    st.header("📅 Today's Measurement")

    if st.session_state.child_info:
        today_date = st.date_input("Measurement Date",
                                    value=date.today(),
                                    min_value=st.session_state.child_info['birth_date'],
                                    max_value=date.today(),
                                    key="today_date")

        age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], today_date)
        st.info(f"Age: {age_months} months ({age_months // 12} years, {age_months % 12} months)")

        # Get 50th percentile defaults for this age and gender
        default_height, default_weight = get_default_measurements(age_months, st.session_state.child_info['gender'], st.session_state.data_source)

        # Use age_months in key to reset defaults when date changes
        today_height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=default_height, step=0.1, key=f"today_height_{age_months}")
        today_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=150.0, value=default_weight, step=0.1, key=f"today_weight_{age_months}")

        # Calculate and display BMI automatically
        if today_height > 0 and today_weight > 0:
            today_bmi = calculate_bmi(today_height, today_weight)
            st.info(f"💡 Calculated BMI: {today_bmi:.2f} kg/m²")

        if st.button("Save Today's Measurement", use_container_width=True):
            today_bmi = calculate_bmi(today_height, today_weight) if today_height > 0 and today_weight > 0 else None
            st.session_state.today_measurement = {
                'date': today_date,
                'age_months': age_months,
                'height': today_height,
                'weight': today_weight,
                'bmi': today_bmi,
                'gender': st.session_state.child_info['gender']
            }
            st.success("Today's measurement saved!")
    else:
        st.warning("Please save child info first")

    st.divider()

    # Historical Measurements Section
    st.header("📊 Historical Data")

    if st.session_state.child_info:
        hist_date = st.date_input("Measurement Date",
                                   value=date.today().replace(year=date.today().year - 1),
                                   min_value=st.session_state.child_info['birth_date'],
                                   max_value=date.today(),
                                   key="hist_date")

        hist_age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], hist_date)
        st.info(f"Age: {hist_age_months} months")

        # Get 50th percentile defaults for this age and gender
        hist_default_height, hist_default_weight = get_default_measurements(hist_age_months, st.session_state.child_info['gender'], st.session_state.data_source)

        # Use age_months in key to reset defaults when date changes
        hist_height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=hist_default_height, step=0.1, key=f"hist_height_{hist_age_months}")
        hist_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=150.0, value=hist_default_weight, step=0.1, key=f"hist_weight_{hist_age_months}")

        # Calculate and display BMI automatically for historical data
        if hist_height > 0 and hist_weight > 0:
            hist_bmi = calculate_bmi(hist_height, hist_weight)
            st.info(f"💡 Calculated BMI: {hist_bmi:.2f} kg/m²")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Point", use_container_width=True):
                hist_bmi = calculate_bmi(hist_height, hist_weight) if hist_height > 0 and hist_weight > 0 else None
                st.session_state.data_points.append({
                    'date': hist_date,
                    'gender': st.session_state.child_info['gender'],
                    'age': hist_age_months,
                    'height': hist_height,
                    'weight': hist_weight,
                    'bmi': hist_bmi
                })
                st.success("Point added!")

        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.data_points = []
                st.rerun()

        # Show current data points
        if st.session_state.data_points:
            st.subheader(f"Historical ({len(st.session_state.data_points)})")
            df_display = pd.DataFrame(st.session_state.data_points)
            df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%Y-%m-%d')
            # Show BMI column if it exists and has values
            columns_to_show = ['date', 'age', 'height', 'weight']
            if 'bmi' in df_display.columns and df_display['bmi'].notna().any():
                columns_to_show.append('bmi')
            st.dataframe(df_display[columns_to_show], use_container_width=True)
    else:
        st.warning("Please save child info first")

# Main content - Today's Measurement Z-scores
if st.session_state.today_measurement:
    st.header("📈 Today's Measurement Analysis")

    today = st.session_state.today_measurement
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📏 Height Analysis")
        height_z, height_perc, height_mean, height_sd = calculate_z_score(
            today['age_months'], today['height'], 'height', today['gender'], st.session_state.data_source
        )
        height_interp, height_status = interpret_z_score(height_z, 'height')

        st.metric("Height", f"{today['height']:.1f} cm")
        st.metric("Z-score", f"{height_z:.2f}")
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
            weight_interp, weight_status = interpret_z_score(weight_z, 'weight')

            st.metric("Weight", f"{today['weight']:.1f} kg")
            st.metric("Z-score", f"{weight_z:.2f}")
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
                    st.metric("Z-score", f"{bmi_z:.2f}")
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
            else:
                if st.session_state.data_source == 'WHO':
                    st.info("💡 WHO BMI-for-age is available from 5-19 years (61-228 months)")
                else:  # CDC
                    st.info("💡 CDC BMI-for-age is available from 2-20 years (24-240 months)")
                st.caption(f"Current: {st.session_state.data_source}, Age: {today['age_months']} months")
        else:
            st.info("💡 Save measurement to calculate BMI")

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

    # Dynamic X-axis range based on age and data source
    # CDC data: 24-240 months (2-20 years)
    # WHO data: 0-228 months (0-19 years)
    if st.session_state.data_source == 'CDC':
        # CDC covers 2-20 years (24-240 months)
        x_range = [24, 240]
        age_group = "2-20 years"
    else:  # WHO
        # Use 0-5 year range for young children, 5-19 year range for older children
        if current_age <= 60:  # 0-5 years
            x_range = [0, 60]  # Show 0-5 years
            age_group = "0-5 years"
        else:  # 5+ years
            x_range = [60, 228]  # Show 5-19 years
            age_group = "5-19 years"

    st.header(f"📊 Growth Charts - {selected_gender} ({st.session_state.data_source}) - {age_group}")

    # For mobile: Use single column layout on narrow screens
    # Streamlit automatically adjusts columns but we can control height
    col1, col2 = st.columns([1, 1])

    # Height-for-Age Chart
    with col1:
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
            fig_height.add_trace(go.Scatter(
                x=df['age'],
                y=df['height'],
                mode='markers+lines',
                name='Historical Measurements',
                marker=dict(size=10, color='blue', symbol='circle'),
                line=dict(color='blue', width=2, dash='dash')
            ))

        # Add today's measurement
        if st.session_state.today_measurement:
            today = st.session_state.today_measurement
            fig_height.add_trace(go.Scatter(
                x=[today['age_months']],
                y=[today['height']],
                mode='markers',
                name="Today's Measurement",
                marker=dict(size=20, color='red', symbol='star', line=dict(color='darkred', width=2))
            ))

        # Calculate Y-axis range to show all percentiles properly
        # Get min from 3rd percentile and max from 97th percentile in visible range
        visible_data = growth_height[growth_height['age_months'].between(x_range[0], x_range[1])]
        if not visible_data.empty:
            y_min = visible_data['p3'].min() * 0.95  # Add 5% padding below
            y_max = visible_data['p97'].max() * 1.02  # Add 2% padding above
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
    with col2:
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
                fig_weight.add_trace(go.Scatter(
                    x=df['age'],
                    y=df['weight'],
                    mode='markers+lines',
                    name='Historical Measurements',
                    marker=dict(size=10, color='blue', symbol='circle'),
                    line=dict(color='blue', width=2, dash='dash')
                ))

            # Add today's measurement
            if st.session_state.today_measurement:
                today = st.session_state.today_measurement
                fig_weight.add_trace(go.Scatter(
                    x=[today['age_months']],
                    y=[today['weight']],
                    mode='markers',
                    name="Today's Measurement",
                    marker=dict(size=20, color='red', symbol='star', line=dict(color='darkred', width=2))
                ))

            # Calculate Y-axis range for weight chart
            visible_weight_data = growth_weight_chart[growth_weight_chart['age_months'].between(x_range[0], x_range[1])]
            if not visible_weight_data.empty:
                y_min_weight = visible_weight_data['p3'].min() * 0.90  # Add 10% padding below
                y_max_weight = visible_weight_data['p97'].max() * 1.05  # Add 5% padding above
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
                    df_with_bmi = df[df['bmi'].notna()]
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
                    marker=dict(size=20, color='red', symbol='star', line=dict(color='darkred', width=2))
                ))

            # Calculate Y-axis range for BMI chart
            visible_bmi_data = growth_bmi[growth_bmi['age_months'].between(x_range[0], x_range[1])]
            if not visible_bmi_data.empty:
                y_min_bmi = visible_bmi_data['p3'].min() * 0.90  # Add 10% padding below
                y_max_bmi = visible_bmi_data['p97'].max() * 1.05  # Add 5% padding above
            else:
                y_min_bmi = None
                y_max_bmi = None

            # Set X-axis range based on data source
            if st.session_state.data_source == 'WHO':
                bmi_x_range = [61, 228]  # WHO: 5-19 years
            else:  # CDC
                bmi_x_range = [24, 240]  # CDC: 2-20 years

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
            if st.session_state.data_source == 'WHO':
                st.caption("💡 BMI-for-age is WHO's recommended indicator for assessing thinness/overweight in children 5-19 years")
            else:
                st.caption("💡 CDC BMI-for-age charts for ages 2-20 years")

    # PDF Export Button
    st.divider()
    st.subheader("📄 Export Report")

    # Generate PDF data
    try:
        pdf_buffer = generate_pdf_report(
            st.session_state.child_info,
            st.session_state.today_measurement,
            st.session_state.data_points,
            fig_height,
            fig_weight,
            fig_bmi,
            st.session_state.data_source
        )

        # Prepare filename
        filename = f"growth_report_{st.session_state.child_info['birth_date'].strftime('%Y%m%d')}.pdf"

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

        with col2:
            st.info("📊 Z-scores, measurements & charts")

    except Exception as e:
        st.error(f"⚠️ Error generating PDF: {str(e)}")
        st.info("Report includes Z-scores, measurements, and growth charts")

else:
    st.info("👈 Please save child information in the sidebar to get started!")

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
