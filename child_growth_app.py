import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from scipy import interpolate
from scipy.stats import norm
import os

# Load WHO Growth Standards from CSV files (based on official WHO LMS parameters)
# Data source: WHO Child Growth Standards (2006) and Growth Reference (2007)

@st.cache_data
def load_who_data_from_csv(filename):
    """Load WHO data from CSV file with caching"""
    filepath = os.path.join('who_data', filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        st.error(f"WHO data file not found: {filename}")
        return pd.DataFrame()

def get_who_data(gender):
    """
    Returns WHO growth percentiles for height-for-age (in cm)
    Ages are in months (0-228 months = 0-19 years)
    Loaded from official WHO LMS-based data
    """
    if gender == "Male":
        df = load_who_data_from_csv('boys_height_full.csv')
    else:  # Female
        df = load_who_data_from_csv('girls_height_full.csv')

    if df.empty:
        return df

    # Return relevant columns
    return df[['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']]

def get_who_weight_data(gender):
    """
    Returns WHO growth percentiles for weight-for-age (in kg)
    Ages are in months (0-120 months for weight data availability)
    Loaded from official WHO LMS-based data
    """
    if gender == "Male":
        df = load_who_data_from_csv('boys_weight_full.csv')
    else:  # Female
        df = load_who_data_from_csv('girls_weight_full.csv')

    if df.empty:
        return df

    # Return relevant columns
    return df[['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']]

def get_who_statistics(gender):
    """
    Returns WHO growth statistics (mean and SD) for Z-score calculation
    This includes both height and weight statistics loaded from CSV
    """
    # Load height data
    if gender == "Male":
        height_df = load_who_data_from_csv('boys_height_full.csv')
        weight_df = load_who_data_from_csv('boys_weight_full.csv')
    else:
        height_df = load_who_data_from_csv('girls_height_full.csv')
        weight_df = load_who_data_from_csv('girls_weight_full.csv')

    if height_df.empty or weight_df.empty:
        return pd.DataFrame()

    # Merge height and weight data
    stats = pd.merge(
        height_df[['age_months', 'mean', 'sd']].rename(columns={'mean': 'height_mean', 'sd': 'height_sd'}),
        weight_df[['age_months', 'mean', 'sd']].rename(columns={'mean': 'weight_mean', 'sd': 'weight_sd'}),
        on='age_months',
        how='outer'
    ).sort_values('age_months')

    return stats

def calculate_z_score(age_months, measurement, measurement_type, gender):
    """
    Calculate Z-score for a given measurement
    measurement_type: 'height' or 'weight'
    """
    stats = get_who_statistics(gender)

    # Interpolate to get mean and SD for exact age
    f_mean = interpolate.interp1d(stats['age_months'], stats[f'{measurement_type}_mean'],
                                   kind='linear', fill_value='extrapolate')
    f_sd = interpolate.interp1d(stats['age_months'], stats[f'{measurement_type}_sd'],
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
    else:  # weight
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

def calculate_age_in_months(birth_date, measurement_date):
    """Calculate age in months between two dates"""
    years = measurement_date.year - birth_date.year
    months = measurement_date.month - birth_date.month
    days = measurement_date.day - birth_date.day

    total_months = years * 12 + months
    if days < 0:
        total_months -= 1

    return total_months

# Streamlit App
st.set_page_config(
    page_title="Child Growth Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Child Growth Tracker with WHO Standards (0-19 years)"
    }
)
st.title("👶 Child Growth Tracker with WHO Percentiles")

st.markdown("""
Track child growth from birth to 19 years with WHO Child Growth Standards and Reference Data.
Charts automatically scale based on your child's age for better visualization.
**Mobile-friendly interface for easy tracking on the go!**
""")

# Initialize session state for storing data points
if 'data_points' not in st.session_state:
    st.session_state.data_points = []
if 'child_info' not in st.session_state:
    st.session_state.child_info = None
if 'today_measurement' not in st.session_state:
    st.session_state.today_measurement = None

# Sidebar for Child Information
with st.sidebar:
    st.header("👤 Child Information")

    child_gender = st.selectbox("Gender", ["Male", "Female"], key="child_gender")
    child_birth_date = st.date_input("Birth Date",
                                      value=date.today().replace(year=date.today().year - 2),
                                      max_value=date.today(),
                                      key="birth_date")

    if st.button("Save Child Info", use_container_width=True):
        st.session_state.child_info = {
            'gender': child_gender,
            'birth_date': child_birth_date
        }
        st.success("Child info saved!")

    st.divider()

    # Today's Measurement Section
    st.header("📅 Today's Measurement")

    if st.session_state.child_info:
        today_date = st.date_input("Measurement Date",
                                    value=date.today(),
                                    max_value=date.today(),
                                    key="today_date")

        age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], today_date)
        st.info(f"Age: {age_months} months ({age_months // 12} years, {age_months % 12} months)")

        today_height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=75.0, step=0.1, key="today_height")
        today_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=150.0, value=10.0, step=0.1, key="today_weight")

        if st.button("Save Today's Measurement", use_container_width=True):
            st.session_state.today_measurement = {
                'date': today_date,
                'age_months': age_months,
                'height': today_height,
                'weight': today_weight,
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
                                   max_value=date.today(),
                                   key="hist_date")

        hist_age_months = calculate_age_in_months(st.session_state.child_info['birth_date'], hist_date)
        st.info(f"Age: {hist_age_months} months")

        hist_height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=70.0, step=0.1, key="hist_height")
        hist_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=150.0, value=9.0, step=0.1, key="hist_weight")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Point", use_container_width=True):
                st.session_state.data_points.append({
                    'date': hist_date,
                    'gender': st.session_state.child_info['gender'],
                    'age': hist_age_months,
                    'height': hist_height,
                    'weight': hist_weight
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
            st.dataframe(df_display[['date', 'age', 'height', 'weight']], use_container_width=True)
    else:
        st.warning("Please save child info first")

# Main content - Today's Measurement Z-scores
if st.session_state.today_measurement:
    st.header("📈 Today's Measurement Analysis")

    today = st.session_state.today_measurement
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Height Analysis")
        height_z, height_perc, height_mean, height_sd = calculate_z_score(
            today['age_months'], today['height'], 'height', today['gender']
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
        weight_z, weight_perc, weight_mean, weight_sd = calculate_z_score(
            today['age_months'], today['weight'], 'weight', today['gender']
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

    st.divider()

# Main content - Visualizations
if st.session_state.child_info:
    st.header(f"📊 Growth Charts - {st.session_state.child_info['gender']}")

    selected_gender = st.session_state.child_info['gender']
    who_height = get_who_data(selected_gender)
    who_weight = get_who_weight_data(selected_gender)

    # Calculate child's current age for dynamic chart scaling
    if st.session_state.today_measurement:
        current_age = st.session_state.today_measurement['age_months']
    else:
        current_age = calculate_age_in_months(st.session_state.child_info['birth_date'], date.today())

    # Dynamic X-axis range based on age
    if current_age <= 12:  # 0-1 year
        x_range = [0, 24]  # Show 0-2 years
    elif current_age <= 60:  # 1-5 years
        x_range = [0, 72]  # Show 0-6 years
    elif current_age <= 120:  # 5-10 years
        x_range = [0, 144]  # Show 0-12 years
    else:  # 10+ years
        x_range = [0, 228]  # Show full range 0-19 years

    # For mobile: Use single column layout on narrow screens
    # Streamlit automatically adjusts columns but we can control height
    col1, col2 = st.columns([1, 1])

    # Height-for-Age Chart
    with col1:
        fig_height = go.Figure()

        # Add WHO percentile lines
        colors = {'p3': 'lightcoral', 'p15': 'lightblue', 'p50': 'green',
                 'p85': 'lightblue', 'p97': 'lightcoral'}
        names = {'p3': '3rd percentile', 'p15': '15th percentile',
                'p50': '50th percentile (median)', 'p85': '85th percentile',
                'p97': '97th percentile'}

        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_height.add_trace(go.Scatter(
                x=who_height['age_months'],
                y=who_height[percentile],
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

        fig_height.update_layout(
            title=f"Height-for-Age ({selected_gender})",
            xaxis_title="Age (months)",
            yaxis_title="Height (cm)",
            hovermode='closest',
            showlegend=True,
            height=500,
            xaxis=dict(range=x_range),
            # Mobile optimization
            font=dict(size=12),
            margin=dict(l=50, r=20, t=50, b=50)
        )

        st.plotly_chart(fig_height, use_container_width=True)

    # Weight-for-Age Chart
    with col2:
        fig_weight = go.Figure()

        # Add WHO percentile lines
        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_weight.add_trace(go.Scatter(
                x=who_weight['age_months'],
                y=who_weight[percentile],
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

        fig_weight.update_layout(
            title=f"Weight-for-Age ({selected_gender})",
            xaxis_title="Age (months)",
            yaxis_title="Weight (kg)",
            hovermode='closest',
            showlegend=True,
            height=500,
            xaxis=dict(range=x_range),
            # Mobile optimization
            font=dict(size=12),
            margin=dict(l=50, r=20, t=50, b=50)
        )

        st.plotly_chart(fig_weight, use_container_width=True)
else:
    st.info("👈 Please save child information in the sidebar to get started!")

    # Show sample charts
    st.subheader("Example: WHO Growth Standards")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Height-for-Age Chart (Male)**")
        who_sample = get_who_data("Male")
        fig_sample = go.Figure()
        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_sample.add_trace(go.Scatter(
                x=who_sample['age_months'],
                y=who_sample[percentile],
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
        who_weight_sample = get_who_weight_data("Male")
        fig_weight_sample = go.Figure()
        for percentile in ['p3', 'p15', 'p50', 'p85', 'p97']:
            fig_weight_sample.add_trace(go.Scatter(
                x=who_weight_sample['age_months'],
                y=who_weight_sample[percentile],
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
**Data Source:** This app uses official WHO Child Growth Standards (2006) and WHO Growth Reference Data (5-19 years, 2007)
based on LMS (Lambda-Mu-Sigma) parameters from WHO publications.

**Note:** This app is for educational and informational purposes.
For clinical use and medical decisions, please consult healthcare professionals and refer to the official
[WHO Child Growth Standards](https://www.who.int/tools/child-growth-standards).
""")
