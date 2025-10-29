# Child Growth Tracker

A Streamlit web application for tracking child growth measurements and comparing them with WHO or CDC growth standards.

## Features

- **Multiple Data Sources**: Choose between WHO (global/multi-ethnic) or CDC (US population) growth charts
- **Smart Data Source Recommendations**: Automatically recommends the best data source based on child's age
- **Automatic Data Source Switching**: When selected data source doesn't have data for child's age, automatically switches to alternative source
- **Child Profile**: Store child's gender and birth date for automatic age calculation
- **Birth Date Tracking**: Automatically calculates age in months based on birth date and measurement date
- **Today's Measurement**: Dedicated section for current measurements with date picker
- **Z-Score Analysis**: Calculate and display Z-scores and percentiles for today's measurements
- **Clinical Interpretation**: Automatic interpretation of measurements (normal, underweight, overweight, stunted, etc.)
- **Growth Percentiles**: Compare measurements against growth percentiles (3rd, 15th, 50th, 85th, 97th)
- **BMI Tracking**: Automatic BMI calculation and BMI-for-age analysis (WHO, 5-19 years)
- **BMI Z-Score Analysis**: Comprehensive BMI analysis with percentiles and clinical interpretation
- **BMI-for-Age Chart**: Dedicated chart showing BMI growth trends (WHO recommended indicator for thinness/overweight)
- **Triple Charts**: Visualize height-for-age, weight-for-age, and BMI-for-age on interactive charts
- **Smart Age-Range Display**: Charts automatically switch between 0-5 years and 5-19 years ranges based on child's age
- **Multi-point Tracking**: Add multiple historical measurements to track growth trends over time
- **Visual Differentiation**: Today's measurements displayed as red stars, historical data as blue circles
- **Gender-specific**: Separate charts and percentiles for male and female children
- **Age Range**:
  - WHO:
    - Height-for-age: 0-228 months (0-19 years)
    - Weight-for-age: 0-120 months (0-10 years only)
    - BMI-for-age: 61-228 months (5-19 years only)
  - CDC: 24-240 months (2-20 years) - BMI not yet implemented
- **PDF Export**: Download comprehensive growth reports with Z-scores and charts
- **Mobile-Friendly**: Optimized for use on mobile devices

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run child_growth_app.py
```

2. The app will open in your default web browser

3. **Setup Child Profile** (Sidebar - First):
   - Select gender (Male/Female)
   - Enter birth date
   - Click "Save Child Info"

4. **Select Data Source** (Sidebar - After child info):
   - App automatically recommends best data source for your child's age
   - Choose between WHO or CDC growth charts
   - WHO: Global multi-ethnic population (0-19 years)
   - CDC: US population data (2-20 years)

5. **Add Today's Measurement** (Sidebar):
   - Select measurement date (defaults to today)
   - Age is automatically calculated from birth date
   - Enter current height in cm
   - Enter current weight in kg
   - Click "Save Today's Measurement"

6. **View Today's Analysis** (Main area - Top):
   - See Z-scores for both height and weight
   - View percentiles
   - Get clinical interpretation (Normal, Underweight, Overweight, etc.)
   - Expand details for expected mean and standard deviation

7. **Add Historical Data** (Sidebar - Bottom):
   - Select past measurement date
   - Age is automatically calculated
   - Enter height and weight
   - Click "Add Point" to add to historical data
   - View historical data table below

7. **View Growth Charts** (Main area - Bottom):
   - Red stars represent today's measurements
   - Blue circles with dashed lines represent historical measurements
   - Colored lines show WHO growth percentiles
   - Hover over points for detailed values

8. Use "Clear All" to remove all historical data points

## Data Points Visualization

- **Red stars**: Today's measurements (prominently displayed)
- **Blue circles with dashed lines**: Historical measurements
- **Green line**: 50th percentile (median)
- **Light blue lines**: 15th and 85th percentiles
- **Light coral lines**: 3rd and 97th percentiles

## Z-Score Interpretation

Z-scores indicate how many standard deviations a measurement is from the WHO reference mean:

### Height-for-Age:
- **Below -3**: Severely stunted (⚠️)
- **-3 to -2**: Stunted (⚠️)
- **-2 to +2**: Normal (✅)
- **+2 to +3**: Tall (⚠️)
- **Above +3**: Very tall (⚠️)

### Weight-for-Age:
- **Below -3**: Severely underweight (⚠️)
- **-3 to -2**: Underweight (⚠️)
- **-2 to +2**: Normal (✅)
- **+2 to +3**: Overweight (⚠️)
- **Above +3**: Obese (⚠️)

## Note

The WHO growth standards data included in this app is simplified sample data for demonstration purposes. For clinical or medical use, please refer to the official [WHO Child Growth Standards](https://www.who.int/tools/child-growth-standards).

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.17.0
- numpy >= 1.24.0
- scipy >= 1.11.0
