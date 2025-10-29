# Child Growth Tracker with WHO Percentiles

A Streamlit web application for tracking child growth measurements and comparing them with WHO (World Health Organization) Child Growth Standards.

## Features

- **Child Profile**: Store child's gender and birth date for automatic age calculation
- **Birth Date Tracking**: Automatically calculates age in months based on birth date and measurement date
- **Today's Measurement**: Dedicated section for current measurements with date picker
- **Z-Score Analysis**: Calculate and display Z-scores and percentiles for today's measurements
- **Clinical Interpretation**: Automatic interpretation of measurements (normal, underweight, overweight, stunted, etc.)
- **WHO Standards**: Compare measurements against WHO growth percentiles (3rd, 15th, 50th, 85th, 97th)
- **Dual Charts**: Visualize both height-for-age and weight-for-age on separate interactive charts
- **Multi-point Tracking**: Add multiple historical measurements to track growth trends over time
- **Visual Differentiation**: Today's measurements displayed as red stars, historical data as blue circles
- **Gender-specific**: Separate charts and percentiles for male and female children
- **Age Range**: Supports children from 0-60 months (0-5 years)

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

3. **Setup Child Profile** (Sidebar - Top Section):
   - Select gender (Male/Female)
   - Enter birth date
   - Click "Save Child Info"

4. **Add Today's Measurement** (Sidebar - Middle Section):
   - Select measurement date (defaults to today)
   - Age is automatically calculated from birth date
   - Enter current height in cm
   - Enter current weight in kg
   - Click "Save Today's Measurement"

5. **View Today's Analysis** (Main area - Top):
   - See Z-scores for both height and weight
   - View percentiles
   - Get clinical interpretation (Normal, Underweight, Overweight, etc.)
   - Expand details for expected mean and standard deviation

6. **Add Historical Data** (Sidebar - Bottom Section):
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
