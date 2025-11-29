"""
E2E tests for complete growth tracking scenarios.

These tests simulate real-world usage scenarios, testing the complete flow
from child information to measurement analysis.
"""
import pytest
import sys
import os
from datetime import date, datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from conftest will set up streamlit mock
from child_growth_app import (
    calculate_age_in_months,
    calculate_bmi,
    calculate_z_score,
    interpret_z_score,
    get_default_measurements,
    get_height_data,
    get_weight_data,
    get_bmi_data,
    get_growth_statistics
)


class TestCompleteInfantTrackingScenario:
    """
    Test complete growth tracking scenario for an infant (0-2 years).
    Uses WHO data source.
    """
    
    def test_complete_infant_tracking_flow(self):
        """Test complete tracking flow for a 12-month-old infant."""
        # Step 1: Set up child info
        child_info = {
            'gender': 'Male',
            'birth_date': date(2022, 6, 15)
        }
        
        # Step 2: Record measurement
        measurement_date = date(2023, 6, 15)  # 12 months old
        age_months = calculate_age_in_months(child_info['birth_date'], measurement_date)
        
        assert age_months == 12, f"Expected 12 months, got {age_months}"
        
        # Step 3: Get actual measurements
        height = 76.0  # cm
        weight = 10.0  # kg
        
        # Step 4: Calculate BMI
        bmi = calculate_bmi(height, weight)
        assert bmi is not None, "BMI calculation should not return None"
        assert 14 < bmi < 20, f"BMI should be in reasonable range for infant, got {bmi}"
        
        # Step 5: Calculate Z-scores using WHO data
        height_z, height_perc, height_mean, height_sd = calculate_z_score(
            age_months, height, 'height', child_info['gender'], 'WHO'
        )
        weight_z, weight_perc, weight_mean, weight_sd = calculate_z_score(
            age_months, weight, 'weight', child_info['gender'], 'WHO'
        )
        
        # All Z-score values should be calculated
        assert height_z is not None, "Height Z-score should not be None"
        assert height_perc is not None, "Height percentile should not be None"
        assert weight_z is not None, "Weight Z-score should not be None"
        assert weight_perc is not None, "Weight percentile should not be None"
        
        # Step 6: Get interpretations
        height_interp, height_status = interpret_z_score(height_z, 'height')
        weight_interp, weight_status = interpret_z_score(weight_z, 'weight')
        
        assert height_interp is not None, "Height interpretation should not be None"
        assert weight_interp is not None, "Weight interpretation should not be None"
        
        # Step 7: Verify the measurement is within normal range for a healthy infant
        # A height of 76cm and weight of 10kg at 12 months is very close to median
        assert abs(height_z) < 1.5, f"Height Z-score should be close to normal, got {height_z}"
        assert abs(weight_z) < 1.5, f"Weight Z-score should be close to normal, got {weight_z}"


class TestCompleteToddlerTrackingScenario:
    """
    Test complete growth tracking scenario for a toddler (2-5 years).
    Tests both WHO and CDC data sources.
    """
    
    def test_complete_toddler_tracking_with_who(self):
        """Test complete tracking flow for a 36-month-old toddler using WHO data."""
        # Set up child info
        child_info = {
            'gender': 'Female',
            'birth_date': date(2020, 3, 10)
        }
        
        # Record measurement
        measurement_date = date(2023, 3, 10)  # 36 months (3 years)
        age_months = calculate_age_in_months(child_info['birth_date'], measurement_date)
        
        assert age_months == 36, f"Expected 36 months, got {age_months}"
        
        # Measurements for average 3-year-old female
        height = 95.0  # cm
        weight = 14.0  # kg
        
        # Calculate Z-scores using WHO
        height_z, height_perc, _, _ = calculate_z_score(age_months, height, 'height', 'Female', 'WHO')
        weight_z, weight_perc, _, _ = calculate_z_score(age_months, weight, 'weight', 'Female', 'WHO')
        
        # Verify Z-scores are calculated
        assert height_z is not None, "WHO height Z-score should be calculated for 36 months"
        assert weight_z is not None, "WHO weight Z-score should be calculated for 36 months"
        
    def test_complete_toddler_tracking_with_cdc(self):
        """Test complete tracking flow for a 36-month-old toddler using CDC data."""
        # Set up child info
        child_info = {
            'gender': 'Male',
            'birth_date': date(2020, 6, 15)
        }
        
        # Record measurement
        measurement_date = date(2023, 6, 15)
        age_months = calculate_age_in_months(child_info['birth_date'], measurement_date)
        
        # Measurements for average 3-year-old male
        height = 97.0  # cm
        weight = 15.0  # kg
        
        # Calculate BMI
        bmi = calculate_bmi(height, weight)
        
        # Calculate Z-scores using CDC
        height_z, _, _, _ = calculate_z_score(age_months, height, 'height', 'Male', 'CDC')
        weight_z, _, _, _ = calculate_z_score(age_months, weight, 'weight', 'Male', 'CDC')
        bmi_z, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Male', 'CDC')
        
        # CDC covers 2-20 years, so all should be available
        assert height_z is not None, "CDC height Z-score should be calculated for 36 months"
        assert weight_z is not None, "CDC weight Z-score should be calculated for 36 months"
        assert bmi_z is not None, "CDC BMI Z-score should be calculated for 36 months"


class TestCompleteSchoolAgeTrackingScenario:
    """
    Test complete growth tracking scenario for school-age child (5-10 years).
    Tests WHO BMI data availability.
    """
    
    def test_complete_school_age_tracking(self):
        """Test complete tracking flow for a 7-year-old child."""
        # Set up child info
        child_info = {
            'gender': 'Male',
            'birth_date': date(2016, 8, 20)
        }
        
        # Record measurement
        measurement_date = date(2023, 8, 20)  # 84 months (7 years)
        age_months = calculate_age_in_months(child_info['birth_date'], measurement_date)
        
        assert age_months == 84, f"Expected 84 months, got {age_months}"
        
        # Measurements for average 7-year-old male
        height = 122.0  # cm
        weight = 23.0  # kg
        bmi = calculate_bmi(height, weight)
        
        # Test WHO data
        height_z_who, _, _, _ = calculate_z_score(age_months, height, 'height', 'Male', 'WHO')
        weight_z_who, _, _, _ = calculate_z_score(age_months, weight, 'weight', 'Male', 'WHO')
        bmi_z_who, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Male', 'WHO')
        
        # WHO height covers 0-19 years
        assert height_z_who is not None, "WHO height Z-score should be available at 7 years"
        
        # WHO weight only covers 0-10 years, so should still be available
        assert weight_z_who is not None, "WHO weight Z-score should be available at 7 years"
        
        # WHO BMI covers 5-19 years, so should be available
        assert bmi_z_who is not None, "WHO BMI Z-score should be available at 7 years (>5 years)"
        
    def test_data_source_consistency(self):
        """Test that growth data from both sources is consistently available."""
        age_months = 84  # 7 years
        gender = 'Female'
        
        # Both sources should have height data
        who_height = get_height_data(gender, 'WHO')
        cdc_height = get_height_data(gender, 'CDC')
        
        assert not who_height.empty, "WHO height data should be available"
        assert not cdc_height.empty, "CDC height data should be available"


class TestCompleteAdolescentTrackingScenario:
    """
    Test complete growth tracking scenario for an adolescent (10-19 years).
    Tests handling of WHO weight-for-age limitation (only up to 10 years).
    """
    
    def test_complete_adolescent_tracking(self):
        """Test complete tracking flow for a 15-year-old adolescent."""
        # Set up child info
        child_info = {
            'gender': 'Female',
            'birth_date': date(2008, 4, 12)
        }
        
        # Record measurement
        measurement_date = date(2023, 4, 12)  # 180 months (15 years)
        age_months = calculate_age_in_months(child_info['birth_date'], measurement_date)
        
        assert age_months == 180, f"Expected 180 months, got {age_months}"
        
        # Measurements for average 15-year-old female
        height = 162.0  # cm
        weight = 52.0  # kg
        bmi = calculate_bmi(height, weight)
        
        # WHO height data covers 0-19 years
        height_z_who, _, _, _ = calculate_z_score(age_months, height, 'height', 'Female', 'WHO')
        assert height_z_who is not None, "WHO height Z-score should be available at 15 years"
        
        # WHO BMI data covers 5-19 years
        bmi_z_who, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Female', 'WHO')
        assert bmi_z_who is not None, "WHO BMI Z-score should be available at 15 years"
        
        # CDC data covers 2-20 years for all metrics
        height_z_cdc, _, _, _ = calculate_z_score(age_months, height, 'height', 'Female', 'CDC')
        weight_z_cdc, _, _, _ = calculate_z_score(age_months, weight, 'weight', 'Female', 'CDC')
        bmi_z_cdc, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Female', 'CDC')
        
        assert height_z_cdc is not None, "CDC height Z-score should be available at 15 years"
        assert weight_z_cdc is not None, "CDC weight Z-score should be available at 15 years"
        assert bmi_z_cdc is not None, "CDC BMI Z-score should be available at 15 years"


class TestGrowthTrackingWithMultipleMeasurements:
    """
    Test growth tracking scenarios with multiple measurements over time.
    """
    
    def test_tracking_growth_over_time(self):
        """Test tracking a child's growth from 6 months to 24 months."""
        child_info = {
            'gender': 'Male',
            'birth_date': date(2022, 1, 15)
        }
        
        # Series of measurements over time
        measurements = [
            {'date': date(2022, 7, 15), 'height': 68.0, 'weight': 8.0},   # 6 months
            {'date': date(2022, 10, 15), 'height': 72.0, 'weight': 9.0},  # 9 months
            {'date': date(2023, 1, 15), 'height': 76.0, 'weight': 10.0},  # 12 months
            {'date': date(2023, 7, 15), 'height': 84.0, 'weight': 12.0},  # 18 months
            {'date': date(2024, 1, 15), 'height': 87.0, 'weight': 12.5},  # 24 months
        ]
        
        previous_height_z = None
        previous_weight_z = None
        
        for measurement in measurements:
            age_months = calculate_age_in_months(child_info['birth_date'], measurement['date'])
            
            height_z, _, _, _ = calculate_z_score(
                age_months, measurement['height'], 'height', 'Male', 'WHO'
            )
            weight_z, _, _, _ = calculate_z_score(
                age_months, measurement['weight'], 'weight', 'Male', 'WHO'
            )
            
            # All Z-scores should be calculable
            assert height_z is not None, f"Height Z-score should be calculated at {age_months} months"
            assert weight_z is not None, f"Weight Z-score should be calculated at {age_months} months"
            
            # Growth should be consistent (Z-scores shouldn't change dramatically)
            if previous_height_z is not None:
                z_change = abs(height_z - previous_height_z)
                # Z-score change should be reasonable (< 2 between measurements)
                assert z_change < 2, f"Z-score change too large: {z_change}"
                
            previous_height_z = height_z
            previous_weight_z = weight_z
            
    def test_growth_measurements_increase_with_age(self):
        """Test that growth measurements increase with age for healthy child."""
        child_info = {
            'gender': 'Female',
            'birth_date': date(2020, 6, 1)
        }
        
        # Get default (median) measurements at different ages
        measurements = []
        for age_months in [12, 24, 36, 48, 60]:
            height, weight = get_default_measurements(age_months, 'Female', 'WHO')
            measurements.append({
                'age': age_months,
                'height': height,
                'weight': weight
            })
            
        # Verify measurements increase with age
        for i in range(1, len(measurements)):
            assert measurements[i]['height'] > measurements[i-1]['height'], \
                f"Height should increase: {measurements[i]['age']} months should be > {measurements[i-1]['age']} months"
            assert measurements[i]['weight'] > measurements[i-1]['weight'], \
                f"Weight should increase: {measurements[i]['age']} months should be > {measurements[i-1]['age']} months"


class TestEdgeCasesInGrowthTracking:
    """
    Test edge cases in growth tracking scenarios.
    """
    
    def test_newborn_tracking(self):
        """Test tracking for a newborn (0 months)."""
        # Newborn male measurements
        age_months = 0
        height = 50.0  # cm
        weight = 3.5  # kg
        
        height_z, _, _, _ = calculate_z_score(age_months, height, 'height', 'Male', 'WHO')
        weight_z, _, _, _ = calculate_z_score(age_months, weight, 'weight', 'Male', 'WHO')
        
        assert height_z is not None, "Height Z-score should be calculable for newborn"
        assert weight_z is not None, "Weight Z-score should be calculable for newborn"
        
    def test_boundary_age_tracking(self):
        """Test tracking at age boundaries between data sources."""
        # Test at 24 months (CDC lower boundary)
        age_months = 24
        height, weight = get_default_measurements(age_months, 'Male', 'CDC')
        
        height_z, _, _, _ = calculate_z_score(age_months, height, 'height', 'Male', 'CDC')
        weight_z, _, _, _ = calculate_z_score(age_months, weight, 'weight', 'Male', 'CDC')
        
        assert height_z is not None, "CDC height Z-score should be available at 24 months"
        assert weight_z is not None, "CDC weight Z-score should be available at 24 months"
        
        # Test at 61 months (WHO BMI lower boundary)
        age_months = 61
        height, weight = get_default_measurements(age_months, 'Female', 'WHO')
        bmi = calculate_bmi(height, weight)
        
        bmi_z, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Female', 'WHO')
        assert bmi_z is not None, "WHO BMI Z-score should be available at 61 months"
        
    def test_extreme_measurements(self):
        """Test tracking with extreme (but possible) measurements."""
        age_months = 60  # 5 years
        
        # Very short child
        height_z_short, _, _, _ = calculate_z_score(age_months, 95.0, 'height', 'Male', 'WHO')
        
        # Very tall child
        height_z_tall, _, _, _ = calculate_z_score(age_months, 120.0, 'height', 'Male', 'WHO')
        
        # Both should be calculable
        assert height_z_short is not None, "Z-score should be calculable for short child"
        assert height_z_tall is not None, "Z-score should be calculable for tall child"
        
        # Short child should have negative Z-score
        assert height_z_short < 0, f"Short child should have negative Z-score, got {height_z_short}"
        
        # Tall child should have positive Z-score
        assert height_z_tall > 0, f"Tall child should have positive Z-score, got {height_z_tall}"


class TestDataSourceTransitions:
    """
    Test scenarios where data source transitions are needed.
    """
    
    def test_who_to_cdc_transition_for_weight(self):
        """Test transition from WHO to CDC for weight data after 10 years."""
        # At 120 months (10 years), WHO weight data is at its limit
        age_months = 120
        
        # WHO weight data should still be available at 10 years
        who_weight_z, _, _, _ = calculate_z_score(age_months, 32.0, 'weight', 'Male', 'WHO')
        assert who_weight_z is not None, "WHO weight should be available at exactly 10 years"
        
        # CDC weight data should also be available
        cdc_weight_z, _, _, _ = calculate_z_score(age_months, 32.0, 'weight', 'Male', 'CDC')
        assert cdc_weight_z is not None, "CDC weight should be available at 10 years"
        
    def test_bmi_availability_transitions(self):
        """Test BMI data availability transitions."""
        # At 60 months (5 years), WHO BMI might not be available
        age_60 = 60
        _, _, _, _ = calculate_z_score(age_60, 15.5, 'bmi', 'Male', 'WHO')
        
        # At 61 months, WHO BMI should be available
        age_61 = 61
        bmi_z_61, _, _, _ = calculate_z_score(age_61, 15.5, 'bmi', 'Male', 'WHO')
        assert bmi_z_61 is not None, "WHO BMI should be available at 61 months"
        
        # CDC BMI should be available at both ages (covers 24-240 months)
        cdc_bmi_60, _, _, _ = calculate_z_score(age_60, 15.5, 'bmi', 'Male', 'CDC')
        cdc_bmi_61, _, _, _ = calculate_z_score(age_61, 15.5, 'bmi', 'Male', 'CDC')
        
        assert cdc_bmi_60 is not None, "CDC BMI should be available at 60 months"
        assert cdc_bmi_61 is not None, "CDC BMI should be available at 61 months"
