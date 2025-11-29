"""
E2E tests for measurement analysis flows.

These tests verify that Z-score calculations, percentile calculations,
and clinical interpretations work correctly.
"""
import pytest
import sys
import os
from datetime import date

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from conftest will set up streamlit mock
from child_growth_app import (
    calculate_z_score,
    interpret_z_score,
    calculate_bmi,
    get_default_measurements
)


class TestZScoreCalculation:
    """Test class for Z-score calculation flows."""
    
    # ===== WHO Height Z-Score Tests =====
    
    def test_who_height_z_score_calculation_for_12_month_male(self):
        """Test Z-score calculation for a 12-month-old male's height."""
        # WHO median for 12-month male is approximately 75.7 cm
        z_score, percentile, mean, sd = calculate_z_score(12, 75.7, 'height', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"
        assert mean is not None, "Mean should not be None"
        assert sd is not None, "SD should not be None"
        
        # Z-score should be close to 0 for median height
        assert abs(z_score) < 0.5, f"Z-score for median height should be close to 0, got {z_score}"
        
    def test_who_height_z_score_calculation_for_12_month_female(self):
        """Test Z-score calculation for a 12-month-old female's height."""
        # WHO median for 12-month female is approximately 74.0 cm
        z_score, percentile, mean, sd = calculate_z_score(12, 74.0, 'height', 'Female', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"
        
        # Z-score should be close to 0 for median height
        assert abs(z_score) < 0.5, f"Z-score for median height should be close to 0, got {z_score}"
        
    def test_who_height_z_score_for_short_child(self):
        """Test Z-score calculation for a short child (below 3rd percentile)."""
        # For a 24-month male, 3rd percentile is around 81-82 cm
        z_score, percentile, mean, sd = calculate_z_score(24, 78.0, 'height', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert z_score < -2, f"Z-score for a short child should be below -2, got {z_score}"
        assert percentile < 5, f"Percentile should be below 5, got {percentile}"
        
    def test_who_height_z_score_for_tall_child(self):
        """Test Z-score calculation for a tall child (above 97th percentile)."""
        # For a 24-month male, 97th percentile is around 93 cm
        z_score, percentile, mean, sd = calculate_z_score(24, 96.0, 'height', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert z_score > 2, f"Z-score for a tall child should be above 2, got {z_score}"
        assert percentile > 95, f"Percentile should be above 95, got {percentile}"

    # ===== WHO Weight Z-Score Tests =====
    
    def test_who_weight_z_score_calculation_for_12_month_male(self):
        """Test Z-score calculation for a 12-month-old male's weight."""
        # WHO median for 12-month male is approximately 10.2 kg
        z_score, percentile, mean, sd = calculate_z_score(12, 10.2, 'weight', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"
        
        # Z-score should be close to 0 for median weight
        assert abs(z_score) < 0.5, f"Z-score for median weight should be close to 0, got {z_score}"
        
    def test_who_weight_z_score_for_underweight_child(self):
        """Test Z-score calculation for an underweight child."""
        # For a 12-month male, median weight is ~10.2 kg, testing with much lower weight
        z_score, percentile, mean, sd = calculate_z_score(12, 7.0, 'weight', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert z_score < -2, f"Z-score for underweight child should be below -2, got {z_score}"

    # ===== CDC Z-Score Tests =====
    
    def test_cdc_height_z_score_calculation_for_36_month_male(self):
        """Test CDC Z-score calculation for a 36-month-old male's height."""
        # CDC median for 36-month male is approximately 95-96 cm
        z_score, percentile, mean, sd = calculate_z_score(36, 95.5, 'height', 'Male', 'CDC')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"
        
    def test_cdc_weight_z_score_calculation_for_36_month_female(self):
        """Test CDC Z-score calculation for a 36-month-old female's weight."""
        z_score, percentile, mean, sd = calculate_z_score(36, 14.0, 'weight', 'Female', 'CDC')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"

    # ===== BMI Z-Score Tests =====
    
    def test_who_bmi_z_score_calculation_for_older_child(self):
        """Test BMI Z-score calculation for an older child (WHO covers 5-19 years)."""
        # For a 72-month (6 year) old, BMI median is approximately 15.2
        z_score, percentile, mean, sd = calculate_z_score(72, 15.2, 'bmi', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"
        
    def test_cdc_bmi_z_score_calculation_for_child(self):
        """Test BMI Z-score calculation using CDC data (covers 2-20 years)."""
        # For a 36-month (3 year) old
        z_score, percentile, mean, sd = calculate_z_score(36, 16.0, 'bmi', 'Male', 'CDC')
        
        assert z_score is not None, "Z-score should not be None"
        assert percentile is not None, "Percentile should not be None"

    # ===== Edge Case Tests =====
    
    def test_z_score_for_newborn(self):
        """Test Z-score calculation for a newborn (0 months)."""
        # WHO median for newborn male is approximately 49.9 cm
        z_score, percentile, mean, sd = calculate_z_score(0, 49.9, 'height', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score for newborn should not be None"
        assert abs(z_score) < 0.5, f"Z-score for median newborn height should be close to 0, got {z_score}"
        
    def test_z_score_for_teenager(self):
        """Test Z-score calculation for a teenager (15 years = 180 months)."""
        # WHO median for 180-month male is approximately 170 cm
        z_score, percentile, mean, sd = calculate_z_score(180, 170.0, 'height', 'Male', 'WHO')
        
        assert z_score is not None, "Z-score for teenager should not be None"


class TestZScoreInterpretation:
    """Test class for Z-score interpretation flows."""
    
    # ===== Height Interpretation Tests =====
    
    def test_height_severely_stunted_interpretation(self):
        """Test interpretation for severely stunted child (Z < -3)."""
        interpretation, status = interpret_z_score(-3.5, 'height')
        
        assert 'Severely stunted' in interpretation, "Should indicate severely stunted"
        assert status == 'danger', "Status should be danger"
        
    def test_height_stunted_interpretation(self):
        """Test interpretation for stunted child (-3 <= Z < -2)."""
        interpretation, status = interpret_z_score(-2.5, 'height')
        
        assert 'Stunted' in interpretation, "Should indicate stunted"
        assert status == 'warning', "Status should be warning"
        
    def test_height_normal_interpretation(self):
        """Test interpretation for normal height (-2 <= Z <= 2)."""
        interpretation, status = interpret_z_score(0.0, 'height')
        
        assert 'Normal' in interpretation, "Should indicate normal"
        assert status == 'success', "Status should be success"
        
    def test_height_tall_interpretation(self):
        """Test interpretation for tall child (2 < Z <= 3)."""
        interpretation, status = interpret_z_score(2.5, 'height')
        
        assert 'Tall' in interpretation, "Should indicate tall"
        assert status == 'warning', "Status should be warning"
        
    def test_height_very_tall_interpretation(self):
        """Test interpretation for very tall child (Z > 3)."""
        interpretation, status = interpret_z_score(3.5, 'height')
        
        assert 'Very tall' in interpretation, "Should indicate very tall"
        assert status == 'danger', "Status should be danger"
        
    # ===== Weight Interpretation Tests =====
    
    def test_weight_severely_underweight_interpretation(self):
        """Test interpretation for severely underweight child (Z < -3)."""
        interpretation, status = interpret_z_score(-3.5, 'weight')
        
        assert 'Severely underweight' in interpretation, "Should indicate severely underweight"
        assert status == 'danger', "Status should be danger"
        
    def test_weight_underweight_interpretation(self):
        """Test interpretation for underweight child (-3 <= Z < -2)."""
        interpretation, status = interpret_z_score(-2.5, 'weight')
        
        assert 'Underweight' in interpretation, "Should indicate underweight"
        assert status == 'warning', "Status should be warning"
        
    def test_weight_normal_interpretation(self):
        """Test interpretation for normal weight (-2 <= Z <= 2)."""
        interpretation, status = interpret_z_score(0.0, 'weight')
        
        assert 'Normal' in interpretation, "Should indicate normal"
        assert status == 'success', "Status should be success"
        
    def test_weight_overweight_interpretation(self):
        """Test interpretation for overweight child (2 < Z <= 3)."""
        interpretation, status = interpret_z_score(2.5, 'weight')
        
        assert 'Overweight' in interpretation, "Should indicate overweight"
        assert status == 'warning', "Status should be warning"
        
    def test_weight_obese_interpretation(self):
        """Test interpretation for obese child (Z > 3)."""
        interpretation, status = interpret_z_score(3.5, 'weight')
        
        assert 'Obese' in interpretation, "Should indicate obese"
        assert status == 'danger', "Status should be danger"
        
    # ===== BMI Interpretation Tests =====
    
    def test_bmi_severely_wasted_interpretation(self):
        """Test interpretation for severely wasted child (Z < -3)."""
        interpretation, status = interpret_z_score(-3.5, 'bmi')
        
        assert 'Severely wasted' in interpretation, "Should indicate severely wasted"
        assert status == 'danger', "Status should be danger"
        
    def test_bmi_wasted_interpretation(self):
        """Test interpretation for wasted child (-3 <= Z < -2)."""
        interpretation, status = interpret_z_score(-2.5, 'bmi')
        
        assert 'Wasted' in interpretation, "Should indicate wasted"
        assert status == 'warning', "Status should be warning"
        
    def test_bmi_normal_interpretation(self):
        """Test interpretation for normal BMI (-2 <= Z <= 1)."""
        interpretation, status = interpret_z_score(0.0, 'bmi')
        
        assert 'Normal' in interpretation, "Should indicate normal"
        assert status == 'success', "Status should be success"
        
    def test_bmi_overweight_interpretation(self):
        """Test interpretation for overweight child (1 < Z <= 2)."""
        interpretation, status = interpret_z_score(1.5, 'bmi')
        
        assert 'Overweight' in interpretation, "Should indicate overweight"
        assert status == 'warning', "Status should be warning"
        
    def test_bmi_obese_interpretation(self):
        """Test interpretation for obese child (Z > 2)."""
        interpretation, status = interpret_z_score(2.5, 'bmi')
        
        assert 'Obese' in interpretation, "Should indicate obese"
        assert status == 'danger', "Status should be danger"


class TestBMICalculation:
    """Test class for BMI calculation flows."""
    
    def test_bmi_calculation_standard_case(self):
        """Test BMI calculation with standard values."""
        # Height: 100 cm (1.0 m), Weight: 20 kg
        # BMI = 20 / (1.0^2) = 20
        bmi = calculate_bmi(100.0, 20.0)
        
        assert bmi == pytest.approx(20.0, rel=0.01), f"Expected BMI of 20.0, got {bmi}"
        
    def test_bmi_calculation_for_infant(self):
        """Test BMI calculation for an infant."""
        # Height: 75 cm (0.75 m), Weight: 10 kg
        # BMI = 10 / (0.75^2) = 17.78
        bmi = calculate_bmi(75.0, 10.0)
        
        assert bmi == pytest.approx(17.78, rel=0.01), f"Expected BMI of ~17.78, got {bmi}"
        
    def test_bmi_calculation_for_older_child(self):
        """Test BMI calculation for an older child."""
        # Height: 140 cm (1.4 m), Weight: 32 kg
        # BMI = 32 / (1.4^2) = 16.33
        bmi = calculate_bmi(140.0, 32.0)
        
        assert bmi == pytest.approx(16.33, rel=0.01), f"Expected BMI of ~16.33, got {bmi}"
        
    def test_bmi_calculation_precision(self):
        """Test BMI calculation maintains appropriate precision."""
        # Height: 120 cm, Weight: 25 kg
        # BMI = 25 / (1.2^2) = 17.361111...
        bmi = calculate_bmi(120.0, 25.0)
        
        # Check that BMI is a float
        assert isinstance(bmi, float), "BMI should be a float"
        
        # Check expected value
        expected_bmi = 25.0 / (1.2 ** 2)
        assert bmi == pytest.approx(expected_bmi, rel=0.001), f"BMI precision issue: expected {expected_bmi}, got {bmi}"


class TestDefaultMeasurements:
    """Test class for default measurement retrieval flows."""
    
    def test_who_default_measurements_for_12_month_male(self):
        """Test getting default measurements for a 12-month-old male from WHO data."""
        default_height, default_weight = get_default_measurements(12, 'Male', 'WHO')
        
        assert default_height is not None, "Default height should not be None"
        assert default_weight is not None, "Default weight should not be None"
        
        # WHO median for 12-month male: height ~75.7 cm, weight ~9.6 kg
        assert 70.0 < default_height < 80.0, f"Default height should be reasonable, got {default_height}"
        assert 8.0 < default_weight < 12.0, f"Default weight should be reasonable, got {default_weight}"
        
    def test_who_default_measurements_for_12_month_female(self):
        """Test getting default measurements for a 12-month-old female from WHO data."""
        default_height, default_weight = get_default_measurements(12, 'Female', 'WHO')
        
        assert default_height is not None, "Default height should not be None"
        assert default_weight is not None, "Default weight should not be None"
        
        # WHO median for 12-month female: height ~74 cm, weight ~8.9 kg
        assert 68.0 < default_height < 78.0, f"Default height should be reasonable, got {default_height}"
        assert 7.5 < default_weight < 11.0, f"Default weight should be reasonable, got {default_weight}"
        
    def test_cdc_default_measurements_for_36_month_male(self):
        """Test getting default measurements for a 36-month-old male from CDC data."""
        default_height, default_weight = get_default_measurements(36, 'Male', 'CDC')
        
        assert default_height is not None, "Default height should not be None"
        assert default_weight is not None, "Default weight should not be None"
        
        # CDC median for 36-month male: height ~95-96 cm, weight ~14-15 kg
        assert 90.0 < default_height < 100.0, f"Default height should be reasonable, got {default_height}"
        assert 12.0 < default_weight < 17.0, f"Default weight should be reasonable, got {default_weight}"
        
    def test_default_measurements_increase_with_age(self):
        """Test that default measurements increase with age."""
        height_12m, weight_12m = get_default_measurements(12, 'Male', 'WHO')
        height_24m, weight_24m = get_default_measurements(24, 'Male', 'WHO')
        height_60m, weight_60m = get_default_measurements(60, 'Male', 'WHO')
        
        assert height_24m > height_12m, "Height at 24 months should be greater than at 12 months"
        assert height_60m > height_24m, "Height at 60 months should be greater than at 24 months"
        assert weight_24m > weight_12m, "Weight at 24 months should be greater than at 12 months"
        assert weight_60m > weight_24m, "Weight at 60 months should be greater than at 24 months"
        
    def test_default_measurements_gender_difference(self):
        """Test that there's a difference between male and female defaults."""
        male_height, male_weight = get_default_measurements(60, 'Male', 'WHO')
        female_height, female_weight = get_default_measurements(60, 'Female', 'WHO')
        
        # Males typically have slightly higher values
        # The important thing is that they're different
        assert male_height != female_height or male_weight != female_weight, \
            "Male and female defaults should differ"
