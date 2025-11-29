"""
Unit tests for utility functions in growth_utils.py
"""

import pytest
from datetime import date
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions to test
from growth_utils import (
    calculate_bmi,
    calculate_age_in_months,
    interpret_z_score
)


class TestCalculateBMI:
    """Tests for calculate_bmi function"""

    def test_calculate_bmi_normal_values(self):
        """Test BMI calculation with normal height and weight"""
        # 170 cm, 70 kg -> BMI = 70 / (1.70^2) = 24.22
        result = calculate_bmi(170, 70)
        assert round(result, 2) == 24.22

    def test_calculate_bmi_child_values(self):
        """Test BMI calculation with child-typical values"""
        # 100 cm, 15 kg -> BMI = 15 / (1.0^2) = 15.0
        result = calculate_bmi(100, 15)
        assert result == 15.0

    def test_calculate_bmi_infant_values(self):
        """Test BMI calculation with infant values"""
        # 50 cm, 3.5 kg -> BMI = 3.5 / (0.5^2) = 14.0
        result = calculate_bmi(50, 3.5)
        assert result == 14.0

    def test_calculate_bmi_tall_adult(self):
        """Test BMI calculation with tall adult values"""
        # 190 cm, 85 kg -> BMI = 85 / (1.9^2) = 23.55
        result = calculate_bmi(190, 85)
        assert round(result, 2) == 23.55

    def test_calculate_bmi_overweight(self):
        """Test BMI calculation indicating overweight"""
        # 160 cm, 80 kg -> BMI = 80 / (1.6^2) = 31.25
        result = calculate_bmi(160, 80)
        assert round(result, 2) == 31.25

    def test_calculate_bmi_underweight(self):
        """Test BMI calculation indicating underweight"""
        # 175 cm, 50 kg -> BMI = 50 / (1.75^2) = 16.33
        result = calculate_bmi(175, 50)
        assert round(result, 2) == 16.33


class TestCalculateAgeInMonths:
    """Tests for calculate_age_in_months function"""

    def test_exact_years(self):
        """Test age calculation for exact years"""
        birth = date(2020, 1, 1)
        measurement = date(2023, 1, 1)
        result = calculate_age_in_months(birth, measurement)
        assert result == 36  # 3 years = 36 months

    def test_partial_months(self):
        """Test age calculation with partial months"""
        birth = date(2020, 1, 15)
        measurement = date(2020, 7, 15)
        result = calculate_age_in_months(birth, measurement)
        assert result == 6

    def test_same_day(self):
        """Test age calculation for birth date"""
        birth = date(2020, 6, 15)
        measurement = date(2020, 6, 15)
        result = calculate_age_in_months(birth, measurement)
        assert result == 0

    def test_one_month(self):
        """Test age calculation for one month old"""
        birth = date(2020, 1, 1)
        measurement = date(2020, 2, 1)
        result = calculate_age_in_months(birth, measurement)
        assert result == 1

    def test_day_adjustment_reduces_month(self):
        """Test that day of month reduces the total when days < 0"""
        birth = date(2020, 1, 25)
        measurement = date(2020, 2, 10)
        result = calculate_age_in_months(birth, measurement)
        # Days difference is negative (10 - 25 = -15), so month count reduces by 1
        assert result == 0

    def test_years_and_months(self):
        """Test age calculation for years and months"""
        birth = date(2018, 3, 10)
        measurement = date(2023, 9, 10)
        result = calculate_age_in_months(birth, measurement)
        # 5 years and 6 months = 66 months
        assert result == 66

    def test_end_of_month_boundary(self):
        """Test age calculation at month boundary"""
        birth = date(2020, 1, 31)
        measurement = date(2020, 3, 1)
        result = calculate_age_in_months(birth, measurement)
        # January to March is 2 months, but day is 1 vs 31 so reduces by 1
        assert result == 1


class TestInterpretZScore:
    """Tests for interpret_z_score function"""

    # Height interpretation tests
    def test_height_severely_stunted(self):
        """Test height interpretation for severely stunted"""
        interpretation, status = interpret_z_score(-3.5, 'height')
        assert "Severely stunted" in interpretation
        assert status == "danger"

    def test_height_stunted(self):
        """Test height interpretation for stunted"""
        interpretation, status = interpret_z_score(-2.5, 'height')
        assert "Stunted" in interpretation
        assert status == "warning"

    def test_height_normal_negative(self):
        """Test height interpretation for normal (negative z)"""
        interpretation, status = interpret_z_score(-1.5, 'height')
        assert "Normal" in interpretation
        assert status == "success"

    def test_height_normal_positive(self):
        """Test height interpretation for normal (positive z)"""
        interpretation, status = interpret_z_score(1.5, 'height')
        assert "Normal" in interpretation
        assert status == "success"

    def test_height_tall(self):
        """Test height interpretation for tall"""
        interpretation, status = interpret_z_score(2.5, 'height')
        assert "Tall" in interpretation
        assert status == "warning"

    def test_height_very_tall(self):
        """Test height interpretation for very tall"""
        interpretation, status = interpret_z_score(3.5, 'height')
        assert "Very tall" in interpretation
        assert status == "danger"

    # Weight interpretation tests
    def test_weight_severely_underweight(self):
        """Test weight interpretation for severely underweight"""
        interpretation, status = interpret_z_score(-3.5, 'weight')
        assert "Severely underweight" in interpretation
        assert status == "danger"

    def test_weight_underweight(self):
        """Test weight interpretation for underweight"""
        interpretation, status = interpret_z_score(-2.5, 'weight')
        assert "Underweight" in interpretation
        assert status == "warning"

    def test_weight_normal(self):
        """Test weight interpretation for normal weight"""
        interpretation, status = interpret_z_score(0, 'weight')
        assert "Normal" in interpretation
        assert status == "success"

    def test_weight_overweight(self):
        """Test weight interpretation for overweight"""
        interpretation, status = interpret_z_score(2.5, 'weight')
        assert "Overweight" in interpretation
        assert status == "warning"

    def test_weight_obese(self):
        """Test weight interpretation for obese"""
        interpretation, status = interpret_z_score(3.5, 'weight')
        assert "Obese" in interpretation
        assert status == "danger"

    # BMI interpretation tests
    def test_bmi_severely_wasted(self):
        """Test BMI interpretation for severely wasted"""
        interpretation, status = interpret_z_score(-3.5, 'bmi')
        assert "Severely wasted" in interpretation
        assert status == "danger"

    def test_bmi_wasted(self):
        """Test BMI interpretation for wasted"""
        interpretation, status = interpret_z_score(-2.5, 'bmi')
        assert "Wasted" in interpretation
        assert status == "warning"

    def test_bmi_normal(self):
        """Test BMI interpretation for normal BMI"""
        interpretation, status = interpret_z_score(0, 'bmi')
        assert "Normal" in interpretation
        assert status == "success"

    def test_bmi_overweight(self):
        """Test BMI interpretation for overweight BMI"""
        interpretation, status = interpret_z_score(1.5, 'bmi')
        assert "Overweight" in interpretation
        assert status == "warning"

    def test_bmi_obese(self):
        """Test BMI interpretation for obese BMI"""
        interpretation, status = interpret_z_score(2.5, 'bmi')
        assert "Obese" in interpretation
        assert status == "danger"

    # Boundary tests
    def test_height_boundary_minus_3(self):
        """Test height at exact -3 boundary (should be stunted, not severely stunted)"""
        interpretation, status = interpret_z_score(-3, 'height')
        # At exactly -3, should still be "Stunted" (< -3 for severely stunted)
        assert "Stunted" in interpretation

    def test_height_boundary_minus_2(self):
        """Test height at exact -2 boundary (should be normal)"""
        interpretation, status = interpret_z_score(-2, 'height')
        # At exactly -2, should be "Normal" (< -2 for stunted)
        assert "Normal" in interpretation

    def test_height_boundary_plus_2(self):
        """Test height at exact +2 boundary (should be normal)"""
        interpretation, status = interpret_z_score(2, 'height')
        assert "Normal" in interpretation

    def test_bmi_boundary_plus_1(self):
        """Test BMI at exact +1 boundary (should be normal)"""
        interpretation, status = interpret_z_score(1, 'bmi')
        assert "Normal" in interpretation

    def test_bmi_boundary_plus_2(self):
        """Test BMI at exact +2 boundary (should be overweight)"""
        interpretation, status = interpret_z_score(2, 'bmi')
        assert "Overweight" in interpretation
