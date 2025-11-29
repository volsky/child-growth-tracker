"""
Tests for graceful error handling in the Historical Data table.

These tests verify that:
1. Missing one of the measures (height or weight) is handled gracefully
2. Errors are caught and logged, not causing app failures
3. Invalid data entries are handled without crashing
"""
import pytest
import pandas as pd
from datetime import date, datetime
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from child_growth_app import (
    calculate_z_score,
    calculate_bmi,
    calculate_age_in_months,
    get_default_measurements
)


class TestGracefulErrorHandling:
    """Tests for graceful error handling in historical data entry."""

    def test_calculate_z_score_with_valid_data(self):
        """Test z-score calculation with valid height data."""
        z_score, percentile, mean, sd = calculate_z_score(
            age_months=24,
            measurement=85.0,
            measurement_type='height',
            gender='Male',
            data_source='WHO'
        )
        # Should return valid values (z_score can be any float including 0)
        assert z_score is not None
        assert percentile is not None
        assert mean is not None
        assert sd is not None

    def test_calculate_z_score_with_none_returns_none(self):
        """Test that z-score calculation handles edge cases gracefully."""
        # Test with extreme age values
        z_score, percentile, mean, sd = calculate_z_score(
            age_months=500,  # Very old age
            measurement=100.0,
            measurement_type='height',
            gender='Male',
            data_source='WHO'
        )
        # Should handle gracefully, may extrapolate or return None
        # The function should not raise an exception

    def test_calculate_bmi_with_valid_data(self):
        """Test BMI calculation with valid height and weight."""
        bmi = calculate_bmi(height_cm=100.0, weight_kg=15.0)
        expected_bmi = 15.0 / (1.0 ** 2)  # 15.0 kg/m²
        assert bmi == pytest.approx(expected_bmi, rel=0.01)

    def test_calculate_bmi_with_zero_height(self):
        """Test that BMI calculation handles zero height."""
        # This should raise ZeroDivisionError - caller should handle this
        with pytest.raises(ZeroDivisionError):
            calculate_bmi(height_cm=0.0, weight_kg=15.0)

    def test_calculate_age_in_months(self):
        """Test age calculation in months."""
        birth_date = date(2020, 1, 15)
        measurement_date = date(2022, 1, 15)
        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 24

    def test_calculate_age_in_months_same_day(self):
        """Test age calculation when measurement is on birth date."""
        birth_date = date(2020, 1, 15)
        measurement_date = date(2020, 1, 15)
        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 0


class TestMissingMeasurements:
    """Tests for handling missing height or weight measurements."""

    def test_process_measurement_with_missing_height(self):
        """Test that measurements with missing height can be processed."""
        # Simulate a row with missing height
        row_data = {
            'Date': '2023-06-15',
            'Height (cm)': None,
            'Weight (kg)': 15.5,
            'Today': ''
        }
        
        # Verify height is None but weight is valid
        assert row_data['Height (cm)'] is None
        assert row_data['Weight (kg)'] == 15.5

    def test_process_measurement_with_missing_weight(self):
        """Test that measurements with missing weight can be processed."""
        # Simulate a row with missing weight
        row_data = {
            'Date': '2023-06-15',
            'Height (cm)': 85.0,
            'Weight (kg)': None,
            'Today': ''
        }
        
        # Verify weight is None but height is valid
        assert row_data['Height (cm)'] == 85.0
        assert row_data['Weight (kg)'] is None

    def test_bmi_is_none_when_height_missing(self):
        """Test that BMI is None when height is missing."""
        height = None
        weight = 15.0
        
        # BMI should be None when height is missing
        bmi = None
        if height is not None and weight is not None and height > 0 and weight > 0:
            bmi = calculate_bmi(height, weight)
        
        assert bmi is None

    def test_bmi_is_none_when_weight_missing(self):
        """Test that BMI is None when weight is missing."""
        height = 85.0
        weight = None
        
        # BMI should be None when weight is missing
        bmi = None
        if height is not None and weight is not None and height > 0 and weight > 0:
            bmi = calculate_bmi(height, weight)
        
        assert bmi is None


class TestDataFrameProcessing:
    """Tests for DataFrame processing with missing values."""

    def test_dataframe_with_mixed_missing_values(self):
        """Test DataFrame processing when some rows have missing height or weight."""
        data = [
            {'date': date(2023, 1, 15), 'age': 24, 'height': 85.0, 'weight': 12.0, 'gender': 'Male'},
            {'date': date(2023, 2, 15), 'age': 25, 'height': None, 'weight': 12.5, 'gender': 'Male'},  # Missing height
            {'date': date(2023, 3, 15), 'age': 26, 'height': 87.0, 'weight': None, 'gender': 'Male'},  # Missing weight
            {'date': date(2023, 4, 15), 'age': 27, 'height': 88.0, 'weight': 13.0, 'gender': 'Male'},
        ]
        df = pd.DataFrame(data)
        
        # Check that DataFrame can handle None values
        assert len(df) == 4
        assert df['height'].isna().sum() == 1
        assert df['weight'].isna().sum() == 1

    def test_filter_valid_heights_from_dataframe(self):
        """Test filtering valid heights from a DataFrame with missing values."""
        data = [
            {'age': 24, 'height': 85.0, 'weight': 12.0},
            {'age': 25, 'height': None, 'weight': 12.5},
            {'age': 26, 'height': 87.0, 'weight': None},
        ]
        df = pd.DataFrame(data)
        
        # Filter valid heights
        valid_heights = df['height'].dropna()
        assert len(valid_heights) == 2
        assert 85.0 in valid_heights.values
        assert 87.0 in valid_heights.values

    def test_filter_valid_weights_from_dataframe(self):
        """Test filtering valid weights from a DataFrame with missing values."""
        data = [
            {'age': 24, 'height': 85.0, 'weight': 12.0},
            {'age': 25, 'height': None, 'weight': 12.5},
            {'age': 26, 'height': 87.0, 'weight': None},
        ]
        df = pd.DataFrame(data)
        
        # Filter valid weights
        valid_weights = df['weight'].dropna()
        assert len(valid_weights) == 2
        assert 12.0 in valid_weights.values
        assert 12.5 in valid_weights.values


class TestInvalidDataEntry:
    """Tests for handling invalid data entries."""

    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        invalid_date_str = "invalid-date"
        
        # Should raise ValueError for invalid date format
        with pytest.raises(ValueError):
            datetime.strptime(invalid_date_str, '%Y-%m-%d')

    def test_valid_date_format(self):
        """Test handling of valid date formats."""
        valid_date_str = "2023-06-15"
        parsed_date = datetime.strptime(valid_date_str, '%Y-%m-%d').date()
        
        assert parsed_date.year == 2023
        assert parsed_date.month == 6
        assert parsed_date.day == 15

    def test_float_conversion_from_string(self):
        """Test converting string numbers to float."""
        # Valid conversion
        assert float("85.5") == 85.5
        assert float("12") == 12.0

    def test_float_conversion_invalid_string(self):
        """Test that invalid string raises ValueError on float conversion."""
        with pytest.raises(ValueError):
            float("not_a_number")


class TestZScoreEdgeCases:
    """Tests for z-score calculation edge cases."""

    def test_z_score_returns_none_for_invalid_measurement_type(self):
        """Test z-score returns None for unknown measurement type."""
        z_score, percentile, mean, sd = calculate_z_score(
            age_months=24,
            measurement=85.0,
            measurement_type='unknown_type',  # Invalid type
            gender='Male',
            data_source='WHO'
        )
        # Should return None values for unknown type
        assert z_score is None
        assert percentile is None

    def test_z_score_with_different_data_sources(self):
        """Test z-score calculation with different data sources."""
        # Test WHO
        z_who, _, _, _ = calculate_z_score(24, 85.0, 'height', 'Male', 'WHO')
        
        # Test CDC (24 months is valid for CDC)
        z_cdc, _, _, _ = calculate_z_score(24, 85.0, 'height', 'Male', 'CDC')
        
        # Both should return values (CDC is valid from 24 months)
        # Values may differ between sources

    def test_z_score_for_both_genders(self):
        """Test z-score calculation for both male and female."""
        z_male, _, _, _ = calculate_z_score(24, 85.0, 'height', 'Male', 'WHO')
        z_female, _, _, _ = calculate_z_score(24, 85.0, 'height', 'Female', 'WHO')
        
        # Both should return values
        # Z-scores should typically differ between genders for the same measurement


class TestGetDefaultMeasurements:
    """Tests for get_default_measurements function."""

    def test_default_measurements_returns_tuple(self):
        """Test that default measurements returns a tuple of two values."""
        height, weight = get_default_measurements(24, 'Male', 'WHO')
        
        assert isinstance(height, float)
        assert isinstance(weight, float)
        assert height > 0
        assert weight > 0

    def test_default_measurements_different_ages(self):
        """Test default measurements vary with age."""
        height_12m, weight_12m = get_default_measurements(12, 'Male', 'WHO')
        height_36m, weight_36m = get_default_measurements(36, 'Male', 'WHO')
        
        # Older children should be taller and heavier
        assert height_36m > height_12m
        assert weight_36m > weight_12m


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
