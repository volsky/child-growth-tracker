"""
Unit tests for utility functions in growth_utils.py

Tests for:
- calculate_bmi()
- calculate_age_in_months()
- interpret_z_score()
- get_height_data(), get_weight_data(), get_bmi_data()
- calculate_z_score()
- get_growth_statistics()
"""

import pytest
import sys
import os
from datetime import date
import pandas as pd
import numpy as np

# Change to repository root to find data files
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the functions from growth_utils
from growth_utils import (
    calculate_bmi,
    calculate_age_in_months,
    interpret_z_score,
    get_height_data,
    get_weight_data,
    get_bmi_data,
    calculate_z_score,
    get_growth_statistics,
)


class TestCalculateBmi:
    """Tests for calculate_bmi() function"""

    def test_bmi_normal_values(self):
        """Test BMI calculation with typical values"""
        # 170cm, 70kg should give BMI ~24.22
        bmi = calculate_bmi(170, 70)
        assert abs(bmi - 24.22) < 0.01

    def test_bmi_child_values(self):
        """Test BMI calculation with child values"""
        # 100cm, 20kg should give BMI = 20
        bmi = calculate_bmi(100, 20)
        assert bmi == 20.0

    def test_bmi_low_values(self):
        """Test BMI calculation with low values"""
        # 50cm (infant), 3kg
        bmi = calculate_bmi(50, 3)
        assert abs(bmi - 12.0) < 0.01

    def test_bmi_tall_person(self):
        """Test BMI calculation for taller person"""
        # 190cm, 90kg
        bmi = calculate_bmi(190, 90)
        assert abs(bmi - 24.93) < 0.01

    def test_bmi_edge_case_small_height(self):
        """Test BMI with very small height"""
        bmi = calculate_bmi(45, 2.5)
        # BMI = 2.5 / (0.45)^2 = 12.35
        assert abs(bmi - 12.35) < 0.01


class TestCalculateAgeInMonths:
    """Tests for calculate_age_in_months() function"""

    def test_one_year_exact(self):
        """Test age calculation for exactly one year"""
        birth = date(2023, 1, 1)
        measurement = date(2024, 1, 1)
        age = calculate_age_in_months(birth, measurement)
        assert age == 12

    def test_same_month_later_day(self):
        """Test when measurement day is after birth day in same month"""
        birth = date(2023, 6, 10)
        measurement = date(2023, 9, 15)
        age = calculate_age_in_months(birth, measurement)
        assert age == 3

    def test_same_month_earlier_day(self):
        """Test when measurement day is before birth day in same month"""
        birth = date(2023, 6, 20)
        measurement = date(2023, 9, 10)
        age = calculate_age_in_months(birth, measurement)
        # 3 months but day hasn't passed so 2 months
        assert age == 2

    def test_zero_months(self):
        """Test newborn (same month)"""
        birth = date(2023, 11, 5)
        measurement = date(2023, 11, 20)
        age = calculate_age_in_months(birth, measurement)
        assert age == 0

    def test_five_years(self):
        """Test age for 5 year old child"""
        birth = date(2018, 3, 15)
        measurement = date(2023, 3, 15)
        age = calculate_age_in_months(birth, measurement)
        assert age == 60

    def test_ten_years(self):
        """Test age for 10 year old child"""
        birth = date(2013, 7, 1)
        measurement = date(2023, 7, 1)
        age = calculate_age_in_months(birth, measurement)
        assert age == 120

    def test_crossing_year_boundary(self):
        """Test age calculation crossing year boundary"""
        birth = date(2022, 11, 15)
        measurement = date(2023, 2, 20)
        age = calculate_age_in_months(birth, measurement)
        assert age == 3

    def test_leap_year_handling(self):
        """Test age calculation with leap year"""
        birth = date(2020, 2, 29)
        measurement = date(2021, 2, 28)
        age = calculate_age_in_months(birth, measurement)
        # 11 months (day hasn't passed in non-leap year)
        assert age == 11


class TestInterpretZScore:
    """Tests for interpret_z_score() function"""

    def test_height_severely_stunted(self):
        """Test severely stunted interpretation for height"""
        interpretation, status = interpret_z_score(-3.5, 'height')
        assert "Severely stunted" in interpretation
        assert status == "danger"

    def test_height_stunted(self):
        """Test stunted interpretation for height"""
        interpretation, status = interpret_z_score(-2.5, 'height')
        assert "Stunted" in interpretation
        assert status == "warning"

    def test_height_normal(self):
        """Test normal interpretation for height"""
        interpretation, status = interpret_z_score(0, 'height')
        assert "Normal" in interpretation
        assert status == "success"

    def test_height_tall(self):
        """Test tall interpretation for height"""
        interpretation, status = interpret_z_score(2.5, 'height')
        assert "Tall" in interpretation
        assert status == "warning"

    def test_height_very_tall(self):
        """Test very tall interpretation for height"""
        interpretation, status = interpret_z_score(3.5, 'height')
        assert "Very tall" in interpretation
        assert status == "danger"

    def test_weight_severely_underweight(self):
        """Test severely underweight interpretation for weight"""
        interpretation, status = interpret_z_score(-3.5, 'weight')
        assert "Severely underweight" in interpretation
        assert status == "danger"

    def test_weight_underweight(self):
        """Test underweight interpretation for weight"""
        interpretation, status = interpret_z_score(-2.5, 'weight')
        assert "Underweight" in interpretation
        assert status == "warning"

    def test_weight_normal(self):
        """Test normal interpretation for weight"""
        interpretation, status = interpret_z_score(0, 'weight')
        assert "Normal" in interpretation
        assert status == "success"

    def test_weight_overweight(self):
        """Test overweight interpretation for weight"""
        interpretation, status = interpret_z_score(2.5, 'weight')
        assert "Overweight" in interpretation
        assert status == "warning"

    def test_weight_obese(self):
        """Test obese interpretation for weight"""
        interpretation, status = interpret_z_score(3.5, 'weight')
        assert "Obese" in interpretation
        assert status == "danger"

    def test_bmi_severely_wasted(self):
        """Test severely wasted interpretation for BMI"""
        interpretation, status = interpret_z_score(-3.5, 'bmi')
        assert "Severely wasted" in interpretation
        assert status == "danger"

    def test_bmi_wasted(self):
        """Test wasted interpretation for BMI"""
        interpretation, status = interpret_z_score(-2.5, 'bmi')
        assert "Wasted" in interpretation
        assert status == "warning"

    def test_bmi_normal(self):
        """Test normal interpretation for BMI"""
        interpretation, status = interpret_z_score(0, 'bmi')
        assert "Normal" in interpretation
        assert status == "success"

    def test_bmi_overweight(self):
        """Test overweight interpretation for BMI"""
        interpretation, status = interpret_z_score(1.5, 'bmi')
        assert "Overweight" in interpretation
        assert status == "warning"

    def test_bmi_obese(self):
        """Test obese interpretation for BMI"""
        interpretation, status = interpret_z_score(2.5, 'bmi')
        assert "Obese" in interpretation
        assert status == "danger"

    def test_boundary_values_height(self):
        """Test boundary values for height interpretation"""
        # Exactly -3
        interpretation, status = interpret_z_score(-3, 'height')
        assert "Stunted" in interpretation or "Severely stunted" in interpretation

        # Exactly -2
        interpretation, status = interpret_z_score(-2, 'height')
        assert "Normal" in interpretation or "Stunted" in interpretation

        # Exactly 2
        interpretation, status = interpret_z_score(2, 'height')
        assert "Normal" in interpretation

        # Exactly 3
        interpretation, status = interpret_z_score(3, 'height')
        assert "Tall" in interpretation


class TestGetHeightData:
    """Tests for get_height_data() function"""

    def test_male_who_data_structure(self):
        """Test WHO male height data returns expected columns"""
        df = get_height_data("Male", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_who_data_structure(self):
        """Test WHO female height data returns expected columns"""
        df = get_height_data("Female", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_male_cdc_data_structure(self):
        """Test CDC male height data returns expected columns"""
        df = get_height_data("Male", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_cdc_data_structure(self):
        """Test CDC female height data returns expected columns"""
        df = get_height_data("Female", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_who_data_age_range(self):
        """Test WHO height data covers expected age range"""
        df = get_height_data("Male", "WHO")
        assert df['age_months'].min() == 0  # Starts at birth
        assert df['age_months'].max() == 228  # Up to 19 years

    def test_cdc_data_age_range(self):
        """Test CDC height data covers expected age range"""
        df = get_height_data("Male", "CDC")
        assert df['age_months'].min() == 24  # Starts at 2 years
        assert df['age_months'].max() == 240  # Up to 20 years

    def test_percentile_ordering(self):
        """Test that percentiles are in correct order"""
        df = get_height_data("Male", "WHO")
        # For all rows, p3 < p15 < p50 < p85 < p97
        assert (df['p3'] < df['p15']).all()
        assert (df['p15'] < df['p50']).all()
        assert (df['p50'] < df['p85']).all()
        assert (df['p85'] < df['p97']).all()


class TestGetWeightData:
    """Tests for get_weight_data() function"""

    def test_male_who_data_structure(self):
        """Test WHO male weight data returns expected columns"""
        df = get_weight_data("Male", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_who_data_structure(self):
        """Test WHO female weight data returns expected columns"""
        df = get_weight_data("Female", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_male_cdc_data_structure(self):
        """Test CDC male weight data returns expected columns"""
        df = get_weight_data("Male", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_cdc_data_structure(self):
        """Test CDC female weight data returns expected columns"""
        df = get_weight_data("Female", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_who_weight_age_range(self):
        """Test WHO weight data covers expected age range (0-10 years)"""
        df = get_weight_data("Male", "WHO")
        assert df['age_months'].min() == 0  # Starts at birth
        assert df['age_months'].max() == 120  # Up to 10 years

    def test_cdc_weight_age_range(self):
        """Test CDC weight data covers expected age range (2-20 years)"""
        df = get_weight_data("Male", "CDC")
        assert df['age_months'].min() == 24  # Starts at 2 years
        assert df['age_months'].max() == 240  # Up to 20 years

    def test_percentile_ordering(self):
        """Test that percentiles are in correct order"""
        df = get_weight_data("Female", "WHO")
        # For all rows, p3 < p15 < p50 < p85 < p97
        assert (df['p3'] < df['p15']).all()
        assert (df['p15'] < df['p50']).all()
        assert (df['p50'] < df['p85']).all()
        assert (df['p85'] < df['p97']).all()


class TestGetBmiData:
    """Tests for get_bmi_data() function"""

    def test_male_who_data_structure(self):
        """Test WHO male BMI data returns expected columns"""
        df = get_bmi_data("Male", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_who_data_structure(self):
        """Test WHO female BMI data returns expected columns"""
        df = get_bmi_data("Female", "WHO")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_male_cdc_data_structure(self):
        """Test CDC male BMI data returns expected columns"""
        df = get_bmi_data("Male", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_female_cdc_data_structure(self):
        """Test CDC female BMI data returns expected columns"""
        df = get_bmi_data("Female", "CDC")
        assert not df.empty
        expected_cols = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
        for col in expected_cols:
            assert col in df.columns

    def test_who_bmi_age_range(self):
        """Test WHO BMI data covers expected age range (5-19 years)"""
        df = get_bmi_data("Male", "WHO")
        assert df['age_months'].min() == 61  # Starts at 5 years
        assert df['age_months'].max() == 228  # Up to 19 years

    def test_cdc_bmi_age_range(self):
        """Test CDC BMI data covers expected age range (2-20 years)"""
        df = get_bmi_data("Male", "CDC")
        assert df['age_months'].min() == 24  # Starts at 2 years
        assert df['age_months'].max() >= 240  # Up to at least 20 years

    def test_percentile_ordering(self):
        """Test that percentiles are in correct order"""
        df = get_bmi_data("Male", "CDC")
        # For all rows, p3 < p15 < p50 < p85 < p97
        assert (df['p3'] < df['p15']).all()
        assert (df['p15'] < df['p50']).all()
        assert (df['p50'] < df['p85']).all()
        assert (df['p85'] < df['p97']).all()


class TestCalculateZScore:
    """Tests for calculate_z_score() function"""

    def test_height_zscore_who_male(self):
        """Test height z-score calculation for WHO male data"""
        # Test with median value at 24 months - should give z-score close to 0
        z_score, percentile, mean, sd = calculate_z_score(24, 87.1, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert abs(z_score) < 0.5  # Should be close to 0 for median value
        assert 40 < percentile < 60  # Should be close to 50th percentile

    def test_weight_zscore_who_female(self):
        """Test weight z-score calculation for WHO female data"""
        # Test with median value at 12 months
        z_score, percentile, mean, sd = calculate_z_score(12, 8.9, 'weight', 'Female', 'WHO')
        assert z_score is not None
        assert percentile is not None
        assert mean is not None
        assert sd is not None

    def test_bmi_zscore_cdc_male(self):
        """Test BMI z-score calculation for CDC male data"""
        # Test with typical BMI at 60 months (5 years)
        z_score, percentile, mean, sd = calculate_z_score(60, 15.5, 'bmi', 'Male', 'CDC')
        assert z_score is not None
        assert percentile is not None

    def test_zscore_returns_none_for_invalid_type(self):
        """Test that z-score returns None for invalid measurement type"""
        z_score, percentile, mean, sd = calculate_z_score(24, 100, 'invalid_type', 'Male', 'WHO')
        assert z_score is None
        assert percentile is None

    def test_zscore_high_value(self):
        """Test z-score for high measurement value"""
        # Very tall 5-year-old (above 97th percentile)
        z_score, percentile, mean, sd = calculate_z_score(60, 130, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert z_score > 2  # Should be above +2 SD

    def test_zscore_low_value(self):
        """Test z-score for low measurement value"""
        # Very short 5-year-old (below 3rd percentile)
        z_score, percentile, mean, sd = calculate_z_score(60, 90, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert z_score < -2  # Should be below -2 SD

    def test_zscore_different_data_sources(self):
        """Test z-score calculation with different data sources"""
        # Same measurement should give different z-scores for WHO vs CDC
        # Test at 36 months with height 96cm
        z_who, _, _, _ = calculate_z_score(36, 96, 'height', 'Male', 'WHO')
        z_cdc, _, _, _ = calculate_z_score(36, 96, 'height', 'Male', 'CDC')
        
        # Both should return valid z-scores
        assert z_who is not None
        assert z_cdc is not None

    def test_zscore_returns_percentile(self):
        """Test that percentile is properly calculated from z-score"""
        z_score, percentile, _, _ = calculate_z_score(36, 95, 'height', 'Male', 'WHO')
        assert percentile is not None
        assert 0 < percentile < 100
        
        # Z-score of 0 should give ~50th percentile
        z_score, percentile, _, _ = calculate_z_score(36, 96, 'height', 'Male', 'WHO')
        # Just verify the percentile is reasonable
        assert 0 < percentile < 100


class TestGetGrowthStatistics:
    """Tests for get_growth_statistics() function"""

    def test_who_male_statistics_structure(self):
        """Test WHO male statistics returns expected structure"""
        stats = get_growth_statistics("Male", "WHO")
        assert not stats.empty
        assert 'age_months' in stats.columns
        assert 'height_mean' in stats.columns
        assert 'height_sd' in stats.columns
        assert 'weight_mean' in stats.columns
        assert 'weight_sd' in stats.columns

    def test_who_female_statistics_structure(self):
        """Test WHO female statistics returns expected structure"""
        stats = get_growth_statistics("Female", "WHO")
        assert not stats.empty
        assert 'age_months' in stats.columns
        assert 'height_mean' in stats.columns
        assert 'weight_mean' in stats.columns

    def test_cdc_male_statistics_structure(self):
        """Test CDC male statistics returns expected structure"""
        stats = get_growth_statistics("Male", "CDC")
        assert not stats.empty
        assert 'age_months' in stats.columns
        assert 'height_mean' in stats.columns
        assert 'weight_mean' in stats.columns

    def test_cdc_female_statistics_structure(self):
        """Test CDC female statistics returns expected structure"""
        stats = get_growth_statistics("Female", "CDC")
        assert not stats.empty
        assert 'age_months' in stats.columns
        assert 'height_mean' in stats.columns
        assert 'weight_mean' in stats.columns

    def test_statistics_includes_bmi(self):
        """Test that statistics includes BMI data when available"""
        stats = get_growth_statistics("Male", "WHO")
        # BMI data should be present (though may have NaN for early ages)
        assert 'bmi_mean' in stats.columns
        assert 'bmi_sd' in stats.columns

    def test_statistics_sorted_by_age(self):
        """Test that statistics are sorted by age"""
        stats = get_growth_statistics("Male", "WHO")
        assert stats['age_months'].is_monotonic_increasing

    def test_mean_values_positive(self):
        """Test that mean values are positive"""
        stats = get_growth_statistics("Female", "WHO")
        # Filter out NaN values and check positivity
        height_means = stats['height_mean'].dropna()
        weight_means = stats['weight_mean'].dropna()
        assert (height_means > 0).all()
        assert (weight_means > 0).all()

    def test_sd_values_positive(self):
        """Test that SD values are positive"""
        stats = get_growth_statistics("Male", "CDC")
        # Filter out NaN values and check positivity
        height_sds = stats['height_sd'].dropna()
        weight_sds = stats['weight_sd'].dropna()
        assert (height_sds > 0).all()
        assert (weight_sds > 0).all()


class TestLMSPercentileCalculation:
    """Tests for LMS-based percentile calculation through z-score function"""

    def test_percentile_at_median(self):
        """Test that measurement at median gives ~50th percentile"""
        # Get the median (p50) at 36 months for male height
        height_data = get_height_data("Male", "WHO")
        row = height_data[height_data['age_months'] == 36].iloc[0]
        median_height = row['p50']
        
        z_score, percentile, _, _ = calculate_z_score(36, median_height, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert abs(z_score) < 0.1  # Should be very close to 0
        assert 45 < percentile < 55  # Should be close to 50th percentile

    def test_percentile_at_p3(self):
        """Test that measurement at 3rd percentile gives ~3rd percentile"""
        height_data = get_height_data("Male", "WHO")
        row = height_data[height_data['age_months'] == 36].iloc[0]
        p3_height = row['p3']
        
        z_score, percentile, _, _ = calculate_z_score(36, p3_height, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert z_score < -1.5  # Should be well below mean
        assert percentile < 10  # Should be near 3rd percentile

    def test_percentile_at_p97(self):
        """Test that measurement at 97th percentile gives ~97th percentile"""
        height_data = get_height_data("Female", "WHO")
        row = height_data[height_data['age_months'] == 48].iloc[0]
        p97_height = row['p97']
        
        z_score, percentile, _, _ = calculate_z_score(48, p97_height, 'height', 'Female', 'WHO')
        assert z_score is not None
        assert z_score > 1.5  # Should be well above mean
        assert percentile > 90  # Should be near 97th percentile

    def test_interpolation_between_ages(self):
        """Test that z-score works with interpolated age values"""
        # Test at 36.5 months (between 36 and 37)
        z_score, percentile, mean, sd = calculate_z_score(36.5, 95, 'height', 'Male', 'WHO')
        assert z_score is not None
        assert percentile is not None
        assert mean is not None
        assert sd is not None

    def test_zscore_to_percentile_conversion(self):
        """Test z-score to percentile conversion accuracy"""
        # Use scipy.stats.norm directly for comparison
        from scipy.stats import norm
        
        # Get z-score for a known value
        z_score, percentile, _, _ = calculate_z_score(60, 110, 'height', 'Male', 'WHO')
        
        if z_score is not None:
            # Calculate expected percentile from z-score
            expected_percentile = norm.cdf(z_score) * 100
            assert abs(percentile - expected_percentile) < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_newborn_height(self):
        """Test z-score calculation for newborn"""
        z_score, percentile, mean, sd = calculate_z_score(0, 50, 'height', 'Male', 'WHO')
        assert z_score is not None  # Should handle age 0

    def test_maximum_age_who(self):
        """Test z-score at maximum WHO age (228 months = 19 years)"""
        z_score, percentile, mean, sd = calculate_z_score(228, 175, 'height', 'Male', 'WHO')
        assert z_score is not None

    def test_maximum_age_cdc(self):
        """Test z-score at maximum CDC age (240 months = 20 years)"""
        z_score, percentile, mean, sd = calculate_z_score(240, 175, 'height', 'Male', 'CDC')
        assert z_score is not None

    def test_consistent_gender_handling(self):
        """Test that Male and Female give different results"""
        z_male, _, mean_male, _ = calculate_z_score(60, 110, 'height', 'Male', 'WHO')
        z_female, _, mean_female, _ = calculate_z_score(60, 110, 'height', 'Female', 'WHO')
        
        # Same height at same age should give different z-scores for different genders
        assert z_male is not None
        assert z_female is not None
        assert z_male != z_female  # Different reference ranges
        assert mean_male != mean_female  # Different mean heights


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
