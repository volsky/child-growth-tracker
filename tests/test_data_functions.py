"""
Unit tests for data loading and z-score calculation functions
"""

import pytest
import os
import pandas as pd

from growth_utils import (
    get_height_data,
    get_weight_data,
    get_bmi_data,
    get_growth_statistics,
    calculate_z_score,
    get_default_measurements,
    calculate_percentile_from_lms
)


# Store original directory and change to repo root for data file access
_original_dir = os.getcwd()
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def change_to_repo_root():
    """Change to repo root directory for tests that need data files"""
    os.chdir(_repo_root)
    yield
    os.chdir(_original_dir)


class TestDataLoading:
    """Tests for data loading functions"""

    def test_load_who_height_male(self):
        """Test loading WHO height data for males"""
        df = get_height_data("Male", "WHO")
        assert not df.empty
        assert 'age_months' in df.columns
        assert 'p3' in df.columns
        assert 'p50' in df.columns
        assert 'p97' in df.columns
        # WHO height covers 0-228 months
        assert df['age_months'].min() == 0
        assert df['age_months'].max() >= 200

    def test_load_who_height_female(self):
        """Test loading WHO height data for females"""
        df = get_height_data("Female", "WHO")
        assert not df.empty
        assert 'age_months' in df.columns
        # WHO height covers 0-228 months
        assert df['age_months'].min() == 0

    def test_load_cdc_height_male(self):
        """Test loading CDC height data for males"""
        df = get_height_data("Male", "CDC")
        assert not df.empty
        # CDC covers 24-240 months (2-20 years)
        assert df['age_months'].min() >= 24

    def test_load_who_weight_male(self):
        """Test loading WHO weight data for males"""
        df = get_weight_data("Male", "WHO")
        assert not df.empty
        # WHO weight covers 0-120 months (0-10 years)
        assert df['age_months'].min() == 0
        assert df['age_months'].max() <= 121

    def test_load_who_weight_female(self):
        """Test loading WHO weight data for females"""
        df = get_weight_data("Female", "WHO")
        assert not df.empty

    def test_load_cdc_weight_male(self):
        """Test loading CDC weight data for males"""
        df = get_weight_data("Male", "CDC")
        assert not df.empty
        # CDC covers 24-240 months
        assert df['age_months'].min() >= 24

    def test_load_who_bmi_male(self):
        """Test loading WHO BMI data for males"""
        df = get_bmi_data("Male", "WHO")
        assert not df.empty
        # WHO BMI covers 61-228 months (5-19 years)
        assert df['age_months'].min() >= 61

    def test_load_who_bmi_female(self):
        """Test loading WHO BMI data for females"""
        df = get_bmi_data("Female", "WHO")
        assert not df.empty

    def test_load_cdc_bmi_male(self):
        """Test loading CDC BMI data for males"""
        df = get_bmi_data("Male", "CDC")
        assert not df.empty
        # CDC BMI covers 24-240 months
        assert df['age_months'].min() >= 24

    def test_growth_statistics_who_male(self):
        """Test loading growth statistics for WHO male"""
        stats = get_growth_statistics("Male", "WHO")
        assert not stats.empty
        assert 'height_mean' in stats.columns
        assert 'height_sd' in stats.columns
        assert 'weight_mean' in stats.columns
        assert 'weight_sd' in stats.columns

    def test_growth_statistics_cdc_female(self):
        """Test loading growth statistics for CDC female"""
        stats = get_growth_statistics("Female", "CDC")
        assert not stats.empty
        assert 'height_mean' in stats.columns
        assert 'weight_mean' in stats.columns


class TestZScoreCalculation:
    """Tests for z-score calculation function"""

    def test_zscore_height_male_who_normal(self):
        """Test height z-score for normal male child using WHO data"""
        # Test at 36 months (3 years)
        z, percentile, mean, sd = calculate_z_score(36, 95.0, 'height', 'Male', 'WHO')
        assert z is not None
        assert percentile is not None
        # 95 cm at 36 months should be around median (z-score near 0)
        assert -2 < z < 2  # Normal range

    def test_zscore_height_female_who_normal(self):
        """Test height z-score for normal female child using WHO data"""
        z, percentile, mean, sd = calculate_z_score(36, 94.0, 'height', 'Female', 'WHO')
        assert z is not None
        assert percentile is not None
        assert -2 < z < 2

    def test_zscore_weight_male_who_normal(self):
        """Test weight z-score for normal male child using WHO data"""
        # At 24 months, normal weight around 12.4 kg
        z, percentile, mean, sd = calculate_z_score(24, 12.0, 'weight', 'Male', 'WHO')
        assert z is not None
        assert percentile is not None
        assert -2 < z < 2

    def test_zscore_height_male_cdc_normal(self):
        """Test height z-score for male child using CDC data"""
        # CDC starts at 24 months
        z, percentile, mean, sd = calculate_z_score(36, 95.0, 'height', 'Male', 'CDC')
        assert z is not None
        assert percentile is not None

    def test_zscore_bmi_male_who(self):
        """Test BMI z-score for male child using WHO data"""
        # WHO BMI starts at 61 months (5 years)
        # At 72 months (6 years), normal BMI around 15-16
        z, percentile, mean, sd = calculate_z_score(72, 15.5, 'bmi', 'Male', 'WHO')
        assert z is not None
        assert percentile is not None

    def test_zscore_bmi_female_cdc(self):
        """Test BMI z-score for female child using CDC data"""
        # CDC BMI starts at 24 months
        z, percentile, mean, sd = calculate_z_score(36, 16.0, 'bmi', 'Female', 'CDC')
        assert z is not None
        assert percentile is not None

    def test_zscore_extreme_high(self):
        """Test z-score for extremely high measurement"""
        # Very tall child
        z, percentile, mean, sd = calculate_z_score(36, 110.0, 'height', 'Male', 'WHO')
        assert z is not None
        assert z > 2  # Should be above normal range
        assert percentile > 97

    def test_zscore_extreme_low(self):
        """Test z-score for extremely low measurement"""
        # Very short child
        z, percentile, mean, sd = calculate_z_score(36, 80.0, 'height', 'Male', 'WHO')
        assert z is not None
        assert z < -2  # Should be below normal range
        assert percentile < 3

    def test_zscore_percentile_50(self):
        """Test that median measurement gives approximately 50th percentile"""
        height_data = get_height_data("Male", "WHO")
        # Get p50 at 24 months
        row = height_data[height_data['age_months'] == 24].iloc[0]
        median_height = row['p50']

        z, percentile, mean, sd = calculate_z_score(24, median_height, 'height', 'Male', 'WHO')
        assert z is not None
        # Z-score should be near 0 for median
        assert -0.5 < z < 0.5
        # Percentile should be near 50
        assert 40 < percentile < 60


class TestDefaultMeasurements:
    """Tests for default measurement retrieval"""

    def test_default_measurements_who_male_infant(self):
        """Test default measurements for infant male using WHO data"""
        height, weight = get_default_measurements(12, "Male", "WHO")
        # 12 month old should have height around 75 cm and weight around 10 kg
        assert 70 < height < 80
        assert 8 < weight < 12

    def test_default_measurements_who_female_toddler(self):
        """Test default measurements for toddler female using WHO data"""
        height, weight = get_default_measurements(36, "Female", "WHO")
        # 36 month old girl should have height around 95 cm and weight around 14 kg
        assert 85 < height < 100
        assert 10 < weight < 18

    def test_default_measurements_cdc_male_child(self):
        """Test default measurements for child using CDC data"""
        height, weight = get_default_measurements(60, "Male", "CDC")
        # 60 month old (5 years) should have height around 110 cm
        assert 100 < height < 120
        assert 15 < weight < 25

    def test_default_measurements_returns_tuple(self):
        """Test that default measurements returns correct type"""
        result = get_default_measurements(24, "Male", "WHO")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestLMSCalculation:
    """Tests for LMS percentile calculation"""

    def test_lms_p50_equals_m(self):
        """Test that 50th percentile equals M parameter"""
        L, M, S = 1.0, 100.0, 0.04
        p50 = calculate_percentile_from_lms(L, M, S, 50)
        # p50 should equal M
        assert abs(p50 - M) < 0.01

    def test_lms_p3_less_than_m(self):
        """Test that 3rd percentile is less than M"""
        L, M, S = 1.0, 100.0, 0.04
        p3 = calculate_percentile_from_lms(L, M, S, 3)
        assert p3 < M

    def test_lms_p97_greater_than_m(self):
        """Test that 97th percentile is greater than M"""
        L, M, S = 1.0, 100.0, 0.04
        p97 = calculate_percentile_from_lms(L, M, S, 97)
        assert p97 > M

    def test_lms_percentile_ordering(self):
        """Test that percentiles are in correct order"""
        L, M, S = 1.0, 100.0, 0.04
        p3 = calculate_percentile_from_lms(L, M, S, 3)
        p15 = calculate_percentile_from_lms(L, M, S, 15)
        p50 = calculate_percentile_from_lms(L, M, S, 50)
        p85 = calculate_percentile_from_lms(L, M, S, 85)
        p97 = calculate_percentile_from_lms(L, M, S, 97)

        assert p3 < p15 < p50 < p85 < p97

    def test_lms_with_l_zero(self):
        """Test LMS calculation when L=0 (uses exponential formula)"""
        L, M, S = 0.0, 15.0, 0.1
        p50 = calculate_percentile_from_lms(L, M, S, 50)
        # When L=0, p50 should still equal M
        assert abs(p50 - M) < 0.01

    def test_lms_realistic_height_values(self):
        """Test LMS with realistic height parameters for 36 month old male"""
        # Using approximate WHO values for 36 month male height
        L, M, S = 1.0, 95.0, 0.04
        p3 = calculate_percentile_from_lms(L, M, S, 3)
        p97 = calculate_percentile_from_lms(L, M, S, 97)

        # 3rd percentile should be roughly 7-8 cm below median
        assert 86 < p3 < 92
        # 97th percentile should be roughly 7-8 cm above median
        assert 98 < p97 < 104

    def test_lms_realistic_weight_values(self):
        """Test LMS with realistic weight parameters for 24 month old male"""
        # Using approximate WHO values for 24 month male weight
        L, M, S = -0.2, 12.2, 0.1
        p3 = calculate_percentile_from_lms(L, M, S, 3)
        p97 = calculate_percentile_from_lms(L, M, S, 97)

        # Weights should be reasonable
        assert 9 < p3 < 11
        assert 14 < p97 < 17
