"""
E2E tests for growth data retrieval flows (WHO + CDC sources).

These tests verify that growth data can be correctly loaded from both WHO and CDC
sources for all measurement types and genders.
"""
import pytest
import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from conftest will set up streamlit mock
# Now we can import from the app
from child_growth_app import (
    load_growth_data_from_csv,
    get_height_data,
    get_weight_data,
    get_bmi_data,
    get_growth_statistics
)


class TestGrowthDataRetrieval:
    """Test class for growth data retrieval flows."""

    # Required columns for all growth data
    REQUIRED_PERCENTILE_COLUMNS = ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97']
    REQUIRED_STATS_COLUMNS = ['age_months', 'mean', 'sd']

    # ===== WHO Height Data Tests =====
    
    def test_who_boys_height_data_loads_successfully(self):
        """Test that WHO boys height data loads without errors."""
        df = get_height_data('Male', 'WHO')
        assert not df.empty, "WHO boys height data should not be empty"
        
    def test_who_girls_height_data_loads_successfully(self):
        """Test that WHO girls height data loads without errors."""
        df = get_height_data('Female', 'WHO')
        assert not df.empty, "WHO girls height data should not be empty"
        
    def test_who_height_data_has_required_columns(self):
        """Test that WHO height data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_height_data(gender, 'WHO')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in WHO {gender} height data"
                
    def test_who_height_data_covers_0_to_19_years(self):
        """Test that WHO height data covers ages 0-228 months (0-19 years)."""
        for gender in ['Male', 'Female']:
            df = get_height_data(gender, 'WHO')
            assert df['age_months'].min() == 0, "WHO height data should start at 0 months"
            assert df['age_months'].max() >= 228, "WHO height data should extend to at least 228 months (19 years)"
            
    # ===== CDC Height Data Tests =====
    
    def test_cdc_boys_height_data_loads_successfully(self):
        """Test that CDC boys height data loads without errors."""
        df = get_height_data('Male', 'CDC')
        assert not df.empty, "CDC boys height data should not be empty"
        
    def test_cdc_girls_height_data_loads_successfully(self):
        """Test that CDC girls height data loads without errors."""
        df = get_height_data('Female', 'CDC')
        assert not df.empty, "CDC girls height data should not be empty"
        
    def test_cdc_height_data_has_required_columns(self):
        """Test that CDC height data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_height_data(gender, 'CDC')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in CDC {gender} height data"
                
    def test_cdc_height_data_covers_2_to_20_years(self):
        """Test that CDC height data covers ages 24-240 months (2-20 years)."""
        for gender in ['Male', 'Female']:
            df = get_height_data(gender, 'CDC')
            # CDC starts at 24 months (2 years)
            assert df['age_months'].min() <= 24, "CDC height data should start at or before 24 months"
            assert df['age_months'].max() >= 240, "CDC height data should extend to at least 240 months (20 years)"
            
    # ===== WHO Weight Data Tests =====
    
    def test_who_boys_weight_data_loads_successfully(self):
        """Test that WHO boys weight data loads without errors."""
        df = get_weight_data('Male', 'WHO')
        assert not df.empty, "WHO boys weight data should not be empty"
        
    def test_who_girls_weight_data_loads_successfully(self):
        """Test that WHO girls weight data loads without errors."""
        df = get_weight_data('Female', 'WHO')
        assert not df.empty, "WHO girls weight data should not be empty"
        
    def test_who_weight_data_has_required_columns(self):
        """Test that WHO weight data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_weight_data(gender, 'WHO')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in WHO {gender} weight data"
                
    def test_who_weight_data_covers_0_to_10_years(self):
        """Test that WHO weight data covers ages 0-120 months (0-10 years)."""
        for gender in ['Male', 'Female']:
            df = get_weight_data(gender, 'WHO')
            assert df['age_months'].min() == 0, "WHO weight data should start at 0 months"
            assert df['age_months'].max() >= 120, "WHO weight data should extend to at least 120 months (10 years)"
            
    # ===== CDC Weight Data Tests =====
    
    def test_cdc_boys_weight_data_loads_successfully(self):
        """Test that CDC boys weight data loads without errors."""
        df = get_weight_data('Male', 'CDC')
        assert not df.empty, "CDC boys weight data should not be empty"
        
    def test_cdc_girls_weight_data_loads_successfully(self):
        """Test that CDC girls weight data loads without errors."""
        df = get_weight_data('Female', 'CDC')
        assert not df.empty, "CDC girls weight data should not be empty"
        
    def test_cdc_weight_data_has_required_columns(self):
        """Test that CDC weight data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_weight_data(gender, 'CDC')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in CDC {gender} weight data"

    # ===== WHO BMI Data Tests =====
    
    def test_who_boys_bmi_data_loads_successfully(self):
        """Test that WHO boys BMI data loads without errors."""
        df = get_bmi_data('Male', 'WHO')
        assert not df.empty, "WHO boys BMI data should not be empty"
        
    def test_who_girls_bmi_data_loads_successfully(self):
        """Test that WHO girls BMI data loads without errors."""
        df = get_bmi_data('Female', 'WHO')
        assert not df.empty, "WHO girls BMI data should not be empty"
        
    def test_who_bmi_data_has_required_columns(self):
        """Test that WHO BMI data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_bmi_data(gender, 'WHO')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in WHO {gender} BMI data"
                
    def test_who_bmi_data_covers_5_to_19_years(self):
        """Test that WHO BMI data covers ages 61-228 months (5-19 years)."""
        for gender in ['Male', 'Female']:
            df = get_bmi_data(gender, 'WHO')
            # WHO BMI starts at 61 months (approximately 5 years)
            assert df['age_months'].min() <= 61, "WHO BMI data should start at or before 61 months"
            assert df['age_months'].max() >= 228, "WHO BMI data should extend to at least 228 months (19 years)"
            
    # ===== CDC BMI Data Tests =====
    
    def test_cdc_boys_bmi_data_loads_successfully(self):
        """Test that CDC boys BMI data loads without errors."""
        df = get_bmi_data('Male', 'CDC')
        assert not df.empty, "CDC boys BMI data should not be empty"
        
    def test_cdc_girls_bmi_data_loads_successfully(self):
        """Test that CDC girls BMI data loads without errors."""
        df = get_bmi_data('Female', 'CDC')
        assert not df.empty, "CDC girls BMI data should not be empty"
        
    def test_cdc_bmi_data_has_required_columns(self):
        """Test that CDC BMI data contains all required columns."""
        for gender in ['Male', 'Female']:
            df = get_bmi_data(gender, 'CDC')
            for col in self.REQUIRED_PERCENTILE_COLUMNS:
                assert col in df.columns, f"Missing column {col} in CDC {gender} BMI data"
                
    def test_cdc_bmi_data_covers_2_to_20_years(self):
        """Test that CDC BMI data covers ages 24-240 months (2-20 years)."""
        for gender in ['Male', 'Female']:
            df = get_bmi_data(gender, 'CDC')
            # CDC BMI starts at 24 months (2 years)
            assert df['age_months'].min() <= 24, "CDC BMI data should start at or before 24 months"
            assert df['age_months'].max() >= 240, "CDC BMI data should extend to at least 240 months (20 years)"

    # ===== Growth Statistics Tests =====
    
    def test_who_growth_statistics_loads_for_male(self):
        """Test that WHO growth statistics load correctly for male."""
        stats = get_growth_statistics('Male', 'WHO')
        assert not stats.empty, "WHO male growth statistics should not be empty"
        
    def test_who_growth_statistics_loads_for_female(self):
        """Test that WHO growth statistics load correctly for female."""
        stats = get_growth_statistics('Female', 'WHO')
        assert not stats.empty, "WHO female growth statistics should not be empty"
        
    def test_cdc_growth_statistics_loads_for_male(self):
        """Test that CDC growth statistics load correctly for male."""
        stats = get_growth_statistics('Male', 'CDC')
        assert not stats.empty, "CDC male growth statistics should not be empty"
        
    def test_cdc_growth_statistics_loads_for_female(self):
        """Test that CDC growth statistics load correctly for female."""
        stats = get_growth_statistics('Female', 'CDC')
        assert not stats.empty, "CDC female growth statistics should not be empty"
        
    def test_growth_statistics_has_height_columns(self):
        """Test that growth statistics contain height mean and sd columns."""
        for data_source in ['WHO', 'CDC']:
            for gender in ['Male', 'Female']:
                stats = get_growth_statistics(gender, data_source)
                assert 'height_mean' in stats.columns, f"Missing height_mean in {data_source} {gender} stats"
                assert 'height_sd' in stats.columns, f"Missing height_sd in {data_source} {gender} stats"
                
    def test_growth_statistics_has_weight_columns(self):
        """Test that growth statistics contain weight mean and sd columns."""
        for data_source in ['WHO', 'CDC']:
            for gender in ['Male', 'Female']:
                stats = get_growth_statistics(gender, data_source)
                assert 'weight_mean' in stats.columns, f"Missing weight_mean in {data_source} {gender} stats"
                assert 'weight_sd' in stats.columns, f"Missing weight_sd in {data_source} {gender} stats"

    # ===== Data Quality Tests =====
    
    def test_percentile_values_are_positive(self):
        """Test that all percentile values are positive numbers."""
        for data_source in ['WHO', 'CDC']:
            for gender in ['Male', 'Female']:
                height_df = get_height_data(gender, data_source)
                weight_df = get_weight_data(gender, data_source)
                
                for col in ['p3', 'p15', 'p50', 'p85', 'p97']:
                    assert (height_df[col] > 0).all(), f"Height {col} should be positive for {data_source} {gender}"
                    assert (weight_df[col] > 0).all(), f"Weight {col} should be positive for {data_source} {gender}"
                    
    def test_percentiles_are_monotonically_increasing(self):
        """Test that percentile values are monotonically increasing (p3 < p15 < p50 < p85 < p97)."""
        for data_source in ['WHO', 'CDC']:
            for gender in ['Male', 'Female']:
                height_df = get_height_data(gender, data_source)
                
                assert (height_df['p3'] < height_df['p15']).all(), "p3 should be less than p15"
                assert (height_df['p15'] < height_df['p50']).all(), "p15 should be less than p50"
                assert (height_df['p50'] < height_df['p85']).all(), "p50 should be less than p85"
                assert (height_df['p85'] < height_df['p97']).all(), "p85 should be less than p97"
                
    def test_age_months_are_positive(self):
        """Test that all age_months values are non-negative."""
        for data_source in ['WHO', 'CDC']:
            for gender in ['Male', 'Female']:
                height_df = get_height_data(gender, data_source)
                weight_df = get_weight_data(gender, data_source)
                bmi_df = get_bmi_data(gender, data_source)
                
                assert (height_df['age_months'] >= 0).all(), "Height age_months should be non-negative"
                assert (weight_df['age_months'] >= 0).all(), "Weight age_months should be non-negative"
                assert (bmi_df['age_months'] >= 0).all(), "BMI age_months should be non-negative"
