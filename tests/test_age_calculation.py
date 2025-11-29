"""
E2E tests for child age calculation flows.

These tests verify that age calculations work correctly for various
date scenarios including edge cases.
"""
import pytest
import sys
import os
from datetime import date

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from conftest will set up streamlit mock
from child_growth_app import calculate_age_in_months


class TestAgeCalculation:
    """Test class for age calculation flows."""
    
    # ===== Basic Age Calculation Tests =====
    
    def test_exact_one_year(self):
        """Test age calculation for exactly one year."""
        birth_date = date(2022, 1, 15)
        measurement_date = date(2023, 1, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 12, f"Expected 12 months, got {age_months}"
        
    def test_exact_two_years(self):
        """Test age calculation for exactly two years."""
        birth_date = date(2021, 6, 20)
        measurement_date = date(2023, 6, 20)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 24, f"Expected 24 months, got {age_months}"
        
    def test_six_months(self):
        """Test age calculation for six months."""
        birth_date = date(2023, 1, 15)
        measurement_date = date(2023, 7, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 6, f"Expected 6 months, got {age_months}"
        
    def test_eighteen_months(self):
        """Test age calculation for eighteen months."""
        birth_date = date(2022, 1, 1)
        measurement_date = date(2023, 7, 1)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 18, f"Expected 18 months, got {age_months}"
        
    def test_zero_months_same_day(self):
        """Test age calculation for same day (birth day)."""
        birth_date = date(2023, 6, 15)
        measurement_date = date(2023, 6, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 0, f"Expected 0 months, got {age_months}"
        
    # ===== Day Adjustment Tests =====
    
    def test_one_month_minus_one_day(self):
        """Test age calculation for one month minus one day."""
        birth_date = date(2023, 1, 15)
        measurement_date = date(2023, 2, 14)  # One day before one month
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 0, f"Expected 0 months (not quite 1 month), got {age_months}"
        
    def test_one_month_exactly(self):
        """Test age calculation for exactly one month."""
        birth_date = date(2023, 1, 15)
        measurement_date = date(2023, 2, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 1, f"Expected 1 month, got {age_months}"
        
    def test_one_month_plus_one_day(self):
        """Test age calculation for one month plus one day."""
        birth_date = date(2023, 1, 15)
        measurement_date = date(2023, 2, 16)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 1, f"Expected 1 month, got {age_months}"
        
    def test_day_before_birthday_month(self):
        """Test age calculation when day is before birth day in the same month."""
        birth_date = date(2022, 3, 20)
        measurement_date = date(2023, 3, 10)  # 10 days before anniversary
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 11, f"Expected 11 months (not quite 12), got {age_months}"
        
    # ===== Year Boundary Tests =====
    
    def test_crossing_year_boundary(self):
        """Test age calculation crossing year boundary."""
        birth_date = date(2022, 10, 15)
        measurement_date = date(2023, 2, 15)  # 4 months later
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 4, f"Expected 4 months, got {age_months}"
        
    def test_born_in_december_measured_in_january(self):
        """Test age calculation for baby born in December, measured in January."""
        birth_date = date(2022, 12, 15)
        measurement_date = date(2023, 1, 15)  # 1 month later
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 1, f"Expected 1 month, got {age_months}"
        
    def test_born_late_december_measured_early_january(self):
        """Test age calculation for baby born Dec 31, measured Jan 1."""
        birth_date = date(2022, 12, 31)
        measurement_date = date(2023, 1, 1)  # Just 1 day later
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 0, f"Expected 0 months, got {age_months}"
        
    # ===== Leap Year Tests =====
    
    def test_born_on_leap_day(self):
        """Test age calculation for child born on February 29 (leap day)."""
        birth_date = date(2020, 2, 29)  # Leap year
        measurement_date = date(2021, 2, 28)  # Non-leap year
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # Should be 11 months (day 28 < day 29)
        assert age_months == 11, f"Expected 11 months, got {age_months}"
        
    def test_born_on_leap_day_measured_march_1(self):
        """Test age calculation for child born on leap day, measured on March 1."""
        birth_date = date(2020, 2, 29)  # Leap year
        measurement_date = date(2021, 3, 1)  # Non-leap year
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 12, f"Expected 12 months, got {age_months}"
        
    def test_born_on_leap_day_one_year_later_leap_year(self):
        """Test age calculation for child born on leap day, measured one leap cycle later."""
        birth_date = date(2020, 2, 29)  # Leap year
        measurement_date = date(2024, 2, 29)  # Also leap year
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 48, f"Expected 48 months (4 years), got {age_months}"
        
    # ===== Longer Duration Tests =====
    
    def test_five_years(self):
        """Test age calculation for five years."""
        birth_date = date(2018, 5, 10)
        measurement_date = date(2023, 5, 10)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 60, f"Expected 60 months, got {age_months}"
        
    def test_ten_years(self):
        """Test age calculation for ten years."""
        birth_date = date(2013, 7, 25)
        measurement_date = date(2023, 7, 25)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 120, f"Expected 120 months, got {age_months}"
        
    def test_fifteen_years(self):
        """Test age calculation for fifteen years."""
        birth_date = date(2008, 3, 15)
        measurement_date = date(2023, 3, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 180, f"Expected 180 months, got {age_months}"
        
    def test_nineteen_years(self):
        """Test age calculation for nineteen years (max WHO range)."""
        birth_date = date(2004, 1, 1)
        measurement_date = date(2023, 1, 1)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 228, f"Expected 228 months, got {age_months}"
        
    # ===== Edge Cases for End of Month =====
    
    def test_born_on_31st_measured_in_month_with_30_days(self):
        """Test age for child born on 31st, measured in month with 30 days."""
        birth_date = date(2023, 1, 31)
        measurement_date = date(2023, 4, 30)  # April has 30 days
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # Day 30 < Day 31, so should be 2 months
        assert age_months == 2, f"Expected 2 months, got {age_months}"
        
    def test_born_on_31st_measured_in_february(self):
        """Test age for child born on 31st, measured on Feb 28."""
        birth_date = date(2023, 1, 31)
        measurement_date = date(2023, 2, 28)  # February has 28 days
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # Day 28 < Day 31, so should be 0 months
        assert age_months == 0, f"Expected 0 months, got {age_months}"
        
    def test_born_on_30th_measured_in_february(self):
        """Test age for child born on 30th, measured on Feb 28."""
        birth_date = date(2023, 1, 30)
        measurement_date = date(2023, 2, 28)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # Day 28 < Day 30, so should be 0 months
        assert age_months == 0, f"Expected 0 months, got {age_months}"
        
    # ===== Month Boundary Edge Cases =====
    
    def test_first_day_of_month(self):
        """Test age for child born on first day of month."""
        birth_date = date(2023, 1, 1)
        measurement_date = date(2023, 4, 1)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 3, f"Expected 3 months, got {age_months}"
        
    def test_last_day_of_month_to_first_day(self):
        """Test age from last day of month to first day of next month."""
        birth_date = date(2023, 1, 31)
        measurement_date = date(2023, 3, 1)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # March 1 is before the 31st of the month
        assert age_months == 1, f"Expected 1 month, got {age_months}"
        
    # ===== Special Date Tests =====
    
    def test_new_years_eve_to_new_years_day(self):
        """Test age from New Year's Eve to New Year's Day."""
        birth_date = date(2022, 12, 31)
        measurement_date = date(2024, 1, 1)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        # From Dec 31, 2022 to Jan 1, 2024 - day 1 < day 31
        assert age_months == 12, f"Expected 12 months, got {age_months}"
        
    def test_birthday_scenario(self):
        """Test common scenario: child's birthday."""
        birth_date = date(2021, 8, 15)
        measurement_date = date(2023, 8, 15)
        
        age_months = calculate_age_in_months(birth_date, measurement_date)
        
        assert age_months == 24, f"Expected 24 months (2nd birthday), got {age_months}"
