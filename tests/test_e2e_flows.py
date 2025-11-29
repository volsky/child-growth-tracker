"""
End-to-end tests for complete growth tracking flows
"""

import pytest
import os
from datetime import date

from growth_utils import (
    calculate_age_in_months,
    calculate_bmi,
    get_height_data,
    get_weight_data,
    get_bmi_data,
    calculate_z_score,
    interpret_z_score,
    get_default_measurements
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


class TestGrowthDataRetrievalFlow:
    """E2E tests for growth data retrieval flow"""

    def test_who_male_complete_data_flow(self):
        """Test complete WHO data flow for male child"""
        gender = "Male"
        data_source = "WHO"

        # Step 1: Load height data
        height_data = get_height_data(gender, data_source)
        assert not height_data.empty
        assert all(col in height_data.columns for col in ['age_months', 'p3', 'p15', 'p50', 'p85', 'p97'])

        # Step 2: Load weight data
        weight_data = get_weight_data(gender, data_source)
        assert not weight_data.empty

        # Step 3: Load BMI data (WHO starts at 61 months)
        bmi_data = get_bmi_data(gender, data_source)
        assert not bmi_data.empty
        assert bmi_data['age_months'].min() >= 61

        # Step 4: Verify data consistency - percentiles should be in order
        sample_height = height_data.iloc[50]  # Pick a sample row
        assert sample_height['p3'] < sample_height['p15'] < sample_height['p50'] < sample_height['p85'] < sample_height['p97']

    def test_who_female_complete_data_flow(self):
        """Test complete WHO data flow for female child"""
        gender = "Female"
        data_source = "WHO"

        height_data = get_height_data(gender, data_source)
        weight_data = get_weight_data(gender, data_source)
        bmi_data = get_bmi_data(gender, data_source)

        assert not height_data.empty
        assert not weight_data.empty
        assert not bmi_data.empty

        # WHO female data should have same structure
        sample_row = height_data.iloc[20]
        assert sample_row['p3'] < sample_row['p50'] < sample_row['p97']

    def test_cdc_male_complete_data_flow(self):
        """Test complete CDC data flow for male child"""
        gender = "Male"
        data_source = "CDC"

        height_data = get_height_data(gender, data_source)
        weight_data = get_weight_data(gender, data_source)
        bmi_data = get_bmi_data(gender, data_source)

        # CDC data starts at 24 months
        assert not height_data.empty
        assert height_data['age_months'].min() >= 24

        assert not weight_data.empty
        assert not bmi_data.empty
        assert bmi_data['age_months'].min() >= 24

    def test_cdc_female_complete_data_flow(self):
        """Test complete CDC data flow for female child"""
        gender = "Female"
        data_source = "CDC"

        height_data = get_height_data(gender, data_source)
        weight_data = get_weight_data(gender, data_source)
        bmi_data = get_bmi_data(gender, data_source)

        assert not height_data.empty
        assert not weight_data.empty
        assert not bmi_data.empty

    def test_data_coverage_across_ages(self):
        """Test that data covers expected age ranges"""
        # WHO height should cover 0-228 months
        who_height = get_height_data("Male", "WHO")
        assert who_height['age_months'].min() == 0
        assert who_height['age_months'].max() >= 200

        # WHO weight only covers 0-120 months
        who_weight = get_weight_data("Male", "WHO")
        assert who_weight['age_months'].min() == 0
        assert who_weight['age_months'].max() <= 121

        # CDC covers 24-240 months
        cdc_height = get_height_data("Male", "CDC")
        assert cdc_height['age_months'].min() >= 24


class TestMeasurementAnalysisFlow:
    """E2E tests for measurement analysis flow"""

    def test_infant_measurement_flow_who(self):
        """Test complete measurement analysis for an infant using WHO data"""
        # Child: Male, 12 months old
        birth_date = date(2022, 6, 15)
        measurement_date = date(2023, 6, 15)
        age_months = calculate_age_in_months(birth_date, measurement_date)
        assert age_months == 12

        height_cm = 76.0
        weight_kg = 10.0

        # Calculate BMI
        bmi = calculate_bmi(height_cm, weight_kg)
        assert 13 < bmi < 20  # Reasonable BMI for infant

        # Get z-scores
        height_z, height_pct, _, _ = calculate_z_score(age_months, height_cm, 'height', 'Male', 'WHO')
        weight_z, weight_pct, _, _ = calculate_z_score(age_months, weight_kg, 'weight', 'Male', 'WHO')

        assert height_z is not None
        assert weight_z is not None

        # Interpret results
        height_interp, height_status = interpret_z_score(height_z, 'height')
        weight_interp, weight_status = interpret_z_score(weight_z, 'weight')

        # Normal values should be in normal range
        assert -2 < height_z < 2
        assert height_status == "success"

    def test_toddler_measurement_flow_who(self):
        """Test complete measurement analysis for a toddler using WHO data"""
        # Child: Female, 30 months old
        birth_date = date(2021, 1, 1)
        measurement_date = date(2023, 7, 1)
        age_months = calculate_age_in_months(birth_date, measurement_date)

        height_cm = 88.0
        weight_kg = 12.5

        bmi = calculate_bmi(height_cm, weight_kg)

        height_z, _, _, _ = calculate_z_score(age_months, height_cm, 'height', 'Female', 'WHO')
        weight_z, _, _, _ = calculate_z_score(age_months, weight_kg, 'weight', 'Female', 'WHO')

        assert height_z is not None
        assert weight_z is not None

    def test_child_measurement_flow_who_with_bmi(self):
        """Test complete measurement analysis for older child with BMI using WHO"""
        # Child: Male, 6 years old (72 months)
        age_months = 72

        height_cm = 116.0
        weight_kg = 21.0

        bmi = calculate_bmi(height_cm, weight_kg)
        assert 14 < bmi < 18  # Normal range for 6 year old

        # Get z-scores for all measurements
        height_z, _, _, _ = calculate_z_score(age_months, height_cm, 'height', 'Male', 'WHO')
        bmi_z, bmi_pct, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Male', 'WHO')

        assert height_z is not None
        assert bmi_z is not None  # WHO BMI available from 61 months

        # Interpret BMI
        bmi_interp, bmi_status = interpret_z_score(bmi_z, 'bmi')
        assert bmi_interp is not None

    def test_child_measurement_flow_cdc(self):
        """Test complete measurement analysis using CDC data"""
        # CDC data starts at 24 months
        age_months = 48  # 4 years old

        height_cm = 102.0
        weight_kg = 16.0

        bmi = calculate_bmi(height_cm, weight_kg)

        # Get z-scores using CDC
        height_z, _, _, _ = calculate_z_score(age_months, height_cm, 'height', 'Female', 'CDC')
        weight_z, _, _, _ = calculate_z_score(age_months, weight_kg, 'weight', 'Female', 'CDC')
        bmi_z, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Female', 'CDC')

        assert height_z is not None
        assert weight_z is not None
        assert bmi_z is not None  # CDC BMI available from 24 months

    def test_underweight_detection_flow(self):
        """Test that underweight child is correctly identified"""
        age_months = 24
        height_cm = 87.0
        weight_kg = 9.0  # Low weight for age

        weight_z, _, _, _ = calculate_z_score(age_months, weight_kg, 'weight', 'Male', 'WHO')
        assert weight_z is not None
        assert weight_z < -2  # Should be underweight

        interp, status = interpret_z_score(weight_z, 'weight')
        assert "Underweight" in interp or "underweight" in interp.lower()

    def test_overweight_detection_flow(self):
        """Test that overweight child is correctly identified"""
        age_months = 48
        bmi = 19.0  # High BMI for 4 year old

        bmi_z, _, _, _ = calculate_z_score(age_months, bmi, 'bmi', 'Male', 'CDC')
        assert bmi_z is not None
        assert bmi_z > 1  # Should be overweight

        interp, status = interpret_z_score(bmi_z, 'bmi')
        assert "Overweight" in interp or "Obese" in interp


class TestChildAgeCalculationFlow:
    """E2E tests for child age calculation flow"""

    def test_age_calculation_newborn(self):
        """Test age calculation for newborn"""
        birth_date = date(2023, 6, 15)
        measurement_date = date(2023, 6, 15)

        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 0

        # Get defaults for newborn
        height, weight = get_default_measurements(age, "Male", "WHO")
        assert 45 < height < 55  # Birth height range
        assert 2 < weight < 5  # Birth weight range

    def test_age_calculation_6_months(self):
        """Test age calculation for 6 month old"""
        birth_date = date(2023, 1, 1)
        measurement_date = date(2023, 7, 1)

        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 6

        height, weight = get_default_measurements(age, "Female", "WHO")
        assert 60 < height < 70
        assert 6 < weight < 9

    def test_age_calculation_1_year(self):
        """Test age calculation for 1 year old"""
        birth_date = date(2022, 3, 10)
        measurement_date = date(2023, 3, 10)

        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 12

    def test_age_calculation_5_years(self):
        """Test age calculation for 5 year old"""
        birth_date = date(2018, 6, 1)
        measurement_date = date(2023, 6, 1)

        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 60

        # For 5 year old, WHO BMI data should be available (starts at 61 months)
        # But we're at exactly 60 months so might not be available
        height, weight = get_default_measurements(age, "Male", "WHO")
        assert 100 < height < 120
        assert 15 < weight < 25

    def test_age_calculation_teenager(self):
        """Test age calculation for teenager"""
        birth_date = date(2008, 1, 15)
        measurement_date = date(2023, 7, 15)

        age = calculate_age_in_months(birth_date, measurement_date)
        assert age == 186  # About 15.5 years

        # For teenager, height data should be available
        height_data = get_height_data("Male", "WHO")
        assert age <= height_data['age_months'].max()


class TestCompleteGrowthTrackingScenario:
    """Full scenario tests simulating real-world usage"""

    def test_track_male_child_growth_over_time(self):
        """Simulate tracking a male child's growth from birth to 3 years"""
        gender = "Male"
        data_source = "WHO"
        birth_date = date(2020, 1, 15)

        # Measurements at different ages
        measurements = [
            {"date": date(2020, 1, 15), "height": 50.0, "weight": 3.5},    # Birth
            {"date": date(2020, 4, 15), "height": 62.0, "weight": 6.0},    # 3 months
            {"date": date(2020, 7, 15), "height": 68.0, "weight": 8.0},    # 6 months
            {"date": date(2021, 1, 15), "height": 76.0, "weight": 10.5},   # 12 months
            {"date": date(2022, 1, 15), "height": 87.0, "weight": 12.5},   # 24 months
            {"date": date(2023, 1, 15), "height": 96.0, "weight": 14.5},   # 36 months
        ]

        results = []
        for m in measurements:
            age = calculate_age_in_months(birth_date, m["date"])
            bmi = calculate_bmi(m["height"], m["weight"])

            height_z, height_pct, _, _ = calculate_z_score(age, m["height"], 'height', gender, data_source)
            weight_z, weight_pct, _, _ = calculate_z_score(age, m["weight"], 'weight', gender, data_source)

            results.append({
                "age": age,
                "height_z": height_z,
                "weight_z": weight_z,
                "bmi": bmi
            })

        # Verify growth pattern is reasonable
        # Heights and weights should increase over time
        for i in range(1, len(measurements)):
            assert measurements[i]["height"] > measurements[i-1]["height"]
            assert measurements[i]["weight"] > measurements[i-1]["weight"]

        # All z-scores should be calculated (within WHO data range)
        for r in results:
            if r["age"] <= 120:  # WHO weight data only up to 120 months
                assert r["height_z"] is not None
                # z-scores should be in reasonable range for normal child
                assert -4 < r["height_z"] < 4

    def test_track_female_child_with_cdc_data(self):
        """Simulate tracking a female child's growth using CDC data"""
        gender = "Female"
        data_source = "CDC"
        birth_date = date(2019, 6, 1)

        # CDC data starts at 24 months, so test from 2 years onwards
        measurements = [
            {"date": date(2021, 6, 1), "height": 86.0, "weight": 12.0},    # 24 months
            {"date": date(2022, 6, 1), "height": 95.0, "weight": 14.0},    # 36 months
            {"date": date(2023, 6, 1), "height": 103.0, "weight": 17.0},   # 48 months
        ]

        for m in measurements:
            age = calculate_age_in_months(birth_date, m["date"])
            bmi = calculate_bmi(m["height"], m["weight"])

            height_z, _, _, _ = calculate_z_score(age, m["height"], 'height', gender, data_source)
            weight_z, _, _, _ = calculate_z_score(age, m["weight"], 'weight', gender, data_source)
            bmi_z, _, _, _ = calculate_z_score(age, bmi, 'bmi', gender, data_source)

            # All CDC z-scores should be available from 24 months
            assert height_z is not None
            assert weight_z is not None
            assert bmi_z is not None

    def test_default_values_match_50th_percentile(self):
        """Test that default measurements are close to 50th percentile"""
        test_ages = [12, 24, 36, 48]

        for age in test_ages:
            default_h, default_w = get_default_measurements(age, "Male", "WHO")

            # Get the actual 50th percentile from data
            height_data = get_height_data("Male", "WHO")
            row = height_data[height_data['age_months'] == age]
            if not row.empty:
                p50_height = row['p50'].values[0]
                # Default should be within 1 cm of p50
                assert abs(default_h - p50_height) < 1
