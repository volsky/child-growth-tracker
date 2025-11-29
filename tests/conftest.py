"""
Pytest configuration and fixtures for child growth tracker tests
"""

import pytest
import sys
import os
from datetime import date

# Add parent directory to path to import from growth_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_child_male():
    """Sample male child data for testing"""
    return {
        'gender': 'Male',
        'birth_date': date(2020, 1, 15)
    }


@pytest.fixture
def sample_child_female():
    """Sample female child data for testing"""
    return {
        'gender': 'Female',
        'birth_date': date(2019, 6, 20)
    }


@pytest.fixture
def sample_measurement():
    """Sample measurement data for testing"""
    return {
        'date': date(2023, 6, 15),
        'age_months': 42,
        'height': 100.5,
        'weight': 15.5,
        'gender': 'Male'
    }
