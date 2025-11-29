"""
Pytest configuration and fixtures for E2E tests.

This module sets up mocks for Streamlit to allow testing the app functions
without running the full Streamlit application.
"""
import pytest
import sys
import os
from datetime import date, datetime
from unittest.mock import MagicMock, PropertyMock

# Add the parent directory to the path so we can import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockSessionState:
    """Mock for Streamlit session_state that supports both dict and attribute access."""
    def __init__(self):
        self._data = {
            'data_points': [],
            'child_info': None,
            'today_measurement': None,
            'data_source': 'WHO'
        }
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._data.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def get(self, key, default=None):
        return self._data.get(key, default)


def create_streamlit_mock():
    """Create a comprehensive Streamlit mock for testing."""
    mock_st = MagicMock()
    
    # Mock cache_data decorator to be a passthrough
    def cache_data_decorator(func):
        return func
    mock_st.cache_data = cache_data_decorator
    
    # Mock columns to return list of MagicMocks
    def mock_columns(widths):
        if isinstance(widths, int):
            return [MagicMock() for _ in range(widths)]
        return [MagicMock() for _ in widths]
    mock_st.columns = mock_columns
    
    # Mock session_state as a dict-like object with attribute access
    mock_st.session_state = MockSessionState()
    
    # Mock other common Streamlit methods
    mock_st.title = MagicMock()
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.write = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.success = MagicMock()
    mock_st.info = MagicMock()
    mock_st.selectbox = MagicMock(return_value='Male')
    mock_st.date_input = MagicMock(return_value=date.today())
    mock_st.number_input = MagicMock(return_value=0.0)
    mock_st.button = MagicMock(return_value=False)
    mock_st.download_button = MagicMock()
    mock_st.file_uploader = MagicMock(return_value=None)
    mock_st.divider = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock_st.data_editor = MagicMock()
    mock_st.plotly_chart = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.set_page_config = MagicMock()
    
    # Mock column_config
    mock_st.column_config = MagicMock()
    mock_st.column_config.TextColumn = MagicMock()
    mock_st.column_config.NumberColumn = MagicMock()
    
    return mock_st


# Create and install the mock before importing the app
mock_st = create_streamlit_mock()
sys.modules['streamlit'] = mock_st


@pytest.fixture
def sample_child_info_male():
    """Sample child info for a male child."""
    return {
        'gender': 'Male',
        'birth_date': date(2020, 1, 15)
    }


@pytest.fixture
def sample_child_info_female():
    """Sample child info for a female child."""
    return {
        'gender': 'Female',
        'birth_date': date(2019, 6, 20)
    }


@pytest.fixture
def sample_measurement_infant():
    """Sample measurement for an infant (12 months)."""
    return {
        'age_months': 12,
        'height': 75.0,  # cm
        'weight': 10.0,  # kg
        'gender': 'Male'
    }


@pytest.fixture
def sample_measurement_child():
    """Sample measurement for a child (5 years)."""
    return {
        'age_months': 60,
        'height': 110.0,  # cm
        'weight': 18.0,  # kg
        'gender': 'Female'
    }


@pytest.fixture
def sample_measurement_older_child():
    """Sample measurement for an older child (10 years)."""
    return {
        'age_months': 120,
        'height': 140.0,  # cm
        'weight': 32.0,  # kg
        'bmi': 16.33,  # kg/m²
        'gender': 'Male'
    }
