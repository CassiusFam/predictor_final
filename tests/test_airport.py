import pytest
from utils.airport import code_to_name, _load
import os
import json

# Define the path to the test data file relative to the tests directory
# This assumes your tests are in a 'tests' directory and 'data' is a sibling of 'utils'
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'airport_codes.json')

@pytest.fixture(scope="module", autouse=True)
def setup_airport_data():
    """Fixture to ensure airport data is loaded before tests run."""
    # Temporarily override the _DATA_FILE_PATH in airport.py for testing if necessary
    # or ensure that the default path used by _load() points to your test data or a copy
    # For this example, we'll assume _load() can be called directly if it uses a relative path
    # or an environment variable that can be set for the test environment.
    
    # If _load uses an absolute path or a fixed relative path not suitable for tests,
    # you might need to mock it or adjust the path it uses.
    # For simplicity, let's assume utils.airport._DATA_FILE_PATH can be temporarily modified
    # or that _load() is robust enough to find the data from the test context.

    # Ensure the data file exists for the tests
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"Test data file not found at {TEST_DATA_PATH}. Run process_airport_data.py first.")
    
    # Pre-load the data to ensure it's available for all tests in this module
    # This relies on _load() using a path that can be resolved correctly or being mockable.
    # If utils.airport._DATA_FILE_PATH is a global variable, you could do:
    # original_path = utils.airport._DATA_FILE_PATH
    # utils.airport._DATA_FILE_PATH = TEST_DATA_PATH
    _load.cache_clear() # Clear cache before test if it's persistent across runs
    _load() # Load data using the (potentially overridden) path
    # yield
    # utils.airport._DATA_FILE_PATH = original_path # Restore original path
    # _load.cache_clear() # Clear cache after tests


@pytest.mark.parametrize("iata_code, expected_name", [
    ("LAX", "Los Angeles International Airport"),
    ("JFK", "John F. Kennedy International Airport"),
    ("LHR", "London Heathrow Airport"),
    ("HND", "Tokyo Haneda International Airport"),
    ("CDG", "Paris Charles de Gaulle Airport"),
    ("AMS", "Amsterdam Airport Schiphol"),
    ("FRA", "Frankfurt am Main Airport"),
    ("DXB", "Dubai International Airport"),
    ("ATL", "Hartsfield-Jackson Atlanta International Airport"),
    ("ORD", "O'Hare International Airport"),
    ("PEK", "Beijing Capital International Airport"),
    ("PVG", "Shanghai Pudong International Airport"),
    ("SYD", "Sydney Kingsford Smith International Airport"),
    ("YYZ", "Toronto Pearson International Airport"),
    ("SIN", "Singapore Changi Airport"),
    ("ICN", "Incheon International Airport"),
    ("DEN", "Denver International Airport"),
    ("SFO", "San Francisco International Airport"),
    ("CLT", "Charlotte Douglas International Airport"),
    ("SEA", "Seattle-Tacoma International Airport"),
    ("ABE", "Lehigh Valley International Airport"), # Specific test case from user
])
def test_code_to_name_known_airports(iata_code, expected_name):
    """Test code_to_name with known IATA codes."""
    assert code_to_name(iata_code) == expected_name, \
        f"Expected '{expected_name}' for IATA code '{iata_code}', but got '{code_to_name(iata_code)}'."

@pytest.mark.parametrize("iata_code", [
    ("ZZZ"),
    ("123"),
    ("INVALID"),
])
def test_code_to_name_unknown_airports(iata_code):
    """Test code_to_name with unknown or invalid IATA codes."""
    expected_default_name = f"Unknown Airport ({iata_code})"
    assert code_to_name(iata_code) == expected_default_name, \
        f"Expected '{expected_default_name}' for unknown IATA code '{iata_code}', but got '{code_to_name(iata_code)}'."

def test_code_to_name_empty_input():
    """Test code_to_name with empty string input."""
    assert code_to_name("") == "Unknown Airport ()", \
        f"Expected 'Unknown Airport ()' for empty IATA code, but got '{code_to_name("")}'."

def test_code_to_name_none_input():
    """Test code_to_name with None input."""
    assert code_to_name(None) == "Unknown Airport (None)", \
        f"Expected 'Unknown Airport (None)' for None IATA code, but got '{code_to_name(None)}'."

# To run these tests, navigate to the 'predictor' directory in your terminal and run:
# python -m pytest
# Ensure you have pytest installed (pip install pytest)
# And that your utils directory is discoverable by Python (e.g., by having an __init__.py in utils, 
# or by setting PYTHONPATH, or by running pytest from the parent directory of utils).
