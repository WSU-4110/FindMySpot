# tests/test_unit_app.py
import pytest
from test_app import clean_plate_text, is_valid_plate

# -------------------------------
# Tests for clean_plate_text()
# -------------------------------
@pytest.mark.parametrize("input_text, expected", [
    ("abc123", "ABC123"),
    (" a!b@c#1$2%3 ", "ABC123"),
    ("---123---", "123"),
    ("xyz!@#", "XYZ"),
])
def test_clean_plate_text(input_text, expected):
    assert clean_plate_text(input_text) == expected


# -------------------------------
# Tests for is_valid_plate()
# -------------------------------
@pytest.mark.parametrize("text, confidence, expected", [
    ("ABCD1234", 0.5, True),      # valid plate
    ("ABC", 0.5, False),          # too short
    ("ABCDEFGH9", 0.5, False),    # too long
    ("PERSON", 0.9, False),       # blocked word
    ("ABCD1234", 0.3, False),     # below min confidence
])
def test_is_valid_plate(text, confidence, expected):
    assert is_valid_plate(text, confidence) == expected