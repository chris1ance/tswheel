import pytest
from tswheel.datawork.strchecks import (
    is_valid_date_format,
    is_valid_monthly_format,
    is_valid_quarterly_format,
)


class TestIsValidDateFormat:
    """Tests for the is_valid_date_format function."""

    def test_valid_dates(self):
        """Test function returns True for valid dates in YYYY-MM-DD format."""
        valid_dates = [
            "2024-01-01",
            "2023-12-31",
            "2000-02-29",  # Leap year
            "1999-07-15",
        ]
        for date in valid_dates:
            assert is_valid_date_format(date), f"Should accept valid date: {date}"

    def test_invalid_dates(self):
        """Test function returns False for invalid dates."""
        invalid_dates = [
            "2023-02-30",  # Invalid day for February
            "2023-13-01",  # Invalid month
            "2023-00-01",  # Invalid month
            "2023-01-00",  # Invalid day
            "2023-01-32",  # Invalid day
        ]
        for date in invalid_dates:
            assert not is_valid_date_format(date), f"Should reject invalid date: {date}"

    def test_invalid_formats(self):
        """Test function returns False for incorrectly formatted strings."""
        invalid_formats = [
            "01-01-2023",  # Wrong order
            "2023/01/01",  # Wrong separator
            "20230101",  # No separators
            "23-01-01",  # Two-digit year
            "abcd-ef-gh",  # Not a date at all
            "",  # Empty string
            "2023-01",  # Incomplete
            "2023-01-01extra",  # Extra characters
        ]
        for date_str in invalid_formats:
            assert not is_valid_date_format(date_str), (
                f"Should reject invalid format: {date_str}"
            )

    def test_non_string_inputs(self):
        """Test function behavior with non-string inputs."""
        with pytest.raises(TypeError):
            is_valid_date_format(20230101)

        with pytest.raises(TypeError):
            is_valid_date_format(None)


class TestIsValidMonthlyFormat:
    """Tests for the is_valid_monthly_format function."""

    def test_valid_months(self):
        """Test function returns True for valid monthly dates in YYYY-MM format."""
        valid_months = [
            "2024-01",
            "2023-12",
            "2000-02",
            "1999-07",
        ]
        for month in valid_months:
            assert is_valid_monthly_format(month), f"Should accept valid month: {month}"

    def test_invalid_months(self):
        """Test function returns False for invalid months."""
        invalid_months = [
            "2023-13",  # Invalid month
            "2023-00",  # Invalid month
        ]
        for month in invalid_months:
            assert not is_valid_monthly_format(month), (
                f"Should reject invalid month: {month}"
            )

    def test_invalid_formats(self):
        """Test function returns False for incorrectly formatted strings."""
        invalid_formats = [
            "01-2023",  # Wrong order
            "2023/01",  # Wrong separator
            "202301",  # No separator
            "23-01",  # Two-digit year
            "abcd-ef",  # Not a month at all
            "",  # Empty string
            "2023",  # Incomplete
            "2023-01-01",  # Too specific (day included)
            "2023-01extra",  # Extra characters
        ]
        for month_str in invalid_formats:
            assert not is_valid_monthly_format(month_str), (
                f"Should reject invalid format: {month_str}"
            )

    def test_non_string_inputs(self):
        """Test function behavior with non-string inputs."""
        with pytest.raises(TypeError):
            is_valid_monthly_format(202301)

        with pytest.raises(TypeError):
            is_valid_monthly_format(None)


class TestIsValidQuarterlyFormat:
    """Tests for the is_valid_quarterly_format function."""

    def test_valid_quarters(self):
        """Test function returns True for valid quarterly dates in YYYYQq format."""
        valid_quarters = [
            "2024Q1",
            "2023Q2",
            "2000Q3",
            "1999Q4",
        ]
        for quarter in valid_quarters:
            assert is_valid_quarterly_format(quarter), (
                f"Should accept valid quarter: {quarter}"
            )

    def test_invalid_quarters(self):
        """Test function returns False for invalid quarters."""
        invalid_quarters = [
            "2023Q0",  # Invalid quarter
            "2023Q5",  # Invalid quarter
            "2023Q9",  # Invalid quarter
        ]
        for quarter in invalid_quarters:
            assert not is_valid_quarterly_format(quarter), (
                f"Should reject invalid quarter: {quarter}"
            )

    def test_invalid_formats(self):
        """Test function returns False for incorrectly formatted strings."""
        invalid_formats = [
            "Q1-2023",  # Wrong order
            "2023-Q1",  # Wrong format (has hyphen)
            "2023/Q1",  # Wrong separator
            "abcdQe",  # Not a quarter at all
            "2023Q1extra",  # Extra characters
        ]
        for quarter_str in invalid_formats:
            assert not is_valid_quarterly_format(quarter_str), (
                f"Should reject invalid format: {quarter_str}"
            )

    def test_short_strings(self):
        """Test function handles short strings properly."""
        short_strings = [
            "",  # Empty string
            "2023",  # Incomplete
            "2023Q",  # Missing quarter number
            "23Q1",  # Two-digit year - too short to access index 4
            "202Q1",  # Three-digit year - only 5 characters long
        ]

        for s in short_strings:
            assert not is_valid_quarterly_format(s), f"Should reject short string: {s}"

    def test_non_string_inputs(self):
        """Test function behavior with non-string inputs."""
        assert not is_valid_quarterly_format(2023)
        assert not is_valid_quarterly_format(None)
