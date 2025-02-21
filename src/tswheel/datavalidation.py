from datetime import datetime


def is_valid_date_format(date_string: str) -> bool:
    """
    Validates if a string matches the YYYY-MM-DD date format and represents a valid date.

    Args:
        date_string (str): The string to validate

    Returns:
        bool: True if the string is in YYYY-MM-DD format and represents a valid date,
              False otherwise
    Examples:
        >>> is_valid_date_format('2024-01-13')
        True
        >>> is_valid_date_format('2024-02-30')  # Invalid date
        False
        >>> is_valid_date_format('2024/01/13')  # Wrong format
        False
        >>> is_valid_date_format('abc')
        False
    """
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def is_valid_monthly_format(date_string: str) -> bool:
    """
    Validates if a string matches the YYYY-MM date format.

    Args:
        date_string (str): The string to validate

    Returns:
        bool: True if the string is in YYYY-MM format, False otherwise

    Examples:
        >>> is_valid_monthly_format('2024-01')
        True
        >>> is_valid_monthly_format('2024-13')  # Invalid month
        False
        >>> is_valid_monthly_format('2024/01')  # Wrong format
        False
        >>> is_valid_monthly_format('abc')
        False
    """
    try:
        datetime.strptime(date_string, "%Y-%m")
        return True
    except ValueError:
        return False


def is_valid_quarterly_format(date_string: str) -> bool:
    """
    Validates if a string matches the YYYYQq quarterly format.

    Args:
        date_string (str): The string to validate

    Returns:
        bool: True if the string is in YYYYQq format, False otherwise

    Examples:
        >>> is_valid_quarterly_format('2024Q1')
        True
        >>> is_valid_quarterly_format('2024Q4')
        True
        >>> is_valid_quarterly_format('2024Q5')  # Invalid quarter
        False
        >>> is_valid_quarterly_format('2024-Q1')  # Wrong format
        False
        >>> is_valid_quarterly_format('abc')
        False
    """
    is_str = isinstance(date_string, str)
    is_six_chars = len(date_string) == 6
    has_q = date_string[4] == "Q"
    is_valid_quarter = date_string[5] in "1234"
    is_valid_year = date_string[:4].isdigit()
    return is_str and is_six_chars and has_q and is_valid_quarter and is_valid_year
