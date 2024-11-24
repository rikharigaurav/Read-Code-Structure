import requests

# Fetch Data from URL
def fetch_data(url: str) -> dict:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx, 5xx)
        return response.json()  # Parse JSON response
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise
    except Exception as err:    
        print(f"Error occurred: {err}")
        raise

# Merge two dictionaries
from typing import TypeVar, Dict

T = TypeVar('T', bound=dict)

def merge(dict1: T, dict2: T) -> T:
    return {**dict1, **dict2}

# Process value (string or number)
def process_value(value):
    if isinstance(value, str):
        return f"String value: {value.upper()}"
    elif isinstance(value, (int, float)):
        return f"Number value: {value * 2}"
    return 'Invalid value'

print(process_value('hello'))  # Output: String value: HELLO
print(process_value(10))       # Output: Number value: 20

# Factorial function
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Output: 120

# Example usage of merging two dictionaries
merged = merge({'name': 'Alice'}, {'age': 30})
print(merged)  # Output: {'name': 'Alice', 'age': 30}

# Usage of fetch_data function
try:
    data = fetch_data('https://api.example.com/data')
    print(data)
except Exception as e:
    print(f"Error fetching data: {e}")
