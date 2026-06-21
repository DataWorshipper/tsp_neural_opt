import pytest
import numpy as np
from src.classical_solvers.exact_dp import solve_exact_dp
def test_square_perimeter_tsp():
    mock_coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0]
    ])
    length, tour = solve_exact_dp(mock_coords)
    assert pytest.approx(length) == 40.0, f"Expected length 40.0, got {length}"
    assert len(tour) == 4, "Tour should contain exactly 4 cities"
    assert set(tour) == {0, 1, 2, 3}, "Tour must visit every city exactly once"
    assert tour[0] == 0, "Tour should start at city 0"


def test_single_city_tsp():
    mock_coords = np.array([[5.0, 5.0]])    
    length, tour = solve_exact_dp(mock_coords)
    assert length == 0.0
    assert tour == [0]