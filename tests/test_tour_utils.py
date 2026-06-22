import pytest
import numpy as np

from src.utils.tour_utils import (
    is_valid_tour,
    calculate_tour_length,
    count_edge_crossings,
)


def test_tour_validation():
    assert is_valid_tour([0, 1, 2, 3], 4) is True
    assert is_valid_tour([0, 1, 2], 4) is False
    assert is_valid_tour([0, 1, 2, 2], 4) is False
    assert is_valid_tour([0, 1, 2, 4], 4) is False


def test_crossing_detection():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    clean_tour = [0, 1, 2, 3]
    assert count_edge_crossings(coords, clean_tour) == 0

    crossing_tour = [0, 2, 1, 3]
    assert count_edge_crossings(coords, crossing_tour) == 1