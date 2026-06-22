import pytest
import numpy as np

from src.classical_solvers.heuristic import (
    solve_nearest_neighbor,
    solve_two_opt,
    solve_nn_two_opt,
)
from src.utils.tour_utils import is_valid_tour, count_edge_crossings


@pytest.fixture
def square_coords():
    return np.array([
        [0.0,  0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0,  10.0],
    ])


@pytest.fixture
def crossing_coords():
    return np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 0.0],
        [2.0, 2.0],
    ])


@pytest.fixture
def large_coords():
    rng = np.random.default_rng(seed=42)
    return rng.uniform(0, 100, size=(50, 2))


class TestNearestNeighbor:

    def test_returns_valid_tour_square(self, square_coords):
        tour, length = solve_nearest_neighbor(square_coords)
        assert is_valid_tour(tour, 4)

    def test_starts_at_city_zero_by_default(self, square_coords):
        tour, _ = solve_nearest_neighbor(square_coords)
        assert tour[0] == 0

    def test_custom_start_city(self, square_coords):
        tour, _ = solve_nearest_neighbor(square_coords, start_city=2)
        assert tour[0] == 2
        assert is_valid_tour(tour, 4)

    def test_square_optimal_length(self, square_coords):
        _, length = solve_nearest_neighbor(square_coords)
        assert pytest.approx(length, rel=1e-5) == 40.0

    def test_single_city(self):
        coords = np.array([[5.0, 5.0]])
        tour, length = solve_nearest_neighbor(coords)
        assert tour == [0]
        assert length == 0.0

    def test_two_cities(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        tour, length = solve_nearest_neighbor(coords)
        assert is_valid_tour(tour, 2)
        assert pytest.approx(length, rel=1e-5) == 10.0

    def test_valid_tour_large(self, large_coords):
        tour, length = solve_nearest_neighbor(large_coords)
        assert is_valid_tour(tour, 50)
        assert length > 0.0

    def test_length_is_positive(self, large_coords):
        _, length = solve_nearest_neighbor(large_coords)
        assert length > 0.0


class TestTwoOpt:

    def test_does_not_break_valid_tour(self, square_coords):
        nn_tour, _ = solve_nearest_neighbor(square_coords)
        improved_tour, _ = solve_two_opt(square_coords, nn_tour)
        assert is_valid_tour(improved_tour, 4)

    def test_never_increases_length(self, large_coords):
        nn_tour, nn_len = solve_nearest_neighbor(large_coords)
        _, improved_len = solve_two_opt(large_coords, nn_tour)

        assert improved_len <= nn_len + 1e-10, \
            f"2-opt made tour worse: {nn_len:.4f} -> {improved_len:.4f}"

    def test_zero_crossings_after_two_opt(self, crossing_coords):
        nn_tour, _ = solve_nearest_neighbor(crossing_coords)
        improved_tour, _ = solve_two_opt(crossing_coords, nn_tour)

        crossings = count_edge_crossings(crossing_coords, improved_tour)

        assert crossings == 0, \
            f"2-opt should eliminate all crossings, found {crossings}"

    def test_zero_crossings_large(self, large_coords):
        nn_tour, _ = solve_nearest_neighbor(large_coords)
        improved_tour, _ = solve_two_opt(large_coords, nn_tour)

        crossings = count_edge_crossings(large_coords, improved_tour)

        assert crossings == 0

    def test_square_stays_optimal(self, square_coords):
        nn_tour, nn_len = solve_nearest_neighbor(square_coords)
        _, improved_len = solve_two_opt(square_coords, nn_tour)

        assert pytest.approx(improved_len, rel=1e-5) == 40.0

    def test_handles_trivial_tours(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        tour = [0, 1, 2]

        improved_tour, length = solve_two_opt(coords, tour)

        assert is_valid_tour(improved_tour, 3)
        assert length > 0.0


class TestNNTwoOpt:

    def test_returns_four_values(self, square_coords):
        result = solve_nn_two_opt(square_coords)

        assert len(result) == 4

    def test_final_tour_valid(self, square_coords):
        tour, nn_len, final_len, t = solve_nn_two_opt(square_coords)

        assert is_valid_tour(tour, 4)

    def test_final_length_lte_nn_length(self, large_coords):
        _, nn_len, final_len, _ = solve_nn_two_opt(large_coords)

        assert final_len <= nn_len + 1e-10

    def test_timing_is_positive(self, large_coords):
        _, _, _, t = solve_nn_two_opt(large_coords)

        assert t > 0.0

    def test_zero_crossings_end_to_end(self, large_coords):
        tour, _, _, _ = solve_nn_two_opt(large_coords)

        assert count_edge_crossings(large_coords, tour) == 0

    def test_square_finds_optimal(self, square_coords):
        _, _, final_len, _ = solve_nn_two_opt(square_coords)

        assert pytest.approx(final_len, rel=1e-5) == 40.0