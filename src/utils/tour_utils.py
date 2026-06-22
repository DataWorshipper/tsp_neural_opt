import numpy as np

def is_valid_tour(tour:list,expected_n:int)->bool:
    if len(tour)!=expected_n:
        return False
    return set(tour)==set(range(expected_n))

def calculate_tour_length(coords:np.ndarray,tour:list)->float:
    if not tour or len(tour) < 2:
        return 0.0
    ordered_coords = coords[tour]
    closed_loop_coords = np.vstack([ordered_coords, ordered_coords[0]])
    diffs = np.diff(closed_loop_coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    return float(np.sum(distances))


def count_edge_crossings(coords: np.ndarray, tour: list) -> int:
    def intersects(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (
            ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and
            ccw(p1, p2, p3) != ccw(p1, p2, p4)
        )

    n = len(tour)
    crossings = 0

    edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]

    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue

            p1, p2 = coords[edges[i][0]], coords[edges[i][1]]
            p3, p4 = coords[edges[j][0]], coords[edges[j][1]]

            if intersects(p1, p2, p3, p4):
                crossings += 1

    return crossings