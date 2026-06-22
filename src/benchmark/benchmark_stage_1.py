import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config.settings import (
    INSTANCE_FILE,
    DP_RESULTS_FILE,
    HEURISTIC_RESULTS_FILE,
    RESULTS_DIR,
)
from src.classical_solvers.exact_dp import solve_exact_dp
from src.classical_solvers.heuristic import solve_nn_two_opt
from src.utils.tour_utils import is_valid_tour, count_edge_crossings
from src.utils.io_utils import read_jsonl, write_jsonl

DP_MAX_N = 15


def time_complexity_label(method: str, n: int) -> str:
    if method == "exact_dp":
        return f"O(n^2 * 2^n) = O({n}^2 * 2^{n})"
    elif method == "nearest_neighbor":
        return f"O(n^2) = O({n}^2)"
    elif method == "nn_two_opt":
        return f"O(n^2 * iterations) approx O({n}^2)"
    return "unknown"


def run_dp_benchmark(instances: list) -> list:
    results = []
    eligible = [inst for inst in instances if inst["n"] <= DP_MAX_N]
    skipped = len(instances) - len(eligible)

    print(f"\n{'='*60}")
    print(f"EXACT DP BENCHMARK")
    print(f"Running on {len(eligible)} instances (n <= {DP_MAX_N})")
    print(f"Skipping {skipped} instances (n > {DP_MAX_N}, infeasible)")
    print(f"{'='*60}")

    for inst in eligible:
        n = inst["n"]
        coords = np.array(inst["coords"])
        instance_id = inst["instance_id"]

        start = time.perf_counter()
        dp_length, dp_tour = solve_exact_dp(coords)
        elapsed = time.perf_counter() - start

        valid = is_valid_tour(dp_tour, n)
        crossings = count_edge_crossings(coords, dp_tour)

        record = {
            "instance_id": instance_id,
            "n": n,
            "method": "exact_dp",
            "time_complexity": time_complexity_label("exact_dp", n),
            "tour": dp_tour,
            "tour_length": round(dp_length, 6),
            "edge_crossings": crossings,
            "solve_time_seconds": round(elapsed, 6),
            "is_optimal": True,
            "optimality_gap_pct": 0.0,
        }
        results.append(record)

        print(
            f"  [{instance_id}] n={n:>3} | "
            f"length={dp_length:>8.4f} | "
            f"crossings={crossings} | "
            f"valid={valid} | "
            f"time={elapsed:.4f}s"
        )

    return results


def run_heuristic_benchmark(instances: list, dp_results: list) -> list:
    dp_lookup = {r["instance_id"]: r["tour_length"] for r in dp_results}

    results = []

    print(f"\n{'='*60}")
    print(f"NN + 2-OPT HEURISTIC BENCHMARK")
    print(f"Running on all {len(instances)} instances")
    print(f"{'='*60}")

    for inst in instances:
        n = inst["n"]
        coords = np.array(inst["coords"])
        instance_id = inst["instance_id"]

        final_tour, nn_length, final_length, elapsed = solve_nn_two_opt(coords)

        valid = is_valid_tour(final_tour, n)
        crossings = count_edge_crossings(coords, final_tour)

        dp_length = dp_lookup.get(instance_id)
        if dp_length is not None:
            gap_pct = round(((final_length - dp_length) / dp_length) * 100, 4)
            is_optimal = gap_pct == 0.0
        else:
            gap_pct = None
            is_optimal = None

        record = {
            "instance_id": instance_id,
            "n": n,
            "method": "nn_two_opt",
            "time_complexity": time_complexity_label("nn_two_opt", n),
            "tour": final_tour,
            "tour_length": round(final_length, 6),
            "nn_length_before_two_opt": round(nn_length, 6),
            "edge_crossings": crossings,
            "solve_time_seconds": round(elapsed, 6),
            "is_optimal": is_optimal,
            "optimality_gap_pct": gap_pct,
        }
        results.append(record)

        gap_str = f"{gap_pct:>+7.3f}%" if gap_pct is not None else "    N/A "
        print(
            f"  [{instance_id}] n={n:>3} | "
            f"nn={nn_length:>8.4f} -> 2opt={final_length:>8.4f} | "
            f"gap={gap_str} | "
            f"crossings={crossings} | "
            f"valid={valid} | "
            f"time={elapsed:.4f}s"
        )

    return results


def print_summary(dp_results: list, heuristic_results: list):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if dp_results:
        dp_times = [r["solve_time_seconds"] for r in dp_results]
        print(f"\nExact DP ({len(dp_results)} instances, n <= {DP_MAX_N})")
        print(f"  Avg solve time : {sum(dp_times)/len(dp_times):.4f}s")
        print(f"  Max solve time : {max(dp_times):.4f}s")

    print(f"\nNN+2opt ({len(heuristic_results)} instances, all N)")
    print(f"\n  {'N':>5} | {'Instances':>9} | {'Avg Length':>10} | {'Avg Gap%':>9} | {'Avg Crossings':>13} | {'Avg Time':>9}")
    print(f"  {'-'*70}")

    sizes = sorted(set(r["n"] for r in heuristic_results))
    for n in sizes:
        group = [r for r in heuristic_results if r["n"] == n]
        avg_len = sum(r["tour_length"] for r in group) / len(group)
        avg_time = sum(r["solve_time_seconds"] for r in group) / len(group)
        avg_crossings = sum(r["edge_crossings"] for r in group) / len(group)

        gaps = [r["optimality_gap_pct"] for r in group if r["optimality_gap_pct"] is not None]
        avg_gap = f"{sum(gaps)/len(gaps):>+.3f}%" if gaps else "   N/A"

        print(
            f"  {n:>5} | {len(group):>9} | {avg_len:>10.4f} | "
            f"{avg_gap:>9} | {avg_crossings:>13.1f} | {avg_time:>8.4f}s"
        )

    gapped = [r for r in heuristic_results if r["optimality_gap_pct"] is not None]
    if gapped:
        gaps = [r["optimality_gap_pct"] for r in gapped]
        exact_matches = sum(1 for g in gaps if g == 0.0)
        print(f"\n  Optimality gap (vs DP, n<={DP_MAX_N}):")
        print(f"     Avg gap    : {sum(gaps)/len(gaps):.4f}%")
        print(f"     Max gap    : {max(gaps):.4f}%")
        print(f"     Found optimal : {exact_matches}/{len(gapped)} instances")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading instances...")
    instances = read_jsonl(INSTANCE_FILE)
    print(f"Loaded {len(instances)} instances across sizes: {sorted(set(i['n'] for i in instances))}")

    dp_results = run_dp_benchmark(instances)
    write_jsonl(DP_RESULTS_FILE, dp_results, append=False)
    print(f"\nDP results saved -> {DP_RESULTS_FILE}")

    heuristic_results = run_heuristic_benchmark(instances, dp_results)
    write_jsonl(HEURISTIC_RESULTS_FILE, heuristic_results, append=False)
    print(f"\nHeuristic results saved -> {HEURISTIC_RESULTS_FILE}")

    print_summary(dp_results, heuristic_results)

    print(f"\nDone. Raw results in results/stage1_classical/")