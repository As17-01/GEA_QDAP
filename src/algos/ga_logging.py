import time
from collections import defaultdict
from contextlib import contextmanager
from typing import List


# Collects and reports GA timing, iteration, and operator statistics.
#
# This is a plain collaborator (composed into BaseGA as `self.logger`), not a mixin --
# all the state it needs (timings, counters) is its own, and everything it reports
# (best cost, diversity) is passed in explicitly rather than reached for via `self`.
class GALogger:
    def __init__(self):
        self._timing = defaultdict(float)
        self._timing_calls = defaultdict(int)
        self._iteration_times: List[float] = []
        self.start_time: float = 0.0

        self._crossover_attempts = 0
        self._crossover_valid = 0
        self._crossover_new_best = 0

        self._mutation_attempts = 0
        self._mutation_valid = 0
        self._mutation_new_best = 0

    @contextmanager
    def timed(self, operation_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timing[operation_name] += time.perf_counter() - start
            self._timing_calls[operation_name] += 1

    def start_run(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def record_iteration(self, iter_time: float) -> None:
        self._iteration_times.append(iter_time)

    def record_crossover(self, attempts: int, valid: int, new_best: int) -> None:
        self._crossover_attempts += attempts
        self._crossover_valid += valid
        self._crossover_new_best += new_best

    def record_mutation(self, attempts: int, valid: int, new_best: int) -> None:
        self._mutation_attempts += attempts
        self._mutation_valid += valid
        self._mutation_new_best += new_best

    def print_iteration_info(self, iteration: int, iter_time: float, best_cost: float, avg_diversity: float) -> None:
        """Print detailed information for every N iterations."""
        print(
            f"Iter {iteration:5d} | Best: {best_cost:12.4f} | Diversity: {avg_diversity:4.4f} | "
            f"Time: {iter_time*1000:6.2f}ms | "
            f"CX: {self._crossover_new_best:3d}/{self._crossover_valid:5d}/{self._crossover_attempts:5d} | "
            f"MUT: {self._mutation_new_best:3d}/{self._mutation_valid:5d}/{self._mutation_attempts:5d}"
        )

    def print_final_report(self, best_cost: float) -> None:
        """Print final statistics after the run completes."""
        total_time = self.elapsed()
        num_iters = len(self._iteration_times)
        avg_iter = sum(self._iteration_times) / num_iters if num_iters > 0 else 0

        print("\n" + "=" * 100)
        print("GENETIC ALGORITHM - FINAL REPORT")
        print("=" * 100)
        print(f"Total runtime            : {total_time:8.4f} seconds")
        print(f"Iterations completed     : {num_iters}")
        print(f"Average time / iteration : {avg_iter*1000:8.2f} ms")
        print(f"Final best cost          : {best_cost:.6f}")

        print("\nTime breakdown:")
        for op, t in sorted(self._timing.items(), key=lambda x: x[1], reverse=True):
            calls = self._timing_calls[op]
            avg = t / calls if calls > 0 else 0
            print(f"  {op:18s} : {t:8.4f}s ({calls:5d} calls, {avg*1000:6.2f} ms avg)")

        print("=" * 100)

    def reset_operator_counters(self) -> None:
        """Reset counters for the next logging block."""
        self._crossover_attempts = self._crossover_valid = self._crossover_new_best = 0
        self._mutation_attempts = self._mutation_valid = self._mutation_new_best = 0
