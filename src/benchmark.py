"""Benchmarking infrastructure for comparing GEPA baseline vs GEPA-MI.

This module provides utilities for:
- Running both methods
- Tracking costs and wall time
- Evaluating final test accuracy
- Comparing speedup
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from .core import StudentModel, TaskEvaluator, TrajectoryFormatter
from .gepa_baseline import gepa_baseline, Candidate
from .gepa_mi import gepa_mi


@dataclass
class BenchmarkResult:
    """Results from running optimizer."""

    method: str
    wall_time: float
    pool_size: int
    total_inference_calls: int
    total_validation_calls: int
    avg_probes_per_candidate: float
    speedup: float
    final_accuracy: float
    iterations: int
    accepted_candidates: int


def evaluate_on_split(
    student: StudentModel,
    evaluator: TaskEvaluator,
    prompt_config: Dict[str, str],
    split: List[Dict],
    inference_config: Dict[str, Any],
    self_consistency_k: int,
) -> float:
    """Evaluate prompt config on split.

    Args:
        student: Student model
        evaluator: Task evaluator
        prompt_config: Prompt configuration to evaluate
        split: Data split to evaluate on
        inference_config: Sampling parameters
        self_consistency_k: Self-consistency samples

    Returns:
        Accuracy (fraction correct)
    """
    correct = 0
    for instance in split:
        if self_consistency_k > 1:
            trajs = student.execute_with_self_consistency(
                instance, prompt_config, inference_config, self_consistency_k
            )
            results = [evaluator.evaluate(t, instance) for t in trajs]
            # Majority vote
            result = max(
                results, key=lambda r: sum(rr.success == r.success for rr in results)
            )
        else:
            traj = student.execute(instance, prompt_config, inference_config)
            result = evaluator.evaluate(traj, instance)

        if result.success:
            correct += 1

    return correct / len(split)


def select_best_candidate(pool: List[Candidate]) -> Candidate:
    """Select best candidate from pool.

    Strategy: Pick candidate with most wins, breaking ties by total score.

    Args:
        pool: List of candidates

    Returns:
        Best candidate
    """
    if not pool:
        raise ValueError("Pool is empty")

    return max(pool, key=lambda c: (len(c.wins), c.total_score))


def run_cost_aware_benchmark(
    student: StudentModel,
    evaluator: TaskEvaluator,
    formatter: TrajectoryFormatter,
    probe_set: List[Dict],
    val_set: List[Dict],
    test_set: List[Dict],
    inference_config: Dict[str, Any],
    time_budget_s: float = 300,
    batch_size: int = 8,
    self_consistency_k: int = 8,
    max_probes: int = 10,
    delta_init: float = 0.05,
    delta_final: float = 0.02,
    verbose: bool = True,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run both methods and compare.

    Args:
        student: Student model
        evaluator: Task evaluator
        formatter: Trajectory formatter
        probe_set: Probe set for minibatch sampling
        val_set: Validation set (should be LARGE: 300-800 for GSM8K)
        test_set: Test set for final evaluation
        inference_config: Sampling parameters
        time_budget_s: Time budget per method
        batch_size: Minibatch size
        self_consistency_k: Self-consistency samples (should be ≥5 for speedup)
        max_probes: Max probes for GEPA-MI
        delta_init: Initial error tolerance
        delta_final: Final error tolerance
        verbose: Print progress

    Returns:
        (baseline_result, mi_result)
    """

    if verbose:
        print("\n" + "=" * 80)
        print("COST-AWARE BENCHMARK")
        print("=" * 80)
        print(f"Validation set size: {len(val_set)}")
        print(f"Test set size: {len(test_set)}")
        print(f"Self-consistency k: {self_consistency_k}")
        print(
            f"Expected per-instance cost: {self_consistency_k} × t_inference"
        )
        print(
            f"Baseline will make: ~N_candidates × {len(val_set)} × {self_consistency_k} calls"
        )
        print(
            f"GEPA-MI will make: ~N_candidates × {max_probes} × {self_consistency_k} calls"
        )
        print(f"Expected speedup: {len(val_set) / max_probes:.1f}x")
        print("=" * 80 + "\n")

    # ============================================================
    # BASELINE
    # ============================================================
    if verbose:
        print("\n" + ">" * 80)
        print("RUNNING BASELINE")
        print(">" * 80 + "\n")

    t0 = time.time()
    pool_baseline, scores_baseline, best_baseline = gepa_baseline(
        student=student,
        evaluator=evaluator,
        formatter=formatter,
        probe_set=probe_set,
        val_set=val_set,
        time_budget_s=time_budget_s,
        batch_size=batch_size,
        inference_config=inference_config,
        self_consistency_k=self_consistency_k,
        verbose=verbose,
    )
    t_baseline = time.time() - t0

    # Evaluate best candidate on test
    if pool_baseline:
        best_candidate = select_best_candidate(pool_baseline)
        test_acc_baseline = evaluate_on_split(
            student,
            evaluator,
            best_candidate.prompt_config,
            test_set,
            inference_config,
            self_consistency_k,
        )
        accepted_baseline = len(pool_baseline) - 1  # Exclude seed
    else:
        test_acc_baseline = 0.0
        accepted_baseline = 0

    total_val_calls_baseline = accepted_baseline * len(val_set) * self_consistency_k

    baseline_result = BenchmarkResult(
        method="GEPA Baseline",
        wall_time=t_baseline,
        pool_size=len(pool_baseline),
        total_inference_calls=total_val_calls_baseline,
        total_validation_calls=total_val_calls_baseline,
        avg_probes_per_candidate=len(val_set),
        speedup=1.0,
        final_accuracy=test_acc_baseline,
        iterations=0,  # Not tracked in current implementation
        accepted_candidates=accepted_baseline,
    )

    # ============================================================
    # GEPA-MI
    # ============================================================
    if verbose:
        print("\n" + ">" * 80)
        print("RUNNING GEPA-MI")
        print(">" * 80 + "\n")

    t0 = time.time()
    pool_mi, scores_mi, best_mi = gepa_mi(
        student=student,
        evaluator=evaluator,
        formatter=formatter,
        probe_set=probe_set,
        val_set=val_set,
        time_budget_s=time_budget_s,
        batch_size=batch_size,
        inference_config=inference_config,
        delta_init=delta_init,
        delta_final=delta_final,
        max_probes=max_probes,
        self_consistency_k=self_consistency_k,
        verbose=verbose,
    )
    t_mi = time.time() - t0

    # Evaluate
    if pool_mi:
        best_candidate = select_best_candidate(pool_mi)
        test_acc_mi = evaluate_on_split(
            student,
            evaluator,
            best_candidate.prompt_config,
            test_set,
            inference_config,
            self_consistency_k,
        )
        accepted_mi = len(pool_mi) - 1  # Exclude seed
    else:
        test_acc_mi = 0.0
        accepted_mi = 0

    # Count actual probes from wins (approximate)
    total_probes_mi = sum(len(c.wins) for c in pool_mi)
    total_val_calls_mi = total_probes_mi * self_consistency_k
    avg_probes = total_probes_mi / max(accepted_mi, 1) if accepted_mi > 0 else 0

    speedup = (
        baseline_result.total_validation_calls / max(total_val_calls_mi, 1)
        if total_val_calls_mi > 0
        else 0
    )

    mi_result = BenchmarkResult(
        method="GEPA-MI",
        wall_time=t_mi,
        pool_size=len(pool_mi),
        total_inference_calls=total_val_calls_mi,
        total_validation_calls=total_val_calls_mi,
        avg_probes_per_candidate=avg_probes,
        speedup=speedup,
        final_accuracy=test_acc_mi,
        iterations=0,  # Not tracked
        accepted_candidates=accepted_mi,
    )

    # ============================================================
    # REPORT
    # ============================================================
    if verbose:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)

        print(f"\n{baseline_result.method}:")
        print(f"  Wall time: {baseline_result.wall_time:.1f}s")
        print(f"  Pool size: {baseline_result.pool_size}")
        print(f"  Accepted candidates: {baseline_result.accepted_candidates}")
        print(f"  Total validation calls: {baseline_result.total_validation_calls}")
        print(f"  Test accuracy: {baseline_result.final_accuracy:.3f}")

        print(f"\n{mi_result.method}:")
        print(f"  Wall time: {mi_result.wall_time:.1f}s")
        print(f"  Pool size: {mi_result.pool_size}")
        print(f"  Accepted candidates: {mi_result.accepted_candidates}")
        print(f"  Total validation calls: {mi_result.total_validation_calls}")
        print(f"  Avg probes/candidate: {mi_result.avg_probes_per_candidate:.1f}")
        print(f"  Speedup: {mi_result.speedup:.2f}x")
        print(f"  Test accuracy: {mi_result.final_accuracy:.3f}")

        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS")
        print("=" * 80)

        if mi_result.speedup < 3.0:
            print("\n⚠️  WARNING: Speedup < 3x suggests:")
            print("    - Validation set too small (need 300-800)")
            print("    - Inference too fast (need k≥5 self-consistency)")
            print("    - Consider switching to costlier benchmark")
        else:
            print(f"\n✅ SUCCESS: {mi_result.speedup:.1f}x speedup achieved!")
            print(
                f"   Saved {baseline_result.total_validation_calls - mi_result.total_validation_calls} "
                f"inference calls"
            )

        print("\n" + "=" * 80 + "\n")

    return baseline_result, mi_result
