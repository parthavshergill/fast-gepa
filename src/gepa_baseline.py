"""GEPA Baseline implementation with full validation.

This module implements the baseline GEPA algorithm that validates
every candidate on the full validation set (expensive but accurate).
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Tuple

from .core import StudentModel, TaskEvaluator, TrajectoryFormatter


@dataclass
class Candidate:
    """A candidate prompt configuration."""

    prompt_config: Dict[str, str]  # Module name -> text
    creation_time: float = field(default_factory=time.time)
    wins: Set[int] = field(default_factory=set)  # Indices where this wins
    total_score: float = 0.0  # Sum of scores across all instances


def gepa_baseline(
    student: StudentModel,
    evaluator: TaskEvaluator,
    formatter: TrajectoryFormatter,
    probe_set: List[Dict],
    val_set: List[Dict],
    time_budget_s: float,
    batch_size: int,
    inference_config: Dict[str, Any],
    self_consistency_k: int = 1,
    verbose: bool = True,
) -> Tuple[List[Candidate], Dict, Dict]:
    """
    Baseline GEPA with full validation.

    Args:
        student: Student model to optimize
        evaluator: Task evaluator
        formatter: Trajectory formatter (unused in baseline, for interface compatibility)
        probe_set: For minibatch probing
        val_set: For Pareto frontier (should be large, 300-800 for GSM8K)
        time_budget_s: Time budget in seconds
        batch_size: Minibatch size
        inference_config: Sampling parameters
        self_consistency_k: Number of samples for self-consistency
        verbose: Print progress

    Returns:
        (pool, scores_dict, best_dict)
        - pool: List of Pareto-optimal candidates
        - scores_dict: {(candidate_id, instance_idx) -> score}
        - best_dict: {instance_idx -> best_score}
    """

    if verbose:
        print("\n" + "=" * 80)
        print("GEPA BASELINE (Full Validation)")
        print("=" * 80)
        print(f"Validation set size: {len(val_set)}")
        print(f"Self-consistency k: {self_consistency_k}")
        print(f"Expected cost per candidate: {len(val_set)} × {self_consistency_k} calls")
        print("=" * 80 + "\n")

    # Initialize
    seed_config = {
        "system": "You are a helpful math tutor.",
        "cot_prompt": "Let's solve this step by step:",
    }

    P = [Candidate(prompt_config=seed_config)]
    Best = {i: 0.0 for i in range(len(val_set))}
    Scores = {}  # Sparse: (candidate_id, instance_idx) -> score

    start = time.time()
    iteration = 0
    total_val_calls = 0
    accepted_count = 0

    while time.time() - start < time_budget_s:
        iteration += 1

        # (1) Select parent uniformly at random
        parent = random.choice(P)

        # (2) Probe on minibatch
        batch = random.sample(probe_set, min(batch_size, len(probe_set)))

        batch_results = []
        for instance in batch:
            if self_consistency_k > 1:
                trajectories = student.execute_with_self_consistency(
                    instance, parent.prompt_config, inference_config, self_consistency_k
                )
                # Majority vote
                results = [evaluator.evaluate(t, instance) for t in trajectories]
                # Pick most common answer
                result = max(
                    results,
                    key=lambda r: sum(rr.success == r.success for rr in results),
                )
            else:
                traj = student.execute(instance, parent.prompt_config, inference_config)
                result = evaluator.evaluate(traj, instance)

            batch_results.append(result)

        parent_acc = sum(r.success for r in batch_results) / len(batch_results)

        # (3) Reflect and mutate (simplified: just add variation)
        # TODO: Replace with actual reflection LLM call
        child_config = parent.prompt_config.copy()
        # Simple mutation: add a hint
        variations = [
            "Focus on identifying the key numbers and operations.",
            "Break down the problem into smaller steps.",
            "Check your arithmetic carefully.",
            "Make sure to show all intermediate calculations.",
            "Verify your answer makes sense in the context.",
        ]
        hint = random.choice(variations)
        child_config["cot_prompt"] = parent.prompt_config["cot_prompt"] + f" {hint}"

        child = Candidate(prompt_config=child_config)

        # (4) Quick check on minibatch
        batch_results_child = []
        for instance in batch:
            if self_consistency_k > 1:
                trajectories = student.execute_with_self_consistency(
                    instance,
                    child.prompt_config,
                    inference_config,
                    self_consistency_k,
                )
                results = [evaluator.evaluate(t, instance) for t in trajectories]
                result = max(
                    results,
                    key=lambda r: sum(rr.success == r.success for rr in results),
                )
            else:
                traj = student.execute(instance, child.prompt_config, inference_config)
                result = evaluator.evaluate(traj, instance)

            batch_results_child.append(result)

        child_acc = sum(r.success for r in batch_results_child) / len(
            batch_results_child
        )

        if child_acc < parent_acc:
            if verbose:
                print(
                    f"[Baseline] Iter {iteration}: Child rejected on minibatch "
                    f"(parent={parent_acc:.2f}, child={child_acc:.2f})"
                )
            continue

        # (5) FULL VALIDATION - expensive!
        if verbose:
            print(
                f"[Baseline] Iter {iteration}: Validating child on full set "
                f"({len(val_set)} instances)..."
            )

        wins = 0
        child_total = 0.0

        for i, instance in enumerate(val_set):
            if self_consistency_k > 1:
                trajectories = student.execute_with_self_consistency(
                    instance,
                    child.prompt_config,
                    inference_config,
                    self_consistency_k,
                )
                results = [evaluator.evaluate(t, instance) for t in trajectories]
                result = max(
                    results,
                    key=lambda r: sum(rr.success == r.success for rr in results),
                )
            else:
                traj = student.execute(instance, child.prompt_config, inference_config)
                result = evaluator.evaluate(traj, instance)

            total_val_calls += self_consistency_k  # Track cost

            score = result.score
            child_total += score
            Scores[(id(child), i)] = score

            if score > Best[i]:
                Best[i] = score
                wins += 1
                child.wins.add(i)

        child.total_score = child_total

        # (6) Add if Pareto-useful
        if wins > 0:
            P.append(child)
            accepted_count += 1
            if verbose:
                print(
                    f"[Baseline] Iter {iteration}: ✓ Accepted child with {wins} wins "
                    f"(total score: {child_total:.2f})"
                )
        else:
            if verbose:
                print(f"[Baseline] Iter {iteration}: ✗ Child had no wins")

    elapsed = time.time() - start

    if verbose:
        print("\n" + "=" * 80)
        print("BASELINE RESULTS")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s")
        print(f"Total iterations: {iteration}")
        print(f"Accepted candidates: {accepted_count}")
        print(f"Final pool size: {len(P)}")
        print(f"Total validation calls: {total_val_calls}")
        print(
            f"Cost: {total_val_calls} inference calls "
            f"({total_val_calls / self_consistency_k:.0f} evaluations × k={self_consistency_k})"
        )
        print("=" * 80 + "\n")

    return P, Scores, Best
