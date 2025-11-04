"""
Self-Reflection baseline algorithm for prompt optimization.

This module implements a simple iterative refinement approach where
the same LLM acts as both student (solving problems) and judge
(reflecting on failures). No validation set or Pareto pool - just
continuous improvement through reflection.
"""

import time
import random
from typing import Dict, List, Any
from dataclasses import dataclass

from .core import (
    AgentTrajectory,
    StudentModel,
    TaskEvaluator,
    TrajectoryFormatter,
    EvaluationResult
)
from .reflection import ReflectionEngine


@dataclass
class PromptCandidate:
    """A prompt configuration with its observed performance."""
    prompt_config: Dict[str, str]
    probe_accuracy: float
    iteration: int


def run_self_reflection(
    student: StudentModel,
    evaluator: TaskEvaluator,
    formatter: TrajectoryFormatter,
    probe_set: List[Dict],
    inference_config: Dict[str, Any],
    time_budget_s: float = 300,
    batch_size: int = 8,
    self_consistency_k: int = 8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run self-reflection optimization.

    Simple iterative refinement: probe -> reflect -> mutate -> repeat.
    No validation set, no Pareto pool. Just continuous improvement via
    self-reflection on failures.

    Args:
        student: Student model for both problem solving and reflection
        evaluator: Task evaluator
        formatter: Trajectory formatter for reflection
        probe_set: Probe instances for evaluation
        inference_config: Sampling parameters
        time_budget_s: Time budget in seconds
        batch_size: Batch size for probing
        self_consistency_k: Self-consistency samples
        verbose: Whether to print progress

    Returns:
        Dictionary with results:
            - best_prompt_config: Best prompt found
            - best_accuracy: Accuracy on probe set
            - iterations: Number of iterations
            - total_inference_calls: Total inference calls
            - wall_time: Total wall time
            - history: List of all candidates tried
    """
    start_time = time.time()

    # Initialize reflection engine
    reflection_engine = ReflectionEngine(student, formatter)

    # Seed prompt (simple baseline)
    seed_config = {
        "system": "You are an expert at solving math word problems.",
        "cot_prompt": "Let's solve this step by step, showing all calculations clearly."
    }

    # Track history
    history: List[PromptCandidate] = []
    total_inference_calls = 0
    iteration = 0

    # Start with seed
    current_config = seed_config
    best_config = seed_config
    best_accuracy = 0.0

    if verbose:
        print("\n" + "=" * 80)
        print("SELF-REFLECTION OPTIMIZATION")
        print("=" * 80)
        print(f"Time budget: {time_budget_s}s")
        print(f"Batch size: {batch_size}")
        print(f"Self-consistency k: {self_consistency_k}")
        print(f"Probe set size: {len(probe_set)}")
        print("=" * 80 + "\n")

    # Main optimization loop
    while (time.time() - start_time) < time_budget_s:
        iteration += 1
        iter_start = time.time()

        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            elapsed = time.time() - start_time
            print(f"Elapsed: {elapsed:.1f}s / {time_budget_s}s")

        # Sample batch from probe set
        batch = random.sample(probe_set, min(batch_size, len(probe_set)))

        # Evaluate current prompt on batch
        batch_results: List[EvaluationResult] = []
        batch_trajectories: List[List[AgentTrajectory]] = []

        for instance in batch:
            if self_consistency_k > 1:
                # Self-consistency sampling
                trajs = student.execute_with_self_consistency(
                    problem=instance,
                    prompt_config=current_config,
                    inference_config=inference_config,
                    k=self_consistency_k
                )
                batch_trajectories.append(trajs)
                total_inference_calls += self_consistency_k

                # Evaluate all trajectories and vote
                results = [evaluator.evaluate(t, instance) for t in trajs]
                # Pick most common result
                result = max(results, key=lambda r: sum(rr.success == r.success for rr in results))
            else:
                # Single trajectory
                traj = student.execute(
                    problem=instance,
                    prompt_config=current_config,
                    inference_config=inference_config
                )
                batch_trajectories.append([traj])
                total_inference_calls += 1

                result = evaluator.evaluate(traj, instance)

            batch_results.append(result)

        # Compute accuracy on batch
        num_correct = sum(1 for r in batch_results if r.success)
        accuracy = num_correct / len(batch_results)

        if verbose:
            print(f"Batch accuracy: {accuracy:.2%} ({num_correct}/{len(batch_results)})")

        # Track this candidate
        candidate = PromptCandidate(
            prompt_config=current_config.copy(),
            probe_accuracy=accuracy,
            iteration=iteration
        )
        history.append(candidate)

        # Update best if improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = current_config.copy()
            if verbose:
                print(f"✓ New best accuracy: {best_accuracy:.2%}")

        # Collect failures for reflection
        failed_trajectories = []
        for i, result in enumerate(batch_results):
            if not result.success:
                # Take first trajectory from self-consistency samples
                traj = batch_trajectories[i][0]
                failed_trajectories.append((traj, batch[i]))

        if verbose:
            print(f"Failures: {len(failed_trajectories)}/{len(batch_results)}")

        # Check time before reflection (reflection adds cost)
        if (time.time() - start_time) >= time_budget_s:
            if verbose:
                print("Time budget exhausted before reflection")
            break

        # Reflect and mutate
        if verbose:
            print("Reflecting on failures...")

        next_config = reflection_engine.reflect_and_mutate(
            parent_config=current_config,
            failed_trajectories=failed_trajectories,
            inference_config=inference_config
        )
        total_inference_calls += 1  # One reflection call

        # Update current prompt for next iteration
        current_config = next_config

        iter_time = time.time() - iter_start
        if verbose:
            print(f"Iteration time: {iter_time:.1f}s")

    # Final summary
    wall_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 80)
        print("SELF-REFLECTION SUMMARY")
        print("=" * 80)
        print(f"Total iterations: {iteration}")
        print(f"Total inference calls: {total_inference_calls}")
        print(f"Wall time: {wall_time:.1f}s")
        print(f"Best accuracy: {best_accuracy:.2%}")
        print("=" * 80 + "\n")

    return {
        "best_prompt_config": best_config,
        "best_accuracy": best_accuracy,
        "iterations": iteration,
        "total_inference_calls": total_inference_calls,
        "wall_time": wall_time,
        "history": history
    }
