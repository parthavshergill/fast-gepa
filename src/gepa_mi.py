"""GEPA-MI implementation with selective validation.

This module implements GEPA with Mutual Information-guided selective
validation, using Bayesian posterior models to decide when to stop probing.
"""

import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .core import StudentModel, TaskEvaluator, TrajectoryFormatter
from .reflection import ReflectionEngine
from .types import GSM8KProblem, MathSolution, PromptConfig, InferenceConfig


@dataclass
class Candidate:
    """A candidate prompt configuration."""

    prompt_config: PromptConfig
    creation_time: float = field(default_factory=time.time)
    wins: Set[int] = field(default_factory=set)  # Indices where this wins
    total_score: float = 0.0  # Sum of scores across all instances


class BetaBernoulliPosterior:
    """Efficient Beta-Bernoulli posterior for binary outcomes."""

    def __init__(self, n_instances: int):
        """Initialize posterior with uniform priors.

        Args:
            n_instances: Number of instances in validation set
        """
        self.n = n_instances
        self.alpha = np.ones(n_instances)
        self.beta = np.ones(n_instances)
        self.probed = np.zeros(n_instances, dtype=bool)

    def seed_from_minibatch(self, accuracy: float):
        """Initialize from minibatch performance.

        Args:
            accuracy: Accuracy on minibatch (0-1)
        """
        strength = 4.0  # Prior strength
        self.alpha = np.full(self.n, accuracy * strength)
        self.beta = np.full(self.n, (1 - accuracy) * strength)

    def update(self, idx: int, win: bool):
        """Bayesian update after observing outcome.

        Args:
            idx: Instance index
            win: Whether candidate won on this instance
        """
        if win:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1
        self.probed[idx] = True

    def mean(self) -> np.ndarray:
        """Expected win probabilities for each instance."""
        return self.alpha / (self.alpha + self.beta)

    def prob_Z_equals_1(self) -> float:
        """Probability that candidate wins on at least one instance.

        Returns:
            P(Z=1) where Z = 1 if wins on ≥1 instance, 0 otherwise
        """
        p_wins = self.mean()
        # P(Z=0) = product of P(loss_i) = product of (1 - p_i)
        log_prob_all_losses = np.sum(np.log(1 - p_wins + 1e-10))
        prob_all_losses = np.exp(log_prob_all_losses)
        return 1 - prob_all_losses


def select_next_instance_uncertainty(
    posterior: BetaBernoulliPosterior, remaining: List[int]
) -> int:
    """Select instance with probability closest to 0.5 (max uncertainty).

    Args:
        posterior: Current posterior distribution
        remaining: List of remaining instance indices

    Returns:
        Selected instance index
    """
    if not remaining:
        return None

    probs = posterior.mean()
    # Uncertainty: closer to 0.5 is more uncertain
    scores = [-abs(probs[i] - 0.5) for i in remaining]
    return remaining[np.argmax(scores)]


def selective_validate(
    student: StudentModel,
    evaluator: TaskEvaluator,
    child: Candidate,
    val_set: List[GSM8KProblem],
    best_scores: Dict[int, float],
    scores_dict: Dict,
    delta: float,
    max_probes: int,
    inference_config: InferenceConfig,
    self_consistency_k: int,
    minibatch_acc: float,
    verbose: bool = False,
) -> Tuple[bool, int]:
    """
    MI-guided selective validation.

    Args:
        student: Student model
        evaluator: Task evaluator
        child: Candidate to validate
        val_set: Validation set
        best_scores: Current best scores per instance
        scores_dict: Score dictionary to update
        delta: Error tolerance (lower = more confident)
        max_probes: Maximum probes before stopping
        inference_config: Sampling parameters
        self_consistency_k: Self-consistency samples
        minibatch_acc: Accuracy on minibatch (for seeding)
        verbose: Print debug info

    Returns:
        (accept: bool, probes_used: int)
    """

    n = len(val_set)
    posterior = BetaBernoulliPosterior(n)
    posterior.seed_from_minibatch(minibatch_acc)

    remaining = list(range(n))
    probes = 0

    while remaining and probes < max_probes:
        # Check stopping condition
        p_Z = posterior.prob_Z_equals_1()

        if probes >= 1:  # Need at least 1 probe
            if p_Z > 1 - delta:
                if verbose:
                    print(f"    ACCEPT: P(Z=1)={p_Z:.4f} > {1-delta:.4f}")
                return True, probes
            if p_Z < delta:
                if verbose:
                    print(f"    REJECT: P(Z=1)={p_Z:.4f} < {delta:.4f}")
                return False, probes

        # Select next instance to probe
        idx = select_next_instance_uncertainty(posterior, remaining)
        if idx is None:
            break

        # Evaluate on selected instance
        problem = val_set[idx]

        if self_consistency_k > 1:
            solutions = student.execute_with_self_consistency(
                problem,
                child.prompt_config,
                inference_config,
                self_consistency_k,
            )
            # Use rigorous self-consistency: vote on extracted answers
            result = evaluator.evaluate_with_self_consistency(solutions, problem)
        else:
            solution = student.execute(problem, child.prompt_config, inference_config)
            result = evaluator.evaluate(solution, problem)

        score = result.score
        best_score = best_scores.get(idx, 0.0)
        win = score > best_score

        # Update posterior
        posterior.update(idx, win)

        if win:
            # Update global state
            best_scores[idx] = score
            child.wins.add(idx)
            scores_dict[(id(child), idx)] = score

        remaining.remove(idx)
        probes += 1

        if verbose:
            print(
                f"    Probe {probes}/{max_probes}: idx={idx}, "
                f"score={score:.2f}, win={win}, P(Z=1)={posterior.prob_Z_equals_1():.4f}"
            )

    # Final decision based on posterior
    final_accept = posterior.prob_Z_equals_1() > 0.5

    if verbose:
        print(
            f"    FINAL: {'ACCEPT' if final_accept else 'REJECT'} "
            f"after {probes} probes (P(Z=1)={posterior.prob_Z_equals_1():.4f})"
        )

    return final_accept, probes


def gepa_mi(
    student: StudentModel,
    evaluator: TaskEvaluator,
    formatter: TrajectoryFormatter,
    probe_set: List[GSM8KProblem],
    val_set: List[GSM8KProblem],
    time_budget_s: float,
    batch_size: int,
    inference_config: InferenceConfig,
    delta_init: float = 0.05,
    delta_final: float = 0.02,
    max_probes: int = 10,
    self_consistency_k: int = 1,
    use_reflection: bool = False,
    verbose: bool = True,
) -> Tuple[List[Candidate], Dict, Dict]:
    """
    GEPA with MI-guided selective validation.

    Args:
        student: Student model to optimize
        evaluator: Task evaluator
        formatter: Trajectory formatter
        probe_set: For minibatch probing
        val_set: For Pareto frontier (should be large, 300-800 for GSM8K)
        time_budget_s: Time budget in seconds
        batch_size: Minibatch size
        inference_config: Sampling parameters
        delta_init: Initial error tolerance
        delta_final: Final error tolerance (adaptive)
        max_probes: Maximum probes per candidate
        self_consistency_k: Number of samples for self-consistency
        use_reflection: Use ReflectionEngine for mutations (default: False, uses random hints)
        verbose: Print progress

    Returns:
        (pool, scores_dict, best_scores)
        - pool: List of Pareto-optimal candidates
        - scores_dict: {(candidate_id, instance_idx) -> score}
        - best_scores: {instance_idx -> best_score}
    """

    if verbose:
        print("\n" + "=" * 80)
        print("GEPA-MI (Selective Validation)")
        print("=" * 80)
        print(f"Validation set size: {len(val_set)}")
        print(f"Self-consistency k: {self_consistency_k}")
        print(f"Max probes per candidate: {max_probes}")
        print(
            f"Expected cost per candidate: ~{max_probes} × {self_consistency_k} calls "
            f"(vs {len(val_set)} × {self_consistency_k} for baseline)"
        )
        print(
            f"Theoretical speedup: {len(val_set) / max_probes:.1f}x "
            f"(if avg probes ≈ {max_probes})"
        )
        print("=" * 80 + "\n")

    # Initialize
    seed_config = PromptConfig(
        system="You are a helpful math tutor.",
        cot_prompt="Let's solve this step by step:"
    )

    P = [Candidate(prompt_config=seed_config)]
    BestScores = {i: 0.0 for i in range(len(val_set))}
    Scores = {}

    # Initialize ReflectionEngine if requested
    reflection_engine = ReflectionEngine(student, formatter) if use_reflection else None

    start = time.time()
    iteration = 0
    total_val_calls = 0
    total_probes = 0
    accepted_count = 0

    while time.time() - start < time_budget_s:
        iteration += 1

        # Adaptive delta (stricter over time)
        progress = (time.time() - start) / time_budget_s
        delta = delta_init + (delta_final - delta_init) * progress

        # (1) Select parent
        parent = random.choice(P)

        # (2) Probe on minibatch
        batch = random.sample(probe_set, min(batch_size, len(probe_set)))

        batch_results = []
        for problem in batch:
            if self_consistency_k > 1:
                solutions = student.execute_with_self_consistency(
                    problem, parent.prompt_config, inference_config, self_consistency_k
                )
                # Use rigorous self-consistency: vote on extracted answers
                result = evaluator.evaluate_with_self_consistency(solutions, problem)
            else:
                solution = student.execute(problem, parent.prompt_config, inference_config)
                result = evaluator.evaluate(solution, problem)

            batch_results.append(result)

        parent_acc = sum(r.success for r in batch_results) / len(batch_results)

        # (3) Reflect and mutate
        if use_reflection and reflection_engine:
            # Collect failures for reflection
            failed_cases = []
            for i, result in enumerate(batch_results):
                if not result.success:
                    # Get the solution that failed
                    if self_consistency_k > 1:
                        # For self-consistency, get first solution
                        solution = student.execute(batch[i], parent.prompt_config, inference_config)
                    else:
                        # Single solution case - need to re-execute to get solution object
                        solution = student.execute(batch[i], parent.prompt_config, inference_config)
                    failed_cases.append((solution, batch[i]))

            # Use ReflectionEngine to generate mutation
            child_config = reflection_engine.reflect_and_mutate(
                parent_config=parent.prompt_config,
                failed_cases=failed_cases,
                inference_config=inference_config
            )
        else:
            # Fallback: random hint mutation
            variations = [
                "Focus on identifying the key numbers and operations.",
                "Break down the problem into smaller steps.",
                "Check your arithmetic carefully.",
                "Make sure to show all intermediate calculations.",
                "Verify your answer makes sense in the context.",
            ]
            hint = random.choice(variations)
            child_config = PromptConfig(
                system=parent.prompt_config.system,
                cot_prompt=parent.prompt_config.cot_prompt + f" {hint}"
            )

        child = Candidate(prompt_config=child_config)

        # (4) Quick check
        batch_results_child = []
        for problem in batch:
            if self_consistency_k > 1:
                solutions = student.execute_with_self_consistency(
                    problem,
                    child.prompt_config,
                    inference_config,
                    self_consistency_k,
                )
                # Use rigorous self-consistency: vote on extracted answers
                result = evaluator.evaluate_with_self_consistency(solutions, problem)
            else:
                solution = student.execute(problem, child.prompt_config, inference_config)
                result = evaluator.evaluate(solution, problem)

            batch_results_child.append(result)

        child_acc = sum(r.success for r in batch_results_child) / len(
            batch_results_child
        )

        if child_acc < parent_acc:
            if verbose:
                print(
                    f"[GEPA-MI] Iter {iteration}: Child rejected on minibatch "
                    f"(parent={parent_acc:.2f}, child={child_acc:.2f})"
                )
            continue

        # (5) SELECTIVE VALIDATION
        if verbose:
            print(
                f"[GEPA-MI] Iter {iteration}: Selective validation "
                f"(delta={delta:.4f}, max_probes={max_probes})..."
            )

        accept, probes_used = selective_validate(
            student=student,
            evaluator=evaluator,
            child=child,
            val_set=val_set,
            best_scores=BestScores,
            scores_dict=Scores,
            delta=delta,
            max_probes=max_probes,
            inference_config=inference_config,
            self_consistency_k=self_consistency_k,
            minibatch_acc=child_acc,
            verbose=verbose,
        )

        total_val_calls += probes_used * self_consistency_k
        total_probes += probes_used

        if accept:
            P.append(child)
            accepted_count += 1
            if verbose:
                print(
                    f"[GEPA-MI] Iter {iteration}: ✓ Accepted after {probes_used} probes "
                    f"(wins: {len(child.wins)})"
                )
        else:
            if verbose:
                print(f"[GEPA-MI] Iter {iteration}: ✗ Rejected after {probes_used} probes")

    elapsed = time.time() - start

    # Calculate speedup
    baseline_cost = accepted_count * len(val_set) * self_consistency_k
    speedup = baseline_cost / max(total_val_calls, 1)
    avg_probes = total_probes / max(accepted_count, 1)

    if verbose:
        print("\n" + "=" * 80)
        print("GEPA-MI RESULTS")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s")
        print(f"Total iterations: {iteration}")
        print(f"Accepted candidates: {accepted_count}")
        print(f"Final pool size: {len(P)}")
        print(f"Total validation calls: {total_val_calls}")
        print(f"Average probes per candidate: {avg_probes:.2f}")
        print(
            f"Baseline would need: {baseline_cost} calls "
            f"({accepted_count} candidates × {len(val_set)} × k={self_consistency_k})"
        )
        print(f"Speedup: {speedup:.2f}x")
        print("=" * 80 + "\n")

    return P, Scores, BestScores
