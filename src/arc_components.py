"""ARC-AGI-2 specific implementations of core interfaces.

This module provides:
- ARCFormatter: Format ARC problem solutions for reflection
- ARCEvaluator: Evaluate ARC solutions with strict types (NO string parsing)

All implementations follow CLAUDE_CONTRACTS.md - strict Pydantic types, fail fast.
"""

import time
from typing import List
from collections import Counter

from .core import (
    AgentTrajectory,
    EvaluationResult,
    TrajectoryFormatter,
    TaskEvaluator,
)
from .types import (
    ARCProblem,
    ARCSolution,
    ARCExample,
    Grid,
)


class ARCFormatter(TrajectoryFormatter):
    """Format ARC problem solutions for reflection.

    This formatter works with ARCSolution (structured output) for the new typed interface.
    """

    def format(self, trajectory: AgentTrajectory, instance: ARCProblem) -> str:
        """Format trajectory for reflection (legacy interface).

        NOTE: This is for backward compatibility with AgentTrajectory.
        New code should use format_solution() with ARCSolution.

        Args:
            trajectory: Legacy AgentTrajectory object
            instance: Typed ARC problem

        Returns:
            Formatted string for reflection
        """
        formatted = f"Task ID: {instance.task_id}\n\n"

        formatted += "Training Examples:\n"
        for i, train_ex in enumerate(instance.train, 1):
            formatted += f"  Example {i}:\n"
            formatted += f"    Input shape: {train_ex.input_shape}\n"
            formatted += f"    Output shape: {train_ex.output_shape}\n"

        formatted += "\nTest Cases:\n"
        for i, test_ex in enumerate(instance.test, 1):
            formatted += f"  Test {i}:\n"
            formatted += f"    Input shape: {test_ex.input_shape}\n"
            formatted += f"    Expected output shape: {test_ex.output_shape}\n"

        formatted += f"\nReasoning:\n{trajectory.final_output}\n"

        return formatted

    def format_solution(self, solution: ARCSolution, problem: ARCProblem) -> str:
        """Format ARCSolution for reflection (new typed interface).

        Args:
            solution: Structured ARC solution with reasoning and grids
            problem: Typed ARC problem

        Returns:
            Formatted string for reflection
        """
        formatted = f"Task ID: {problem.task_id}\n\n"

        formatted += "Training Examples:\n"
        for i, train_ex in enumerate(problem.train, 1):
            formatted += f"  Example {i}:\n"
            formatted += f"    Input:  {train_ex.input_shape} -> {self._format_grid_preview(train_ex.input)}\n"
            formatted += f"    Output: {train_ex.output_shape} -> {self._format_grid_preview(train_ex.output)}\n"

        formatted += "\nReasoning:\n"
        formatted += f"{solution.reasoning}\n\n"

        formatted += "Rule Description:\n"
        formatted += f"{solution.rule_description}\n\n"

        formatted += "Test Predictions:\n"
        for i, (test_ex, pred_grid) in enumerate(zip(problem.test, solution.output_grids), 1):
            formatted += f"  Test {i}:\n"
            formatted += f"    Input shape: {test_ex.input_shape}\n"
            formatted += f"    Expected: {test_ex.output_shape} -> {self._format_grid_preview(test_ex.output)}\n"
            formatted += f"    Predicted: {self._get_grid_shape(pred_grid)} -> {self._format_grid_preview(pred_grid)}\n"
            formatted += f"    Match: {'✓' if self._grids_equal(pred_grid, test_ex.output) else '✗'}\n"

        return formatted

    def _format_grid_preview(self, grid: Grid, max_width: int = 10) -> str:
        """Format grid preview for display (first row only)."""
        if not grid or not grid[0]:
            return "[]"
        first_row = grid[0][:max_width]
        preview = str(first_row)
        if len(grid[0]) > max_width or len(grid) > 1:
            preview += "..."
        return preview

    def _get_grid_shape(self, grid: Grid) -> tuple:
        """Get shape of grid."""
        if not grid:
            return (0, 0)
        return (len(grid), len(grid[0]) if grid[0] else 0)

    def _grids_equal(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if two grids are exactly equal."""
        if len(grid1) != len(grid2):
            return False
        for row1, row2 in zip(grid1, grid2):
            if row1 != row2:
                return False
        return True


class ARCEvaluator(TaskEvaluator):
    """Evaluate ARC solutions with strict types.

    IMPORTANT: This evaluator works with ARCSolution (structured output).
    NO string parsing, NO normalization heuristics - direct grid comparison only.

    ARC evaluation rules:
    - Each test case must have exactly correct grid
    - ALL test cases must be correct for success
    - Score = fraction of test cases correct
    """

    def evaluate(
        self, trajectory: AgentTrajectory, ground_truth: ARCProblem
    ) -> EvaluationResult:
        """Evaluate legacy trajectory against ground truth.

        NOTE: This is for backward compatibility. New code should use evaluate_solution().
        """
        start = time.time()

        # For legacy trajectories, we can't properly evaluate without structured output
        # This is a placeholder that always fails
        return EvaluationResult(
            success=False,
            score=0.0,
            execution_time=time.time() - start,
            trajectory=trajectory,
            error_message="Legacy trajectory evaluation not supported for ARC. Use evaluate_solution() with ARCSolution."
        )

    def evaluate_solution(
        self, solution: ARCSolution, problem: ARCProblem
    ) -> EvaluationResult:
        """Evaluate structured ARCSolution against ground truth.

        This is the CORRECT way - strict type comparison, no parsing.

        Args:
            solution: Structured solution from LLM (via instructor)
            problem: Ground truth problem with test cases

        Returns:
            EvaluationResult with success flag (True only if ALL test cases correct)
        """
        start = time.time()

        # Check correct number of output grids
        num_test_cases = len(problem.test)
        num_predictions = len(solution.output_grids)

        if num_predictions != num_test_cases:
            # Create minimal trajectory for compatibility
            trajectory = AgentTrajectory(
                reasoning_steps=[solution.reasoning, solution.rule_description],
                tool_calls=[],
                observations=[],
                final_output=f"Predicted {num_predictions} grids, expected {num_test_cases}",
                metadata={"confidence": solution.confidence}
            )

            return EvaluationResult(
                success=False,
                score=0.0,
                execution_time=time.time() - start,
                trajectory=trajectory,
                error_message=f"Wrong number of outputs: predicted {num_predictions}, expected {num_test_cases}"
            )

        # Evaluate each test case
        correct_count = 0
        for i, (predicted_grid, test_case) in enumerate(zip(solution.output_grids, problem.test)):
            if self._grids_equal(predicted_grid, test_case.output):
                correct_count += 1

        # Success only if ALL test cases correct
        success = correct_count == num_test_cases
        score = correct_count / num_test_cases if num_test_cases > 0 else 0.0

        # Create minimal trajectory for compatibility with EvaluationResult
        trajectory = AgentTrajectory(
            reasoning_steps=[solution.reasoning, solution.rule_description],
            tool_calls=[],
            observations=[],
            final_output=f"Correct: {correct_count}/{num_test_cases}",
            metadata={"confidence": solution.confidence}
        )

        return EvaluationResult(
            success=success,
            score=score,
            execution_time=time.time() - start,
            trajectory=trajectory,
        )

    def evaluate_with_self_consistency(
        self, trajectories: List[AgentTrajectory], ground_truth: ARCProblem
    ) -> EvaluationResult:
        """Evaluate with self-consistency (legacy interface).

        NOTE: This is for backward compatibility. New code should use
        evaluate_solutions_with_self_consistency().
        """
        if not trajectories:
            raise ValueError("Cannot evaluate empty trajectory list")

        # Legacy interface not properly supported for ARC
        return self.evaluate(trajectories[0], ground_truth)

    def evaluate_solutions_with_self_consistency(
        self, solutions: List[ARCSolution], problem: ARCProblem
    ) -> EvaluationResult:
        """Evaluate with self-consistency using structured solutions.

        This is the CORRECT way for ARC - vote on grids per test case.

        Strategy:
        1. For each test case position, collect all predicted grids
        2. Vote on the most common grid (using grid equality)
        3. Create combined solution with majority grids
        4. Evaluate the combined solution

        Args:
            solutions: List of k structured solutions from self-consistency sampling
            problem: Ground truth problem

        Returns:
            EvaluationResult for the majority-voted solution
        """
        if not solutions:
            raise ValueError("Cannot evaluate empty solutions list")

        if len(solutions) == 1:
            return self.evaluate_solution(solutions[0], problem)

        num_test_cases = len(problem.test)

        # Check all solutions have correct number of grids
        valid_solutions = [
            s for s in solutions
            if len(s.output_grids) == num_test_cases
        ]

        if not valid_solutions:
            # No valid solutions - use first solution and let it fail
            return self.evaluate_solution(solutions[0], problem)

        # Vote on each test case separately
        majority_grids = []
        for test_idx in range(num_test_cases):
            # Collect all predictions for this test case
            predictions_for_test = [
                s.output_grids[test_idx] for s in valid_solutions
            ]

            # Vote by finding most common grid
            # Since grids are lists (unhashable), we need custom voting
            majority_grid = self._vote_on_grids(predictions_for_test)
            majority_grids.append(majority_grid)

        # Create combined solution with majority grids
        # Use reasoning from first solution that matches majority on first test case
        best_solution_idx = 0
        for i, sol in enumerate(valid_solutions):
            if self._grids_equal(sol.output_grids[0], majority_grids[0]):
                best_solution_idx = i
                break

        majority_solution = ARCSolution(
            reasoning=valid_solutions[best_solution_idx].reasoning,
            rule_description=valid_solutions[best_solution_idx].rule_description,
            output_grids=majority_grids,
            confidence=valid_solutions[best_solution_idx].confidence
        )

        # Evaluate the majority solution
        return self.evaluate_solution(majority_solution, problem)

    def _grids_equal(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if two grids are exactly equal.

        Args:
            grid1: First grid
            grid2: Second grid

        Returns:
            True if grids are identical
        """
        if len(grid1) != len(grid2):
            return False

        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
            if row1 != row2:  # List equality
                return False

        return True

    def _vote_on_grids(self, grids: List[Grid]) -> Grid:
        """Vote on most common grid from list of grids.

        Since grids (lists) are unhashable, we convert to tuples for counting.

        Args:
            grids: List of grid predictions

        Returns:
            Most common grid (or first if tie)
        """
        if not grids:
            raise ValueError("Cannot vote on empty grid list")

        # Convert grids to hashable tuples for counting
        grid_tuples = [
            tuple(tuple(row) for row in grid)
            for grid in grids
        ]

        # Count occurrences
        counter = Counter(grid_tuples)

        # Get most common
        most_common_tuple, _ = counter.most_common(1)[0]

        # Convert back to list format
        majority_grid = [list(row) for row in most_common_tuple]

        return majority_grid
