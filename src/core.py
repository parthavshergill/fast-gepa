"""Core abstractions for GEPA/GEPA-MI implementation.

This module defines the base interfaces for:
- AgentTrajectory: Raw execution traces
- EvaluationResult: Evaluation outcomes
- TrajectoryFormatter: Converting traces to natural language
- StudentModel: Executing inference on problems
- TaskEvaluator: Evaluating solutions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentTrajectory:
    """Raw trajectory from agent execution."""
    reasoning_steps: List[str]
    tool_calls: List[Dict[str, Any]]
    observations: List[str]
    final_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating trajectory on instance."""
    success: bool  # Binary: correct/incorrect
    score: float   # Continuous score if available
    execution_time: float
    trajectory: AgentTrajectory
    error_message: Optional[str] = None


class TrajectoryFormatter(ABC):
    """Convert agent trajectory to string for reflection."""

    @abstractmethod
    def format(self, trajectory: AgentTrajectory, instance: Dict) -> str:
        """
        Format trajectory into natural language description.

        Args:
            trajectory: Raw agent execution trace
            instance: The problem instance

        Returns:
            Formatted string for reflection LLM
        """
        pass


class StudentModel(ABC):
    """Abstract student model interface."""

    @abstractmethod
    def execute(self,
                problem: Dict,
                prompt_config: Dict[str, str],
                inference_config: Dict[str, Any]) -> AgentTrajectory:
        """
        Execute student model on problem.

        Args:
            problem: Problem instance (e.g., {'question': ..., 'answer': ...})
            prompt_config: Module texts (e.g., {'system': ..., 'cot': ...})
            inference_config: Sampling params (temp, top_p, max_tokens, etc.)

        Returns:
            Complete agent trajectory
        """
        pass

    @abstractmethod
    def execute_with_self_consistency(self,
                                      problem: Dict,
                                      prompt_config: Dict[str, str],
                                      inference_config: Dict[str, Any],
                                      k: int) -> List[AgentTrajectory]:
        """
        Execute k times and return all trajectories.

        For self-consistency: sample k times, return majority vote.

        Args:
            problem: Problem instance
            prompt_config: Module texts
            inference_config: Sampling parameters
            k: Number of samples

        Returns:
            List of k trajectories
        """
        pass


class TaskEvaluator(ABC):
    """Task-specific evaluation logic."""

    @abstractmethod
    def evaluate(self,
                 trajectory: AgentTrajectory,
                 ground_truth: Dict) -> EvaluationResult:
        """
        Evaluate trajectory against ground truth.

        Args:
            trajectory: Agent's output
            ground_truth: Correct answer/solution

        Returns:
            Evaluation result with success flag
        """
        pass
