"""
Reflection engine for prompt improvement via self-reflection.

This module provides a reusable reflection mechanism that can be used
by any optimization algorithm (GEPA, Self-Reflection, etc.) to improve
prompts by analyzing failed trajectories.

Uses strict Pydantic types and structured outputs - no string parsing.
"""

from typing import List, Tuple

from .core import StudentModel
from .types import (
    ReflectionOutput,
    MathSolution,
    GSM8KProblem,
    PromptConfig,
    InferenceConfig,
)


class ReflectionEngine:
    """
    Generates prompt mutations by reflecting on failed trajectories.

    Uses the same LLM (student model) as both the problem solver and the
    judge/reflector, providing a simple baseline for prompt optimization.

    Now uses strict Pydantic types with instructor for structured outputs.
    """

    def __init__(self, student: StudentModel):
        """
        Initialize reflection engine.

        Args:
            student: The student model to use for reflection (must support instructor)
        """
        self.student = student
        self._setup_instructor_client()

    def _setup_instructor_client(self):
        """Setup instructor-wrapped client from the student model."""
        import instructor

        # Check if student model has instructor-compatible client
        if hasattr(self.student, 'client'):
            # Wrap existing client with instructor if not already wrapped
            if not hasattr(self.student.client, 'chat'):
                # Need to determine provider type and wrap accordingly
                # This is a simplified approach - in production, student model
                # should already provide instructor-wrapped client
                self.instructor_client = instructor.patch(self.student.client)
            else:
                # Assume it's already compatible with instructor
                self.instructor_client = instructor.patch(self.student.client)
        else:
            raise ValueError(
                "Student model must have a 'client' attribute compatible with instructor"
            )

    def reflect_and_mutate(
        self,
        parent_config: PromptConfig,
        failed_cases: List[Tuple[MathSolution, GSM8KProblem]],
        inference_config: InferenceConfig,
    ) -> PromptConfig:
        """
        Reflect on failures and generate improved prompt.

        Uses instructor to get structured ReflectionOutput - no string parsing.

        Args:
            parent_config: Current prompt configuration
            failed_cases: List of (solution, problem) pairs that failed
            inference_config: Sampling parameters for reflection call

        Returns:
            Mutated prompt configuration (PromptConfig)

        Raises:
            ValidationError: If LLM doesn't produce valid ReflectionOutput
            ValueError: If no failed cases provided
        """
        if not failed_cases:
            raise ValueError("reflect_and_mutate requires at least one failed case")

        # Build reflection prompt with top 3 failures
        reflection_prompt = self._build_reflection_prompt(
            parent_config, failed_cases[:3]
        )

        # Build messages for reflection call
        messages = [
            {
                "role": "system",
                "content": "You are an expert at improving prompts for math problem solving. "
                "Analyze failed attempts and suggest specific improvements.",
            },
            {"role": "user", "content": reflection_prompt},
        ]

        # Call LLM with instructor for structured output
        reflection = self.instructor_client.chat.completions.create(
            model=self.student.model_name,
            messages=messages,
            response_model=ReflectionOutput,  # Automatic Pydantic validation
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
        )

        # Apply improvement to create new config
        new_config_dict = parent_config.model_dump()
        if reflection.target == "system":
            new_config_dict["system"] += f" {reflection.improvement}"
        else:  # target == "cot"
            new_config_dict["cot_prompt"] += f" {reflection.improvement}"

        # Return new PromptConfig
        return PromptConfig(**new_config_dict)

    def _build_reflection_prompt(
        self, parent_config: PromptConfig, failed_cases: List[Tuple[MathSolution, GSM8KProblem]]
    ) -> str:
        """
        Build prompt for reflection LLM.

        Args:
            parent_config: Current prompt configuration
            failed_cases: List of (solution, problem) tuples that failed

        Returns:
            Reflection prompt asking LLM to analyze and improve
        """
        prompt = "You are analyzing failed attempts at solving math problems. "
        prompt += "Your goal is to improve the prompts given to the problem solver.\n\n"

        prompt += "=== CURRENT SYSTEM PROMPT ===\n"
        prompt += f"{parent_config.system}\n\n"

        prompt += "=== CURRENT CHAIN-OF-THOUGHT PROMPT ===\n"
        prompt += f"{parent_config.cot_prompt}\n\n"

        prompt += "=== FAILED ATTEMPTS ===\n"
        for i, (solution, problem) in enumerate(failed_cases, 1):
            prompt += f"\n--- Failure {i} ---\n"
            prompt += f"Problem: {problem.question}\n\n"

            prompt += "Reasoning:\n"
            for j, step in enumerate(solution.reasoning_steps, 1):
                prompt += f"{j}. {step}\n"
            prompt += "\n"

            prompt += f"Predicted Answer: {solution.final_answer}\n"
            prompt += f"Expected Answer: {problem.numeric_answer}\n"

        prompt += "\n=== YOUR TASK ===\n"
        prompt += "Analyze these failures and suggest ONE specific, actionable improvement "
        prompt += "to either the system prompt or the chain-of-thought prompt. "
        prompt += "Focus on addressing the most common or critical error pattern.\n\n"

        prompt += "You must provide your response in a structured format with:\n"
        prompt += "- analysis: Brief explanation of what went wrong\n"
        prompt += "- improvement: Specific text to add to the prompt\n"
        prompt += "- target: Either 'system' or 'cot' to indicate which prompt to modify\n"

        return prompt

