"""GSM8K-specific implementations of core interfaces.

This module provides:
- GSM8KFormatter: Format math problem trajectories
- GSM8KEvaluator: Evaluate math solutions with strict types (NO string parsing)
- OpenAIStudentModel: Student model using OpenAI API with instructor
- GeminiStudentModel: Student model using Google Gemini API with JSON schema validation

All implementations follow CLAUDE_CONTRACTS.md - strict Pydantic types, fail fast.
"""

import os
import json
import time
from typing import List
from collections import Counter

import instructor
from openai import OpenAI

from .core import (
    AgentTrajectory,
    EvaluationResult,
    TrajectoryFormatter,
    StudentModel,
    TaskEvaluator,
)
from .types import (
    GSM8KProblem,
    MathSolution,
    InferenceConfig,
    PromptConfig,
)


class GSM8KFormatter(TrajectoryFormatter):
    """Format math problem trajectories for reflection.

    Note: This formatter bridges legacy AgentTrajectory with new MathSolution types.
    For new code using MathSolution directly, create format_solution() method.
    """

    def format(self, trajectory: AgentTrajectory, instance: GSM8KProblem) -> str:
        """Format trajectory for reflection (legacy interface).

        Args:
            trajectory: Legacy AgentTrajectory object
            instance: Typed GSM8K problem

        Returns:
            Formatted string for reflection
        """
        formatted = f"Problem: {instance.question}\n\n"

        if trajectory.reasoning_steps:
            formatted += "Reasoning:\n"
            for i, step in enumerate(trajectory.reasoning_steps, 1):
                formatted += f"{i}. {step}\n"
            formatted += "\n"

        formatted += f"Final Answer: {trajectory.final_output}\n"
        formatted += f"Expected: {instance.answer}\n"

        return formatted

    def format_solution(self, solution: MathSolution, problem: GSM8KProblem) -> str:
        """Format MathSolution for reflection (new typed interface).

        Args:
            solution: Structured math solution with reasoning and answer
            problem: Typed GSM8K problem

        Returns:
            Formatted string for reflection
        """
        formatted = f"Problem: {problem.question}\n\n"

        formatted += "Reasoning:\n"
        for i, step in enumerate(solution.reasoning_steps, 1):
            formatted += f"{i}. {step}\n"
        formatted += "\n"

        formatted += f"Final Answer: {solution.final_answer}\n"
        formatted += f"Expected: {problem.numeric_answer}\n"

        return formatted


class GSM8KEvaluator(TaskEvaluator):
    """Evaluate math problem solutions with strict types.

    IMPORTANT: This evaluator works with MathSolution (structured output).
    NO string parsing, NO normalization heuristics - direct comparison only.
    """

    def evaluate(
        self, trajectory: AgentTrajectory, ground_truth: GSM8KProblem
    ) -> EvaluationResult:
        """Evaluate legacy trajectory against ground truth.

        NOTE: This is for backward compatibility. New code should use evaluate_solution().
        """
        start = time.time()

        try:
            # For legacy trajectories, we still need to parse :(
            # This should be removed once all code uses MathSolution
            import re
            if "####" in trajectory.final_output:
                predicted_str = trajectory.final_output.split("####")[-1].strip()
            else:
                numbers = re.findall(r"-?\d+(?:\.\d+)?", trajectory.final_output)
                predicted_str = numbers[-1] if numbers else trajectory.final_output.strip()

            predicted = float(predicted_str.replace(",", "").replace("$", "").strip())
            success = predicted == ground_truth.numeric_answer

            return EvaluationResult(
                success=success,
                score=1.0 if success else 0.0,
                execution_time=time.time() - start,
                trajectory=trajectory,
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                score=0.0,
                execution_time=time.time() - start,
                trajectory=trajectory,
                error_message=str(e),
            )

    def evaluate_solution(
        self, solution: MathSolution, problem: GSM8KProblem
    ) -> EvaluationResult:
        """Evaluate structured MathSolution against ground truth.

        This is the NEW, CORRECT way - strict type comparison, no parsing.

        Args:
            solution: Structured solution from LLM (via instructor)
            problem: Ground truth problem

        Returns:
            EvaluationResult with success flag
        """
        start = time.time()

        # Simple, direct comparison - no normalization needed
        success = solution.final_answer == problem.numeric_answer

        # Create minimal trajectory for compatibility with EvaluationResult
        trajectory = AgentTrajectory(
            reasoning_steps=solution.reasoning_steps,
            tool_calls=[],
            observations=[],
            final_output=str(solution.final_answer),
            metadata={"confidence": solution.confidence}
        )

        return EvaluationResult(
            success=success,
            score=1.0 if success else 0.0,
            execution_time=time.time() - start,
            trajectory=trajectory,
        )

    def evaluate_with_self_consistency(
        self, trajectories: List[AgentTrajectory], ground_truth: GSM8KProblem
    ) -> EvaluationResult:
        """Evaluate with self-consistency: vote on extracted answers (legacy interface).

        NOTE: This is for backward compatibility. New code should use
        evaluate_solutions_with_self_consistency().
        """
        if not trajectories:
            raise ValueError("Cannot evaluate empty trajectory list")

        if len(trajectories) == 1:
            return self.evaluate(trajectories[0], ground_truth)

        # Extract and vote (still requires parsing for legacy trajectories)
        import re
        answers = []
        for t in trajectories:
            if "####" in t.final_output:
                ans_str = t.final_output.split("####")[-1].strip()
            else:
                numbers = re.findall(r"-?\d+(?:\.\d+)?", t.final_output)
                ans_str = numbers[-1] if numbers else t.final_output.strip()
            try:
                answers.append(float(ans_str.replace(",", "").replace("$", "").strip()))
            except:
                answers.append(0.0)

        # Count occurrences
        answer_counts = Counter(answers)

        # Pick majority answer
        majority_answer, count = answer_counts.most_common(1)[0]

        # Find first trajectory that produced the majority answer
        majority_idx = answers.index(majority_answer)
        majority_traj = trajectories[majority_idx]

        # Evaluate the majority trajectory
        return self.evaluate(majority_traj, ground_truth)

    def evaluate_solutions_with_self_consistency(
        self, solutions: List[MathSolution], problem: GSM8KProblem
    ) -> EvaluationResult:
        """Evaluate with self-consistency using structured solutions.

        This is the NEW, CORRECT way - vote on typed final_answer values.

        Args:
            solutions: List of k structured solutions from self-consistency sampling
            problem: Ground truth problem

        Returns:
            EvaluationResult for the majority answer
        """
        if not solutions:
            raise ValueError("Cannot evaluate empty solutions list")

        if len(solutions) == 1:
            return self.evaluate_solution(solutions[0], problem)

        # Vote on final_answer values (no parsing needed - already floats!)
        answers = [s.final_answer for s in solutions]

        # Count occurrences
        answer_counts = Counter(answers)

        # Pick majority answer
        majority_answer, count = answer_counts.most_common(1)[0]

        # Find first solution that produced the majority answer
        majority_idx = answers.index(majority_answer)
        majority_solution = solutions[majority_idx]

        # Evaluate the majority solution
        return self.evaluate_solution(majority_solution, problem)


class OpenAIStudentModel(StudentModel):
    """Student model using OpenAI API with instructor for structured outputs.

    This implementation follows CLAUDE_CONTRACTS.md Pattern 2:
    - Uses instructor.from_openai() for automatic Pydantic validation
    - Returns MathSolution directly, no string parsing
    - Supports base_url for custom OpenAI-compatible endpoints
    - Fail fast on validation errors
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None
    ):
        """Initialize OpenAI student model.

        Args:
            model_name: OpenAI model to use (default: gpt-4o-mini)
            api_key: OpenAI API key (reads from OPENAI_API_KEY env var if None)
            base_url: Custom endpoint (reads from OPENAI_BASE_URL env var if None)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize instructor-wrapped client
        raw_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.client = instructor.from_openai(raw_client)

    def execute(
        self,
        problem: GSM8KProblem,
        prompt_config: PromptConfig,
        inference_config: InferenceConfig
    ) -> MathSolution:
        """Execute single trajectory - returns structured MathSolution.

        Args:
            problem: Typed GSM8K problem
            prompt_config: Typed prompt configuration
            inference_config: Typed inference parameters

        Returns:
            MathSolution validated by Pydantic via instructor
        """
        # Build messages from typed config
        messages = [
            {"role": "system", "content": prompt_config.system},
            {"role": "user", "content": f"{prompt_config.cot_prompt}\n\n{problem.question}"}
        ]

        # Call with instructor for structured output
        solution = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=MathSolution,  # Automatic Pydantic validation
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
            top_p=inference_config.top_p
        )

        return solution  # Already validated MathSolution

    def execute_with_self_consistency(
        self,
        problem: GSM8KProblem,
        prompt_config: PromptConfig,
        inference_config: InferenceConfig,
        k: int
    ) -> List[MathSolution]:
        """Execute k times for self-consistency.

        Args:
            problem: Typed GSM8K problem
            prompt_config: Typed prompt configuration
            inference_config: Typed inference parameters
            k: Number of samples

        Returns:
            List of k validated MathSolution objects
        """
        return [self.execute(problem, prompt_config, inference_config) for _ in range(k)]


class GeminiStudentModel(StudentModel):
    """Student model using Google Gemini API with JSON schema validation.

    NOTE: Gemini does not have native instructor support, so we use JSON schema
    in the prompt and validate with Pydantic. This is less robust than OpenAI's
    structured outputs but follows the same fail-fast principle.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str | None = None,
        base_url: str | None = None
    ):
        """Initialize Gemini student model.

        Args:
            model_name: Gemini model to use (default: gemini-1.5-flash)
                       Options: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp
            api_key: Google API key (reads from GEMINI_API_KEY env var if None)
            base_url: Custom endpoint (reads from GEMINI_ENDPOINT env var if None)
                     Note: base_url is not typically used with Gemini but included for consistency
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_ENDPOINT")

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Gemini client
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def execute(
        self,
        problem: GSM8KProblem,
        prompt_config: PromptConfig,
        inference_config: InferenceConfig
    ) -> MathSolution:
        """Execute single trajectory - returns structured MathSolution.

        Args:
            problem: Typed GSM8K problem
            prompt_config: Typed prompt configuration
            inference_config: Typed inference parameters

        Returns:
            MathSolution validated by Pydantic after JSON parsing
        """
        # Build prompt with JSON schema instructions
        prompt = f"""{prompt_config.system}

{prompt_config.cot_prompt}

Problem: {problem.question}

You MUST respond with valid JSON matching this exact schema:
{{
    "reasoning_steps": ["step 1", "step 2", ...],
    "final_answer": <numeric value as a number, not a string>,
    "confidence": <float between 0.0 and 1.0>
}}

Do not include any text outside the JSON object. The final_answer MUST be a numeric value (not a string).
"""

        # Configure generation parameters
        generation_config = {
            "temperature": inference_config.temperature,
            "top_p": inference_config.top_p,
            "max_output_tokens": inference_config.max_tokens,
        }

        # Call API
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        output = response.text

        # Parse JSON and validate with Pydantic
        try:
            # Clean output (remove markdown code blocks if present)
            cleaned = output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            data = json.loads(cleaned)

            # Validate with Pydantic (fail fast if invalid)
            solution = MathSolution(**data)
            return solution

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Gemini output is not valid JSON. Output: {output[:200]}\nError: {e}"
            )
        except Exception as e:
            raise ValueError(
                f"Gemini output does not match MathSolution schema. Output: {output[:200]}\nError: {e}"
            )

    def execute_with_self_consistency(
        self,
        problem: GSM8KProblem,
        prompt_config: PromptConfig,
        inference_config: InferenceConfig,
        k: int
    ) -> List[MathSolution]:
        """Execute k times for self-consistency.

        Args:
            problem: Typed GSM8K problem
            prompt_config: Typed prompt configuration
            inference_config: Typed inference parameters
            k: Number of samples

        Returns:
            List of k validated MathSolution objects
        """
        return [self.execute(problem, prompt_config, inference_config) for _ in range(k)]
