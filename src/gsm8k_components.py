"""GSM8K-specific implementations of core interfaces.

This module provides:
- GSM8KFormatter: Format math problem trajectories
- GSM8KEvaluator: Evaluate math solutions
- OpenAIStudentModel: Student model using OpenAI API
"""

import os
import re
import time
from typing import Dict, List, Any

from .core import (
    AgentTrajectory,
    EvaluationResult,
    TrajectoryFormatter,
    StudentModel,
    TaskEvaluator,
)


class GSM8KFormatter(TrajectoryFormatter):
    """Format math problem trajectories for reflection."""

    def format(self, trajectory: AgentTrajectory, instance: Dict) -> str:
        """Format for reflection."""
        formatted = f"Problem: {instance['question']}\n\n"

        if trajectory.reasoning_steps:
            formatted += "Reasoning:\n"
            for i, step in enumerate(trajectory.reasoning_steps, 1):
                formatted += f"{i}. {step}\n"
            formatted += "\n"

        formatted += f"Final Answer: {trajectory.final_output}\n"
        formatted += f"Expected: {instance['answer']}\n"

        return formatted


class GSM8KEvaluator(TaskEvaluator):
    """Evaluate math problem solutions."""

    def evaluate(
        self, trajectory: AgentTrajectory, ground_truth: Dict
    ) -> EvaluationResult:
        """Evaluate trajectory against ground truth."""
        start = time.time()

        try:
            predicted = self._extract_answer(trajectory.final_output)
            expected = self._extract_answer(ground_truth["answer"])

            success = self._normalize(predicted) == self._normalize(expected)

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

    def _extract_answer(self, text: str) -> str:
        """Extract numeric answer after ####."""
        if "####" in text:
            return text.split("####")[-1].strip()

        # Fallback: extract last number
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else text.strip()

    def _normalize(self, answer: str) -> str:
        """Normalize for comparison."""
        cleaned = re.sub(r"[^\d\.\-]", "", answer)
        try:
            # Handle floats and integers
            num = float(cleaned)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except:
            return cleaned


class OpenAIStudentModel(StudentModel):
    """Student model using OpenAI API."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        """Initialize OpenAI student model.

        Args:
            model_name: OpenAI model to use (default: gpt-4o-mini)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)

    def execute(
        self,
        problem: Dict,
        prompt_config: Dict[str, str],
        inference_config: Dict[str, Any],
    ) -> AgentTrajectory:
        """Execute single trajectory."""

        # Build messages from prompt config
        messages = []
        if "system" in prompt_config:
            messages.append({"role": "system", "content": prompt_config["system"]})

        # Add problem with CoT prompt
        user_msg = problem["question"]
        if "cot_prompt" in prompt_config:
            user_msg = f"{prompt_config['cot_prompt']}\n\n{user_msg}"

        messages.append({"role": "user", "content": user_msg})

        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=inference_config.get("temperature", 0.7),
            max_tokens=inference_config.get("max_new_tokens", 256),
            top_p=inference_config.get("top_p", 1.0),
        )

        output = response.choices[0].message.content

        # Parse reasoning steps (basic: split by newlines)
        steps = [line.strip() for line in output.split("\n") if line.strip()]

        return AgentTrajectory(
            reasoning_steps=steps,
            tool_calls=[],
            observations=[],
            final_output=output,
            metadata={"model": self.model_name},
        )

    def execute_with_self_consistency(
        self,
        problem: Dict,
        prompt_config: Dict[str, str],
        inference_config: Dict[str, Any],
        k: int,
    ) -> List[AgentTrajectory]:
        """Execute k times for self-consistency."""
        trajectories = []
        for _ in range(k):
            traj = self.execute(problem, prompt_config, inference_config)
            trajectories.append(traj)
        return trajectories
