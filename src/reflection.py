"""
Reflection engine for prompt improvement via self-reflection.

This module provides a reusable reflection mechanism that can be used
by any optimization algorithm (GEPA, Self-Reflection, etc.) to improve
prompts by analyzing failed trajectories.
"""

import random
from typing import Dict, List, Tuple, Any

from .core import AgentTrajectory, StudentModel, TrajectoryFormatter


class ReflectionEngine:
    """
    Generates prompt mutations by reflecting on failed trajectories.

    Uses the same LLM (student model) as both the problem solver and the
    judge/reflector, providing a simple baseline for prompt optimization.
    """

    def __init__(self, student: StudentModel, formatter: TrajectoryFormatter):
        """
        Initialize reflection engine.

        Args:
            student: The student model to use for reflection
            formatter: Formatter to convert trajectories to natural language
        """
        self.student = student
        self.formatter = formatter

    def reflect_and_mutate(
        self,
        parent_config: Dict[str, str],
        failed_trajectories: List[Tuple[AgentTrajectory, Dict]],
        inference_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Reflect on failures and generate improved prompt.

        Args:
            parent_config: Current prompt configuration (system, cot_prompt, etc.)
            failed_trajectories: List of (trajectory, instance) pairs that failed
            inference_config: Sampling parameters for reflection call

        Returns:
            Mutated prompt configuration
        """
        if not failed_trajectories:
            # No failures - apply minor random variation
            return self._random_mutation(parent_config)

        # Format top 3 failures for reflection
        formatted_failures = []
        for traj, instance in failed_trajectories[:3]:
            formatted = self.formatter.format(traj, instance)
            formatted_failures.append(formatted)

        # Build reflection prompt
        reflection_prompt = self._build_reflection_prompt(
            parent_config, formatted_failures
        )

        # Call LLM for reflection (use same student model)
        try:
            reflection_result = self.student.execute(
                problem={"question": reflection_prompt},
                prompt_config={
                    "system": "You are an expert at improving prompts for math problem solving.",
                    "cot_prompt": ""
                },
                inference_config=inference_config
            )

            # Parse suggestions and apply mutation
            child_config = self._apply_mutation(
                parent_config, reflection_result.final_output
            )

            return child_config

        except Exception as e:
            # Graceful fallback on any error
            print(f"Warning: Reflection failed ({e}), using random mutation")
            return self._random_mutation(parent_config)

    def _build_reflection_prompt(
        self, parent_config: Dict[str, str], formatted_failures: List[str]
    ) -> str:
        """
        Build prompt for reflection LLM.

        Args:
            parent_config: Current prompt configuration
            formatted_failures: List of formatted failure descriptions

        Returns:
            Reflection prompt asking LLM to analyze and improve
        """
        prompt = "You are analyzing failed attempts at solving math problems. "
        prompt += "Your goal is to improve the prompts given to the problem solver.\n\n"

        prompt += "=== CURRENT SYSTEM PROMPT ===\n"
        prompt += f"{parent_config.get('system', 'None')}\n\n"

        prompt += "=== CURRENT CHAIN-OF-THOUGHT PROMPT ===\n"
        prompt += f"{parent_config.get('cot_prompt', 'None')}\n\n"

        prompt += "=== FAILED ATTEMPTS ===\n"
        for i, failure in enumerate(formatted_failures, 1):
            prompt += f"\n--- Failure {i} ---\n{failure}\n"

        prompt += "\n=== YOUR TASK ===\n"
        prompt += "Analyze these failures and suggest ONE specific, actionable improvement "
        prompt += "to either the system prompt or the chain-of-thought prompt. "
        prompt += "Focus on addressing the most common or critical error pattern.\n\n"
        prompt += "Format your response EXACTLY as follows:\n"
        prompt += "ANALYSIS: [Brief explanation of what went wrong across these failures]\n"
        prompt += "IMPROVEMENT: [Specific text to add or modify in the prompt]\n"
        prompt += "TARGET: [Either 'system' or 'cot' to indicate which prompt to modify]\n"

        return prompt

    def _apply_mutation(
        self, parent_config: Dict[str, str], reflection_output: str
    ) -> Dict[str, str]:
        """
        Parse reflection output and apply mutation.

        Args:
            parent_config: Current prompt configuration
            reflection_output: LLM's reflection response

        Returns:
            Mutated prompt configuration
        """
        child_config = parent_config.copy()

        try:
            # Extract structured sections
            lines = reflection_output.strip().split('\n')
            analysis = ""
            improvement = ""
            target = "cot"  # Default to CoT prompt

            for line in lines:
                line = line.strip()
                if line.startswith("ANALYSIS:"):
                    analysis = line.replace("ANALYSIS:", "").strip()
                elif line.startswith("IMPROVEMENT:"):
                    improvement = line.replace("IMPROVEMENT:", "").strip()
                elif line.startswith("TARGET:"):
                    target_str = line.replace("TARGET:", "").strip().lower()
                    if "system" in target_str:
                        target = "system"
                    else:
                        target = "cot"

            # Apply improvement if we successfully extracted it
            if improvement:
                if target == "system":
                    child_config["system"] = parent_config.get("system", "") + " " + improvement
                else:
                    child_config["cot_prompt"] = parent_config.get("cot_prompt", "") + " " + improvement

                return child_config
            else:
                # Parsing failed - use fallback
                return self._random_mutation(parent_config)

        except Exception:
            # Any parsing error - use fallback
            return self._random_mutation(parent_config)

    def _random_mutation(self, parent_config: Dict[str, str]) -> Dict[str, str]:
        """
        Apply random mutation as fallback.

        Args:
            parent_config: Current prompt configuration

        Returns:
            Randomly mutated configuration
        """
        child_config = parent_config.copy()

        variations = [
            "Focus on identifying the key numbers and operations.",
            "Break down the problem into smaller, manageable steps.",
            "Double-check your arithmetic at each step.",
            "Make sure to show all intermediate calculations clearly.",
            "Verify your final answer makes sense in the context of the problem.",
            "Pay careful attention to units and what the question is asking for.",
            "Consider whether you need to add, subtract, multiply, or divide at each step.",
            "Read the problem carefully to identify all relevant information.",
        ]

        hint = random.choice(variations)
        child_config["cot_prompt"] = parent_config.get("cot_prompt", "") + f" {hint}"

        return child_config
