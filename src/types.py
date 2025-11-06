"""Strict type definitions using Pydantic.

This module defines all structured types used throughout the codebase.
All LLM outputs are validated against these schemas - no heuristic parsing.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class GSM8KProblem(BaseModel):
    """A GSM8K math word problem with ground truth answer."""

    question: str = Field(..., description="The math word problem")
    answer: str = Field(..., description="Ground truth answer (may contain #### delimiter)")

    @property
    def numeric_answer(self) -> float:
        """Extract the numeric answer from the answer string.

        GSM8K answers are formatted as: "explanation #### numeric_answer"
        """
        if "####" in self.answer:
            answer_str = self.answer.split("####")[-1].strip()
        else:
            answer_str = self.answer.strip()

        # Remove any remaining non-numeric characters (commas, dollar signs, etc.)
        answer_str = answer_str.replace(",", "").replace("$", "").strip()
        return float(answer_str)


class MathSolution(BaseModel):
    """Structured output from LLM when solving a math problem.

    This is the schema that LLMs must output (enforced via instructor).
    No string parsing or normalization needed.
    """

    reasoning_steps: List[str] = Field(
        ...,
        description="Step-by-step reasoning leading to the answer",
        min_length=1
    )
    final_answer: float = Field(
        ...,
        description="The numeric answer to the problem (no units, no formatting)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0-1), defaults to 1.0"
    )

    @field_validator('reasoning_steps')
    @classmethod
    def validate_non_empty_steps(cls, v):
        if not v or all(not step.strip() for step in v):
            raise ValueError("reasoning_steps must contain at least one non-empty step")
        return v


class ReflectionOutput(BaseModel):
    """Structured output from reflection LLM analyzing failures.

    This replaces the fragile string parsing in ReflectionEngine.
    """

    analysis: str = Field(
        ...,
        description="Brief explanation of what went wrong in the failed attempts"
    )
    improvement: str = Field(
        ...,
        description="Specific text to add or modify in the prompt"
    )
    target: Literal["system", "cot"] = Field(
        ...,
        description="Which prompt to modify: 'system' for system prompt, 'cot' for chain-of-thought prompt"
    )

    @field_validator('improvement')
    @classmethod
    def validate_improvement_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("improvement must be non-empty")
        return v.strip()


class InferenceConfig(BaseModel):
    """Configuration for LLM inference parameters."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096, alias="max_new_tokens")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True  # Allow both 'max_tokens' and 'max_new_tokens'


class PromptConfig(BaseModel):
    """Configuration for prompt modules."""

    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt setting the model's behavior"
    )
    cot_prompt: str = Field(
        default="Let's solve this step by step:",
        description="Chain-of-thought prompt for reasoning"
    )

    @field_validator('system', 'cot_prompt')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt fields must be non-empty")
        return v.strip()


# ARC-AGI-2 Types

Grid = List[List[int]]  # 2D grid of integers 0-9

class ARCExample(BaseModel):
    """Input-output demonstration pair for ARC task."""

    input: Grid = Field(..., description="Input grid (2D array of 0-9)")
    output: Grid = Field(..., description="Output grid (2D array of 0-9)")

    @field_validator('input', 'output')
    @classmethod
    def validate_grid(cls, grid: Grid) -> Grid:
        if not grid:
            raise ValueError("Grid cannot be empty")
        if not all(isinstance(row, list) for row in grid):
            raise ValueError("Grid must be list of lists")

        width = len(grid[0])
        if not all(len(row) == width for row in grid):
            raise ValueError("Grid must be rectangular")

        height = len(grid)
        if not (1 <= height <= 30 and 1 <= width <= 30):
            raise ValueError(f"Grid dimensions must be 1-30, got {height}x{width}")

        for row in grid:
            for val in row:
                if not isinstance(val, int) or not (0 <= val <= 9):
                    raise ValueError(f"Grid values must be 0-9, got {val}")

        return grid

    @property
    def input_shape(self) -> tuple:
        return (len(self.input), len(self.input[0]) if self.input else 0)

    @property
    def output_shape(self) -> tuple:
        return (len(self.output), len(self.output[0]) if self.output else 0)


class ARCProblem(BaseModel):
    """ARC-AGI task with training examples and test cases."""

    task_id: str = Field(..., description="Unique task ID")
    train: List[ARCExample] = Field(..., min_length=2, description="Training demonstrations")
    test: List[ARCExample] = Field(..., min_length=1, description="Test cases with ground truth")


class ARCSolution(BaseModel):
    """LLM-generated solution for ARC task (via instructor)."""

    reasoning: str = Field(..., description="Analysis of transformation rule")
    rule_description: str = Field(..., description="Explicit rule description")
    output_grids: List[Grid] = Field(..., min_length=1, description="Predicted grids for each test")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator('reasoning', 'rule_description')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text fields must be non-empty")
        return v.strip()

    @field_validator('output_grids')
    @classmethod
    def validate_output_grids(cls, grids: List[Grid]) -> List[Grid]:
        if not grids:
            raise ValueError("Must provide at least one output grid")

        for i, grid in enumerate(grids):
            if not grid:
                raise ValueError(f"Grid {i} cannot be empty")
            if not all(isinstance(row, list) for row in grid):
                raise ValueError(f"Grid {i} must be list of lists")

            width = len(grid[0])
            if not all(len(row) == width for row in grid):
                raise ValueError(f"Grid {i} must be rectangular")

            height = len(grid)
            if not (1 <= height <= 30 and 1 <= width <= 30):
                raise ValueError(f"Grid {i} dimensions must be 1-30")

            for row in grid:
                for val in row:
                    if not isinstance(val, int) or not (0 <= val <= 9):
                        raise ValueError(f"Grid {i} values must be 0-9, got {val}")

        return grids
