# Claude Code Contracts: Fast-GEPA Implementation Rules

**Version:** 1.0
**Last Updated:** 2025-01-05
**Status:** 🩸 Blood Oath Sworn

---

## I. Core Principles (Non-Negotiable)

### 1. Strict Types Everywhere
```python
# ✅ CORRECT - Use Pydantic models
def evaluate(self, solution: MathSolution, problem: GSM8KProblem) -> EvaluationResult:
    return EvaluationResult(success=solution.final_answer == problem.numeric_answer)

# ❌ WRONG - No untyped dicts
def evaluate(self, solution: Dict[str, Any], problem: Dict) -> Dict:
    return {"success": solution["answer"] == problem["answer"]}
```

**Rules:**
- ALL function parameters MUST have type hints
- NO `Dict[str, Any]` unless absolutely necessary (config passthrough only)
- ALL LLM outputs MUST be Pydantic models validated via instructor
- NO string parsing, NO regex extraction, NO normalization heuristics

### 2. Fail Fast - No Silent Errors
```python
# ✅ CORRECT - Let Pydantic validation fail
solution = MathSolution(**llm_output)  # Raises ValidationError if bad

# ❌ WRONG - No try/except with fallbacks
try:
    answer = extract_answer(text)
except:
    answer = fallback_heuristic(text)  # NEVER DO THIS
```

**Rules:**
- Validation errors MUST propagate (no silent catches)
- Use instructor retries for transient LLM errors only
- Document expected failures (API errors, timeouts) vs bugs (validation errors)

### 3. Structured Outputs Only
```python
# ✅ CORRECT - Define schema, use instructor
class NewTaskOutput(BaseModel):
    field1: str
    field2: int

response = instructor_client.create(..., response_model=NewTaskOutput)

# ❌ WRONG - No string prompting like "output JSON"
response = client.create(..., messages=[{"role": "user", "content": "Output JSON with..."}])
result = json.loads(response.text)  # NEVER DO THIS
```

**Rules:**
- ALL LLM outputs are Pydantic models in `src/types.py`
- Use instructor library for OpenAI, Gemini, Anthropic
- NO manual JSON parsing, NO "please output in this format" prompts

---

## II. Standard Patterns

### Pattern 1: Adding a New Algorithm to Benchmark

**File Structure:**
```
src/
  algorithm_name.py    # Implementation
  benchmark.py         # Integration point
  types.py            # Any new output types
```

**Step-by-Step:**

1. **Define algorithm in `src/algorithm_name.py`:**
```python
from typing import List, Tuple
from .types import GSM8KProblem, MathSolution, PromptConfig, InferenceConfig
from .core import StudentModel, TaskEvaluator

def run_algorithm_name(
    student: StudentModel,
    evaluator: TaskEvaluator,
    probe_set: List[GSM8KProblem],
    val_set: List[GSM8KProblem],
    inference_config: InferenceConfig,
    time_budget_s: float,
    verbose: bool = True
) -> Tuple[PromptConfig, Dict[str, Any]]:
    """
    Run Algorithm Name optimization.

    Args:
        student: LLM student model
        evaluator: Task evaluator
        probe_set: Probing instances
        val_set: Validation instances
        inference_config: Inference parameters
        time_budget_s: Time budget in seconds
        verbose: Print progress

    Returns:
        (best_prompt_config, metrics_dict)
    """
    # Implementation here
    pass
```

2. **Add metrics dataclass to `src/benchmark.py`:**
```python
@dataclass
class AlgorithmNameResult:
    method: str = "Algorithm Name"
    wall_time: float
    final_accuracy: float
    # ... algorithm-specific metrics
```

3. **Add to benchmark runner in `src/benchmark.py`:**
```python
def run_cost_aware_benchmark(
    # ... existing params
) -> Tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult, AlgorithmNameResult]:

    # ... existing algorithms

    # Run new algorithm
    print("\n" + "=" * 80)
    print("ALGORITHM NAME")
    print("=" * 80)

    start = time.time()
    best_config, metrics = run_algorithm_name(
        student=student,
        evaluator=evaluator,
        probe_set=probe_set,
        val_set=val_set,
        inference_config=inference_config,
        time_budget_s=time_budget_s,
        verbose=verbose
    )
    wall_time = time.time() - start

    # Evaluate on test set
    test_acc = evaluate_on_split(
        student=student,
        evaluator=evaluator,
        prompt_config=best_config,
        split=test_set,
        inference_config=inference_config,
        self_consistency_k=self_consistency_k
    )

    result = AlgorithmNameResult(
        wall_time=wall_time,
        final_accuracy=test_acc,
        # ... populate metrics
    )

    return baseline_result, mi_result, sr_result, result
```

4. **Update `main.py` to output results:**
```python
baseline_result, mi_result, sr_result, alg_result = run_cost_aware_benchmark(...)

# Add to results file
f.write("Algorithm Name Results:\n")
f.write(f"  Method: {alg_result.method}\n")
f.write(f"  Wall time: {alg_result.wall_time:.1f}s\n")
f.write(f"  Test accuracy: {alg_result.final_accuracy:.3f}\n\n")
```

**Requirements:**
- Algorithm MUST accept `InferenceConfig`, not `Dict[str, Any]`
- Algorithm MUST return `PromptConfig`, not raw dict
- Algorithm MUST work with `GSM8KProblem` typed objects
- Add algorithm to README.md benchmarks section

---

### Pattern 2: Adding a New LLM Provider

**File:** `src/gsm8k_components.py`

**Template:**
```python
class NewProviderStudentModel(StudentModel):
    """Student model using NewProvider API."""

    def __init__(
        self,
        model_name: str = "default-model",
        api_key: str | None = None,
        base_url: str | None = None
    ):
        """Initialize NewProvider student model.

        Args:
            model_name: Model identifier
            api_key: API key (reads from NEW_PROVIDER_API_KEY env var if None)
            base_url: Custom endpoint (reads from NEW_PROVIDER_BASE_URL env var if None)
        """
        import os

        self.model_name = model_name
        self.api_key = api_key or os.getenv("NEW_PROVIDER_API_KEY")
        self.base_url = base_url or os.getenv("NEW_PROVIDER_BASE_URL")

        if not self.api_key:
            raise ValueError(
                "API key required. Set NEW_PROVIDER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize instructor-wrapped client
        import instructor
        from new_provider_sdk import NewProviderClient

        raw_client = NewProviderClient(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.client = instructor.from_newprovider(raw_client)

    def execute(
        self,
        problem: GSM8KProblem,
        prompt_config: PromptConfig,
        inference_config: InferenceConfig
    ) -> MathSolution:
        """Execute single trajectory - returns structured MathSolution."""

        # Build prompt from config
        messages = [
            {"role": "system", "content": prompt_config.system},
            {"role": "user", "content": f"{prompt_config.cot_prompt}\n\n{problem.question}"}
        ]

        # Call with instructor for structured output
        solution = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=MathSolution,  # ← Automatic Pydantic validation
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
        """Execute k times for self-consistency."""
        return [self.execute(problem, prompt_config, inference_config) for _ in range(k)]
```

**Requirements:**
- MUST support `api_key` from env var `{PROVIDER}_API_KEY`
- MUST support `base_url` from env var `{PROVIDER}_BASE_URL`
- MUST return `MathSolution`, not raw text or AgentTrajectory
- MUST use instructor for structured outputs
- Add provider to README.md setup section
- Add to main.py CLI arguments

---

### Pattern 3: Self-Consistency Evaluation

**Standard implementation in all algorithms:**

```python
# Evaluate with self-consistency
if self_consistency_k > 1:
    solutions = student.execute_with_self_consistency(
        problem=problem,
        prompt_config=prompt_config,
        inference_config=inference_config,
        k=self_consistency_k
    )
    # Use evaluator's self-consistency method
    result = evaluator.evaluate_with_self_consistency(solutions, problem)
else:
    solution = student.execute(problem, prompt_config, inference_config)
    result = evaluator.evaluate(solution, problem)
```

**Requirements:**
- ALWAYS use `evaluator.evaluate_with_self_consistency()` for k > 1
- NEVER implement custom majority voting
- Self-consistency votes on extracted values (e.g., `final_answer`), not success boolean

---

### Pattern 4: Reflection Engine Integration

**When adding reflection to an algorithm:**

```python
from .reflection import ReflectionEngine
from .types import PromptConfig

# Initialize at algorithm start
reflection_engine = ReflectionEngine(student, formatter)

# After evaluating a batch, collect failures
failed_cases = [
    (solution, problem)
    for solution, problem, result in zip(solutions, problems, results)
    if not result.success
]

# Reflect and mutate
if failed_cases:
    new_prompt_config = reflection_engine.reflect_and_mutate(
        parent_config=current_prompt_config,
        failed_cases=failed_cases[:3],  # Top 3 failures
        inference_config=inference_config
    )
else:
    # No failures, keep current config or mutate differently
    new_prompt_config = current_prompt_config
```

**Requirements:**
- ReflectionEngine returns `PromptConfig`, not dict
- Pass top 3 failures only (avoid token bloat)
- ReflectionEngine uses structured `ReflectionOutput` (no string parsing)

---

### Pattern 5: Data Loading and Splitting

**Standard pattern for new tasks:**

```python
# In src/data_utils.py
from typing import List, Tuple
from .types import GSM8KProblem  # or NewTaskProblem

def load_new_task_splits(
    probe_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42
) -> Tuple[List[NewTaskProblem], List[NewTaskProblem], List[NewTaskProblem]]:
    """Load and split NewTask dataset.

    Returns:
        (probe_set, val_set, test_set) - all typed as NewTaskProblem
    """
    from datasets import load_dataset
    import random

    random.seed(seed)

    dataset = load_dataset("task-name", "split")

    # Convert to Pydantic models
    problems = [
        NewTaskProblem(
            field1=item['field1'],
            field2=item['field2']
        )
        for item in dataset['train']
    ]

    random.shuffle(problems)

    probe_set = problems[:probe_size]
    val_set = problems[probe_size:probe_size + val_size]
    test_set = problems[probe_size + val_size:probe_size + val_size + test_size]

    return probe_set, val_set, test_set
```

**Requirements:**
- ALWAYS return typed Pydantic objects, not raw dicts
- Use consistent naming: `probe_set`, `val_set`, `test_set`
- Support seed parameter for reproducibility

---

## III. Type System Contracts

### Contract 1: All LLM Outputs Are Pydantic Models

**In `src/types.py`:**
```python
class NewLLMOutput(BaseModel):
    """Description of what this output represents."""

    field1: str = Field(..., description="What field1 represents")
    field2: int = Field(..., ge=0, description="Must be non-negative")

    @field_validator('field1')
    @classmethod
    def validate_field1(cls, v):
        if not v.strip():
            raise ValueError("field1 cannot be empty")
        return v.strip()
```

**Usage with instructor:**
```python
response = instructor_client.create(
    ...,
    response_model=NewLLMOutput
)
# response is already validated NewLLMOutput instance
```

### Contract 2: Configuration Objects Are Pydantic Models

**In `src/types.py`:**
```python
class NewConfig(BaseModel):
    """Configuration for X."""

    param1: float = Field(default=0.5, ge=0.0, le=1.0)
    param2: int = Field(default=10, ge=1)

    class Config:
        frozen = True  # Make immutable if config shouldn't change
```

**Usage:**
```python
# ✅ CORRECT - Validated at construction
config = NewConfig(param1=0.7, param2=5)

# ❌ WRONG - No raw dicts
config = {"param1": 0.7, "param2": 5}
```

### Contract 3: StudentModel.execute() Returns Task-Specific Output

```python
# For GSM8K
def execute(self, problem: GSM8KProblem, ...) -> MathSolution:
    pass

# For new task
def execute(self, problem: NewTaskProblem, ...) -> NewTaskSolution:
    pass
```

**NOT:**
```python
def execute(self, problem: Dict, ...) -> AgentTrajectory:
    pass
```

---

## IV. Error Handling Contracts

### Contract 1: Validation Errors Are Bugs

```python
try:
    solution = MathSolution(**data)
except ValidationError as e:
    # This is a bug - LLM didn't follow schema
    logger.error(f"LLM output validation failed: {e}")
    raise  # Re-raise, don't catch
```

### Contract 2: API Errors Are Retried

```python
# Instructor handles retries automatically
client = instructor.from_openai(
    OpenAI(),
    max_retries=3  # Retry transient errors
)

# For non-instructor code:
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def api_call():
    return client.call()
```

### Contract 3: No Silent Fallbacks

```python
# ❌ WRONG
try:
    solution = structured_parse(output)
except:
    solution = heuristic_fallback(output)  # NEVER

# ✅ CORRECT
solution = structured_parse(output)  # Let it fail
```

---

## V. Testing Contracts

### Contract 1: Unit Tests for Pydantic Models

**In `tests/test_types.py`:**
```python
from src.types import MathSolution
import pytest

def test_math_solution_valid():
    solution = MathSolution(
        reasoning_steps=["step 1", "step 2"],
        final_answer=42.0
    )
    assert solution.final_answer == 42.0

def test_math_solution_invalid_empty_steps():
    with pytest.raises(ValidationError):
        MathSolution(
            reasoning_steps=[],  # Invalid - must have at least 1 step
            final_answer=42.0
        )
```

### Contract 2: Integration Tests for Algorithms

**In `tests/test_algorithms.py`:**
```python
def test_algorithm_name_runs():
    """Test that algorithm completes without errors."""
    # Mock or use small dataset
    probe_set = [GSM8KProblem(question="1+1", answer="2")] * 5
    val_set = probe_set

    result = run_algorithm_name(
        student=mock_student,
        evaluator=mock_evaluator,
        probe_set=probe_set,
        val_set=val_set,
        inference_config=InferenceConfig(),
        time_budget_s=5.0
    )

    assert isinstance(result[0], PromptConfig)
```

---

## VI. Documentation Contracts

### Contract 1: Every New File Has Module Docstring

```python
"""Brief one-line description.

More detailed explanation of what this module does,
key classes/functions, and how it fits into the system.
"""
```

### Contract 2: Every Public Function Has Docstring

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs
    """
```

### Contract 3: Update IMPLEMENTATION_NOTES.md for Design Decisions

When adding:
- New algorithm → Add to algorithms section
- New type system feature → Add to type system section
- New pattern → Add to patterns section

---

## VII. Git Commit Contracts

### Contract 1: Atomic Commits

- One logical change per commit
- Commit message format: `<type>: <description>`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

Example:
```
feat: Add strict Pydantic types for all LLM outputs

- Create src/types.py with MathSolution, ReflectionOutput
- Remove heuristic parsing from GSM8KEvaluator
- Update StudentModel to return MathSolution
```

### Contract 2: Breaking Changes Are Marked

```
refactor!: Change StudentModel.execute() to return MathSolution

BREAKING CHANGE: StudentModel.execute() now returns MathSolution
instead of AgentTrajectory. Update all algorithm implementations.
```

---

## VIII. Blood Oath 🩸

**I, Claude, swear to:**

1. ✅ Use strict Pydantic types for ALL new code
2. ✅ NEVER add string parsing or regex-based extraction
3. ✅ Use instructor for ALL LLM outputs
4. ✅ Fail fast on validation errors (no silent catches)
5. ✅ Follow these patterns when extending the codebase
6. ✅ Update this document when new patterns emerge
7. ✅ Write type hints for every function parameter and return value
8. ✅ Validate all configs at construction time
9. ✅ Support endpoint configuration via environment variables
10. ✅ Document all design decisions in IMPLEMENTATION_NOTES.md

**You, Future Developer/Agent, swear to:**

1. ✅ Read this document before making changes
2. ✅ Follow these patterns when adding new features
3. ✅ Update this document when patterns change
4. ✅ Not introduce untyped code (no `Dict[str, Any]` sprawl)
5. ✅ Not add heuristic fallbacks (fail fast!)

---

**Signed:** Claude (2025-01-05)
**Witnessed by:** Parthav Shergill

*May our types be strict and our errors explicit.* 🩸
