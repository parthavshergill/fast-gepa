"""Data loading and splitting utilities for ARC-AGI-2.

This module provides functions to load ARC-AGI-2 JSON files and convert them
to typed Pydantic models following CLAUDE_CONTRACTS.md patterns.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

from .types import ARCProblem, ARCExample


def load_arc_splits(
    probe_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
    data_dir: str = "ARC-AGI-2/data"
) -> Tuple[List[ARCProblem], List[ARCProblem], List[ARCProblem]]:
    """Load and split ARC-AGI-2 dataset.

    Args:
        probe_size: Size of probe set for minibatch sampling
        val_size: Size of validation set (Pareto frontier)
        test_size: Size of test set for final evaluation
        seed: Random seed for reproducibility
        data_dir: Path to ARC-AGI-2/data directory (relative to project root)

    Returns:
        (probe_set, val_set, test_set) - all typed as ARCProblem

    Raises:
        ValueError: If requested sizes exceed available data
        FileNotFoundError: If data directory does not exist
    """
    print("Loading ARC-AGI-2 dataset...")

    # Resolve data directory
    project_root = Path(__file__).parent.parent
    training_dir = project_root / data_dir / "training"
    eval_dir = project_root / data_dir / "evaluation"

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    # Load all training problems
    training_problems = []
    for json_file in sorted(training_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Convert to ARCProblem with validation
        problem = ARCProblem(
            task_id=task_id,
            train=[
                ARCExample(input=ex['input'], output=ex['output'])
                for ex in data['train']
            ],
            test=[
                ARCExample(input=ex['input'], output=ex['output'])
                for ex in data['test']
            ]
        )
        training_problems.append(problem)

    # Load all evaluation problems
    eval_problems = []
    for json_file in sorted(eval_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file, 'r') as f:
            data = json.load(f)

        problem = ARCProblem(
            task_id=task_id,
            train=[
                ARCExample(input=ex['input'], output=ex['output'])
                for ex in data['train']
            ],
            test=[
                ARCExample(input=ex['input'], output=ex['output'])
                for ex in data['test']
            ]
        )
        eval_problems.append(problem)

    print(f"Loaded {len(training_problems)} training tasks, {len(eval_problems)} evaluation tasks")

    # Combine all problems and shuffle
    all_problems = training_problems + eval_problems
    random.seed(seed)
    random.shuffle(all_problems)

    # Check we have enough data
    total_requested = probe_size + val_size + test_size
    if total_requested > len(all_problems):
        raise ValueError(
            f"Requested total size ({total_requested}) exceeds available tasks ({len(all_problems)}). "
            f"probe_size={probe_size}, val_size={val_size}, test_size={test_size}"
        )

    # Split into probe, val, test
    probe_set = all_problems[:probe_size]
    val_set = all_problems[probe_size:probe_size + val_size]
    test_set = all_problems[probe_size + val_size:probe_size + val_size + test_size]

    print(f"\nSplit sizes:")
    print(f"  Probe: {len(probe_set)} (for minibatch sampling)")
    print(f"  Val:   {len(val_set)} (for Pareto frontier)")
    print(f"  Test:  {len(test_set)} (for final evaluation)")

    return probe_set, val_set, test_set


def verify_arc_data_quality(
    probe_set: List[ARCProblem],
    val_set: List[ARCProblem],
    test_set: List[ARCProblem]
) -> None:
    """Verify ARC data quality and format.

    Args:
        probe_set: Probe split
        val_set: Validation split
        test_set: Test split

    Raises:
        ValueError: If data quality checks fail
    """
    print("\nVerifying ARC data quality...")

    # Check all sets are non-empty and contain ARCProblem objects
    for name, dataset in [
        ("probe", probe_set),
        ("val", val_set),
        ("test", test_set),
    ]:
        if not dataset:
            raise ValueError(f"{name} set is empty")

        for i, problem in enumerate(dataset[:3]):  # Check first 3
            if not isinstance(problem, ARCProblem):
                raise ValueError(f"{name}[{i}] is not an ARCProblem instance")
            if not problem.task_id:
                raise ValueError(f"{name}[{i}] has empty task_id")
            if not problem.train:
                raise ValueError(f"{name}[{i}] has no training examples")
            if not problem.test:
                raise ValueError(f"{name}[{i}] has no test cases")

            # Verify training examples
            for j, train_ex in enumerate(problem.train):
                if not train_ex.input:
                    raise ValueError(f"{name}[{i}] train[{j}] has empty input")
                if not train_ex.output:
                    raise ValueError(f"{name}[{i}] train[{j}] has empty output")

            # Verify test cases
            for j, test_ex in enumerate(problem.test):
                if not test_ex.input:
                    raise ValueError(f"{name}[{i}] test[{j}] has empty input")
                if not test_ex.output:
                    raise ValueError(f"{name}[{i}] test[{j}] has empty output")

    # Print sample
    print("\nSample task from probe set:")
    sample = probe_set[0]
    print(f"  Task ID: {sample.task_id}")
    print(f"  Training examples: {len(sample.train)}")
    print(f"  Test cases: {len(sample.test)}")
    print(f"  First train input shape: {sample.train[0].input_shape}")
    print(f"  First train output shape: {sample.train[0].output_shape}")
    print(f"  First test input shape: {sample.test[0].input_shape}")
    print(f"  First test output shape: {sample.test[0].output_shape}")

    # Print grid statistics
    total_train_examples = sum(len(p.train) for p in probe_set + val_set + test_set)
    total_test_cases = sum(len(p.test) for p in probe_set + val_set + test_set)
    print(f"\nDataset statistics:")
    print(f"  Total tasks: {len(probe_set) + len(val_set) + len(test_set)}")
    print(f"  Total training examples: {total_train_examples}")
    print(f"  Total test cases: {total_test_cases}")

    print("\n✓ ARC data quality verified")
