"""Data loading and splitting utilities for GSM8K."""

import random
from typing import List, Tuple
from datasets import load_dataset

from .types import GSM8KProblem


def load_gsm8k_splits(
    probe_size: int = 200,
    val_size: int = 600,
    test_size: int = 200,
    seed: int = 42,
) -> Tuple[List[GSM8KProblem], List[GSM8KProblem], List[GSM8KProblem]]:
    """
    Load and split GSM8K dataset.

    Args:
        probe_size: Size of probe set for minibatch sampling
        val_size: Size of validation set (Pareto frontier)
        test_size: Size of test set for final evaluation
        seed: Random seed for reproducibility

    Returns:
        (probe_set, val_set, test_set) - all typed as GSM8KProblem
    """
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")

    train_full = list(ds["train"])
    test_full = list(ds["test"])

    print(f"Loaded {len(train_full)} train, {len(test_full)} test instances")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(train_full)
    random.shuffle(test_full)

    # Split train into probe and val
    if probe_size + val_size > len(train_full):
        raise ValueError(
            f"probe_size ({probe_size}) + val_size ({val_size}) "
            f"exceeds train size ({len(train_full)})"
        )

    # Convert to Pydantic models
    probe_set = [
        GSM8KProblem(question=item['question'], answer=item['answer'])
        for item in train_full[:probe_size]
    ]
    val_set = [
        GSM8KProblem(question=item['question'], answer=item['answer'])
        for item in train_full[probe_size : probe_size + val_size]
    ]

    # Use subset of test
    test_set = [
        GSM8KProblem(question=item['question'], answer=item['answer'])
        for item in test_full[:test_size]
    ]

    print(f"\nSplit sizes:")
    print(f"  Probe: {len(probe_set)} (for minibatch sampling)")
    print(f"  Val:   {len(val_set)} (for Pareto frontier)")
    print(f"  Test:  {len(test_set)} (for final evaluation)")

    return probe_set, val_set, test_set


def verify_data_quality(
    probe_set: List[GSM8KProblem], val_set: List[GSM8KProblem], test_set: List[GSM8KProblem]
) -> None:
    """Verify data quality and format."""
    print("\nVerifying data quality...")

    # Check all sets are non-empty and contain GSM8KProblem objects
    for name, dataset in [
        ("probe", probe_set),
        ("val", val_set),
        ("test", test_set),
    ]:
        if not dataset:
            raise ValueError(f"{name} set is empty")

        for i, problem in enumerate(dataset[:3]):  # Check first 3
            if not isinstance(problem, GSM8KProblem):
                raise ValueError(f"{name}[{i}] is not a GSM8KProblem instance")
            if not problem.question.strip():
                raise ValueError(f"{name}[{i}] has empty question")
            if not problem.answer.strip():
                raise ValueError(f"{name}[{i}] has empty answer")

    # Print sample
    print("\nSample instance from probe set:")
    sample = probe_set[0]
    print(f"  Question: {sample.question[:80]}...")
    print(f"  Answer: {sample.answer[:80]}...")

    print("\n✓ Data quality verified")
