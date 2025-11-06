"""Main entry point for GEPA/GEPA-MI/Self-Reflection experiments.

Usage:
    python main.py --mode quick --task gsm8k    # Quick test run on GSM8K
    python main.py --mode full --task arc       # Full benchmark on ARC
"""

import os
import argparse
import time
from dotenv import load_dotenv

from src.gsm8k_components import (
    OpenAIStudentModel,
    GeminiStudentModel,
)
from src.benchmark import run_cost_aware_benchmark


def load_task_data(task_name, config, seed):
    """Load task-specific data and components.

    Returns:
        tuple: (probe_set, val_set, test_set, evaluator, formatter)
    """
    if task_name == "gsm8k":
        from src.data_utils import load_gsm8k_splits, verify_data_quality
        from src.gsm8k_components import GSM8KEvaluator, GSM8KFormatter

        probe, val, test = load_gsm8k_splits(
            probe_size=config["probe_size"],
            val_size=config["val_size"],
            test_size=config["test_size"],
            seed=seed,
        )
        verify_data_quality(probe, val, test)
        evaluator = GSM8KEvaluator()
        formatter = GSM8KFormatter()

    elif task_name == "arc":
        from src.arc_data_utils import load_arc_splits, verify_arc_data_quality
        from src.arc_components import ARCEvaluator, ARCFormatter

        probe, val, test = load_arc_splits(
            probe_size=config["probe_size"],
            val_size=config["val_size"],
            test_size=config["test_size"],
            seed=seed,
        )
        verify_arc_data_quality(probe, val, test)
        evaluator = ARCEvaluator()
        formatter = ARCFormatter()

    return probe, val, test, evaluator, formatter


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run GEPA/GEPA-MI experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Experiment mode: 'quick' for fast validation, 'full' for complete benchmark",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["gsm8k", "arc"],
        default="gsm8k",
        help="Task: 'gsm8k' (math) or 'arc' (abstract reasoning)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider: 'openai' or 'gemini' (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY or GEMINI_API_KEY env var). "
             "For custom endpoints, set OPENAI_BASE_URL or GEMINI_ENDPOINT.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gpt-4o-mini for OpenAI, gemini-1.5-flash for Gemini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Configuration based on mode
    if args.mode == "quick":
        print(f"\n🚀 QUICK MODE: Fast validation run ({args.task.upper()})")
        config = {
            "probe_size": 50,
            "val_size": 100,  # Smaller for quick test
            "test_size": 50,
            "time_budget_s": 60,  # 1 minute per method
            "batch_size": 4,
            "self_consistency_k": 2,  # Lower for speed
            "max_probes": 5,
            "delta_init": 0.1,
            "delta_final": 0.05,
        }
    else:  # full
        print(f"\n🔬 FULL MODE: Complete benchmark ({args.task.upper()})")
        config = {
            "probe_size": 200,
            "val_size": 600,  # LARGE validation set for speedup
            "test_size": 200,
            "time_budget_s": 600,  # 10 minutes per method
            "batch_size": 8,
            "self_consistency_k": 8,  # High k for expensive inference
            "max_probes": 10,
            "delta_init": 0.05,
            "delta_final": 0.02,
        }

    # Adjust config for ARC task (harder, needs more samples)
    if args.task == "arc" and args.mode == "quick":
        config.update({
            "probe_size": 20,
            "val_size": 50,
            "test_size": 20,
            "self_consistency_k": 2,
        })

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print("\n" + "=" * 80)
    print(f"LOADING DATA ({args.task.upper()})")
    print("=" * 80)

    probe_set, val_set, test_set, evaluator, formatter = load_task_data(
        task_name=args.task,
        config=config,
        seed=args.seed,
    )

    # Initialize components
    print("\n" + "=" * 80)
    print("INITIALIZING COMPONENTS")
    print("=" * 80)

    # Determine model name if not specified
    if args.model is None:
        if args.provider == "openai":
            model_name = "gpt-4o-mini"
        else:  # gemini
            model_name = "gemini-1.5-flash"
    else:
        model_name = args.model

    # Initialize student model based on provider
    if args.provider == "openai":
        student = OpenAIStudentModel(model_name=model_name, api_key=args.api_key)
    else:  # gemini
        student = GeminiStudentModel(model_name=model_name, api_key=args.api_key)

    print(f"✓ Provider: {args.provider}")
    print(f"✓ Student model: {model_name}")
    print(f"✓ Task: {args.task}")
    print(f"✓ Evaluator: {type(evaluator).__name__}")
    print(f"✓ Formatter: {type(formatter).__name__}")

    # Inference config - using typed InferenceConfig
    from src.types import InferenceConfig

    # ARC needs more tokens for grid representations
    if args.task == "arc":
        inference_config = InferenceConfig(
            temperature=0.7,
            max_tokens=2048,  # Grids need more space
            top_p=0.9
        )
    else:
        inference_config = InferenceConfig(
            temperature=0.7,
            max_tokens=256,
            top_p=0.9
        )

    print(f"\nInference config:")
    print(f"  temperature: {inference_config.temperature}")
    print(f"  max_tokens: {inference_config.max_tokens}")
    print(f"  top_p: {inference_config.top_p}")

    # Run benchmark
    print("\n" + "=" * 80)
    print("STARTING BENCHMARK")
    print("=" * 80)

    start_time = time.time()

    baseline_result, mi_result, sr_result = run_cost_aware_benchmark(
        student=student,
        evaluator=evaluator,
        formatter=formatter,
        probe_set=probe_set,
        val_set=val_set,
        test_set=test_set,
        inference_config=inference_config,
        time_budget_s=config["time_budget_s"],
        batch_size=config["batch_size"],
        self_consistency_k=config["self_consistency_k"],
        max_probes=config["max_probes"],
        delta_init=config["delta_init"],
        delta_final=config["delta_final"],
        verbose=True,
    )

    total_time = time.time() - start_time

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"results/experiment_{args.task}_{args.mode}_{timestamp}.txt"

    with open(results_file, "w") as f:
        f.write("GEPA vs GEPA-MI vs Self-Reflection Experimental Results\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Task: {args.task}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Provider: {args.provider}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Total time: {total_time:.1f}s\n\n")

        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("GEPA Baseline Results:\n")
        f.write(f"  Method: {baseline_result.method}\n")
        f.write(f"  Wall time: {baseline_result.wall_time:.1f}s\n")
        f.write(f"  Pool size: {baseline_result.pool_size}\n")
        f.write(f"  Accepted candidates: {baseline_result.accepted_candidates}\n")
        f.write(f"  Total validation calls: {baseline_result.total_validation_calls}\n")
        f.write(f"  Test accuracy: {baseline_result.final_accuracy:.3f}\n\n")

        f.write("GEPA-MI Results:\n")
        f.write(f"  Method: {mi_result.method}\n")
        f.write(f"  Wall time: {mi_result.wall_time:.1f}s\n")
        f.write(f"  Pool size: {mi_result.pool_size}\n")
        f.write(f"  Accepted candidates: {mi_result.accepted_candidates}\n")
        f.write(f"  Total validation calls: {mi_result.total_validation_calls}\n")
        f.write(f"  Avg probes/candidate: {mi_result.avg_probes_per_candidate:.1f}\n")
        f.write(f"  Speedup: {mi_result.speedup:.2f}x\n")
        f.write(f"  Test accuracy: {mi_result.final_accuracy:.3f}\n\n")

        f.write("Self-Reflection Results:\n")
        f.write(f"  Method: {sr_result.method}\n")
        f.write(f"  Wall time: {sr_result.wall_time:.1f}s\n")
        f.write(f"  Iterations: {sr_result.iterations}\n")
        f.write(f"  Total inference calls: {sr_result.total_inference_calls}\n")
        f.write(f"  Test accuracy: {sr_result.final_accuracy:.3f}\n\n")

        f.write("Analysis:\n")
        if mi_result.speedup < 3.0:
            f.write("  ⚠️ Speedup < 3x (consider larger val set or higher k)\n")
        else:
            f.write(f"  ✅ Achieved {mi_result.speedup:.1f}x speedup!\n")
            saved_calls = (
                baseline_result.total_validation_calls
                - mi_result.total_validation_calls
            )
            f.write(f"  Saved {saved_calls} inference calls\n")

        f.write("\nTest Accuracy Comparison:\n")
        f.write(f"  GEPA Baseline:    {baseline_result.final_accuracy:.3f}\n")
        f.write(f"  GEPA-MI:          {mi_result.final_accuracy:.3f}\n")
        f.write(f"  Self-Reflection:  {sr_result.final_accuracy:.3f}\n")

    print(f"✓ Results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
