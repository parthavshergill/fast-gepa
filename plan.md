# GEPA & GEPA-MI Implementation Plan

## Objective
Implement and benchmark GEPA (Genetic-Pareto prompt optimization) and GEPA-MI (with Mutual Information-guided selective validation) on GSM8K math problems.

## Status: IMPLEMENTATION COMPLETE → READY FOR EXPERIMENTS

---

## Phase 1: Project Setup ✅
- [x] Create plan.md
- [x] Set up project structure
- [x] Create requirements.txt with dependencies
- [x] Verify environment and dependencies
- [x] Create uv virtual environment

## Phase 2: Core Abstractions (Modular Architecture) ✅
- [x] Implement base interfaces:
  - [x] AgentTrajectory dataclass
  - [x] EvaluationResult dataclass
  - [x] TrajectoryFormatter abstract class
  - [x] StudentModel abstract class
  - [x] TaskEvaluator abstract class
- [x] Create core.py with all interfaces

## Phase 3: GSM8K-Specific Implementations ✅
- [x] Implement GSM8KFormatter (trajectory formatting)
- [x] Implement GSM8KEvaluator (answer extraction & comparison)
- [x] Implement StudentModel implementations:
  - [x] OpenAIStudentModel (using GPT-4o-mini)
  - [ ] Optional: LocalLLMStudentModel (for later)
- [x] Create gsm8k_components.py

## Phase 4: Data Loading & Splitting ✅
- [x] Load GSM8K dataset
- [x] Split into probe/val/test sets:
  - [x] Probe: 200 instances (minibatch sampling)
  - [x] Val: 600 instances (large validation set for Pareto frontier)
  - [x] Test: 200 instances (final evaluation)
- [x] Verify splits and data quality
- [x] Create data_utils.py

## Phase 5: GEPA Baseline Implementation ✅
- [x] Implement Candidate dataclass
- [x] Implement gepa_baseline function:
  - [x] Parent selection
  - [x] Minibatch probing
  - [x] Reflection & mutation (simplified for MVP)
  - [x] Quick check on minibatch
  - [x] FULL validation (expensive)
  - [x] Pareto frontier management
- [x] Add cost tracking (total validation calls)
- [x] Create gepa_baseline.py

## Phase 6: GEPA-MI Selective Validation ✅
- [x] Implement BetaBernoulliPosterior class:
  - [x] Bayesian updates
  - [x] Mean estimation
  - [x] P(Z=1) calculation
- [x] Implement selection strategies:
  - [x] Uncertainty sampling (p closest to 0.5)
- [x] Implement selective_validate function:
  - [x] MI-guided probing
  - [x] Early stopping (ACCEPT/REJECT)
  - [x] Adaptive delta
- [x] Implement gepa_mi function
- [x] Create gepa_mi.py

## Phase 7: Benchmarking Infrastructure ✅
- [x] Implement BenchmarkResult dataclass
- [x] Implement run_cost_aware_benchmark:
  - [x] Run both baseline and MI methods
  - [x] Track wall time, validation calls, speedup
  - [x] Evaluate final test accuracy
- [x] Implement evaluate_on_split helper
- [x] Create benchmark.py

## Phase 8: Integration & Main Script ✅
- [x] Create main.py with complete example
- [x] Add configuration management
- [x] Add proper logging
- [x] Add argument parsing for experiments
- [x] Create README.md
- [x] Create .env.example

## Phase 9: Initial Experimental Run 🧪 ⬅️ NEXT
- [ ] Set up OpenAI API key
- [ ] Run with minimal settings first (quick validation):
  - [ ] Smaller val set (100)
  - [ ] Lower k (2)
  - [ ] Short time budget (60s)
- [ ] Debug any issues
- [ ] Verify basic functionality

## Phase 10: Full Experimental Run 🚀
- [ ] Run with proper settings:
  - [ ] Val set: 600
  - [ ] Self-consistency k: 8
  - [ ] Time budget: 600s (10 min)
  - [ ] Max probes: 10
- [ ] Collect results
- [ ] Analyze speedup
- [ ] Generate report

## Phase 11: Analysis & Documentation 📊
- [ ] Compare baseline vs GEPA-MI
- [ ] Validate speedup expectations (5-15x)
- [ ] Document findings
- [ ] Create results.md

---

## Git Commits Made
1. ✅ Initial project setup with core abstractions
2. ✅ Implement GSM8K components and data utilities
3. ✅ Implement GEPA baseline and GEPA-MI algorithms
4. ✅ Add benchmarking infrastructure and main entry point
5. ✅ Add documentation and environment configuration

---

## Key Implementation Notes

### Critical Requirements for Speedup
1. **Large validation set**: 600 instances (not 100!)
2. **Expensive inference**: k ≥ 8 for self-consistency
3. **Per-instance time**: Should be > 500ms with k=8

### Project Structure
```
fast-gepa/
├── plan.md                 # This file
├── requirements.txt        # Dependencies
├── src/
│   ├── __init__.py
│   ├── core.py            # Base interfaces
│   ├── gsm8k_components.py # GSM8K implementations
│   ├── data_utils.py      # Data loading
│   ├── gepa_baseline.py   # Baseline GEPA
│   ├── gepa_mi.py         # GEPA-MI
│   └── benchmark.py       # Benchmarking
├── main.py                # Main entry point
└── results/               # Experimental results
```

### Dependencies
- datasets (HuggingFace)
- numpy
- scipy
- openai
- python-dotenv (for API keys)
- tqdm (progress bars)

---

## Current Focus
**Next Step**: Phase 1 - Project Setup
