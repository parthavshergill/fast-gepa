# GEPA & GEPA-MI Implementation Plan

## Objective
Implement and benchmark GEPA (Genetic-Pareto prompt optimization) and GEPA-MI (with Mutual Information-guided selective validation) on GSM8K math problems.

## Status: PLANNING → IMPLEMENTATION

---

## Phase 1: Project Setup ✅
- [x] Create plan.md
- [ ] Set up project structure
- [ ] Create requirements.txt with dependencies
- [ ] Verify environment and dependencies

## Phase 2: Core Abstractions (Modular Architecture) 🔄
- [ ] Implement base interfaces:
  - [ ] AgentTrajectory dataclass
  - [ ] EvaluationResult dataclass
  - [ ] TrajectoryFormatter abstract class
  - [ ] StudentModel abstract class
  - [ ] TaskEvaluator abstract class
- [ ] Create core.py with all interfaces

## Phase 3: GSM8K-Specific Implementations 📝
- [ ] Implement GSM8KFormatter (trajectory formatting)
- [ ] Implement GSM8KEvaluator (answer extraction & comparison)
- [ ] Implement StudentModel implementations:
  - [ ] OpenAIStudentModel (using GPT-4o-mini)
  - [ ] Optional: LocalLLMStudentModel (for later)
- [ ] Test basic inference pipeline
- [ ] Create gsm8k_components.py

## Phase 4: Data Loading & Splitting 📊
- [ ] Load GSM8K dataset
- [ ] Split into probe/val/test sets:
  - [ ] Probe: 200 instances (minibatch sampling)
  - [ ] Val: 600 instances (large validation set for Pareto frontier)
  - [ ] Test: 200 instances (final evaluation)
- [ ] Verify splits and data quality
- [ ] Create data_utils.py

## Phase 5: GEPA Baseline Implementation 🧬
- [ ] Implement Candidate dataclass
- [ ] Implement gepa_baseline function:
  - [ ] Parent selection
  - [ ] Minibatch probing
  - [ ] Reflection & mutation (simplified for MVP)
  - [ ] Quick check on minibatch
  - [ ] FULL validation (expensive)
  - [ ] Pareto frontier management
- [ ] Add cost tracking (total validation calls)
- [ ] Create gepa_baseline.py

## Phase 6: GEPA-MI Selective Validation 🎯
- [ ] Implement BetaBernoulliPosterior class:
  - [ ] Bayesian updates
  - [ ] Mean estimation
  - [ ] P(Z=1) calculation
- [ ] Implement selection strategies:
  - [ ] Uncertainty sampling (p closest to 0.5)
- [ ] Implement selective_validate function:
  - [ ] MI-guided probing
  - [ ] Early stopping (ACCEPT/REJECT)
  - [ ] Adaptive delta
- [ ] Implement gepa_mi function
- [ ] Create gepa_mi.py

## Phase 7: Benchmarking Infrastructure 📈
- [ ] Implement BenchmarkResult dataclass
- [ ] Implement run_cost_aware_benchmark:
  - [ ] Run both baseline and MI methods
  - [ ] Track wall time, validation calls, speedup
  - [ ] Evaluate final test accuracy
- [ ] Implement evaluate_on_split helper
- [ ] Create benchmark.py

## Phase 8: Integration & Main Script 🔌
- [ ] Create main.py with complete example
- [ ] Add configuration management
- [ ] Add proper logging
- [ ] Add argument parsing for experiments

## Phase 9: Initial Experimental Run 🧪
- [ ] Run with minimal settings first (quick validation):
  - [ ] Smaller val set (100)
  - [ ] Lower k (1-2)
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
