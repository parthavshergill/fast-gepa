# GEPA, GEPA-MI & Self-Reflection Implementation Plan

## Objective
Implement and benchmark three prompt optimization methods on GSM8K math problems:
- GEPA (Genetic-Pareto with full validation)
- GEPA-MI (with Mutual Information-guided selective validation)
- Self-Reflection (simple iterative refinement baseline)

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

## Phase 9: Self-Reflection Baseline ✅
- [x] Implement ReflectionEngine class:
  - [x] Format failed trajectories
  - [x] Build reflection prompt
  - [x] Call LLM for analysis
  - [x] Parse structured suggestions (ANALYSIS/IMPROVEMENT/TARGET)
  - [x] Apply mutations with graceful fallback
- [x] Implement Self-Reflection algorithm:
  - [x] Simple iterative refinement (no validation, no pool)
  - [x] Probe → evaluate → reflect → mutate loop
  - [x] Track best prompt by probe accuracy
- [x] Integrate into benchmarking:
  - [x] Update run_cost_aware_benchmark to run three methods
  - [x] Return three BenchmarkResult objects
  - [x] Add comparison metrics
- [x] Update main.py to handle three methods
- [x] Update documentation:
  - [x] README.md with algorithm descriptions
  - [x] IMPLEMENTATION_NOTES.md with design rationale
  - [x] plan.md with phase completion
- [x] Create src/reflection.py
- [x] Create src/self_reflection.py

## Phase 10: Initial Experimental Run 🧪 ⬅️ NEXT
- [ ] Set up OpenAI API key
- [ ] Run with minimal settings first (quick validation):
  - [ ] Smaller val set (100)
  - [ ] Lower k (2)
  - [ ] Short time budget (60s)
- [ ] Debug any issues
- [ ] Verify basic functionality

## Phase 11: Full Experimental Run 🚀
- [ ] Run with proper settings:
  - [ ] Val set: 600
  - [ ] Self-consistency k: 8
  - [ ] Time budget: 600s (10 min)
  - [ ] Max probes: 10
- [ ] Collect results for all three methods
- [ ] Analyze speedup
- [ ] Compare reflection vs random mutation
- [ ] Generate report

## Phase 12: Analysis & Documentation 📊
- [ ] Compare all three methods:
  - [ ] GEPA Baseline (full validation)
  - [ ] GEPA-MI (selective validation + speedup)
  - [ ] Self-Reflection (no validation + overfitting risk)
- [ ] Validate speedup expectations (5-15x for GEPA-MI)
- [ ] Analyze reflection effectiveness
- [ ] Document findings
- [ ] Create results.md

---

## Git Commits Made
1. ✅ Initial project setup with core abstractions
2. ✅ Implement GSM8K components and data utilities
3. ✅ Implement GEPA baseline and GEPA-MI algorithms
4. ✅ Add benchmarking infrastructure and main entry point
5. ✅ Add documentation and environment configuration
6. ✅ Add detailed implementation notes and simplifications
7. ✅ Add Google Gemini API support as default provider
8. ✅ Implement Self-Reflection baseline with ReflectionEngine

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
├── README.md               # User documentation
├── IMPLEMENTATION_NOTES.md # Technical deep-dive
├── src/
│   ├── __init__.py
│   ├── core.py            # Base interfaces
│   ├── gsm8k_components.py # GSM8K implementations
│   ├── data_utils.py      # Data loading
│   ├── gepa_baseline.py   # Baseline GEPA
│   ├── gepa_mi.py         # GEPA-MI
│   ├── reflection.py      # ReflectionEngine
│   ├── self_reflection.py # Self-Reflection algorithm
│   └── benchmark.py       # Benchmarking (all 3 methods)
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
**Current Phase**: Phase 9 - Self-Reflection Baseline ✅ COMPLETE
**Next Step**: Phase 10 - Initial Experimental Run 🧪

All three methods are now implemented and integrated:
- ✅ GEPA Baseline with full validation
- ✅ GEPA-MI with selective validation and speedup
- ✅ Self-Reflection with ReflectionEngine for iterative refinement

Ready to run experiments and compare all three approaches!
