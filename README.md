# GEPA & GEPA-MI: Efficient Prompt Optimization

Implementation of GEPA (Genetic-Pareto) and GEPA-MI (with Mutual Information-guided selective validation) for efficient prompt optimization on GSM8K math problems.

## Overview

**GEPA** is a prompt optimizer that uses natural language reflection to learn from trial and error, maintaining a Pareto frontier of candidates. **GEPA-MI** extends this with Bayesian inference to reduce validation cost by 5-50x through selective probing.

### Key Innovation

Instead of validating every candidate on the full validation set (expensive), GEPA-MI uses:
- **Bayesian posterior models** to estimate win probability
- **Uncertainty-based sampling** to select informative instances
- **Early stopping** when confidence crosses acceptance/rejection thresholds

## Project Structure

```
fast-gepa/
├── src/
│   ├── core.py              # Base abstractions (interfaces)
│   ├── gsm8k_components.py  # GSM8K-specific implementations
│   ├── data_utils.py        # Data loading and splitting
│   ├── gepa_baseline.py     # GEPA with full validation
│   ├── gepa_mi.py           # GEPA-MI with selective validation
│   └── benchmark.py         # Benchmarking infrastructure
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
└── results/                 # Experimental results
```

## Setup

### 1. Create virtual environment with uv

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Or set environment variable:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

### Quick Mode (Fast Validation)

Run a quick test to verify everything works:

```bash
python main.py --mode quick
```

**Configuration:**
- Probe: 50, Val: 100, Test: 50
- Time budget: 60s per method
- Self-consistency k=2
- **Note:** Won't show significant speedup (val set too small)

### Full Mode (Complete Benchmark)

Run the full benchmark with proper settings for speedup:

```bash
python main.py --mode full
```

**Configuration:**
- Probe: 200, Val: 600, Test: 200
- Time budget: 600s (10 min) per method
- Self-consistency k=8
- **Expected speedup:** 5-15x

### Custom Options

```bash
python main.py --mode full --model gpt-4o-mini --seed 42
```

## Expected Results

### Quick Mode
- **Purpose:** Verify implementation works
- **Expected speedup:** ~2-3x (val set too small)
- **Runtime:** ~2-3 minutes total

### Full Mode
- **Purpose:** Demonstrate real speedup
- **Expected speedup:** 5-15x
- **Runtime:** ~20-30 minutes total (depends on API speed)

### Key Metrics

The benchmark reports:
- **Wall time**: Total execution time
- **Pool size**: Number of Pareto-optimal candidates
- **Validation calls**: Total inference calls made
- **Avg probes/candidate**: Average instances probed per candidate (GEPA-MI)
- **Speedup**: Validation call reduction (baseline calls / MI calls)
- **Test accuracy**: Final accuracy on held-out test set

## When GEPA-MI Shows Speedup

GEPA-MI works best when:
- ✅ **Large validation set**: N ≥ 300-800
- ✅ **Expensive inference**: k ≥ 5 self-consistency
- ✅ **Per-instance time**: > 500ms

GEPA-MI won't show speedup when:
- ❌ Small validation set (N ≤ 100)
- ❌ Cheap inference (k=1, fast model)
- ❌ Per-instance time < 200ms

## Architecture

### Core Abstractions

All components implement clean interfaces:

- **StudentModel**: Execute inference on problems
- **TaskEvaluator**: Evaluate solutions against ground truth
- **TrajectoryFormatter**: Format execution traces
- **Candidate**: Prompt configuration with Pareto wins

### GSM8K Components

- **GSM8KFormatter**: Format math problem trajectories
- **GSM8KEvaluator**: Extract and compare numeric answers
- **OpenAIStudentModel**: Student model using OpenAI API

### Algorithms

**GEPA Baseline:**
1. Select parent from Pareto pool
2. Probe on minibatch
3. Mutate (add hint/variation)
4. Quick check on minibatch
5. **Full validation** on entire val set (expensive!)
6. Add to pool if Pareto-useful

**GEPA-MI:**
1-4. Same as baseline
5. **Selective validation** with MI guidance:
   - Seed Bayesian posterior from minibatch
   - Select uncertain instances (p ≈ 0.5)
   - Update posterior after each probe
   - Stop early when P(Z=1) crosses threshold
6. Add to pool if accepted

## Results Location

Results are saved to `results/experiment_{mode}_{timestamp}.txt` with:
- Configuration
- Baseline metrics
- GEPA-MI metrics
- Speedup analysis

## Citation

Based on the GEPA methodology from prompt optimization research, extended with Bayesian mutual information guidance for efficient validation.

## License

MIT
