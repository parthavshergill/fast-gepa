# GEPA, GEPA-MI & Self-Reflection: Prompt Optimization

Implementation of three prompt optimization methods for GSM8K math problems:
- **GEPA Baseline**: Genetic-Pareto with full validation
- **GEPA-MI**: Bayesian mutual information-guided selective validation
- **Self-Reflection**: Simple iterative refinement baseline

## Overview

**GEPA** is a prompt optimizer that maintains a Pareto frontier of candidates, using mutation to explore prompt variations. **GEPA-MI** extends this with Bayesian inference to reduce validation cost by 5-50x through selective probing. **Self-Reflection** provides a simpler baseline using iterative refinement without validation.

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
│   ├── reflection.py        # ReflectionEngine for prompt improvement
│   ├── self_reflection.py   # Self-Reflection algorithm
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

The implementation supports both **Google Gemini** (default) and **OpenAI** models.

#### Option A: Using Gemini (Recommended - Free Tier Available)

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Or set environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### Option B: Using OpenAI

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

**Using Gemini (default):**
```bash
python main.py --mode quick
```

**Using OpenAI:**
```bash
python main.py --mode quick --provider openai
```

**Configuration:**
- Probe: 50, Val: 100, Test: 50
- Time budget: 60s per method
- Self-consistency k=2
- **Note:** Won't show significant speedup (val set too small)

### Full Mode (Complete Benchmark)

Run the full benchmark with proper settings for speedup:

**Using Gemini:**
```bash
python main.py --mode full
```

**Using OpenAI:**
```bash
python main.py --mode full --provider openai
```

**Configuration:**
- Probe: 200, Val: 600, Test: 200
- Time budget: 600s (10 min) per method
- Self-consistency k=8
- **Expected speedup:** 5-15x

### Custom Options

**Specify a different model:**
```bash
# Gemini models
python main.py --mode full --model gemini-1.5-pro
python main.py --mode full --model gemini-2.0-flash-exp

# OpenAI models
python main.py --mode full --provider openai --model gpt-4o-mini
python main.py --mode full --provider openai --model gpt-4o

# With custom seed
python main.py --mode full --seed 42
```

### Available Models

**Gemini Models (Default - Free Tier):**
- `gemini-1.5-flash` (default) - Fast and efficient
- `gemini-1.5-pro` - More capable, slower
- `gemini-2.0-flash-exp` - Experimental, latest features

**OpenAI Models (Paid):**
- `gpt-4o-mini` (default for OpenAI) - Cost-effective
- `gpt-4o` - Most capable
- `gpt-4-turbo` - Balanced performance

**Cost Comparison (Approximate):**
- **Gemini 1.5 Flash**: FREE (60 RPM, 1M TPM limit)
- **GPT-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **GPT-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

For this experiment with ~1000-1500 total inference calls (quick mode) or ~10000-15000 calls (full mode), Gemini is free while OpenAI would cost $5-50 depending on the model.

## Expected Results

The benchmark runs all three methods and compares their performance.

### Quick Mode
- **Purpose:** Verify implementation works
- **GEPA-MI speedup:** ~2-3x (val set too small for real speedup)
- **Runtime:** ~3-5 minutes total (all three methods)

### Full Mode
- **Purpose:** Demonstrate real speedup and compare approaches
- **GEPA-MI speedup:** 5-15x over GEPA Baseline
- **Self-Reflection:** Cheaper per iteration (no validation), may overfit
- **Runtime:** ~30-45 minutes total (depends on API speed)

### Key Metrics

The benchmark reports for each method:

**GEPA Baseline:**
- Wall time, pool size, validation calls
- Test accuracy on held-out set

**GEPA-MI:**
- Wall time, avg probes/candidate
- Speedup vs baseline
- Test accuracy

**Self-Reflection:**
- Wall time, iterations completed
- Total inference calls (no validation)
- Test accuracy

### Method Comparison

**GEPA Baseline:**
- Full validation (expensive but thorough)
- Pareto pool diversity

**GEPA-MI:**
- Selective validation (5-15x cheaper)
- Same pool diversity as baseline

**Self-Reflection:**
- No validation (cheapest per iteration)
- Single trajectory (no pool)
- Risk of overfitting to probe set

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

**Self-Reflection:**
1. Sample batch from probe set
2. Evaluate with self-consistency
3. Collect failed trajectories
4. Use ReflectionEngine to analyze failures:
   - Format failures with context
   - Ask LLM to suggest improvements
   - Parse structured suggestions (ANALYSIS/IMPROVEMENT/TARGET)
   - Apply mutation to prompt
5. Track best prompt by probe accuracy
6. Repeat until time budget exhausted

## Results Location

Results are saved to `results/experiment_{mode}_{timestamp}.txt` with:
- Configuration (model, seed, parameters)
- GEPA Baseline metrics
- GEPA-MI metrics and speedup
- Self-Reflection metrics
- Method comparison (test accuracy, inference calls)

## Citation

Based on the GEPA methodology from prompt optimization research, extended with Bayesian mutual information guidance for efficient validation.

## License

MIT
