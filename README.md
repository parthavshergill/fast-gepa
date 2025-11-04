# GEPA, GEPA-MI & Self-Reflection

Implementation and comparison of three prompt optimization methods for GSM8K math problems:
- **GEPA Baseline**: Genetic-Pareto with full validation
- **GEPA-MI**: Bayesian mutual information-guided selective validation (5-15x speedup)
- **Self-Reflection**: Iterative refinement baseline

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

**Gemini (default, free):**
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

**OpenAI (alternative):**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

### Quick Mode
Test that everything works (~5 min):
```bash
python main.py --mode quick
```

### Full Mode
Run complete benchmark with 5-15x speedup demonstration (~30-45 min):
```bash
python main.py --mode full
```

### Options
```bash
# Use OpenAI instead of Gemini
python main.py --mode full --provider openai

# Specify model
python main.py --mode full --model gemini-1.5-pro
python main.py --mode full --provider openai --model gpt-4o-mini

# Set seed
python main.py --mode full --seed 42
```

## Results

Results are saved to `results/experiment_{mode}_{timestamp}.txt` with metrics for all three methods:
- Test accuracy
- Wall time
- Validation calls
- GEPA-MI speedup over baseline

## License

MIT
