# ManyIH Benchmark

Benchmark for evaluating how well language models resolve conflicting instructions based on privilege hierarchies.

When instructions at different privilege levels conflict, models must follow the highest-privilege instruction. ManyIH tests this across two subsets:

- **Coding** (427 samples): Code generation tasks with conflicting style instructions at different privilege levels
- **Instruction-Following** (426 samples): Agentic scenarios with privilege-annotated constraints that conflict

## Installation

```bash
pip install manyih               # Core (OpenRouter + vLLM)
pip install manyih[bedrock]      # + AWS Bedrock support
pip install manyih[all]          # All backends
```

Or install from source:

```bash
git clone <repo-url>
cd manyih
pip install -e .
```

## Quick Start

### Evaluate on the coding subset

```bash
# OpenRouter
export OPENROUTER_API_KEY='your-key'
python -m manyih evaluate --subset coding --model anthropic/claude-3.5-sonnet

# AWS Bedrock
python -m manyih evaluate --subset coding --model bedrock:sonnet-4.6

# Local vLLM server
python -m manyih evaluate --subset coding --model vllm:localhost:8000:meta-llama/Llama-3.1-8B-Instruct
```

### Evaluate on the instruction-following subset

```bash
# Predict with one model, judge with another
python -m manyih evaluate --subset if \
    --model anthropic/claude-3.5-sonnet \
    --judge-model bedrock:sonnet-4.6

# Use the same model for both
python -m manyih evaluate --subset if --model bedrock:opus-4.6
```

### Evaluate on both subsets

```bash
python -m manyih evaluate --subset all --model bedrock:sonnet-4.6
```

### View benchmark data

```bash
python -m manyih view --subset coding --output coding_viewer.html
python -m manyih view --subset if --output if_viewer.html
```

## Model String Formats

| Format | Backend | Example |
|--------|---------|---------|
| `<model_name>` | OpenRouter | `anthropic/claude-3.5-sonnet` |
| `bedrock:<alias>` | AWS Bedrock | `bedrock:sonnet-4.6`, `bedrock:opus-4.6` |
| `bedrock:<region>:<model_id>` | AWS Bedrock (full) | `bedrock:us-east-1:us.anthropic.claude-opus-4-6-v1` |
| `vllm:<host>:<port>:<model>` | Local vLLM | `vllm:localhost:8000:Qwen/Qwen3.5-32B` |

## Subset-Specific Options

### Coding subset

```bash
python -m manyih.coding.evaluator --help
```

Key options:
- `--max_samples N` — evaluate only first N samples
- `--concurrency N` — parallel API calls (default: 50)
- `--no-cache` — disable response caching
- `--resume` — resume from existing output file
- `--analyze RESULTS_FILE` — re-analyze an existing results file

### Instruction-following subset

```bash
python -m manyih.instruction_following.evaluator --help
```

Key options:
- `--model MODEL` — model for generating predictions
- `--judge-model MODEL` — model for LLM-judge evaluation (default: same as `--model`)
- `--predictions-file FILE` — skip prediction, evaluate from existing file
- `--skip-eval` — only generate predictions, skip evaluation
- `--max-samples N` — limit number of samples
- `--num-workers N` — parallel workers (default: 8)

## Output Format

### Coding subset

Results are saved as JSON with:
- `stats`: overall pass rates (test, style, overall)
- `results`: per-sample evaluation with `evaluation.overall_passed`

### Instruction-following subset

Results are saved as:
- `results.json`: per-sample constraint scores
- `accuracy.json`: ISR (Instruction Success Rate) and CSR (Constraint Success Rate)

## Security Note

The coding subset evaluation executes model-generated Python code using `exec()` to check functional correctness. **Run evaluations in a sandboxed environment** (container, VM) when evaluating untrusted models.

## Citation

If you use ManyIH in your research, please cite:

```bibtex
@article{manyih2026,
    title={ManyIH: Benchmarking Instruction Hierarchy Resolution in Language Models},
    year={2026}
}
```
