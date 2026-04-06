# ManyIH Benchmark

Benchmark for evaluating how well language models resolve conflicting instructions based on privilege hierarchies.

When instructions at different privilege levels conflict, models must follow the highest-privilege instruction. ManyIH tests this across two subsets:

- **Coding** (427 samples): Code generation tasks with conflicting style instructions at different privilege levels
- **Instruction-Following** (426 samples): Agentic scenarios with privilege-annotated constraints that conflict

## Installation and Setup

### Install the package

```bash
pip install manyih               # Core: OpenRouter + vLLM backends
pip install manyih[bedrock]      # + AWS Bedrock support (installs boto3)
pip install manyih[all]          # All backends
```

The core package supports querying models via [OpenRouter](https://openrouter.ai/) (cloud API aggregator) and local [vLLM](https://docs.vllm.ai/) servers. The `bedrock` extra adds support for calling models through AWS Bedrock.

To install from source instead:

```bash
git clone <repo-url>
cd manyih
pip install -e .                 # or pip install -e ".[all]"
```

### API keys

Set the appropriate environment variables for whichever backend(s) you plan to use:

**OpenRouter** — required for cloud models via OpenRouter (e.g., `openai/gpt-5.4`):

```bash
export OPENROUTER_API_KEY='your-key'
```

**AWS Bedrock** — required for `bedrock:` model strings (e.g., `bedrock:sonnet-4.6`). Configure standard AWS credentials so that `boto3` can authenticate:

```bash
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'       # or your preferred region
```

Alternatively, configure credentials via `aws configure` or an IAM instance profile. See the [boto3 credentials docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) for all options.

**vLLM** — no API key needed. Just point to your running server using the `vllm:host:port:model` format.

## Quick Start

### Evaluate on the coding subset

```bash
# OpenRouter
export OPENROUTER_API_KEY='your-key'
python -m manyih evaluate --subset coding --model openai/gpt-5.4

# AWS Bedrock
python -m manyih evaluate --subset coding --model bedrock:sonnet-4.6

# Local vLLM server
python -m manyih evaluate --subset coding --model vllm:localhost:8000:meta-llama/Llama-3.1-8B-Instruct
```

### Evaluate on the instruction-following subset

```bash
# Predict with one model, judge with another
python -m manyih evaluate --subset if \
    --model openai/gpt-5.4 \
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
| `<model_name>` | OpenRouter | `openai/gpt-5.4` |
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
TODO
```
