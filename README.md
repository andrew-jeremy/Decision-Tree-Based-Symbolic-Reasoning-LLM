# Tree-LLM Reasoner

A runnable reference implementation of a hybrid symbolic-neural reasoning architecture: decision-tree oracle + LLM agent + trainable orchestrator. The implementation targets ProofWriter-style entailment data and includes training, validation, test evaluation, plots, rule export, and single-example inference.

## What is implemented

- Perception/feature extraction for ProofWriter-style logical examples.
- Decision tree or random forest symbolic oracle.
- Human-readable rule export and tree plot visualization.
- LLM agent abstraction:
  - `heuristic` offline stand-in for quick local tests.
  - `gemini` Google Gemini API agent with JSONL response cache.
- Orchestrators:
  - `tree_only`
  - `llm_only`
  - `veto_low_confidence`
  - `weighted_vote`
  - `policy` trainable PyTorch choose-agent controller.
- Metrics: accuracy, precision, recall, F1, confusion matrix, disagreement rate, override rate.
- Artifacts: `metrics.json`, `tree_rules.txt`, `tree_oracle.png`, `test_trace.jsonl`, policy curves, performance bars.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick local run with included sample data

```bash
python scripts/train_eval.py --config configs/default.yaml
```

Artifacts will be written to:

```text
outputs/sample_run/
```

## Full ProofWriter dataset options

### Option A: local JSONL

Put your ProofWriter split files here:

```text
data/proofwriter/train.jsonl
data/proofwriter/val.jsonl
data/proofwriter/test.jsonl
```

Each JSONL row can contain fields such as:

```json
{"context": "All cats are mammals. Felix is a cat.", "question": "Felix is a mammal", "answer": "yes"}
```

Then run:

```bash
python scripts/train_eval.py --config configs/gemini.yaml
```

### Option B: Hugging Face datasets

Edit `configs/default.yaml`:

```yaml
data:
  source: hf
  hf_name: tasksource/proofwriter
  hf_config: null
```

Then run:

```bash
python scripts/train_eval.py --config configs/default.yaml
```

Hugging Face dataset names/configs can vary, so adjust `hf_name` and `hf_config` to the ProofWriter variant you use.

## Gemini mode

Set your API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

Run:

```bash
python scripts/train_eval.py --config configs/gemini.yaml
```

Gemini responses are cached in `outputs/gemini_cache.jsonl` to avoid repeated calls.

## Single-example inference

After training:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --context "All cats are mammals. Felix is a cat." \
  --question "Felix is a mammal" \
  --run-dir outputs/sample_run
```

## Repository layout

```text
src/tree_llm_reasoner/
  config.py          YAML config loader
  data.py            ProofWriter/local/HF data loading and normalization
  features.py        symbolic feature extraction
  tree_oracle.py     decision tree/random forest oracle
  llm_agents.py      heuristic and Gemini LLM agents
  orchestrator.py    fixed and trainable orchestrators
  plots.py           training/evaluation plots
  pipeline.py        end-to-end train/eval pipeline
scripts/
  train_eval.py      train/evaluate pipeline
  infer.py           single-example inference
  make_sample_dataset.py
configs/
  default.yaml       offline sample run
  gemini.yaml        Gemini API run template
```

## Notes on benchmarking

The included sample data is only a smoke test. Use a full ProofWriter split for meaningful results. The repository logs all model decisions so you can analyze when the symbolic oracle corrects, vetoes, or disagrees with the LLM agent.
