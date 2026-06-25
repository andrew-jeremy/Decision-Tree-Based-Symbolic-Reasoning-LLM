# AI Agent Instructions for Tree-LLM Reasoner

## Purpose
This repository implements a hybrid symbolic/neural reasoner for ProofWriter-style entailment examples. The code is organized around:

- `src/tree_llm_reasoner/`: core package
- `scripts/train_eval.py`: train/evaluate pipeline entrypoint
- `scripts/infer.py`: single-example inference
- `configs/`: YAML configuration templates
- `data/`: sample data and local dataset paths

Use this file to understand the repo structure, common workflows, and conventions without reinventing the architecture.

## Recommended workflows

1. Install and activate a Python environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -e .`

2. Run the default sample experiment:
   - `python scripts/train_eval.py --config configs/default.yaml`

3. Run a single example after training:
   - `python scripts/infer.py --config configs/default.yaml --context "..." --question "..." --run-dir outputs/sample_run`

4. Run tests:
   - `pytest`

## Key concepts and conventions

### Config-driven execution

The repository uses YAML configs loaded by `src/tree_llm_reasoner/config.py`.

Important config sections:

- `data`: sample, Hugging Face dataset, or local JSONL source
- `tree`: decision tree / random forest hyperparameters
- `llm`: provider settings, including `heuristic` and `gemini`
- `orchestrator`: mode and weights for combining tree and LLM outputs
- `run.output_dir`: output directory for artifacts

### Supported orchestrator modes

- `tree_only`
- `llm_only`
- `veto_low_confidence`
- `weighted_vote`
- `policy`

When `policy` is selected, a trainable PyTorch policy network may be saved as `orchestrator_policy.pt`.

### LLM providers

- `heuristic`: local stand-in for fast testing
- `gemini`: Google Gemini API agent

If using `gemini`, the API key must be provided via `GEMINI_API_KEY` and responses are cached in `outputs/gemini_cache.jsonl`.

### Data and outputs

- Sample data lives under `data/sample/`.
- Full ProofWriter runs may use `data/proofwriter/*.jsonl` or Hugging Face datasets.
- Artifacts are written to `outputs/<run_name>/`.

## Important files

- `src/tree_llm_reasoner/config.py`: YAML config loader
- `src/tree_llm_reasoner/data.py`: data loading and normalization
- `src/tree_llm_reasoner/features.py`: symbolic feature extraction
- `src/tree_llm_reasoner/tree_oracle.py`: decision tree / random forest oracle
- `src/tree_llm_reasoner/llm_agents.py`: heuristic and Gemini agents
- `src/tree_llm_reasoner/orchestrator.py`: combine predictions and policy logic
- `src/tree_llm_reasoner/pipeline.py`: end-to-end experiment orchestration
- `docs/ARCHITECTURE.md`: architecture overview

## Notes for AI agents

- Prefer updating configuration rather than creating new CLI wrappers.
- Preserve existing output layout and avoid breaking `run.output_dir` semantics.
- Avoid inventing new dataset formats; use sample JSONL, local JSONL or Hugging Face dataset config patterns already present.
- Keep changes consistent with the hybrid decision-tree + LLM orchestrator architecture.

## References

- `README.md`
- `docs/ARCHITECTURE.md`
- `configs/default.yaml`
- `configs/gemini.yaml`
