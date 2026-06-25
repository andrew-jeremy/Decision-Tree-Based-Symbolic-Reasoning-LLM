from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    source: str = "sample"  # sample | local_jsonl | hf
    train_path: str = "data/sample/train.jsonl"
    val_path: str = "data/sample/val.jsonl"
    test_path: str = "data/sample/test.jsonl"
    hf_name: str = "tasksource/proofwriter"
    hf_config: str | None = None
    max_examples: int | None = None


@dataclass
class TreeConfig:
    kind: str = "random_forest"  # decision_tree | random_forest
    max_depth: int = 8
    n_estimators: int = 200
    random_state: int = 13


@dataclass
class LLMConfig:
    provider: str = "heuristic"  # heuristic | gemini
    gemini_model: str = "models/gemini-1.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    batch_size: int = 8
    cache_path: str = "outputs/gemini_cache.jsonl"
    temperature: float = 0.0
    max_output_tokens: int = 32


@dataclass
class OrchestratorConfig:
    mode: str = "policy"  # policy | veto_low_confidence | weighted_vote | tree_only | llm_only
    veto_confidence_threshold: float = 0.70
    llm_weight: float = 0.65
    tree_weight: float = 0.35
    policy_epochs: int = 40
    policy_lr: float = 1e-2
    policy_hidden: int = 32
    seed: int = 13


@dataclass
class RunConfig:
    output_dir: str = "outputs/run"
    save_plots: bool = True
    save_tree_plot: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _merge_dataclass(cfg, raw)
    Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg
