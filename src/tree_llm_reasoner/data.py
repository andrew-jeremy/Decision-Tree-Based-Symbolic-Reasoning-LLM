from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass
class ReasoningExample:
    uid: str
    context: str
    question: str
    label: int
    raw: dict

    @property
    def text(self) -> str:
        return f"Context:\n{self.context}\n\nQuestion:\n{self.question}".strip()


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    s = str(value).strip().lower()
    if s in {"yes", "true", "entailed", "entailment", "1", "supports", "correct"}:
        return 1
    if s in {"no", "false", "not entailed", "contradiction", "0", "unsupported", "incorrect"}:
        return 0
    if "yes" in s or "true" in s:
        return 1
    return 0


def _pick(raw: dict, keys: Iterable[str], default: str = "") -> str:
    for k in keys:
        if k in raw and raw[k] is not None:
            v = raw[k]
            if isinstance(v, (list, tuple)):
                return "\n".join(map(str, v))
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return str(v)
    return default


def parse_record(raw: dict, idx: int) -> ReasoningExample:
    uid = str(raw.get("id", raw.get("uid", idx)))
    # ProofWriter variants use fields like theory, facts/rules, context, question, hypothesis, answer.
    context = _pick(raw, ["context", "theory", "facts", "rules", "premises", "passage", "input"])
    question = _pick(raw, ["question", "hypothesis", "query", "goal", "conclusion"])
    if not context and "triples" in raw:
        context = json.dumps(raw["triples"], ensure_ascii=False)
    if not question and "statement" in raw:
        question = str(raw["statement"])
    label_raw = raw.get("answer", raw.get("label", raw.get("target", raw.get("truth_value", raw.get("is_correct", 0)))))
    return ReasoningExample(uid=uid, context=context, question=question, label=normalize_label(label_raw), raw=raw)


def load_jsonl(path: str, max_examples: int | None = None) -> list[ReasoningExample]:
    examples: list[ReasoningExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            examples.append(parse_record(json.loads(line), i))
    return examples


def save_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hf_dataset(name: str, config: str | None, split: str, max_examples: int | None = None) -> list[ReasoningExample]:
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [parse_record(dict(row), i) for i, row in enumerate(ds)]


def load_splits(cfg) -> tuple[list[ReasoningExample], list[ReasoningExample], list[ReasoningExample]]:
    if cfg.data.source == "sample":
        return (
            load_jsonl(cfg.data.train_path, cfg.data.max_examples),
            load_jsonl(cfg.data.val_path, cfg.data.max_examples),
            load_jsonl(cfg.data.test_path, cfg.data.max_examples),
        )
    if cfg.data.source == "local_jsonl":
        return (
            load_jsonl(cfg.data.train_path, cfg.data.max_examples),
            load_jsonl(cfg.data.val_path, cfg.data.max_examples),
            load_jsonl(cfg.data.test_path, cfg.data.max_examples),
        )
    if cfg.data.source == "hf":
        return (
            load_hf_dataset(cfg.data.hf_name, cfg.data.hf_config, "train", cfg.data.max_examples),
            load_hf_dataset(cfg.data.hf_name, cfg.data.hf_config, "validation", cfg.data.max_examples),
            load_hf_dataset(cfg.data.hf_name, cfg.data.hf_config, "test", cfg.data.max_examples),
        )
    raise ValueError(f"Unsupported data.source={cfg.data.source}")
