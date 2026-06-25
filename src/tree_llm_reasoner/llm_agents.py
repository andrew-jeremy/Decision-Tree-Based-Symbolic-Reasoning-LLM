from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Protocol


class LLMAgent(Protocol):
    def predict_batch(self, texts: list[str]) -> tuple[list[int], list[float], list[str]]:
        """Return labels, confidences, and raw rationales."""


def parse_yes_no(text: str) -> tuple[int, float]:
    s = text.strip().lower()
    # Structured responses like: answer: yes confidence: 0.82
    conf_match = re.search(r"confidence\s*[:=]\s*([0-9.]+)", s)
    confidence = float(conf_match.group(1)) if conf_match else 0.60
    if re.search(r"\b(yes|true|entailed|supports)\b", s):
        return 1, min(max(confidence, 0.0), 1.0)
    if re.search(r"\b(no|false|not entailed|unsupported)\b", s):
        return 0, min(max(confidence, 0.0), 1.0)
    return (0, 0.50)


class HeuristicLLMAgent:
    """Offline stand-in for an LLM agent.

    This makes the repo runnable without API keys. It approximates an LLM answer
    from language cues, not from hidden benchmark labels.
    """

    def predict_batch(self, texts: list[str]) -> tuple[list[int], list[float], list[str]]:
        preds, confs, raws = [], [], []
        for text in texts:
            lower = text.lower()
            question = lower.split("question:")[-1]
            pos = sum(w in lower for w in ["therefore", "because", "if", "then", "all"])
            neg = sum(w in question.split() for w in ["not", "no", "never", "false"])
            pred = int(pos >= neg)
            conf = 0.55 + min(0.35, 0.05 * abs(pos - neg))
            preds.append(pred)
            confs.append(conf)
            raws.append(f"heuristic answer={'yes' if pred else 'no'} confidence={conf:.2f}")
        return preds, confs, raws


class JsonlCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.store[row["key"]] = row

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> dict | None:
        return self.store.get(self.key(text))

    def put(self, text: str, raw: str, pred: int, confidence: float) -> None:
        row = {"key": self.key(text), "raw": raw, "pred": int(pred), "confidence": float(confidence)}
        self.store[row["key"]] = row
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class GeminiLLMAgent:
    """Google Gemini API LLM agent with JSONL cache.

    Requires: export GEMINI_API_KEY="..."
    """

    def __init__(self, model_name: str, api_key_env: str, cache_path: str, temperature: float = 0.0, max_output_tokens: int = 32, sleep_s: float = 0.1):
        import google.generativeai as genai
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set environment variable {api_key_env}.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache = JsonlCache(cache_path)
        self.sleep_s = sleep_s

    def _prompt(self, text: str) -> str:
        return (
            "You are solving a ProofWriter-style logical entailment task.\n"
            "Return exactly two lines:\n"
            "answer: yes|no\nconfidence: number between 0 and 1\n\n"
            f"{text}\n"
        )

    def predict_batch(self, texts: list[str]) -> tuple[list[int], list[float], list[str]]:
        preds, confs, raws = [], [], []
        for text in texts:
            cached = self.cache.get(text)
            if cached:
                preds.append(int(cached["pred"])); confs.append(float(cached["confidence"])); raws.append(cached["raw"])
                continue
            response = self.model.generate_content(
                self._prompt(text),
                generation_config={"temperature": self.temperature, "max_output_tokens": self.max_output_tokens},
            )
            raw = getattr(response, "text", "") or ""
            pred, conf = parse_yes_no(raw)
            self.cache.put(text, raw, pred, conf)
            preds.append(pred); confs.append(conf); raws.append(raw)
            time.sleep(self.sleep_s)
        return preds, confs, raws


def build_llm_agent(cfg) -> LLMAgent:
    if cfg.llm.provider == "heuristic":
        return HeuristicLLMAgent()
    if cfg.llm.provider == "gemini":
        return GeminiLLMAgent(
            model_name=cfg.llm.gemini_model,
            api_key_env=cfg.llm.api_key_env,
            cache_path=cfg.llm.cache_path,
            temperature=cfg.llm.temperature,
            max_output_tokens=cfg.llm.max_output_tokens,
        )
    raise ValueError(f"Unsupported llm.provider={cfg.llm.provider}")
