from __future__ import annotations

import re
import numpy as np

FEATURE_NAMES = [
    "n_tokens", "n_sentences", "n_rules_if", "n_then", "n_not", "n_all", "n_some",
    "n_because", "n_therefore", "n_entities", "has_query_neg", "rule_density",
]

ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_]*\b")


def featurize_text(text: str) -> np.ndarray:
    lower = text.lower()
    tokens = re.findall(r"\w+", lower)
    n_tokens = max(len(tokens), 1)
    n_sentences = max(len(re.findall(r"[.!?]\s+|\n", text)), 1)
    n_rules_if = lower.count(" if ") + int(lower.startswith("if "))
    n_then = lower.count(" then ") + lower.count("therefore")
    n_not = tokens.count("not") + tokens.count("no") + tokens.count("never")
    n_all = tokens.count("all") + tokens.count("every")
    n_some = tokens.count("some") + tokens.count("exists")
    n_because = tokens.count("because")
    n_therefore = tokens.count("therefore") + tokens.count("hence")
    n_entities = len(set(ENTITY_RE.findall(text)))
    # rough query negation: check final question chunk
    question = text.split("Question:")[-1].lower()
    has_query_neg = int(any(w in question.split() for w in ["not", "no", "never", "false"]))
    rule_density = (n_rules_if + n_then + n_because + n_therefore) / n_tokens
    return np.array([
        n_tokens, n_sentences, n_rules_if, n_then, n_not, n_all, n_some,
        n_because, n_therefore, n_entities, has_query_neg, rule_density,
    ], dtype=np.float32)


def featurize_many(texts: list[str]) -> np.ndarray:
    return np.vstack([featurize_text(t) for t in texts])
