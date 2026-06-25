#!/usr/bin/env python
# Generate small sample ProofWriter-style train/val/test datasets for local smoke tests.
#
# Andrew Kiruluta, UC Berkeley

from __future__ import annotations
import random
from pathlib import Path
from tree_llm_reasoner.data import save_jsonl

RULES = [
    ("All cats are mammals. All mammals are animals.", "cats are animals", 1),
    ("All birds are animals. No reptiles are birds.", "reptiles are birds", 0),
    ("If someone is kind then they are trusted. Alice is kind.", "Alice is trusted", 1),
    ("If an object is red then it is visible. The box is blue.", "The box is visible", 0),
    ("All doctors are trained. Mira is a doctor.", "Mira is trained", 1),
    ("No cold things are hot. Ice is cold.", "Ice is hot", 0),
    ("If a sample reacts then it is unstable. Sample A reacts.", "Sample A is unstable", 1),
    ("All squares are shapes. Circles are shapes.", "Circles are squares", 0),
    ("If the alarm rings then evacuate. The alarm rings.", "evacuate", 1),
    ("If water is frozen then it is solid. This water is not frozen.", "This water is solid", 0),
]

def build(n: int, seed: int):
    random.seed(seed)
    rows = []
    for i in range(n):
        c, q, y = random.choice(RULES)
        rows.append({"id": f"sample-{seed}-{i}", "context": c, "question": q, "answer": "yes" if y else "no"})
    return rows

if __name__ == "__main__":
    root = Path("data/sample")
    save_jsonl(str(root / "train.jsonl"), build(80, 1))
    save_jsonl(str(root / "val.jsonl"), build(30, 2))
    save_jsonl(str(root / "test.jsonl"), build(30, 3))
    print("Wrote sample data to data/sample")
