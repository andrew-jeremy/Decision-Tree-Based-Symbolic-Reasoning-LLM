#!/usr/bin/env python
# Train and evaluate the hybrid Tree-LLM orchestrator on ProofWriter-style data.
#
# Andrew Kiruluta, UC Berkeley

from __future__ import annotations
import argparse
from tree_llm_reasoner.pipeline import run_experiment


def main():
    ap = argparse.ArgumentParser(description="Train/evaluate tree + LLM orchestrator on ProofWriter-style data.")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = ap.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
