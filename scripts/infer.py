#!/usr/bin/env python
# Run single-example hybrid Tree-LLM inference using a trained run directory.
#
# Andrew Kiruluta, UC Berkeley, May 2026

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from tree_llm_reasoner.config import load_config
from tree_llm_reasoner.llm_agents import build_llm_agent
from tree_llm_reasoner.orchestrator import AgentOutputs, combine_predictions, PolicyOrchestrator
from tree_llm_reasoner.tree_oracle import TreeOracle
import torch


def main():
    ap = argparse.ArgumentParser(description="Run single-example inference with the hybrid reasoner.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--context", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--run-dir", default=None, help="Directory containing tree_oracle.joblib and optional orchestrator_policy.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or cfg.run.output_dir)
    tree = TreeOracle.load(run_dir / "tree_oracle.joblib")
    llm = build_llm_agent(cfg)
    text = f"Context:\n{args.context}\n\nQuestion:\n{args.question}"
    tree_pred = tree.predict([text])
    tree_conf = tree.confidence([text])
    llm_pred, llm_conf, raw = llm.predict_batch([text])
    outputs = AgentOutputs(
        tree_pred=tree_pred,
        tree_conf=tree_conf,
        llm_pred=np.asarray(llm_pred),
        llm_conf=np.asarray(llm_conf),
        labels=None,
        texts=[text],
        llm_raw=raw,
    )
    policy = None
    if cfg.orchestrator.mode == "policy" and (run_dir / "orchestrator_policy.pt").exists():
        policy = PolicyOrchestrator(hidden=cfg.orchestrator.policy_hidden)
        policy.load_state_dict(torch.load(run_dir / "orchestrator_policy.pt", map_location="cpu"))
    elif cfg.orchestrator.mode == "policy":
        cfg.orchestrator.mode = "weighted_vote"
    final, choice = combine_predictions(outputs, cfg, policy)
    print({
        "tree_pred": "yes" if int(tree_pred[0]) else "no",
        "tree_conf": float(tree_conf[0]),
        "llm_pred": "yes" if int(llm_pred[0]) else "no",
        "llm_conf": float(llm_conf[0]),
        "chosen_agent": "llm" if int(choice[0]) == 1 else "tree",
        "final_pred": "yes" if int(final[0]) else "no",
        "llm_raw": raw[0],
    })


if __name__ == "__main__":
    main()
