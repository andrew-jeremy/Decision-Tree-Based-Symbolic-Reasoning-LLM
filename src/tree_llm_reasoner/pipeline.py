from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .config import load_config
from .data import load_splits, ReasoningExample
from .llm_agents import build_llm_agent
from .orchestrator import AgentOutputs, train_policy, combine_predictions, metrics_dict, write_trace
from .plots import plot_policy_curves, plot_agent_bars, save_metrics_json
from .tree_oracle import TreeOracle


def examples_text_labels(examples: list[ReasoningExample]) -> tuple[list[str], np.ndarray]:
    return [e.text for e in examples], np.asarray([e.label for e in examples], dtype=np.int64)


def collect_outputs(examples: list[ReasoningExample], tree: TreeOracle, llm_agent, batch_size: int) -> AgentOutputs:
    texts, labels = examples_text_labels(examples)
    tree_pred = tree.predict(texts).astype(np.int64)
    tree_conf = tree.confidence(texts).astype(np.float32)
    llm_pred_all, llm_conf_all, raw_all = [], [], []
    for i in range(0, len(texts), batch_size):
        p, c, r = llm_agent.predict_batch(texts[i:i + batch_size])
        llm_pred_all.extend(p); llm_conf_all.extend(c); raw_all.extend(r)
    return AgentOutputs(
        tree_pred=tree_pred,
        tree_conf=tree_conf,
        llm_pred=np.asarray(llm_pred_all, dtype=np.int64),
        llm_conf=np.asarray(llm_conf_all, dtype=np.float32),
        labels=labels,
        texts=texts,
        llm_raw=raw_all,
    )


def run_experiment(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ex, val_ex, test_ex = load_splits(cfg)
    print(f"Loaded splits: train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    tree = TreeOracle(kind=cfg.tree.kind, max_depth=cfg.tree.max_depth, n_estimators=cfg.tree.n_estimators, random_state=cfg.tree.random_state)
    train_texts, train_labels = examples_text_labels(train_ex)
    tree.fit(train_texts, train_labels.tolist())
    tree.save(out_dir / "tree_oracle.joblib")
    tree.save_rules(out_dir / "tree_rules.txt")
    if cfg.run.save_tree_plot:
        tree.save_plot(out_dir / "tree_oracle.png")

    llm_agent = build_llm_agent(cfg)

    print("Collecting validation agent outputs...")
    val_outputs = collect_outputs(val_ex, tree, llm_agent, cfg.llm.batch_size)
    print("Collecting test agent outputs...")
    test_outputs = collect_outputs(test_ex, tree, llm_agent, cfg.llm.batch_size)

    policy = None
    policy_losses, policy_accs = [], []
    if cfg.orchestrator.mode == "policy":
        policy, policy_losses, policy_accs = train_policy(
            val_outputs,
            hidden=cfg.orchestrator.policy_hidden,
            epochs=cfg.orchestrator.policy_epochs,
            lr=cfg.orchestrator.policy_lr,
            seed=cfg.orchestrator.seed,
        )
        torch.save(policy.state_dict(), out_dir / "orchestrator_policy.pt")
        if cfg.run.save_plots:
            plot_policy_curves(policy_losses, policy_accs, out_dir)

    val_final, val_choice = combine_predictions(val_outputs, cfg, policy)
    test_final, test_choice = combine_predictions(test_outputs, cfg, policy)

    write_trace(out_dir / "validation_trace.jsonl", val_outputs, val_final, val_choice)
    write_trace(out_dir / "test_trace.jsonl", test_outputs, test_final, test_choice)

    metrics = {
        "validation": {
            "tree": metrics_dict(val_outputs.labels, val_outputs.tree_pred),
            "llm": metrics_dict(val_outputs.labels, val_outputs.llm_pred),
            "hybrid": metrics_dict(val_outputs.labels, val_final),
            "override_rate": float(np.mean(val_choice == 0)) if cfg.orchestrator.mode != "tree_only" else 1.0,
            "disagreement_rate": float(np.mean(val_outputs.tree_pred != val_outputs.llm_pred)),
        },
        "test": {
            "tree": metrics_dict(test_outputs.labels, test_outputs.tree_pred),
            "llm": metrics_dict(test_outputs.labels, test_outputs.llm_pred),
            "hybrid": metrics_dict(test_outputs.labels, test_final),
            "override_rate": float(np.mean(test_choice == 0)) if cfg.orchestrator.mode != "tree_only" else 1.0,
            "disagreement_rate": float(np.mean(test_outputs.tree_pred != test_outputs.llm_pred)),
        },
        "policy_training": {"losses": policy_losses, "choose_agent_acc": policy_accs},
    }
    save_metrics_json(metrics, out_dir)
    if cfg.run.save_plots:
        plot_agent_bars(metrics["test"], out_dir)

    print("\nTest metrics")
    for name in ["tree", "llm", "hybrid"]:
        m = metrics["test"][name]
        print(f"  {name:>6}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")
    print(f"Artifacts written to: {out_dir}")
    return metrics
