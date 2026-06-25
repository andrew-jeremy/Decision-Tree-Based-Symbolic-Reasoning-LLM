from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_policy_curves(losses: list[float], accs: list[float], out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="policy train loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Orchestrator Policy Training Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "policy_loss.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(accs, label="policy choose-agent accuracy")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Orchestrator Policy Training Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "policy_accuracy.png", dpi=180); plt.close()


def plot_agent_bars(metrics: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = [k for k in metrics if k in {"tree", "llm", "hybrid"}]
    acc = [metrics[k]["accuracy"] for k in names]
    f1 = [metrics[k]["f1"] for k in names]
    x = np.arange(len(names)); width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, acc, width, label="accuracy")
    plt.bar(x + width/2, f1, width, label="f1")
    plt.xticks(x, names); plt.ylim(0, 1); plt.title("Agent vs Hybrid Performance")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "performance_bars.png", dpi=180); plt.close()


def save_metrics_json(metrics: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
