from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


@dataclass
class AgentOutputs:
    tree_pred: np.ndarray
    tree_conf: np.ndarray
    llm_pred: np.ndarray
    llm_conf: np.ndarray
    labels: np.ndarray | None = None
    texts: list[str] | None = None
    llm_raw: list[str] | None = None

    def feature_matrix(self) -> np.ndarray:
        disagree = (self.tree_pred != self.llm_pred).astype(np.float32)
        return np.vstack([
            self.tree_pred.astype(np.float32), self.tree_conf.astype(np.float32),
            self.llm_pred.astype(np.float32), self.llm_conf.astype(np.float32), disagree,
            np.abs(self.tree_conf - self.llm_conf).astype(np.float32),
        ]).T


class PolicyOrchestrator(nn.Module):
    """Small policy network that chooses whether to trust tree or LLM.

    Input features: [tree_pred, tree_conf, llm_pred, llm_conf, disagree, conf_gap]
    Output: probability of choosing LLM. A choice of 0 uses tree, 1 uses LLM.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_choice(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.tensor(features, dtype=torch.float32))
            return torch.argmax(logits, dim=1).cpu().numpy()


def desired_choice(outputs: AgentOutputs) -> np.ndarray:
    """Supervised target for policy learning: choose an agent that is correct.

    If both are correct or both wrong, prefer higher confidence. This is a practical
    offline proxy for reward learning and can be replaced by online REINFORCE/PPO.
    """
    assert outputs.labels is not None
    y = outputs.labels
    tree_ok = outputs.tree_pred == y
    llm_ok = outputs.llm_pred == y
    choice = np.zeros(len(y), dtype=np.int64)
    choice[llm_ok & ~tree_ok] = 1
    choice[tree_ok & ~llm_ok] = 0
    both_same_quality = tree_ok == llm_ok
    choice[both_same_quality] = (outputs.llm_conf[both_same_quality] >= outputs.tree_conf[both_same_quality]).astype(np.int64)
    return choice


def train_policy(outputs: AgentOutputs, hidden: int = 32, epochs: int = 40, lr: float = 1e-2, seed: int = 13) -> tuple[PolicyOrchestrator, list[float], list[float]]:
    torch.manual_seed(seed)
    model = PolicyOrchestrator(hidden=hidden)
    x = torch.tensor(outputs.feature_matrix(), dtype=torch.float32)
    target = torch.tensor(desired_choice(outputs), dtype=torch.long)
    opt = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == target).float().mean().item()
        losses.append(float(loss.item()))
        accs.append(acc)
    return model, losses, accs


def combine_predictions(outputs: AgentOutputs, cfg, policy: PolicyOrchestrator | None = None) -> tuple[np.ndarray, np.ndarray]:
    mode = cfg.orchestrator.mode
    n = len(outputs.tree_pred)
    if mode == "tree_only":
        choice = np.zeros(n, dtype=np.int64)
        return outputs.tree_pred.copy(), choice
    if mode == "llm_only":
        choice = np.ones(n, dtype=np.int64)
        return outputs.llm_pred.copy(), choice
    if mode == "veto_low_confidence":
        final = outputs.llm_pred.copy()
        choice = np.ones(n, dtype=np.int64)
        mask = (outputs.tree_pred != outputs.llm_pred) & (outputs.llm_conf < cfg.orchestrator.veto_confidence_threshold)
        final[mask] = outputs.tree_pred[mask]
        choice[mask] = 0
        return final, choice
    if mode == "weighted_vote":
        tree_score = cfg.orchestrator.tree_weight * outputs.tree_conf * (2 * outputs.tree_pred - 1)
        llm_score = cfg.orchestrator.llm_weight * outputs.llm_conf * (2 * outputs.llm_pred - 1)
        final = ((tree_score + llm_score) >= 0).astype(np.int64)
        choice = (np.abs(llm_score) >= np.abs(tree_score)).astype(np.int64)
        return final, choice
    if mode == "policy":
        if policy is None:
            raise ValueError("policy orchestrator requested but policy is None")
        choice = policy.predict_choice(outputs.feature_matrix())
        final = np.where(choice == 1, outputs.llm_pred, outputs.tree_pred)
        return final, choice
    raise ValueError(f"Unsupported orchestrator.mode={mode}")


def metrics_dict(labels: np.ndarray, preds: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def write_trace(path: str | Path, outputs: AgentOutputs, final: np.ndarray, choice: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(final)):
            row = {
                "idx": i,
                "label": None if outputs.labels is None else int(outputs.labels[i]),
                "tree_pred": int(outputs.tree_pred[i]),
                "tree_conf": float(outputs.tree_conf[i]),
                "llm_pred": int(outputs.llm_pred[i]),
                "llm_conf": float(outputs.llm_conf[i]),
                "final_pred": int(final[i]),
                "chosen_agent": "llm" if int(choice[i]) == 1 else "tree",
                "disagreement": bool(outputs.tree_pred[i] != outputs.llm_pred[i]),
                "text": None if outputs.texts is None else outputs.texts[i],
                "llm_raw": None if outputs.llm_raw is None else outputs.llm_raw[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
