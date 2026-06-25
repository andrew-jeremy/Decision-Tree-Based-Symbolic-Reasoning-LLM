from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

from .features import FEATURE_NAMES, featurize_many


class TreeOracle:
    """Callable symbolic oracle using a decision tree or random forest.

    The oracle is intentionally interpretable: every prediction is accompanied by
    a probability estimate and, when the base model is a single tree, a rule trace.
    """

    def __init__(self, kind: str = "random_forest", max_depth: int = 8, n_estimators: int = 200, random_state: int = 13):
        self.kind = kind
        if kind == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        else:
            raise ValueError("kind must be 'decision_tree' or 'random_forest'")

    def fit(self, texts: list[str], labels: list[int]) -> "TreeOracle":
        x = featurize_many(texts)
        self.model.fit(x, np.asarray(labels))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = featurize_many(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        preds = self.model.predict(x)
        return np.vstack([1 - preds, preds]).T

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)

    def confidence(self, texts: list[str]) -> np.ndarray:
        return np.max(self.predict_proba(texts), axis=1)

    def rules_text(self) -> str:
        if isinstance(self.model, DecisionTreeClassifier):
            return export_text(self.model, feature_names=FEATURE_NAMES)
        # Forest summary: export first few estimators for auditability.
        chunks = ["RandomForest symbolic oracle. Showing first 3 tree rules.\n"]
        for i, estimator in enumerate(self.model.estimators_[:3]):
            chunks.append(f"\n--- estimator_{i} ---\n")
            chunks.append(export_text(estimator, feature_names=FEATURE_NAMES))
        return "".join(chunks)

    def save_rules(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.rules_text(), encoding="utf-8")

    def save_plot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(22, 10))
        model = self.model if isinstance(self.model, DecisionTreeClassifier) else self.model.estimators_[0]
        plot_tree(model, feature_names=FEATURE_NAMES, class_names=["no", "yes"], filled=True, rounded=True, max_depth=4)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TreeOracle":
        return joblib.load(path)
