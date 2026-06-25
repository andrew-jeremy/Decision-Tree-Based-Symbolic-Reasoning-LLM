from tree_llm_reasoner.config import load_config
from tree_llm_reasoner.data import load_splits
from tree_llm_reasoner.tree_oracle import TreeOracle


def test_load_sample_and_tree():
    cfg = load_config("configs/default.yaml")
    train, val, test = load_splits(cfg)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    tree = TreeOracle(kind="decision_tree", max_depth=3)
    tree.fit([e.text for e in train], [e.label for e in train])
    preds = tree.predict([e.text for e in test[:3]])
    assert len(preds) == 3
