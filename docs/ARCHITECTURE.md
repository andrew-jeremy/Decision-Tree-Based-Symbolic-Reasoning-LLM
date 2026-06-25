# Architecture Notes

This repository implements the hybrid architecture described in the uploaded paper, centered on five modules:

1. **Perception Agent** converts ProofWriter-style records into normalized text and symbolic features.
2. **Tree Oracle** trains a decision tree or random forest over symbolic features and exports human-readable rule traces.
3. **LLM Agent** provides abductive language reasoning through either a local heuristic stand-in or Google Gemini API.
4. **Central Orchestrator** combines tree and LLM outputs. It supports fixed rules, confidence vetoes, weighted voting, and a trainable PyTorch policy network.
5. **External Tool Interface** is represented by the LLM agent abstraction and cache layer; Gemini can be plugged in without changing the tree/orchestrator code.

The data flow is:

```text
raw ProofWriter example -> normalized text -> symbolic features -> tree oracle
                                       \-> LLM agent
(tree prediction, LLM prediction, confidences) -> orchestrator -> final answer + trace log
```

The trainable orchestrator policy receives:

```text
[tree_pred, tree_conf, llm_pred, llm_conf, disagreement, confidence_gap]
```

It learns to choose either the tree oracle or the LLM agent using validation labels as reward supervision. This can be replaced by online REINFORCE/PPO if the agent is deployed in an interactive environment.
