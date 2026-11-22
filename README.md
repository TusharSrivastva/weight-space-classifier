# Weight-Space Programming on a Small Classifier

Can an LLM write or edit a neural network's weights directly, with no gradients, and does it need to know how the data works to do it?

## Weight-space programming

*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL; blog post, Nov 2026) claims frontier LLMs can read and write network weights directly. Given the rules and source code of MinAtar games, but no data, an LLM wrote policy weights in one reply that beat a PPO-trained baseline. [Post](https://rikiyat.github.io/blog/weight-space-programming/) · [Logs](https://github.com/RikiyaT/lm_understand-logs)

## Objective

Test whether this carries over to a small tabular classifier, both by **writing** all the weights from scratch and by **editing** a model trained with gradient descent.

## Hypothesis

Weight-space programming carries knowledge into weights. An LLM that knows how the data works can compile it into a working network without seeing data; one that doesn't, can't.

## Methodology

- **Data:** synthetic, in an invented domain (whether batches of fictional "velmorite" crystals fail a stress test), so the model cannot have seen it.
  - 569 rows and 30 features: 10 true properties, each measured 3 ways with noise.
  - The label comes from a rule with linear terms, a threshold and an interaction, plus noise.
  - Stratified 60/20/20 split.
- **Model:** MLP 30 → 16 (ReLU) → 1 (sigmoid), 513 weights.
- **LLM:** Claude Opus 5.5 (effort high) through headless Claude Code (`claude -p`). Each call is isolated: no tools, no repository context, one-line system prompt. Weights go in and out as JSON.
- **Two conditions:** every LLM arm runs with the **rule known** (the exact rule, `wsp/data.py:RULE`, in the prompt) and **rule not known** (task and feature names only).

| Arm | What happens |
|---|---|
| 1. GD | Adam + early stopping on validation loss (shared by both conditions) |
| 2. AI from scratch | The LLM writes all 513 weights in one reply, with **no data and no feedback** |
| 3. GD → AI tweak | The LLM edits the GD weights over 10 rounds, one validation forward pass per round, seeing validation results only |

- **Temperature scaling (TS):** each final model is also reported with one temperature fitted on validation, to separate real gains from plain calibration.
- **Test set:** used once, at the end, for every model.
- **Seeds:** 3 GD seeds; 3 trials of arm 2; one arm-3 run per GD seed.

## Results (test set, mean ± std over 3)

| | Accuracy | AUC | Log loss | Log loss + TS |
|---|---|---|---|---|
| Always predict "passes" | 0.667 | 0.500 | – | – |
| Logistic regression | 0.746 | 0.764 | 0.602 | – |
| 1. GD | 0.792 ± 0.018 | 0.814 ± 0.014 | 0.628 ± 0.071 | 0.547 ± 0.037 |
| **Rule known** | | | | |
| 2. AI from scratch | 0.930 ± 0.000 | 0.983 ± 0.001 | 0.212 ± 0.044 | 0.187 ± 0.016 |
| 3. GD → AI tweak | **0.924 ± 0.005** | **0.986 ± 0.002** | **0.163 ± 0.009** | 0.164 ± 0.010 |
| **Rule not known** | | | | |
| 2. AI from scratch | 0.646 ± 0.018 | 0.722 ± 0.018 | 0.773 ± 0.010 | 0.650 ± 0.010 |
| 3. GD → AI tweak | 0.798 ± 0.009 | 0.813 ± 0.009 | 0.549 ± 0.026 | 0.554 ± 0.035 |

**What arm 3 did to each GD model** (test set, same seed):

| Seed | Rule known: accuracy | Rule known: log loss | Rule not known: accuracy | Rule not known: log loss vs GD + TS |
|---|---|---|---|---|
| 0 | 0.772 → 0.921 | 0.694 → 0.174 | 0.772 → 0.789 | −1.8% |
| 1 | 0.798 → 0.930 | 0.553 → 0.156 | 0.798 → 0.798 | +2.0% |
| 2 | 0.807 → 0.921 | 0.636 → 0.160 | 0.807 → 0.807 | +0.8% |

- **Rule known, from scratch:** the LLM compiled the rule into the network (linear units, an `anneal_temp` hinge, and a piecewise-linear approximation of the interaction). With no data, it beat GD: 0.93 vs 0.79 accuracy.
- **Rule known, tweak:** in every seed the LLM replaced the GD network with the rule network, then refined it on validation. It produced the best models in the study.
- **Rule not known, from scratch:** below always guessing "passes". Guessing from feature names wasn't enough.
- **Rule not known, tweak:** temperature scaling and small shrinkage. No accuracy or AUC gain, and no log-loss gain beyond what one fitted temperature gives.

## Conclusion

Weight-space programming is a way to put knowledge into a network, not a way to learn from data. When the LLM knew how the data worked, it wrote weights that beat gradient descent on a small dataset, in a single reply. When it didn't, it could do no more than standard calibration.

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter lab weight_space_classifier.ipynb
```

Every LLM prompt and reply is saved in `logs/` (rule known) and `logs/norule/` (rule not known). `wsp/llm.py` replays a saved reply instead of calling the model, so the notebook re-runs fully offline and every number can be checked against the raw replies. Delete a log file to make a fresh call (needs the `claude` CLI, logged in).

## Layout

- `weight_space_classifier.ipynb`: the experiment and results
- `wsp/data.py`: synthetic data generator, task description and the rule
- `wsp/model.py`: numpy forward pass, metrics, temperature scaling, weights ↔ JSON
- `wsp/gd.py`: arm 1
- `wsp/arm2.py`: arm 2 (`rule=True/False`)
- `wsp/arm3.py`: arm 3 (`rule=True/False`)
- `wsp/llm.py`: the LLM call, with logging and replay
- `results/gd/`: trained GD weights
- `logs/`: every LLM call
