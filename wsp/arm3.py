"""Arm 3: the LLM edits GD-trained weights over rounds, seeing validation results only.
It is also given the exact data-generating rule (wsp.data.RULE); rule=False is
the control without it, logged under logs/norule/.

One round = one LLM call + one forward pass on the validation set. Each call is
stateless: the prompt carries the best weights so far plus a table of all
previous attempts. The LLM returns a list of edits as JSON (no code). The
kept model is the one with the lowest validation loss so far.
"""
import numpy as np

from .data import RULE, TASK
from .llm import call_llm, extract_json
from .model import metrics, weights_to_json

TOTAL_ROUNDS = 10


def _fmt(m: dict) -> str:
    return f"loss={m['loss']:.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}"


def build_prompt(splits, gd_run, best_w, best_val, history, rnd, rule: bool = True) -> str:
    feats = "\n".join(f"{i:2d}. {n}" for i, n in enumerate(splits.feature_names))
    rows = "\n".join(
        f"| {h['round']} | {h['n_changed']} | {_fmt(h['val']) if h['val'] else 'invalid reply: ' + h['error']} | {h['note']} |"
        for h in history
    ) or "| - | - | (no attempts yet) | - |"
    given = "You cannot run code or see data, but you are given the exact rule that generated the data." if rule else "You cannot run code or see data."
    rule_block = f"\n\n{RULE}" if rule else ""
    return f"""You are improving a trained neural network classifier by editing its weights directly. {given} Each round you propose edits; we apply them to the current best weights, run one forward pass on the validation set and report the metrics. This is round {rnd} of {TOTAL_ROUNDS}. The test set is held out and never shown.

{TASK} Validation set: 114 samples.

Inputs: 30 features in this order, each z-scored with training-set statistics:
{feats}{rule_block}

Architecture: h = relu(x @ W1 + b1), p = sigmoid(h @ W2 + b2)
Shapes: W1 30x16 (row i = feature i, column j = hidden unit j), b1 16, W2 16x1, b2 1. Threshold p >= 0.5.

The starting weights were trained with Adam and early stopping on validation loss (best epoch {gd_run['best_epoch']}).
Starting train metrics: {_fmt(gd_run['train'])}
Starting validation metrics: {_fmt(gd_run['val'])}

Goal: lower validation loss (ties broken by accuracy). The weights below are the best so far; validation: {_fmt(best_val)}.

```json
{weights_to_json(best_w)}
```

Previous attempts (each applied to the best weights at that time):
| round | weights changed | validation result | note |
|---|---|---|---|
{rows}

Reply with a short explanation, then one ```json block of the form
{{"note": "<one line describing the change>", "edits": [{{"param": "W1", "index": [i, j], "value": 0.123}}, {{"param": "b2", "index": [0], "value": -0.5}}]}}
Each edit sets one weight to a new absolute value. Plain numbers only."""


def apply_edits(w: dict, edits: list) -> dict:
    new = {k: v.copy() for k, v in w.items()}
    for e in edits:
        new[e["param"]][tuple(e["index"])] = float(e["value"])
    return new


def run_arm3(splits, gd_run, rounds: int = TOTAL_ROUNDS, rule: bool = True) -> dict:
    seed = gd_run["seed"]
    best_w, best_val = gd_run["weights"], gd_run["val"]
    history, best_curve, cost, wall, out_tok = [], [best_val["loss"]], 0.0, 0.0, 0

    for rnd in range(1, rounds + 1):
        prompt = build_prompt(splits, gd_run, best_w, best_val, history, rnd, rule)
        rec = call_llm(prompt, tag=f"{'' if rule else 'norule/'}arm3_seed{seed}_round{rnd:02d}")
        cost, wall, out_tok = cost + rec["cost_usd"], wall + rec["wall_s"], out_tok + rec["output_tokens"]

        h = {"round": rnd, "n_changed": 0, "val": None, "note": "", "error": ""}
        try:
            reply = extract_json(rec["text"])
            cand = apply_edits(best_w, reply["edits"])
            h["note"] = str(reply.get("note", ""))[:200]
            h["n_changed"] = int(sum((cand[k] != best_w[k]).sum() for k in cand))
            h["val"] = metrics(cand, splits.X_val, splits.y_val)  # the one forward pass
            if (h["val"]["loss"], -h["val"]["acc"]) < (best_val["loss"], -best_val["acc"]):
                best_w, best_val = cand, h["val"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            h["error"] = f"{type(e).__name__}: {e}"[:200]  # an invalid reply still uses up the round
        history.append(h)
        best_curve.append(best_val["loss"])

    return {"seed": seed, "weights": best_w, "val": best_val, "history": history,
            "best_curve": best_curve, "cost_usd": cost, "wall_s": wall, "output_tokens": out_tok,
            "n_changed_total": int(sum((best_w[k] != gd_run["weights"][k]).sum() for k in best_w))}


def best_at(run: dict, budget: int) -> float:
    """Best validation loss after `budget` forward passes."""
    return float(np.min(run["best_curve"][: budget + 1]))
