"""Arm 2: the LLM writes all 513 weights from scratch in one reply, with no data.

It gets the task description, feature names and the exact data-generating rule
(wsp.data.RULE), but no data. rule=False is the control: the same prompt without
the rule. Its calls are logged under logs/norule/.
"""
from .data import RULE, TASK
from .llm import call_llm, extract_json
from .model import metrics, validate_weights


def build_prompt(splits, rule: bool = True) -> str:
    feats = "\n".join(f"{i:2d}. {n}" for i, n in enumerate(splits.feature_names))
    given = "You get the exact rule that generated the data, but no data and no feedback" if rule else "You get no training data and no feedback"
    rule_block = f"\n\n{RULE}" if rule else ""
    return f"""Write the weights of a small neural network classifier by hand. {given}; your reply is evaluated as-is on held-out data.

{TASK}

Inputs: 30 features in this order, each z-scored with training-set statistics (you are not given them):
{feats}{rule_block}

Architecture: h = relu(x @ W1 + b1), p = sigmoid(h @ W2 + b2)
Shapes: W1 30x16 (row i = feature i), b1 16, W2 16x1, b2 1. Threshold p >= 0.5. Scored on accuracy, F1 and ROC AUC.

Briefly explain your approach, then give the weights as one ```json block with keys "W1", "b1", "W2", "b2" (nested lists of plain numbers)."""


def run_arm2(splits, trial: int, rule: bool = True) -> dict:
    rec = call_llm(build_prompt(splits, rule), tag=f"{'' if rule else 'norule/'}arm2_trial{trial}")
    w = validate_weights(extract_json(rec["text"]))
    return {"trial": trial, "weights": w,
            "val": metrics(w, splits.X_val, splits.y_val),
            **{k: rec[k] for k in ("input_tokens", "output_tokens", "cost_usd", "wall_s")}}
