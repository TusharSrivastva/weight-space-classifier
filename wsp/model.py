"""MLP 30 -> 16 (ReLU) -> 1 (sigmoid) as plain numpy weights, JSON I/O, metrics.

Layout matches sklearn's MLPClassifier (coefs_ / intercepts_), so GD-trained
weights drop straight in: W1 row i = input feature i, column j = hidden unit j.
"""
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

N_IN, N_HIDDEN, N_OUT = 30, 16, 1

SHAPES = {
    "W1": (N_IN, N_HIDDEN),
    "b1": (N_HIDDEN,),
    "W2": (N_HIDDEN, N_OUT),
    "b2": (N_OUT,),
}
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES.values())  # 513


# ---------- forward pass ----------

def predict_proba(w: dict, X: np.ndarray) -> np.ndarray:
    """P(y=1 | x) = sigmoid(relu(X @ W1 + b1) @ W2 + b2)."""
    h = np.maximum(X @ w["W1"] + w["b1"], 0.0)
    z = (h @ w["W2"] + w["b2"]).ravel()
    return 1.0 / (1.0 + np.exp(-z))


def metrics(w: dict, X: np.ndarray, y: np.ndarray) -> dict:
    p = predict_proba(w, X)
    pred = (p >= 0.5).astype(int)
    # AUC is undefined if all scores tie (e.g. all-zero weights); report 0.5.
    auc = roc_auc_score(y, p) if np.unique(p).size > 1 else 0.5
    return {
        "loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(auc),
    }


# ---------- temperature scaling ----------

def temperature_scale(w: dict, X_val: np.ndarray, y_val: np.ndarray) -> tuple[dict, float]:
    """Fit one temperature T on validation log loss; return (scaled weights, T).

    p = sigmoid(z / T), applied by dividing W2 and b2 by T, so the result is
    ordinary 513-weight MLP weights. Accuracy, F1 and AUC are unchanged; only
    log loss moves.
    """
    from scipy.optimize import minimize_scalar

    def scaled(t):
        return {**w, "W2": w["W2"] / t, "b2": w["b2"] / t}

    res = minimize_scalar(lambda log_t: metrics(scaled(np.exp(log_t)), X_val, y_val)["loss"],
                          bounds=(np.log(0.05), np.log(20)), method="bounded")
    t = float(np.exp(res.x))
    return scaled(t), t


# ---------- weights <-> sklearn / JSON ----------

def from_sklearn(clf) -> dict:
    return validate_weights({
        "W1": clf.coefs_[0], "b1": clf.intercepts_[0],
        "W2": clf.coefs_[1], "b2": clf.intercepts_[1],
    })


def validate_weights(w: dict) -> dict:
    """Coerce to float64 arrays and check names, shapes and finiteness.

    Used on every LLM reply, so it must reject anything malformed.
    """
    if set(w) != set(SHAPES):
        raise ValueError(f"expected keys {sorted(SHAPES)}, got {sorted(w)}")
    out = {}
    for k, shape in SHAPES.items():
        a = np.asarray(w[k], dtype=np.float64)
        if a.shape != shape:
            raise ValueError(f"{k}: expected shape {shape}, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{k}: contains NaN/inf")
        out[k] = a.copy()
    return out


def zero_weights() -> dict:
    return {k: np.zeros(s) for k, s in SHAPES.items()}


def weights_to_json(w: dict, decimals: int | None = 4) -> str:
    """Compact JSON for prompts (rounded to keep ~513 numbers short).

    decimals=None keeps full float64 precision (exact round-trip, for saving).
    """
    r = (lambda a: a) if decimals is None else (lambda a: np.round(a, decimals))
    return json.dumps({k: r(np.asarray(v)).tolist() for k, v in w.items()})


def weights_from_json(s: str) -> dict:
    return validate_weights(json.loads(s))


def count_changed(w_old: dict, w_new: dict, tol: float = 1e-6) -> int:
    return int(sum((np.abs(w_old[k] - w_new[k]) > tol).sum() for k in SHAPES))
