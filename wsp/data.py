"""Synthetic dataset in an invented domain, so no LLM can have seen it.

Task: predict whether a batch of (fictional) 'velmorite' crystals fails a stress
test. Ten true crystal properties are each measured three ways (_mean,
_peak, _spread) = 30 features. The label comes from a nonlinear rule on the
true properties plus noise. RULE states that rule and the measurement model
exactly (the classifier analogue of giving game rules / source code). With
rule=True the LLM gets it in the prompt; with rule=False it does not. The LLM
never sees any data.

Stratified 60/20/20 split, standardized with training-set statistics.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

N_ROWS = 569
GEN_SEED = 2026
PROPERTIES = ["lattice_density", "grain_spread", "impurity_ratio", "anneal_temp", "cooling_rate",
              "facet_count", "luminance", "hydration", "edge_curvature", "isotope_shift"]
FEATURE_NAMES = [f"{p}_{m}" for m in ("mean", "peak", "spread") for p in PROPERTIES]

TASK = """Task: predict whether a batch of velmorite crystals fails a stress test. Ten properties of each batch are measured three ways: _mean (average over samples), _peak (maximum over samples) and _spread (variability over samples). Label y = 1 means the batch **fails**, y = 0 means it passes; the model outputs P(y = 1)."""


RULE = """How the data is generated (exact):
- Each batch has 10 true properties, each independently standard normal (mean 0, sd 1).
- Failure logit = 1.4*impurity_ratio + 0.9*grain_spread - 1.1*lattice_density + 3.5*max(anneal_temp - 1, 0) + 2.0*impurity_ratio*cooling_rate - 1.3, using the TRUE properties.
- y = 1 (fail) if logit + 0.2*e > 0, where e is standard logistic noise. The other 5 properties do not affect the label.
- Measurements of true property z, each with its own independent standard normal noise n: _mean = z + 0.15*n; _peak = 0.9*z + 0.45*n + 1.5; _spread = 0.4*z + 0.9*n + 3.0.
- About a third of batches fail."""


def _generate():
    rng = np.random.default_rng(GEN_SEED)
    z = rng.standard_normal((N_ROWS, len(PROPERTIES)))  # true properties
    zi = {p: z[:, i] for i, p in enumerate(PROPERTIES)}

    # The rule (given to the LLM verbatim as RULE).
    logit = (1.4 * zi["impurity_ratio"] + 0.9 * zi["grain_spread"] - 1.1 * zi["lattice_density"]
             + 3.5 * np.maximum(zi["anneal_temp"] - 1.0, 0)
             + 2.0 * zi["impurity_ratio"] * zi["cooling_rate"]
             - 1.3)
    y = (logit + rng.logistic(size=N_ROWS) * 0.2 > 0).astype(np.float32)

    # Noisy measurements of the true properties.
    mean = z + 0.15 * rng.standard_normal(z.shape)
    peak = 0.9 * z + 0.45 * rng.standard_normal(z.shape) + 1.5
    spread = 0.4 * z + 0.90 * rng.standard_normal(z.shape) + 3.0
    X = np.hstack([mean, peak, spread]).astype(np.float32)
    return X, y


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    mean: np.ndarray
    std: np.ndarray


def load_splits(seed: int = 0) -> Splits:
    X, y = _generate()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    z = lambda a: ((a - mean) / std).astype(np.float32)
    return Splits(z(X_tr), y_tr, z(X_va), y_va, z(X_te), y_te, FEATURE_NAMES, mean, std)
