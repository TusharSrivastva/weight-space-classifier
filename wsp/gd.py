"""Arm 1: gradient descent baseline (sklearn MLPClassifier + Adam).

Trained one epoch at a time with partial_fit so early stopping uses OUR
validation split (sklearn's built-in early_stopping carves its own split out
of the training data instead). The best-val-loss weights are kept.
"""
import json
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from .model import N_HIDDEN, from_sklearn, metrics, validate_weights, weights_to_json

GD_DIR = Path("results/gd")


def train_gd(splits, seed: int, max_epochs: int = 500, patience: int = 20,
             lr: float = 1e-3, batch_size: int = 32, alpha: float = 1e-4) -> dict:
    clf = MLPClassifier(
        hidden_layer_sizes=(N_HIDDEN,), activation="relu", solver="adam",
        learning_rate_init=lr, batch_size=batch_size, alpha=alpha,
        random_state=seed,
    )
    y_tr = splits.y_train.astype(int)
    history, best, best_loss, bad_epochs = [], None, float("inf"), 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            clf.partial_fit(splits.X_train, y_tr, classes=[0, 1])
            w = from_sklearn(clf)
            tr = metrics(w, splits.X_train, splits.y_train)
            va = metrics(w, splits.X_val, splits.y_val)
            history.append({"epoch": epoch, "train": tr, "val": va})
            if va["loss"] < best_loss - 1e-6:
                best, best_loss, best_epoch, bad_epochs = w, va["loss"], epoch, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    return {
        "seed": seed,
        "weights": best,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train": metrics(best, splits.X_train, splits.y_train),
        "val": metrics(best, splits.X_val, splits.y_val),
        "history": history,
        "hparams": dict(max_epochs=max_epochs, patience=patience, lr=lr,
                        batch_size=batch_size, alpha=alpha),
    }


def save_gd(run: dict, out_dir: Path = GD_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{run['seed']}.json"
    payload = {k: v for k, v in run.items() if k != "weights"}
    payload["weights"] = json.loads(weights_to_json(run["weights"], decimals=None))
    path.write_text(json.dumps(payload))
    return path


def load_gd(seed: int, out_dir: Path = GD_DIR) -> dict:
    run = json.loads((out_dir / f"seed{seed}.json").read_text())
    run["weights"] = validate_weights(run["weights"])
    return run
