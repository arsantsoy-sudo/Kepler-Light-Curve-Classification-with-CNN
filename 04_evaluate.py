from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
RESULTS = ROOT / "results"
MODEL = ROOT / "models" / "best_cnn.keras"

CLASS_NAMES = ["candidate", "binary", "nontransit"]

X = np.load(PROCESSED / "X.npy").astype(np.float32)
y = np.load(PROCESSED / "y.npy").astype(np.int64)

with np.load(PROCESSED / "split_indices.npz") as split:
    test_idx = split["test_idx"]

X_test = X[test_idx]
y_test = y[test_idx]

model = tf.keras.models.load_model(MODEL)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

metrics = {
    "test_objects": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "macro_precision": float(
        precision_score(y_test, y_pred, average="macro", zero_division=0)
    ),
    "macro_recall": float(
        recall_score(y_test, y_pred, average="macro", zero_division=0)
    ),
    "macro_f1": float(
        f1_score(y_test, y_pred, average="macro", zero_division=0)
    ),
}

RESULTS.mkdir(exist_ok=True)

with open(RESULTS / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    normalize="true",
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    matrix,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    colorbar=False,
)

ax.set_title("Normalized confusion matrix — test set")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
plt.close()

print("\nEvaluation completed")
for name, value in metrics.items():
    print(f"{name}: {value}")