from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
CLASS_COUNT = 3


def load_data():
    X = np.load(PROCESSED / "X.npy").astype(np.float32)
    y = np.load(PROCESSED / "y.npy").astype(np.int64)

    with np.load(PROCESSED / "split_indices.npz") as split:
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(CLASS_COUNT, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_curve(train_values, val_values, title, ylabel, filename):
    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, "o-", label=f"Training {ylabel}")
    plt.plot(epochs, val_values, "o-", label=f"Validation {ylabel}")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=200)
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()
    model = build_model(X_train.shape[1:])
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                MODELS / "best_cnn.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        ],
        verbose=1,
    )

    save_curve(
        history.history["loss"],
        history.history["val_loss"],
        "Loss History Profiles",
        "Loss",
        "loss_curve.png",
    )

    save_curve(
        history.history["accuracy"],
        history.history["val_accuracy"],
        "Accuracy History Profiles",
        "Accuracy",
        "accuracy_curve.png",
    )

    print("\nTraining completed")
    print("Saved model: models/best_cnn.keras")
    print("Saved graphs: results/loss_curve.png, results/accuracy_curve.png")


if __name__ == "__main__":
    main()